"""WanModelHandle — one handle per (mode, generation, size) tuple.

Lifecycle:
  1. ensure_loaded()   — lazy-build the pipeline from the mounted path, attach LoRA if available
  2. configure_preset()— toggle Fast/Quality via set_adapters() / disable_lora(); return inference kwargs
  3. generate()        — call the pipeline (called from within @spaces.GPU)
  4. unload_to_cpu()   — move transformers to CPU + empty_cache() when switching modes

Only ONE handle's pipeline lives on GPU at a time (managed by app.py orchestrator).
"""
from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from pipelines.preset import Preset, PresetKwargs, resolve
from pipelines.registry import BY_KEY, ModelCard
from utils.backend import detect


# Path on the deployed Space where mounted volumes appear:
#   /models/wan2.1-t2v-14b/, /models/wan2.2-t2v-a14b/, etc.
# Locally (MPS dev), fall back to standard HF cache.
SPACE_MOUNT_ROOT = Path(os.getenv("WAN_STUDIO_MOUNT_ROOT", "/models"))

# Local-converted bf16 checkpoints live here (outside the HF cache so the
# "evict non-Wan cache" disk step can never touch them). One subdir per slug,
# e.g. ~/wan-bf16/wan2.1-vace-14b/{transformer,scheduler,...}. Option-1 of the
# push ruling: convert locally, load locally, defer the HF push.
LOCAL_BF16_ROOT = Path(os.getenv("WAN_STUDIO_LOCAL_BF16_ROOT",
                                 str(Path.home() / "wan-bf16")))

# Bundled metadata (configs, tokenizer, .safetensors.index.json) shipped with
# the Space repo itself. Working around an HF Volume bug where small JSON
# files mount truncated — transformer/config.json shows 290B instead of 495B.
# We ship correct copies of every small file here (~21 MB for tokenizer).
META_ROOT = Path(__file__).parent.parent / "models_meta"

# Stitched dirs combine mounted weights (read-only mount, no disk cost) with
# bundled metadata (writable /tmp copy). One per slug. Built lazily on first
# call to stitch_local_dir. Symlinks for big files = ~0 disk cost.
STITCH_ROOT = Path("/tmp/wan-stitched")

# File extensions classified as "weights" — symlinked from the mount.
# Everything else (json/txt/md/etc.) is copied from META_ROOT or skipped.
WEIGHT_EXTS = (".safetensors", ".bin", ".pth", ".pt", ".onnx", ".gguf", ".ckpt")


def _slug_for(card: ModelCard) -> str:
    """Compute the directory slug used for the duplicated mirror.

    Convention: underscores → dashes, dots preserved. Examples:
      wan2.1_t2v_14b   → wan2.1-t2v-14b   (mirror: techfreakworm/wan2.1-t2v-14b)
      wan2.2_i2v_a14b  → wan2.2-i2v-a14b
    This MUST match Volume(mount_path=...) in scripts/create_space.py.
    """
    return card.key.replace("_", "-")


SHARED_MIRROR_REPO = "techfreakworm/wan-shared-encoders"


def _build_stitch(repo: str, mount: Path, name: str) -> str | None:
    """Stitch a mounted checkpoint into a /tmp dir that from_pretrained can read.

    Returns the stitched dir, or None if the mount is absent (local MPS dev →
    callers fall back to the repo id and let from_pretrained download).

    Idempotent via a marker file. The HF Volume mount TRUNCATES sub-~1 KB JSON
    config files (verified on the live Space: vae/config.json arrives invalid),
    so the small text files are fetched fresh from the mirror REPO (always
    correct, tiny, xet-accelerated) while the big weight files are SYMLINKED
    from the read-only mount (zero container-disk cost — the whole point of the
    volume mount). This makes per-model bundled `models_meta/` optional.
    """
    if not mount.exists():
        return None

    stitched = STITCH_ROOT / name
    marker = stitched / ".wan_studio_stitched"
    if marker.exists():
        return str(stitched)

    stitched.mkdir(parents=True, exist_ok=True)

    # Small files (configs, tokenizer, *.index.json) from the repo — bypasses
    # the mount truncation bug.
    from huggingface_hub import hf_hub_download, list_repo_files
    for f in list_repo_files(repo):
        if f.startswith(".") or Path(f).suffix.lower() in WEIGHT_EXTS:
            continue
        try:
            src = hf_hub_download(repo, f)
        except Exception as e:  # noqa: BLE001 — best-effort per small file
            print(f"  [stitch] skip {repo}:{f} — {type(e).__name__}: {e}", flush=True)
            continue
        dst = stitched / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    # Big weight files symlinked from the mount.
    for src in mount.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in WEIGHT_EXTS:
            continue
        rel = src.relative_to(mount)
        dst = stitched / rel
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)

    marker.touch()
    return str(stitched)


def stitch_local_dir(card: ModelCard) -> str | None:
    """Stitch a model checkpoint mount (small files from card.mirror_repo)."""
    slug = _slug_for(card)
    return _build_stitch(card.mirror_repo, SPACE_MOUNT_ROOT / slug, slug)


def stitch_shared_dir() -> str | None:
    """Stitch the shared-encoders mount (UMT5/VAE/CLIP) — configs from the repo,
    weights symlinked from the mount. Used by pipelines.shared on the Space."""
    return _build_stitch(SHARED_MIRROR_REPO, SPACE_MOUNT_ROOT / "wan-shared-encoders",
                         "wan-shared-encoders")


def _mount_path(card: ModelCard) -> str:
    """Resolve where the checkpoint lives for from_pretrained().

    Priority:
      0. Local-converted bf16 dir (LOCAL_BF16_ROOT/<slug>) — the Option-1 path:
         weights converted locally, OUTSIDE the HF cache (eviction-safe), never
         pushed. Preferred locally so missing-mirror modes work with no push and
         a local save_pretrained writes complete configs (no stitch hack needed).
      1. Stitched local dir (mount + bundled metadata) — zero disk on ZeroGPU.
      2. ZeroGPU + no mount → RAISE (never silently download fp32 into /tmp).
      3. Local + no mount → the bf16 mirror repo (downloads once to persistent cache).
    """
    if os.getenv("SPACES_ZERO_GPU") is None:
        local = LOCAL_BF16_ROOT / _slug_for(card)
        if (local / "transformer").is_dir() or (local / "model_index.json").exists():
            return str(local)
    stitched = stitch_local_dir(card)
    if stitched:
        # ZeroGPU: copy the stitched checkpoint (symlinks → real bytes) onto the
        # local ephemeral disk ONCE, then load from there. The HF Volume mount
        # is slow (~76s/shard); a safetensors mmap stays backed by the mount, so
        # `pipe.to('cuda')` later page-faults all 28 GB from the mount and blows
        # the GPU duration budget. Backing the weights with local NVMe makes the
        # host→GPU copy fault from fast disk instead. See pipelines/trace.py.
        if os.getenv("SPACES_ZERO_GPU") is not None and os.getenv("WAN_STUDIO_TIER2", "1") == "1":
            try:
                return tier2_warm_copy(_slug_for(card), stitched)
            except Exception as e:  # never let the fast-path break loading
                print(f"=== TIER2 copy failed for {_slug_for(card)} ({e}); using mount ===", flush=True)
                return stitched
        return stitched
    if os.getenv("SPACES_ZERO_GPU") is not None:
        raise RuntimeError(
            f"mount /models/{_slug_for(card)} missing — check create_space.py manifest"
        )
    return card.mirror_repo


# Tier-2 warm cache root. Hot model shards are copied here (real local bytes)
# once, so repeat reads and forked workers avoid slow HF-mount page-faults.
TIER2_ROOT = Path("/tmp/wan-hot")


def tier2_warm_copy(slug: str, src_dir: str) -> str:
    """Copy a stitched/mounted checkpoint dir into local /tmp once.

    The first read of a >10 GB model over the HF mount is slow (network
    page-faults); copying to local disk makes repeat reads (and forked
    workers) fast. Idempotent via a marker. Caller is responsible for
    LRU-evicting prior hot copies to stay under the 150 GB disk cap.
    """
    import time as _t
    src = Path(src_dir)
    dst = TIER2_ROOT / slug
    marker = dst / ".wan_hot_done"
    if marker.exists():
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = _t.time()
    nbytes = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(f, target)  # resolves symlinks → real local bytes
            try:
                nbytes += target.stat().st_size
            except OSError:
                pass
    marker.touch()
    dt = max(0.001, _t.time() - t0)
    gb = nbytes / 1e9
    print(f"=== TIER2 warm-copy {slug}: {gb:.1f} GB mount→local in {dt:.0f}s "
          f"({gb / dt * 1000:.0f} MB/s) → {dst} ===", flush=True)
    return str(dst)


def tier2_evict(keep_slug: str) -> None:
    """Remove every hot copy except keep_slug (LRU bound = 1 model)."""
    if not TIER2_ROOT.exists():
        return
    for child in TIER2_ROOT.iterdir():
        if child.is_dir() and child.name != keep_slug:
            shutil.rmtree(child, ignore_errors=True)


# Path to the mounted Lightning LoRA bundle on ZeroGPU.
LIGHTNING_MIRROR_MOUNT = SPACE_MOUNT_ROOT / "wan-lightning-loras"
LIGHTNING_MIRROR_REPO = "techfreakworm/wan-lightning-loras"


def _lora_repo_for(card: ModelCard) -> str:
    """Resolve where to load Lightning LoRA weights from.

    On ZeroGPU: read from the mounted /models/wan-lightning-loras consolidated mirror.
    Locally: the consolidated HF mirror repo holds the canonical Lightning weights
    in a subdir-per-model layout (weight_name = "<slug>/lightning*.safetensors").
    Use it whenever the card's weight path is one of those consolidated subpaths;
    otherwise fall back to the card's upstream repo (e.g. FLF2V reuses a Kijai file
    that isn't in the consolidated mirror).
    """
    backend = detect()
    if backend.is_zerogpu and LIGHTNING_MIRROR_MOUNT.exists():
        return str(LIGHTNING_MIRROR_MOUNT)
    slug = _slug_for(card)
    if (card.lightning_high_lora or "").startswith(slug + "/"):
        return LIGHTNING_MIRROR_REPO
    assert card.lightning_lora_repo, f"{card.key} missing lightning_lora_repo upstream fallback"
    return card.lightning_lora_repo


def diffusers_step_callback(on_step, total_steps):
    """Adapt a simple `on_step(done:int, total:int)` into a diffusers
    `callback_on_step_end` so the Gradio progress bar advances per denoise step.

    Returns None if no callback — the caller then omits the kwarg. Never lets a
    progress-update error break generation."""
    if on_step is None:
        return None

    def _cb(pipe, step, timestep, callback_kwargs):
        try:
            on_step(step + 1, total_steps)
        except Exception:
            pass
        return callback_kwargs

    return _cb


class WanModelHandle:
    """Wraps a single (mode, generation, size) combo.

    Concrete pipeline construction is delegated to subclasses by mode
    (T2VHandle, I2VHandle, etc.) via `_build_pipeline()`.
    """

    def __init__(self, card: ModelCard):
        self.card = card
        self.pipe: Any = None  # set in ensure_loaded
        self.lora_loaded: bool = False
        self.current_preset: Preset | None = None
        self.cuda_attached: bool = False  # set by ensure_cuda_attached (inside @spaces.GPU)
        self.offload_enabled: bool = False  # accelerate model_cpu_offload hooks active

    @classmethod
    def for_key(cls, key: str) -> "WanModelHandle":
        """Look up a card by key and return a fresh handle."""
        if key not in BY_KEY:
            raise KeyError(f"Unknown model key: {key!r}")
        return cls(BY_KEY[key])

    def ensure_loaded(self) -> None:
        """Build the pipeline + attach Lightning LoRA. Loads into CPU RAM only.

        Safe to call in main process at app startup (before any @spaces.GPU
        worker exists). The CUDA dance happens lazily in ensure_cuda_attached.
        """
        if self.pipe is not None:
            from pipelines.trace import trace
            trace(f"ensure_loaded[{self.card.key}] CACHE-HIT (already built)")
            return
        from pipelines.trace import trace
        trace(f"ensure_loaded[{self.card.key}] BUILD start (disk→CPU)")
        self.pipe = self._build_pipeline()
        trace(f"ensure_loaded[{self.card.key}] pipeline built")
        self._configure_scheduler()
        if self.card.lightning_available:
            self._load_lightning_lora()
            self.lora_loaded = True
        trace(f"ensure_loaded[{self.card.key}] BUILD done (LoRA={self.lora_loaded})")

    def _needs_offload(self) -> bool:
        """Decide between model-cpu-offload (low peak VRAM, slower per-step from
        component streaming) and plain resident `.to('cuda')` (fast, but the
        whole pipe must fit the 48 GB slice + the fp32 activation spike).

        - MoE (two 14B experts, ~56 GB) → always offload.
        - I2V/FLF2V/Animate (`requires_image_encoder`): 28 GB transformer + ~13 GB
          shared UMT5/CLIP ≈ 42 GB resident leaves no room for activations → OOM
          (the NVML `CUDACachingAllocator` assert on the MIG slice) → offload.
        - T2V/V2V single-14B: no CLIP image encoder, so ~28 GB transformer + ~11 GB
          UMT5 ≈ 39 GB — fits resident at 480p with headroom for activations, and
          resident is much faster (no per-step 28 GB stream), which is what makes
          14B-T2V *Quality* (50 steps) fit the GPU window at all. Stays resident.

        NOTE: at 720p the activation spike is larger; 14B-T2V there may still need
        offload — handled by the per-call resolution gate, not this card-level
        default."""
        return self.card.is_moe or self.card.requires_image_encoder

    def ensure_cuda_attached(self) -> None:
        """Move the loaded pipeline to GPU. MUST be called from inside @spaces.GPU.

        Large models (`_needs_offload`) use accelerate's model_cpu_offload hooks
        so only the active component sits on the 48 GB `large` slice. Small ones
        (1.3B / 5B) just `.to(device)`. Idempotent — re-entry from a warm worker
        is a no-op.
        """
        from pipelines.trace import trace
        if self.cuda_attached:
            trace(f"ensure_cuda_attached[{self.card.key}] already on GPU")
            return
        import torch
        backend = detect()
        if self._needs_offload() and torch.cuda.is_available():
            trace(f"ensure_cuda_attached[{self.card.key}] enable_model_cpu_offload start")
            self.pipe.enable_model_cpu_offload()
            self.offload_enabled = True
            trace(f"ensure_cuda_attached[{self.card.key}] offload hooks installed")
        else:
            trace(f"ensure_cuda_attached[{self.card.key}] .to({backend.device}) start (host→GPU)")
            self.pipe.to(backend.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            trace(f"ensure_cuda_attached[{self.card.key}] on GPU")
        self.cuda_attached = True

    def configure_preset(self, preset: Preset) -> PresetKwargs:
        """Apply Fast/Quality preset; return inference kwargs."""
        self.ensure_loaded()
        self.ensure_cuda_attached()
        kwargs = resolve(self.card, preset)

        # Key the scheduler shift to the ACTIVE preset (Fast/Lightning runs at its
        # distilled shift ≈5, Quality at the card's shift). ensure_loaded() set a
        # default; this refines it now that the preset is known.
        self._configure_scheduler(shift=kwargs.flow_shift)

        if not self.lora_loaded:
            self.current_preset = kwargs.effective_preset
            return kwargs

        if kwargs.effective_preset == "fast":
            # MUST enable_lora() BEFORE set_adapters(): if a prior Quality run called
            # disable_lora(), set_adapters() alone does NOT re-enable the turned-off
            # layers — the Lightning LoRA silently has ZERO effect, and a 4-step/cfg1.0
            # run with a dead distillation LoRA decodes to under-denoised garbage (reads
            # as "Lightning broken"). enable_lora() is a no-op on first/fresh load.
            self.pipe.enable_lora()
            if self.card.is_moe:
                self.pipe.set_adapters(["lightning_high", "lightning_low"], [1.0, 1.0])
            else:
                self.pipe.set_adapters(["lightning"], [1.0])
        else:
            self.pipe.disable_lora()

        self.current_preset = kwargs.effective_preset
        return kwargs

    def unload_to_cpu(self) -> None:
        """Move transformers off GPU but keep them resident in CPU RAM.

        Called when switching the GPU-active mode. The pipeline stays built and
        LoRA-attached (warm in CPU) so re-acquiring this key is instant — only
        the GPU residency moves. Resetting `cuda_attached` is essential: it lets
        a later `ensure_cuda_attached()` re-move this same warm handle back onto
        the GPU when the user swaps back (without it, the swap-back would skip
        the `.to(device)` and try to run a CPU-resident pipe on CUDA inputs)."""
        if self.pipe is None:
            return
        if self.offload_enabled:
            # Tear down the accelerate offload hooks (also from the shared
            # encoders this pipe borrowed) so the next pipe gets clean modules
            # and the components return to CPU. A re-acquire re-installs them.
            try:
                self.pipe.maybe_free_model_hooks()
            except Exception:
                pass
            self.offload_enabled = False
        else:
            if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
                self.pipe.transformer.to("cpu")
            if hasattr(self.pipe, "transformer_2") and self.pipe.transformer_2 is not None:
                self.pipe.transformer_2.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cuda_attached = False

    # --- Mode-specific overrides (implemented in subclasses) ---

    def _build_pipeline(self) -> Any:
        raise NotImplementedError("Subclass must implement _build_pipeline()")

    def _configure_scheduler(self, shift: float | None = None) -> None:
        """Set FlowMatchEulerDiscreteScheduler with the given shift (defaults to the
        card's QUALITY flow_shift). configure_preset() re-calls this with the ACTIVE
        preset's shift so Fast/Lightning runs at its distilled shift, not quality's.

        ROOT CAUSE (MPS oversaturation/neon, this whole campaign): diffusers'
        UniPCMultistepScheduler — its multistep CORRECTOR diverges on MPS (it integrates
        the history of DiT forward outputs with higher-order coefficients; the forward
        itself is MPS==CPU clean). Swapping to a plain Euler flow-match integrator (the
        diffusers equivalent of ComfyUI's euler + ModelSamplingDiscreteFlow, which is
        clean) fixes it. Decisive A/B: same 1.3B / @21 / g5 that neoned all session →
        clean photoreal, per-frame sat ~0.52 flat (vs UniPC ~0.85). flow_shift maps to
        FlowMatchEuler's `shift` (same time_snr_shift formula).
        """
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            shift=self.card.flow_shift if shift is None else shift,
        )

    def _load_lightning_lora(self) -> None:
        """Attach Lightning LoRA(s) to the transformer(s).

        Wan 2.1 (single transformer): one LoRA call.
        Wan 2.2 MoE: two LoRA calls — HIGH onto transformer, LOW onto transformer_2 with
        `load_into_transformer_2=True`.

        Resolves the LoRA source via `_lora_repo_for()` so the same code path works
        for both ZeroGPU (mounted mirror) and local dev (upstream hub).
        """
        if not self.card.lightning_available:
            return

        lora_repo = _lora_repo_for(self.card)

        # HIGH-noise / single-transformer LoRA
        self.pipe.load_lora_weights(
            lora_repo,
            weight_name=self.card.lightning_high_lora,
            adapter_name="lightning_high" if self.card.is_moe else "lightning",
        )

        if self.card.is_moe:
            assert self.card.lightning_low_lora, "MoE card missing low-noise LoRA path"
            self.pipe.load_lora_weights(
                lora_repo,
                weight_name=self.card.lightning_low_lora,
                adapter_name="lightning_low",
                load_into_transformer_2=True,  # diffusers PR #12074
            )


class ModelRegistry:
    """At-most-one GPU-attached handle, up to `max_warm` handles kept warm in CPU.

    Two distinct residencies, decoupled:
      • GPU-attached — exactly one (`warm_key`). The 48 GB slice holds one
        transformer family at a time; switching keys moves the prior one off
        the GPU via `unload_to_cpu()` (GPU→CPU, frees VRAM) but leaves it built
        and LoRA-attached.
      • CPU-warm — up to `max_warm` handles stay resident in CPU RAM (LRU). A
        re-acquire of any warm key is a cache hit, so it never re-pays the slow
        disk→CPU shard load on the GPU clock. Only when the warm set exceeds
        `max_warm` is the least-recently-used handle fully freed (pipe=None +
        gc + tier-2 disk evict).

    This is what makes the cross-call warm-swap work on ZeroGPU: the served
    checkpoints are preloaded to CPU at startup (app.py), every fork inherits
    them copy-on-write, and a T2V↔I2V swap only moves GPU residency — no cold
    load lands inside a `@spaces.GPU` window.

    factory(key) -> WanModelHandle builds a fresh handle for a registry key
    (injected so tests can stub it; production passes the HANDLER_REGISTRY
    builder).
    """

    def __init__(self, factory, max_warm: int | None = None):
        self._factory = factory
        self._handles: dict[str, WanModelHandle] = {}
        self.warm_key: str | None = None
        if max_warm is None:
            max_warm = int(os.getenv("WAN_STUDIO_MAX_WARM", "2"))
        self.max_warm = max(1, max_warm)
        # MRU-ordered key list (most-recently-used last) for CPU-warm eviction.
        self._lru: list[str] = []

    def _touch(self, key: str) -> None:
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def _evict_cpu_overflow(self) -> None:
        """Free least-recently-used warm handles beyond `max_warm` (never the
        GPU-attached `warm_key`)."""
        while len(self._handles) > self.max_warm:
            victim = next((k for k in self._lru if k != self.warm_key), None)
            if victim is None:
                break
            handle = self._handles.pop(victim, None)
            self._lru.remove(victim)
            if handle is not None:
                handle.unload_to_cpu()
                handle.pipe = None
            gc.collect()
            # Drop tier-2 hot copies for every checkpoint no longer warm.
            self._prune_tier2()

    def _prune_tier2(self) -> None:
        # Always keep the shared-encoders hot copy — it's injected into every
        # pipe, not owned by any single model handle.
        keep = {_slug_for(h.card) for h in self._handles.values()} | {"wan-shared-encoders"}
        if not TIER2_ROOT.exists():
            return
        for child in TIER2_ROOT.iterdir():
            if child.is_dir() and child.name not in keep:
                shutil.rmtree(child, ignore_errors=True)

    def acquire(self, key: str) -> WanModelHandle:
        from pipelines.trace import trace
        if key not in BY_KEY:
            raise KeyError(f"Unknown model key: {key!r}")
        trace(f"acquire[{key}] warm_key={self.warm_key} warm_set={list(self._handles)}")
        if self.warm_key == key and key in self._handles:
            self._touch(key)
            trace(f"acquire[{key}] FULL WARM HIT (on-GPU resident)")
            return self._handles[key]
        # Switching GPU residency: move the prior attached handle off the GPU,
        # but keep it warm in CPU (instant swap-back). It is freed only later if
        # it falls out of the LRU window.
        if self.warm_key is not None and self.warm_key != key:
            prev = self._handles.get(self.warm_key)
            if prev is not None:
                prev.unload_to_cpu()
        handle = self._handles.get(key) or self._factory(key)
        handle.ensure_loaded()
        self._handles[key] = handle
        self.warm_key = key
        self._touch(key)
        self._evict_cpu_overflow()
        return handle
