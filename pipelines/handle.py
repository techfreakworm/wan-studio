"""WanModelHandle — one handle per (mode, generation, size) tuple.

Lifecycle:
  1. ensure_loaded()   — lazy-build the pipeline from the mounted path, attach LoRA if available
  2. configure_preset()— toggle Fast/Quality via set_adapters() / disable_lora(); return inference kwargs
  3. generate()        — call the pipeline (called from within @spaces.GPU)
  4. unload_to_cpu()   — move transformers to CPU + empty_cache() when switching modes

Only ONE handle's pipeline lives on GPU at a time (managed by app.py orchestrator).
"""
from __future__ import annotations

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


def stitch_local_dir(card: ModelCard) -> str | None:
    """Build a stitched local dir = mounted weights + bundled JSONs.

    Returns the stitched dir path, or None if either the mount or the
    bundled metadata is missing (e.g. local MPS dev — no /models mount).

    Idempotent: a marker file guards against re-stitching. Once stitched,
    subsequent calls return the path immediately.

    Big binary files (`.safetensors` etc.) are SYMLINKED from the mount
    so they cost zero disk on the container's writable layer (the entire
    point of using space_volumes for model weights). Small text files
    (JSONs, tokenizer files) are COPIED from `models_meta/<slug>/` because
    the volume mount serves truncated copies of them.
    """
    slug = _slug_for(card)
    mount = SPACE_MOUNT_ROOT / slug
    meta = META_ROOT / slug
    stitched = STITCH_ROOT / slug

    if not mount.exists() or not meta.exists():
        return None

    marker = stitched / ".wan_studio_stitched"
    if marker.exists():
        return str(stitched)

    stitched.mkdir(parents=True, exist_ok=True)

    # Copy bundled small files first (configs, tokenizer, safetensors.index.json).
    for src in meta.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(meta)
        dst = stitched / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    # Symlink large weight files from the mount.
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


def _mount_path(card: ModelCard) -> str:
    """Resolve where the checkpoint lives for from_pretrained().

    Priority:
      1. Stitched local dir (mount + bundled metadata) — zero disk cost
         on ZeroGPU, big files served via the read-only volume mount.
      2. Upstream mirror repo ID — local MPS dev or when stitching fails.
    """
    stitched = stitch_local_dir(card)
    if stitched:
        return stitched
    return card.repo


# Path to the mounted Lightning LoRA bundle on ZeroGPU.
LIGHTNING_MIRROR_MOUNT = SPACE_MOUNT_ROOT / "wan-lightning-loras"
LIGHTNING_MIRROR_REPO = "techfreakworm/wan-lightning-loras"


def _lora_repo_for(card: ModelCard) -> str:
    """Resolve where to load Lightning LoRA weights from.

    On ZeroGPU: read from the mounted /models/wan-lightning-loras consolidated mirror.
    Locally: fall back to whatever `card.lightning_lora_repo` points at upstream.
    """
    backend = detect()
    if backend.is_zerogpu and LIGHTNING_MIRROR_MOUNT.exists():
        return str(LIGHTNING_MIRROR_MOUNT)
    assert card.lightning_lora_repo, f"{card.key} missing lightning_lora_repo upstream fallback"
    return card.lightning_lora_repo


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
            return
        self.pipe = self._build_pipeline()
        self._configure_scheduler()
        if self.card.lightning_available:
            self._load_lightning_lora()
            self.lora_loaded = True

    def ensure_cuda_attached(self) -> None:
        """Move the loaded pipeline to GPU. MUST be called from inside @spaces.GPU.

        On MoE cards we use accelerate's model_cpu_offload hooks so the two
        14B transformers fit on the 48 GB `large` slice. On single-transformer
        cards we just `.to(device)`. Idempotent — re-entry from a warm worker
        is a no-op.
        """
        if self.cuda_attached:
            return
        import torch
        backend = detect()
        if self.card.is_moe and torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(backend.device)
        self.cuda_attached = True

    def configure_preset(self, preset: Preset) -> PresetKwargs:
        """Apply Fast/Quality preset; return inference kwargs."""
        self.ensure_loaded()
        self.ensure_cuda_attached()
        kwargs = resolve(self.card, preset)

        if not self.lora_loaded:
            self.current_preset = kwargs.effective_preset
            return kwargs

        if kwargs.effective_preset == "fast":
            if self.card.is_moe:
                self.pipe.set_adapters(["lightning_high", "lightning_low"], [1.0, 1.0])
            else:
                self.pipe.set_adapters(["lightning"], [1.0])
        else:
            self.pipe.disable_lora()

        self.current_preset = kwargs.effective_preset
        return kwargs

    def unload_to_cpu(self) -> None:
        """Move transformers off GPU. Called when switching active mode."""
        if self.pipe is None:
            return
        if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            self.pipe.transformer.to("cpu")
        if hasattr(self.pipe, "transformer_2") and self.pipe.transformer_2 is not None:
            self.pipe.transformer_2.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Mode-specific overrides (implemented in subclasses) ---

    def _build_pipeline(self) -> Any:
        raise NotImplementedError("Subclass must implement _build_pipeline()")

    def _configure_scheduler(self) -> None:
        """Set UniPCMultistepScheduler with the mode's flow_shift. Override if needed."""
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            flow_shift=self.card.flow_shift,
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
