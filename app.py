"""Wan Studio — Gradio entry point (design v2/2: Linear-inspired).

Refined dev-tool dark: warm near-black surface, Geist typography, hairline
borders, restrained accent.

T2V + I2V Generate buttons are wired to the real `pipelines.{t2v,i2v}` handles
(Wave F+G). All other tabs (TI2V, FLF2V, V2V, VACE, S2V, Animate) still fire a
no-op toast until their pipelines are wired in later waves.
"""
from __future__ import annotations

# ── HF cache redirect (must precede every huggingface_hub touch) ────────
# /home/user/.cache/ on ZeroGPU isn't owned by the runtime user (preload
# daemon owns it) so xet_get / snapshot_download permission-deny on writes.
# /tmp/hf_cache is world-writable. Space-level env vars are also set via
# api.add_space_variable for redundancy — these defaults are a belt+braces.
import os as _os
_os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache")
_os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
_os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    _os.makedirs("/tmp/hf_cache", exist_ok=True)
except (PermissionError, OSError):
    pass

# IMPORTANT: import `spaces` BEFORE any CUDA-related package (torch, diffusers,
# transformers, peft) so the ZeroGPU runtime can fork CUDA correctly.  Once
# torch has touched CUDA, `import spaces` raises:
#   RuntimeError: CUDA has been initialized before importing the `spaces` package.
# `pipelines` and `utils` both transitively import torch, so this MUST stay
# at the top of the entry-point file.  Outside ZeroGPU the import is a no-op.
try:
    import spaces  # noqa: F401
except ImportError:
    pass

import os
from pathlib import Path as _Path

import gradio as gr

# ── Boot probe: assert every expected /models/<slug> mount is present ────
# create_space.py provisions the mount set from provisioning.manifest; if the
# Space's actual volumes drift from that manifest we want to fail loud at boot
# rather than 404 mid-generation. MOUNT_ROOT is "/" in production and is
# monkeypatched to a tmp dir in tests.
MOUNT_ROOT = _Path("/")


def assert_expected_mounts() -> None:
    """Fail loud at boot if any expected /models/<slug> mount is missing."""
    from provisioning.manifest import expected_mount_paths
    missing = [p for p in expected_mount_paths()
               if not (MOUNT_ROOT / p.lstrip("/")).exists()]
    if missing:
        raise RuntimeError(f"missing mount(s): {missing} — check create_space.py manifest")

from pipelines import modes_in
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.handle import ModelRegistry
from pipelines.registry import BY_KEY
from ui import build_all_tabs, build_header, build_sidebar, MODE_PILLS
from utils import detect


# ── Single warm-handle LRU registry (replaces the per-mode handle dicts) ──
# `_build_handle(key)` resolves the right HandlerSpec for a registry key and
# returns a fresh handle via its `handle_cls.for_key(key)`. `ModelRegistry`
# keeps at-most-one warmed handle and evicts the prior one on a key switch.
def _build_handle(key: str):
    """Map a registry key → a fresh handle via the owning HandlerSpec.

    Look up the card by `key` in `BY_KEY`, then return the handle from the spec
    whose `mode` equals `card.mode` (e.g. `wan2.1_i2v_14b_480p` resolves to the
    spec with `mode == "i2v"`). Raises `KeyError` if no spec matches the mode.
    """
    card = BY_KEY[key]
    for spec in HANDLER_REGISTRY.values():
        if spec.mode == card.mode:
            return spec.handle_cls.for_key(key)
    raise KeyError(f"No HandlerSpec registered for key {key!r} (mode {card.mode!r})")


REGISTRY = ModelRegistry(factory=_build_handle)


# ── Startup probe: log filesystem permissions for cache paths ───────────
def _probe_filesystem() -> None:
    if os.getenv("SPACES_ZERO_GPU") is None:
        return
    print(f"=== FS PROBE: uid={os.getuid()} gid={os.getgid()} ===", flush=True)
    paths = ["/", "/tmp", "/home/user", "/home/user/.cache",
             "/home/user/.cache/huggingface", "/home/user/app",
             "/tmp/hf_cache", "/data", "/models",
             "/models/wan-lightning-loras"]
    for p in paths:
        try:
            st = os.stat(p)
            owner = f"uid={st.st_uid} gid={st.st_gid} mode={oct(st.st_mode)[-3:]}"
            try:
                tmp = os.path.join(p, f".test_write_{os.getpid()}")
                with open(tmp, "w") as f:
                    f.write("ok")
                os.unlink(tmp)
                wr = "WRITABLE"
            except (PermissionError, OSError) as e:
                wr = f"NO-WRITE ({type(e).__name__})"
            print(f"  {p:<45} {owner:<45} {wr}", flush=True)
        except FileNotFoundError:
            print(f"  {p:<45} NOT-FOUND", flush=True)
    print("=== END FS PROBE ===", flush=True)
    # FS logging is done — now fail loud if any expected mount is absent.
    assert_expected_mounts()


# ── Stitch the Wan 2.2 T2V dir at startup ──────────────────────────────
# Combines the read-only /models/<slug>/ volume mount (weights) with
# bundled models_meta/<slug>/ JSONs (correct configs — mount truncates
# small text files). Result is a /tmp/wan-stitched/<slug>/ dir that
# from_pretrained can read directly. Zero downloads, zero container disk
# for weights — they're symlinks pointing into the read-only mount.
def _stitch_default_model() -> None:
    if os.getenv("SPACES_ZERO_GPU") is None:
        return
    try:
        from pipelines.handle import stitch_local_dir
        from pipelines.registry import BY_KEY
        from provisioning.manifest import resolve_available_key
        import time as _t
        key = resolve_available_key("wan2.2_t2v_a14b", "t2v")
        print(f"=== STITCH {key} ===", flush=True)
        t0 = _t.time()
        path = stitch_local_dir(BY_KEY[key])
        if path:
            print(f"=== STITCH done in {int(_t.time()-t0)}s → {path} ===", flush=True)
        else:
            print(f"=== STITCH SKIPPED ({key} mount or meta missing) ===", flush=True)
    except Exception as e:
        import traceback
        print(f"=== STITCH FAILED: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()


# ── Preload the served handles into CPU RAM at app startup ─────────────
# Worker forks inherit these via copy-on-write so each @spaces.GPU click
# skips the slow disk→CPU shard load that was blowing the GPU duration
# budget (a cold 14B load takes >90s — past the I2V reservation, so the
# first click was aborting). The registry keeps up to `max_warm` handles
# resident, so a T2V↔I2V swap only moves GPU residency: no cold load ever
# lands inside a @spaces.GPU window. Bounded at REGISTRY.max_warm so a
# larger served set never blows CPU RAM at boot.
def _preload_served_handles() -> None:
    if os.getenv("SPACES_ZERO_GPU") is None:
        return
    try:
        from provisioning.manifest import available_keys, resolve_available_key
        import time as _t
        # Default T2V first (the mode the UI opens on → initial GPU resident),
        # then the rest of the served set, capped at the warm budget.
        default_t2v = resolve_available_key("wan2.2_t2v_a14b", "t2v")
        served = available_keys() or {default_t2v}  # None ⇒ full deploy: warm the default
        ordered = [default_t2v] + sorted(k for k in served if k != default_t2v)
        ordered = ordered[: REGISTRY.max_warm]
        for key in ordered:
            print(f"=== PRELOAD handle to CPU: {key} ===", flush=True)
            t0 = _t.time()
            # acquire() builds via _build_handle + ensure_loaded (disk → CPU RAM
            # only; no CUDA touch) and keeps it warm in the bounded registry.
            REGISTRY.acquire(key)
            print(f"=== PRELOAD {key} done in {int(_t.time()-t0)}s ===", flush=True)
        # Leave the default T2V as the nominal GPU-resident key so the first
        # T2V click is a pure cache hit (the UI opens on T2V).
        if default_t2v in REGISTRY._handles:
            REGISTRY.warm_key = default_t2v
        print(f"=== PRELOAD complete — warm: {list(REGISTRY._handles.keys())} ===", flush=True)
    except Exception as e:
        import traceback
        print(f"=== PRELOAD FAILED: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()


_probe_filesystem()
_stitch_default_model()
_preload_served_handles()




# ────────────────────────────────────────────────────────────────────────────
# Generate-handler helpers — kept at module scope so the `@spaces.GPU(...)`
# decorator can reference `duration=`/`size=` callables that match the wrapped
# function's signature.  Everything heavy (diffusers, torch, model handles)
# stays lazy: nothing imports diffusers or instantiates a handle at module
# import time, so `from app import build; build()` is cheap.
# ────────────────────────────────────────────────────────────────────────────

def _key_for(mode: str, generation: str, resolution_label: str = "") -> str:
    """Resolve the registry key for a (mode, generation, resolution) via the
    owning HandlerSpec, then apply the local-dev override if present.

    The per-mode `key_for` lives on the HandlerSpec (pipelines/t2v.py,
    pipelines/i2v.py), so this single helper drives both tiers.
    """
    spec = HANDLER_REGISTRY[mode]
    key = spec.key_for(generation, resolution_label=resolution_label)
    # Local-dev override: allow forcing a smaller checkpoint for faster MPS smoke.
    local_override = os.getenv(f"WAN_STUDIO_{mode.upper()}_LOCAL_KEY")
    if local_override and os.getenv("SPACES_ZERO_GPU") is None:
        key = local_override
    # Subset deploy: if the ideal checkpoint isn't served by this Space, fall back
    # to an available one for this mode (full deploy returns the key unchanged).
    from provisioning.manifest import resolve_available_key
    return resolve_available_key(key, mode)


def _parse_resolution(label: str) -> tuple[int, int]:
    """'1280 × 720  (16:9)' or '1280x720 (16:9)' → (height, width)."""
    import re
    m = re.search(r"(\d+)\s*[x×]\s*(\d+)", label or "")
    if not m:
        return 720, 1280
    w, h = int(m.group(1)), int(m.group(2))
    return h, w


# --- @spaces.GPU(duration=callable) — duration is dynamic per-args, but size
# MUST be a static literal ('large' | 'xlarge'). Passing a callable for size
# silently serializes the function object into the /schedule POST body and HF
# rejects with 422. Both T2V + I2V modes are bounded to 'large' on PRO tier
# (utils.budget.MODE_BUDGET) so we hard-code 'large' on the decorator below.

def _get_duration(mode, *ui_args, **kwargs):
    """ZeroGPU `duration=` callable for both tier entrypoints.

    Receives the same `(mode, *ui_args)` the entrypoint is called with. The
    per-mode UI layout differs (I2V leads with an image), so we resolve the
    registry key from the slots that carry (generation, resolution, duration_s)
    by mode before scaling.
    """
    from utils.budget import reserve_seconds
    generation, resolution_label, duration_s, preset_label = _ui_dispatch(mode, ui_args)
    key = _key_for(mode, generation, resolution_label)
    steps = _effective_steps(key, preset_label, ui_args)
    return reserve_seconds(key, steps=steps, resolution_label=resolution_label,
                           duration_s=float(duration_s or 3.0))


def _effective_steps(key: str, preset_label, ui_args: tuple) -> int:
    """Resolved denoise-step count: the Advanced override (last-3 slot for every
    wired mode) if set, else the preset's steps for this card."""
    steps_override = ui_args[-3] if len(ui_args) >= 3 else 0
    if steps_override and int(steps_override) > 0:
        return int(steps_override)
    from pipelines.preset import resolve
    from pipelines.registry import BY_KEY
    return resolve(BY_KEY[key], _coerce_preset(preset_label)).num_inference_steps


def _coerce_preset(preset_label: str) -> str:
    """Header `preset_state` may carry either the literal 'fast'/'quality'
    string or a user-facing 'Fast'/'Quality' label — normalize to lower."""
    return "fast" if preset_label and str(preset_label).lower().startswith("fast") else "quality"


def _raise_user_error(e: BaseException) -> None:
    """Translate a low-level exception into a gr.Error toast for the user."""
    # OOM: hard to recover from in-handler, so message the user with a hint.
    try:
        import torch
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
            raise gr.Error(f"GPU out of memory. Try a smaller resolution or shorter duration. ({e})") from e
    except ImportError:
        pass
    if isinstance(e, FileNotFoundError):
        raise gr.Error(f"Model files not found — volume mount may be missing on the Space. ({e})") from e
    import traceback
    print(traceback.format_exc())
    raise gr.Error(f"Generation failed: {type(e).__name__}: {e}") from e


# ── UI-arg layout per mode ────────────────────────────────────────────────
# The Generate wiring in build() binds a per-mode `inputs=[...]` list. Both
# wired modes share the same trailing Advanced block (negative_prompt, seed,
# randomize, steps, cfg, cfg_2); they differ only in the leading slots:
#   t2v: (prompt, generation, preset, resolution, duration, *advanced)
#   i2v: (image, prompt, generation, preset, resolution, duration, *advanced)
# `_inputs_for` (in build) builds these lists; `_ui_dispatch` reads back the
# (generation, resolution, duration_s) needed for key + duration resolution.
def _ui_dispatch(mode: str, ui_args: tuple) -> tuple[str, str, float, str]:
    """Extract (generation, resolution_label, duration_s, preset_label) from a
    mode's raw UI arg tuple. Lets the shared duration callable + key resolver +
    ETA work for both the image-led (i2v) and prompt-led (t2v) layouts."""
    if mode == "i2v":
        # (image, prompt, generation, preset, resolution, duration, ...)
        return ui_args[2], ui_args[4], ui_args[5], ui_args[3]
    if mode == "v2v":
        # (video, generation, preset, prompt, strength, ...) — no resolution
        # dropdown; fixed ~3s reserve so _get_duration scales sanely.
        return ui_args[1], "", 3.0, ui_args[2]
    if mode == "flf2v":
        # (start_frame, generation, preset, end_uploaded, end_generated, ...) —
        # no resolution dropdown; FLF2V is an 81-frame (~5s) fixed clip.
        return ui_args[1], "", 5.0, ui_args[2]
    if mode == "vace":
        # (submode, generation, preset, source_video, references, prompt, ...) —
        # no resolution dropdown; default ~4s reserve (length comes from the
        # source video at runtime, falling back to 81 frames).
        return ui_args[1], "", 4.0, ui_args[2]
    # t2v: (prompt, generation, preset, resolution, duration, ...)
    return ui_args[1], ui_args[3], ui_args[4], ui_args[2]


def _build_inference_kwargs(preset_kwargs, steps_override, cfg_override, cfg_2_override):
    """Common Fast/Quality → pipeline-kwargs merge shared by both modes."""
    inference_kwargs = {
        "num_inference_steps": (
            int(steps_override) if steps_override and int(steps_override) > 0
            else preset_kwargs.num_inference_steps
        ),
        "guidance_scale": (
            float(cfg_override) if cfg_override and float(cfg_override) > 0
            else preset_kwargs.guidance_scale
        ),
    }
    if preset_kwargs.guidance_scale_2 is not None:
        inference_kwargs["guidance_scale_2"] = (
            float(cfg_2_override) if cfg_2_override and float(cfg_2_override) > 0
            else preset_kwargs.guidance_scale_2
        )
    return inference_kwargs


def _assert_fits(key: str, steps, resolution_label: str, duration_s) -> None:
    """Fail fast if this (model, res, steps, length) can't finish inside the
    ZeroGPU window — so the user gets an immediate, actionable error instead of
    watching a progress bar for minutes only to abort near the end (product-brain
    D5: never wait then abort)."""
    from utils.budget import fits_window, ZEROGPU_DURATION_CAP
    if not fits_window(key, steps=int(steps), resolution_label=resolution_label,
                       duration_s=float(duration_s or 3.0)):
        raise gr.Error(
            f"This setting ({resolution_label or 'this length'}, {int(steps)} steps) needs "
            f"more than the {ZEROGPU_DURATION_CAP}s GPU limit. Lower the resolution or steps, "
            f"or switch to Fast."
        )


def _assert_mps_memory_safe(key: str, num_frames: int, resolution_label: str = "") -> None:
    """LOCAL MPS only: refuse a clip whose estimated peak driver memory would risk
    thrashing/crashing the OS. The app has no ZeroGPU isolation locally — a prior
    over-long run swap-thrashed the box. CUDA/ZeroGPU has discrete VRAM + a releasing
    allocator, so this is a no-op there. Calibrated to measured driver_allocated."""
    from utils.backend import detect
    if detect().device != "mps":
        return
    import sys as _sys
    _scripts = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scripts")
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from memcheck import (estimate_mps_real_gb, estimate_peak_gb, free_gb,
                          HARD_CAP_GB, MPS_REAL_GB_PER_LATENT)
    res = "720p" if ("720" in (resolution_label or "") or "1280" in (resolution_label or "")) else "480p"
    # PRIMARY co-gate (wan-brain C1): the conservative STATIC envelope is the only one
    # PROVEN safe. The real model UNDER-estimates VAE decode at high latent-frame counts
    # (a 13-latent 480p decode OOM-killed the process historically) — so a pure real-model
    # gate would authorize a panic-class monolithic decode. Refuse anything outside the
    # static-safe envelope; real-length needs temporal-chunked decode (out of scope).
    static = estimate_peak_gb(key, int(num_frames), res)
    if not static["safe"]:
        # largest static-safe frame count for this model/res (latent steps of 4 px frames).
        safe_f = int(num_frames)
        while safe_f > 5 and not estimate_peak_gb(key, safe_f, res)["safe"]:
            safe_f -= 4
        raise gr.Error(
            f"On local MPS, {int(num_frames)} frames @ {res} exceeds the safe VAE-decode "
            f"envelope (static peak ~{static['peak_gb']}GB > {static['hard_cap_gb']:.0f}GB cap) — "
            f"a monolithic decode this large can OOM/crash the machine. Try ~{safe_f} frames or "
            f"fewer (real-length output needs chunked VAE decode, not yet implemented)."
        )
    est = estimate_mps_real_gb(key, int(num_frames), res)
    # C1 backstop (wan-brain): an ABSOLUTE ceiling independent of momentary free RAM.
    # A monolithic (non-chunked) VAE decode beyond this is panic-class on this 128GB
    # box (e.g. 720p@81f ≈ 144GB real). The free-margin check below is necessary but
    # not sufficient — on a freshly-booted box with lots of free RAM it could otherwise
    # authorize a panic-class clip. Real-length clips need temporal-chunked decode (TODO).
    ABS_REAL_CAP_GB = 100.0
    if est["real_gb"] > ABS_REAL_CAP_GB:
        raise gr.Error(
            f"On local MPS, ~{int(num_frames)} frames @ {res} needs ~{est['real_gb']:.0f}GB for a "
            f"monolithic VAE decode — above the safe {ABS_REAL_CAP_GB:.0f}GB ceiling. Use fewer "
            f"frames or lower resolution (real-length output needs chunked VAE decode)."
        )
    # Gate on REAL free system memory (vm_stat-based), not the lying driver counter. Leave
    # an OS+app headroom margin. driver_allocated over-reports ~6x; using it would refuse
    # clips that actually use a fraction of the claimed memory.
    free = free_gb()
    margin = 24.0
    if est["real_gb"] > free - margin:
        budget = max(0.0, free - margin - (est["real_gb"] - est["runtime_gb"]))
        safe_lat = max(1, int(budget / max(0.1, MPS_REAL_GB_PER_LATENT)))
        safe_frames = max(17, (safe_lat - 1) * 4 + 1)
        raise gr.Error(
            f"On local MPS, ~{int(num_frames)} frames needs ~{est['real_gb']:.0f}GB but only "
            f"~{free:.0f}GB is free. Close other apps, shorten to ~{(safe_frames - 1) / 16:.1f}s, "
            f"or use the lighter 1.3B model."
        )


def _make_step_progress(progress, lo: float = 0.2, hi: float = 0.92):
    """Build an `on_step(done, total)` that advances the Gradio progress bar per
    denoise step. Passed to handle.generate → diffusers callback_on_step_end, so
    the UI shows live progress instead of a frozen panel (track_tqdm stays off to
    avoid the SSE DOM-accumulation bug)."""
    def on_step(done: int, total: int) -> None:
        frac = lo + (hi - lo) * (done / max(1, int(total or 1)))
        progress(min(hi, frac), desc=f"Denoising · step {done}/{total}")
    return on_step


def _export(frames, mode: str, fallback_message: str = "") -> str:
    """Encode generated frames to a temp MP4 (`wan_<mode>_*.mp4`, fps=16) and,
    when a Fast→Quality fallback was applied, surface the message as a toast.
    Shared by every `_run_<mode>` runner."""
    import os
    import tempfile
    from pipelines.video_io import save_video  # imageio; diffusers export_to_video corrupts frames on MPS

    fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix=f"wan_{mode}_")
    os.close(fd)
    save_video(frames, out_path, fps=16)
    if fallback_message:
        gr.Info(fallback_message, duration=8)
    return out_path


def _run_t2v(spec, ui_args, progress):
    """T2V body: resolve key → acquire (LRU) → configure preset → generate →
    export. Behaviour-identical to the prior `generate_t2v` closure."""
    import random

    (prompt, generation, preset_label, resolution_label, duration_s,
     negative_prompt, seed, randomize, steps_override, cfg_override,
     cfg_2_override) = ui_args

    # Worker-side filesystem + env diagnostics. Logs once per worker fork so we
    # can confirm /tmp/hf_cache is visible from inside the ZeroGPU sandbox.
    print(
        f"=== WORKER PROBE: uid={os.getuid()} "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')} "
        f"WAN_STUDIO_WAN22_T2V_LOCAL_PATH={os.environ.get('WAN_STUDIO_WAN22_T2V_LOCAL_PATH')} "
        f"tmp_hf_cache_exists={os.path.exists('/tmp/hf_cache')} ===",
        flush=True,
    )
    if os.path.exists("/tmp/hf_cache"):
        try:
            listing = os.listdir("/tmp/hf_cache")
            print(f"=== WORKER PROBE /tmp/hf_cache listing: {listing} ===", flush=True)
        except Exception as e:
            print(f"=== WORKER PROBE /tmp/hf_cache listdir error: {e} ===", flush=True)

    if not prompt or not str(prompt).strip():
        raise gr.Error("Prompt is required.")

    if randomize:
        seed = random.randint(0, 2**31 - 1)

    preset = _coerce_preset(preset_label)
    key = _key_for(spec.mode, generation, resolution_label)
    handle = REGISTRY.acquire(key)
    progress(0.05, desc="Loading model to GPU (first run is slower)…")
    preset_kwargs = handle.configure_preset(preset)
    inference_kwargs = _build_inference_kwargs(
        preset_kwargs, steps_override, cfg_override, cfg_2_override
    )
    _assert_fits(key, inference_kwargs["num_inference_steps"], resolution_label, duration_s)

    height, width = _parse_resolution(resolution_label)
    # Wan VAE temporal patching: num_frames must be 4k+1.
    num_frames = max(17, int(float(duration_s) * 16) // 4 * 4 + 1)
    _assert_mps_memory_safe(key, num_frames, resolution_label)

    progress(0.2, desc="Generating frames…")
    frames = handle.generate(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        height=height,
        width=width,
        num_frames=num_frames,
        seed=int(seed),
        preset_kwargs=inference_kwargs,
        step_callback=_make_step_progress(progress),
    )

    progress(0.95, desc="Encoding MP4…")
    return _export(frames, "t2v", preset_kwargs.fallback_message)


def _run_i2v(spec, ui_args, progress):
    """I2V body: auto-picks the checkpoint by (generation × resolution) and
    coerces filepath → PIL.Image. Behaviour-identical to `generate_i2v`."""
    import random
    from PIL import Image
    from pipelines.trace import trace, reset_trace

    reset_trace()
    trace("=== _run_i2v ENTER (GPU worker) ===")

    (image, prompt, generation, preset_label, resolution_label, duration_s,
     negative_prompt, seed, randomize, steps_override, cfg_override,
     cfg_2_override) = ui_args

    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not str(prompt).strip():
        raise gr.Error("Motion prompt is required.")

    # Coerce filepath → PIL.Image (gr.Image type="pil" already yields a
    # PIL.Image, but tolerate strings for callers that bind type="filepath").
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")

    if randomize:
        seed = random.randint(0, 2**31 - 1)

    preset = _coerce_preset(preset_label)
    key = _key_for(spec.mode, generation, resolution_label)
    handle = REGISTRY.acquire(key)
    trace("after REGISTRY.acquire")
    progress(0.05, desc="Loading model to GPU (first run is slower)…")
    preset_kwargs = handle.configure_preset(preset)
    trace("after configure_preset (preset + cuda attach done)")
    inference_kwargs = _build_inference_kwargs(
        preset_kwargs, steps_override, cfg_override, cfg_2_override
    )
    _assert_fits(key, inference_kwargs["num_inference_steps"], resolution_label, duration_s)

    # max_area drives `aspect_ratio_resize` — clamp to the picked res.
    h_label, w_label = _parse_resolution(resolution_label)
    max_area = h_label * w_label
    num_frames = max(17, int(float(duration_s) * 16) // 4 * 4 + 1)
    _assert_mps_memory_safe(key, num_frames, resolution_label)

    progress(0.2, desc="Generating frames…")
    trace(f"generate START (num_frames={num_frames}, steps={inference_kwargs.get('num_inference_steps')})")
    frames = handle.generate(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        max_area=max_area,
        num_frames=num_frames,
        seed=int(seed),
        preset_kwargs=inference_kwargs,
        step_callback=_make_step_progress(progress),
    )
    trace("generate DONE")

    progress(0.95, desc="Encoding MP4…")
    return _export(frames, "i2v", preset_kwargs.fallback_message)


def _run_v2v(spec, ui_args, progress):
    """V2V body: decode the source video to PIL frames on the VAE grid, then
    restyle via WanVideoToVideoPipeline (video + strength). Wan 2.1, Quality-only."""
    import random
    from pipelines.video_io import decode_video

    (video, generation, preset_label, prompt, strength,
     negative_prompt, seed, randomize, steps, cfg, cfg_2) = ui_args

    if not video:
        raise gr.Error("Please upload a video.")
    if not prompt or not str(prompt).strip():
        raise gr.Error("Restyle prompt is required.")

    if randomize:
        seed = random.randint(0, 2**31 - 1)

    handle = REGISTRY.acquire(_key_for(spec.mode, generation))
    progress(0.05, desc="Loading model to GPU (first run is slower)…")
    pk = handle.configure_preset(_coerce_preset(preset_label))
    inf = _build_inference_kwargs(pk, steps, cfg, cfg_2)

    progress(0.2, desc="Decoding source video…")
    frames_in, _, _ = decode_video(video, handle.pipe, 480 * 832)
    _assert_mps_memory_safe(_key_for(spec.mode, generation), len(frames_in))

    progress(0.3, desc="Restyling frames…")
    out = handle.generate(
        frames_in,
        prompt,
        negative_prompt=negative_prompt or "",
        strength=float(strength),
        seed=int(seed),
        preset_kwargs=inf,
    )

    progress(0.9, desc="Encoding MP4…")
    return _export(out, "v2v", pk.fallback_message)


def _run_flf2v(spec, ui_args, progress):
    """FLF2V body: first + last frame → 81-frame transition via WanImageToVideoPipeline
    (last_image=). End frame may be uploaded or T2I-generated. Wan 2.1 720p-locked."""
    import random

    (start_frame, generation, preset_label, end_uploaded, end_generated, prompt,
     negative_prompt, seed, randomize, steps, cfg, cfg_2) = ui_args

    if start_frame is None:
        raise gr.Error("Please provide a start frame.")
    # NOTE: both end-frame inputs are type='pil' (ui/tabs.py) — end_frame_uploaded
    # and end_frame_generated alike — so center_crop_resize's image.convert('RGB')
    # gets a PIL.Image, never a raw numpy frame. generate_end IS wired to
    # generate_end_frame in build(), so end_generated is reachable; type='pil' on
    # that gr.Image is what coerces the T2V handle's numpy frame to PIL on the way back.
    last = end_uploaded if end_uploaded is not None else end_generated
    if last is None:
        raise gr.Error("Please upload or generate an end frame.")
    if not prompt or not str(prompt).strip():
        raise gr.Error("Transition prompt is required.")

    if randomize:
        seed = random.randint(0, 2**31 - 1)

    key = _key_for(spec.mode, generation)
    handle = REGISTRY.acquire(key)
    progress(0.05, desc="Loading model to GPU (first run is slower)…")
    pk = handle.configure_preset(_coerce_preset(preset_label))
    # Let the card drive CFG per preset (quality_guidance=5.5 / lightning_guidance=1.0)
    # exactly like t2v/i2v — no hardcode, so the Fast/Lightning path isn't forced
    # to a CFG-distilled-hostile 5.5 when the user didn't override.
    inf = _build_inference_kwargs(pk, steps, cfg, cfg_2)

    # FLF2V's handle defaults to 81 frames @720p — a monolithic MPS decode that would
    # OOM. Cap to the verified-safe 720p length locally + guard (this runner previously
    # had NO memcheck). HF Spaces (real GPU) keeps the full 81.
    num_frames = 81 if os.getenv("SPACES_ZERO_GPU") else 13
    _assert_mps_memory_safe(key, num_frames, "720p")

    progress(0.2, desc="Generating frames…")
    out = handle.generate(
        start_frame,
        last,
        prompt,
        negative_prompt=negative_prompt or "",
        num_frames=num_frames,
        seed=int(seed),
        preset_kwargs=inf,
    )

    progress(0.9, desc="Encoding MP4…")
    return _export(out, "flf2v", pk.fallback_message)


def _snap_vace_frames(seq, num_frames):
    """Trim or pad a conditioning frame list to exactly `num_frames` items.

    WanVACEPipeline rounds num_frames DOWN to the nearest 4k+1 but keeps every
    conditioning frame, so a raw clip length (e.g. 150) leaves video/mask longer
    than the noise latents → the control signal misaligns vs the latents. We snap
    the lists to the same 4k+1 length the pipeline will use: trim the tail when
    too long, repeat the last frame when too short (only when num_frames slightly
    exceeds a short clip)."""
    if not seq:
        return seq
    if len(seq) > num_frames:
        return seq[:num_frames]
    if len(seq) < num_frames:
        return seq + [seq[-1]] * (num_frames - len(seq))
    return seq


def _vace_spatial_base(pipe):
    """The spatial divisibility unit WanVACEPipeline.check_inputs enforces on
    height/width: vae_scale_factor_spatial * transformer.patch_size[1] (8 * 2 = 16
    for Wan 2.1). Derived from the pipe so it tracks the actual loaded config; falls
    back to 16 when the transformer/vae aren't introspectable."""
    spatial = getattr(pipe, "vae_scale_factor_spatial", None) or 8
    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_2", None)
    patch = getattr(getattr(transformer, "config", None), "patch_size", None)
    psize = patch[1] if patch else 2
    return (spatial * psize) or 16


def _run_vace(spec, ui_args, progress):
    """VACE body: resolve the sub-mode → build video/mask/reference_images via the
    pure builders, then run WanVACEPipeline. Wan 2.1, Quality-only. Control-extraction
    sub-modes (Depth/Pose/Sketch/Flow) take the uploaded source AS the control video in
    v1 (auto-extraction is #1b-preproc); deferred sub-modes raise an actionable error."""
    import random
    from pipelines.vace_inputs import gallery_to_references, resolve_submode
    from pipelines.video_io import decode_video

    (submode, generation, preset_label, source_video, references, prompt,
     negative_prompt, seed, randomize, steps, cfg, cfg_2) = ui_args

    if not str(prompt or "").strip():
        raise gr.Error("Prompt is required.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)

    handle = REGISTRY.acquire(_key_for(spec.mode, generation))
    progress(0.05, desc="Loading model to GPU (first run is slower)…")
    pk = handle.configure_preset(_coerce_preset(preset_label))

    refs = gallery_to_references(references)
    src_frames, h, w = None, 480, 832
    if source_video:
        progress(0.15, desc="Decoding source video…")
        src_frames, h, w = decode_video(source_video, handle.pipe, 480 * 832)

    # VACE's native length is 81 frames; on local MPS that decode is far past the safe
    # envelope (~568GB static), so with no source video (Reference/T2V-style submodes)
    # default to an MPS-safe length. HF Spaces (real GPU) keeps 81.
    _vace_default_n = 81 if os.getenv("SPACES_ZERO_GPU") else 17
    try:
        plan = resolve_submode(
            submode, source_frames=src_frames, references=refs,
            height=h, width=w, num_frames=(len(src_frames) if src_frames else _vace_default_n),
        )
    except ValueError as e:
        raise gr.Error(str(e))

    # Snap num_frames to the 4k+1 the pipeline enforces (vae_scale_factor_temporal)
    # and trim/extend video+mask to MATCH it. Otherwise the pipeline silently rounds
    # num_frames down but keeps every conditioning frame → conditioning_latents has
    # more temporal frames than the noise latents and the control signal misaligns
    # (pipeline_wan_vace.py:836-840 / 941 warn-only). video/mask are already equal-length.
    raw_n = len(plan.video) if plan.video else _vace_default_n
    vsft = getattr(handle.pipe, "vae_scale_factor_temporal", 4) or 4
    num_frames = max(raw_n // vsft * vsft + 1, 1)
    _assert_mps_memory_safe(_key_for(spec.mode, generation), num_frames)
    plan_video = _snap_vace_frames(plan.video, num_frames)
    plan_mask = _snap_vace_frames(plan.mask, num_frames)

    # Outpaint (and any sub-mode that changes frame dims) enlarges the canvas to
    # (w+2*pad, h+2*pad). Pass the PADDED dims to generate so the area budget keeps
    # the enlarged canvas; passing the original h,w lets preprocess_conditions'
    # area-budget rescale (pipeline_wan_vace.py:434-436) shrink it back, squeezing
    # the extension instead of enlarging the frame.
    out_h, out_w = h, w
    if plan_video:
        pw, ph = plan_video[0].size  # PIL .size is (width, height)
        out_h, out_w = ph, pw
    # The padded outpaint canvas (h+2*(h//4)) is only guaranteed %8, but check_inputs
    # hard-raises unless height/width are %base (16 for Wan 2.1). Snap DOWN to the same
    # base the pipeline uses so a non-480x832 aspect ratio (e.g. 624x624 -> 936x936,
    # 936%16==8) no longer ValueErrors before generation — it loses at most base-1 px
    # of the extension, which the model regenerates anyway, vs the pre-commit squeeze.
    base = _vace_spatial_base(handle.pipe)
    out_h = max(out_h // base * base, base)
    out_w = max(out_w // base * base, base)

    inf = _build_inference_kwargs(pk, steps, cfg, cfg_2)

    progress(0.2, desc="Generating frames…")
    out = handle.generate(
        prompt=prompt,
        video=plan_video,
        mask=plan_mask,
        reference_images=plan.reference_images,
        negative_prompt=negative_prompt or "",
        height=out_h,
        width=out_w,
        num_frames=num_frames,
        seed=int(seed),
        preset_kwargs=inf,
    )

    progress(0.9, desc="Encoding MP4…")
    return _export(out, "vace", pk.fallback_message)


def _run_ti2v(spec, ui_args, progress):
    """TI2V body (Wan 2.2-5B): text+image→video. Fixed-resolution via an orientation
    radio (1280×704 / 704×1280, no resolution/duration sliders). Short safe length —
    real-length (121f) needs temporal-chunked VAE decode (out of scope), and the
    C1 backstop + memcheck would refuse a monolithic long 720p decode anyway."""
    import random
    from PIL import Image

    (image, prompt, generation, preset_label, orientation,
     negative_prompt, seed, randomize, steps_override, cfg_override,
     cfg_2_override) = ui_args

    if image is None:
        raise gr.Error("TI2V needs an init image (text+image→video).")
    if not prompt or not str(prompt).strip():
        raise gr.Error("Prompt is required.")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")
    if randomize:
        seed = random.randint(0, 2**31 - 1)

    # Orientation radio → the only two supported resolutions.
    height, width = (1280, 704) if "Portrait" in (orientation or "") else (704, 1280)

    preset = _coerce_preset(preset_label)
    key = _key_for(spec.mode, generation)
    handle = REGISTRY.acquire(key)
    progress(0.05, desc="Loading model to GPU (first run is slower)…")
    preset_kwargs = handle.configure_preset(preset)
    inference_kwargs = _build_inference_kwargs(
        preset_kwargs, steps_override, cfg_override, cfg_2_override
    )
    num_frames = 13  # safe verified length on this MPS box; real-length = future chunked decode
    _assert_mps_memory_safe(key, num_frames, "720p")

    progress(0.2, desc="Generating frames…")
    frames = handle.generate(
        image, prompt,
        negative_prompt=negative_prompt or "",
        height=height, width=width, num_frames=num_frames,
        seed=int(seed), preset_kwargs=inference_kwargs,
        step_callback=_make_step_progress(progress),
    )
    progress(0.9, desc="Encoding MP4…")
    return _export(frames, "ti2v", preset_kwargs.fallback_message)


def _run_s2v(spec, ui_args, progress):
    """S2V body (Wan 2.2): reference image + driving audio → talking-character video.
    Runs the upstream wan.WanS2V port as a SCOPED SUBPROCESS (scripts/s2v_smoke.py) so
    its process-global MPS shims (device cuda→mps, float64→fp32, autocast remap,
    flash_attention replacement) stay isolated from the diffusers modes. Frees the app's
    warm models first — the subprocess needs ~110GB of unified memory."""
    import os
    import tempfile
    from PIL import Image
    from pipelines.s2v import run_s2v_subprocess

    (reference_image, audio, generation, preset_label, resolution, prompt,
     negative_prompt, seed, randomize, steps_override, cfg_override,
     cfg_2_override) = ui_args

    if reference_image is None:
        raise gr.Error("S2V needs a reference character image.")
    if not audio:
        raise gr.Error("S2V needs a driving audio clip (upload or record).")

    img = reference_image
    if isinstance(img, str):
        img_path = img
    else:
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        fd, img_path = tempfile.mkstemp(suffix=".png", prefix="s2v_ref_")
        os.close(fd)
        img.save(img_path)
    audio_path = audio if isinstance(audio, str) else str(audio)

    # S2V on MPS is ~75s/step (chunked-flash + RoPE CPU fallback). Default to a
    # tractable interactive step count; override via the Advanced "Steps" slider or
    # WAN_S2V_APP_STEPS. 8 steps gives valid talking-head output (between the proven
    # 4-step smoke and the 16-step quality run).
    steps = int(steps_override) if steps_override else int(os.getenv("WAN_S2V_APP_STEPS", "8"))
    fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="wan_s2v_")
    os.close(fd)

    progress(0.05, desc="Freeing app models for S2V subprocess…")
    REGISTRY.free_all()
    progress(0.15, desc="S2V subprocess: loading 14B + generating (several minutes)…")
    run_s2v_subprocess(
        img_path, audio_path, prompt, out_path,
        frames=17, steps=steps, size="832*480",
        progress_cb=lambda s: progress(0.5, desc=f"S2V sampling: {s}"),
    )
    progress(0.95, desc="Done.")
    return out_path


def _run_animate(spec, ui_args, progress):
    """Animate body (Wan 2.2): character image + driving video → animated character.
    ViTPose/YOLO pose+face preprocessing runs on CPU (scripts/animate_preprocess.py
    subprocess — isolates its vendored-utils sys.path collision), then the
    WanAnimatePipeline on MPS via the in-process handle."""
    import os
    import random
    import subprocess
    import sys
    import tempfile
    import imageio.v2 as imageio
    from PIL import Image

    repo_root = os.path.dirname(os.path.abspath(__file__))
    (character, driving, mode, res, prompt,
     negative_prompt, seed, randomize, steps_override, cfg_override,
     cfg_2_override) = ui_args

    if character is None:
        raise gr.Error("Animate needs a character reference image.")
    if not driving:
        raise gr.Error("Animate needs a driving / template video.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)

    img = character
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif hasattr(img, "convert"):
        img = img.convert("RGB")

    is_720 = "720" in (res or "")
    height, width = (1280, 720) if is_720 else (832, 480)
    num_frames = 13  # safe verified length on this MPS box

    progress(0.05, desc="Pose+face preprocessing (CPU, ~30s)…")
    pre_out = tempfile.mkdtemp(prefix="animate_pre_")
    env = dict(os.environ)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    pre = subprocess.run(
        [sys.executable, os.path.join(repo_root, "scripts", "animate_preprocess.py"),
         "--video", driving, "--out", pre_out, "--frames", str(num_frames),
         "--area", str(height * width)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    pose_mp4 = os.path.join(pre_out, "src_pose.mp4")
    face_mp4 = os.path.join(pre_out, "src_face.mp4")
    if pre.returncode != 0 or not os.path.exists(pose_mp4):
        raise gr.Error(f"Pose preprocessing failed: {(pre.stderr or pre.stdout)[-300:]}")
    pose_video = [Image.fromarray(f) for f in imageio.mimread(pose_mp4, memtest=False)]
    face_video = [Image.fromarray(f) for f in imageio.mimread(face_mp4, memtest=False)]

    key = _key_for(spec.mode, "wan2.2")
    handle = REGISTRY.acquire(key)
    progress(0.4, desc="Loading Animate model…")
    pk = handle.configure_preset(_coerce_preset("quality"))
    inf = _build_inference_kwargs(pk, steps_override, cfg_override, cfg_2_override)
    _assert_mps_memory_safe(key, num_frames, "720p" if is_720 else "480p")

    progress(0.5, desc="Animating…")
    out = handle.generate(
        img, pose_video, face_video,
        prompt=prompt or "a person, natural motion, cinematic",
        negative_prompt=negative_prompt or "",
        height=height, width=width, segment_frame_length=num_frames,
        seed=int(seed), preset_kwargs=inf,
        step_callback=_make_step_progress(progress),
    )
    progress(0.9, desc="Encoding MP4…")
    return _export(out, "animate", pk.fallback_message)


# Per-mode body dispatch. Append-only: a new wired mode adds its `_run_*` here.
_MODE_RUNNERS = {
    "t2v": _run_t2v, "i2v": _run_i2v, "flf2v": _run_flf2v, "v2v": _run_v2v,
    "vace": _run_vace, "ti2v": _run_ti2v, "s2v": _run_s2v, "animate": _run_animate,
}


def _run(spec, *ui_args, progress=None):
    """Generic Generate body — resolve key, acquire (LRU), configure preset,
    generate, export. Dispatches to the mode-specific runner so T2V/I2V keep
    their distinct input layouts while sharing the registry + LRU plumbing."""
    if progress is None:
        progress = gr.Progress(track_tqdm=False)
    runner = _MODE_RUNNERS[spec.mode]
    try:
        return runner(spec, ui_args, progress)
    except gr.Error:
        raise
    except Exception as e:
        _raise_user_error(e)


# ── Per-tier @spaces.GPU entrypoints — size MUST be a static literal ───────
# `@spaces.GPU(duration=callable)` accepts a dynamic per-args duration, but
# `size=` must be a literal ('large' | 'xlarge'); passing a callable serializes
# the function into the /schedule POST body and HF rejects it (422). The two
# wired modes (T2V/I2V) are both tier `large`; the `xlarge` entrypoint exists
# for later modes (Animate) that the HANDLER_REGISTRY routes there.
from utils.backend import spaces_gpu_or_noop  # noqa: E402


@spaces_gpu_or_noop()(duration=_get_duration, size="large")
def generate_large(mode, *args, progress=gr.Progress(track_tqdm=False)):
    return _run(HANDLER_REGISTRY[mode], *args, progress=progress)


@spaces_gpu_or_noop()(duration=_get_duration, size="xlarge")
def generate_xlarge(mode, *args, progress=gr.Progress(track_tqdm=False)):
    return _run(HANDLER_REGISTRY[mode], *args, progress=progress)


@spaces_gpu_or_noop()(duration=lambda *a, **k: 30, size="large")
def generate_end_frame(end_frame_prompt, generation, progress=gr.Progress(track_tqdm=False)):
    """Synthesize an FLF2V end frame via Wan T2I (num_frames=1). Returns one frame.

    Runs its own short load→generate→unload on the T2V backbone (risk R21: avoid
    two 14B transformers warm at once). The LRU registry evicts the warm FLF2V
    transformer if needed, so the secondary button can't double-load."""
    if not end_frame_prompt or not str(end_frame_prompt).strip():
        raise gr.Error("Enter a prompt to generate an end frame.")
    key = _key_for("t2v", generation)              # reuse the T2V backbone
    handle = REGISTRY.acquire(key)                 # LRU: evicts the warm transformer if needed
    progress(0.2, desc="Generating end frame…")
    pk = handle.configure_preset("quality")
    frames = handle.generate(
        prompt=end_frame_prompt,
        negative_prompt="",
        height=720,
        width=1280,
        num_frames=1,
        seed=0,
        preset_kwargs=_build_inference_kwargs(pk, 0, 0, 0),
    )
    return frames[0]                               # single frame → gr.Image


# ────────────────────────────────────────────────────────────────────────────
# Theme — Linear-faithful warm near-black palette with electric-blue accent.
# Built off gr.themes.Base (lowest preset noise) and overridden in CSS below.
# ────────────────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eef2ff", c100="#e0e7ff", c200="#c7d2fe", c300="#a5b4fc",
        c400="#818cf8", c500="#5e84ff", c600="#4f6fe5", c700="#4055bf",
        c800="#323f95", c900="#23306e", c950="#161e4a",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f7f8f8", c100="#e6e7e9", c200="#c1c4cc", c300="#8a8f98",
        c400="#62666d", c500="#44484e", c600="#2a2d33", c700="#1e2024",
        c800="#141518", c900="#0d0e10", c950="#08090a",
    ),
    font=(gr.themes.GoogleFont("Geist"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"),
    radius_size=gr.themes.sizes.radius_sm,
    spacing_size=gr.themes.sizes.spacing_md,
    text_size=gr.themes.sizes.text_sm,
).set(
    body_background_fill="#08090a",
    body_background_fill_dark="#08090a",
    background_fill_primary="#08090a",
    background_fill_primary_dark="#08090a",
    background_fill_secondary="#0d0e10",
    background_fill_secondary_dark="#0d0e10",
    block_background_fill="#0d0e10",
    block_background_fill_dark="#0d0e10",
    block_border_color="#1e2024",
    block_border_color_dark="#1e2024",
    block_border_width="1px",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#8a8f98",
    block_label_text_color_dark="#8a8f98",
    block_label_text_size="11px",
    block_label_text_weight="500",
    block_title_text_color="#f7f8f8",
    block_title_text_color_dark="#f7f8f8",
    body_text_color="#f7f8f8",
    body_text_color_dark="#f7f8f8",
    body_text_color_subdued="#8a8f98",
    body_text_color_subdued_dark="#8a8f98",
    border_color_primary="#1e2024",
    border_color_primary_dark="#1e2024",
    border_color_accent="#5e84ff",
    border_color_accent_dark="#5e84ff",
    input_background_fill="#101114",
    input_background_fill_dark="#101114",
    input_background_fill_focus="#141518",
    input_background_fill_focus_dark="#141518",
    input_border_color="#1e2024",
    input_border_color_dark="#1e2024",
    input_border_color_focus="#5e84ff",
    input_border_color_focus_dark="#5e84ff",
    input_placeholder_color="#62666d",
    input_placeholder_color_dark="#62666d",
    button_primary_background_fill="#5e84ff",
    button_primary_background_fill_dark="#5e84ff",
    button_primary_background_fill_hover="#7497ff",
    button_primary_background_fill_hover_dark="#7497ff",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",
    button_secondary_background_fill="#141518",
    button_secondary_background_fill_dark="#141518",
    button_secondary_background_fill_hover="#1c1e22",
    button_secondary_background_fill_hover_dark="#1c1e22",
    button_secondary_text_color="#e6e7e9",
    button_secondary_text_color_dark="#e6e7e9",
    button_secondary_border_color="#1e2024",
    button_secondary_border_color_dark="#1e2024",
    panel_background_fill="#0d0e10",
    panel_background_fill_dark="#0d0e10",
    panel_border_color="#1e2024",
    panel_border_color_dark="#1e2024",
    color_accent="#5e84ff",
    color_accent_soft="#1a2238",
    color_accent_soft_dark="#1a2238",
    link_text_color="#7497ff",
    link_text_color_dark="#7497ff",
    link_text_color_hover="#a8bcff",
    link_text_color_hover_dark="#a8bcff",
)


# ────────────────────────────────────────────────────────────────────────────
# CSS — Linear-faithful chrome.  Heavy use of !important to defeat Gradio's
# default specificity.
# ────────────────────────────────────────────────────────────────────────────
CSS = """
/* ─── Root surface ─────────────────────────────────────────────────── */
:root, html, body, gradio-app, .gradio-container {
  --ws-bg: #08090a !important;
  --ws-surface: #0d0e10 !important;
  --ws-surface-2: #101114 !important;
  --ws-elev: #16181c !important;
  --ws-border: #1e2024 !important;
  --ws-border-strong: #2a2d33 !important;
  --ws-fg: #f7f8f8 !important;
  --ws-fg-dim: #c1c4cc !important;
  --ws-fg-muted: #8a8f98 !important;
  --ws-fg-faint: #62666d !important;
  --ws-accent: #5e84ff !important;
  --ws-accent-soft: rgba(94, 132, 255, 0.12) !important;
  --ws-accent-line: rgba(94, 132, 255, 0.30) !important;
  --ws-amber: #f5a524 !important;
  --ws-pad: 16px !important;
}

body, gradio-app, .gradio-container {
  background: #08090a !important;
  color: #f7f8f8 !important;
  font-family: "Geist", "Inter", ui-sans-serif, system-ui, sans-serif !important;
  font-feature-settings: "cv11", "ss01", "ss03" !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
}

.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
}

.gradio-container .main, .gradio-container > .wrap, .gradio-container .contain {
  background: #08090a !important;
}

/* Reset Gradio's universal ".block" wrapper — we apply card styling selectively below. */
.gradio-container .block {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* Status-tracker overlay: KEPT VISIBLE — it renders the live per-step progress
   bar (driven by handle.generate's callback_on_step_end → gr.Progress). The
   demo-mode suppression that used to hide it was the real cause of the "no
   progress bar" report. Only soften its chrome to match the Linear theme. */
#ws-content .wrap.generating,
#ws-content .wrap.full.generating {
  background: rgba(10, 12, 16, 0.72) !important;
  backdrop-filter: blur(2px);
}

/* Hide Gradio's footer noise. */
footer, .footer, .gradio-container > footer { display: none !important; }
.api-docs, .built-with, .show-api { display: none !important; }

/* ─── Header chrome ────────────────────────────────────────────────── */
#ws-header {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  background: rgba(8, 9, 10, 0.92) !important;
  backdrop-filter: saturate(180%) blur(14px) !important;
  -webkit-backdrop-filter: saturate(180%) blur(14px) !important;
  border-bottom: 1px solid var(--ws-border) !important;
  padding: 10px 20px !important;
  margin: 0 !important;
  position: sticky !important;
  top: 0 !important;
  z-index: 50 !important;
  align-items: center !important;
  gap: 22px !important;
  min-height: 64px !important;
  border-radius: 0 !important;
  width: 100% !important;
  overflow: visible !important;
}

#ws-header > * {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.ws-chrome-col {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
  min-width: 0 !important;
  flex: 0 0 auto !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  gap: 2px !important;
}
.ws-chrome-col > * { background: transparent !important; border: 0 !important; padding: 0 !important; box-shadow: none !important; }
.ws-chrome-col .block, .ws-chrome-col .form { background: transparent !important; border: 0 !important; padding: 0 !important; }

.ws-brand-col { flex: 1 1 auto !important; min-width: 0 !important; }
.ws-chrome-right { margin-left: auto !important; }

#ws-brand-html {
  background: transparent !important;
  padding: 0 !important;
  border: 0 !important;
}

.ws-brand {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
}

.ws-brand-mark {
  width: 22px !important; height: 22px !important;
  border-radius: 6px !important;
  background:
    radial-gradient(120% 80% at 30% 20%, #aeb8ff 0%, #5e84ff 40%, #3a52b8 100%) !important;
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,0.18),
    0 1px 2px rgba(0,0,0,0.6) !important;
}

.ws-brand-text {
  display: flex !important;
  flex-direction: column !important;
  line-height: 1.1 !important;
}

.ws-brand-name {
  color: var(--ws-fg) !important;
  font-weight: 510 !important;
  font-size: 14px !important;
  letter-spacing: -0.012em !important;
}

.ws-brand-sub {
  color: var(--ws-fg-muted) !important;
  font-size: 11px !important;
  font-weight: 400 !important;
  letter-spacing: 0 !important;
}

/* ─── Field labels in header ───────────────────────────────────────── */
.ws-field-label {
  color: var(--ws-fg-muted) !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  font-weight: 510 !important;
  margin: 0 0 6px 0 !important;
  padding: 0 !important;
}

/* Hide gradio's own labels for the header dropdown (we provide our own visual hierarchy). */
.ws-chrome-col .ws-dropdown label > span,
.ws-chrome-col .ws-dropdown .label-wrap,
.ws-chrome-col .ws-dropdown > label > span:first-child {
  display: none !important;
}
.ws-chrome-col .ws-dropdown > label { gap: 0 !important; }

/* ─── Generation dropdown ──────────────────────────────────────────── */
.ws-dropdown {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
.ws-dropdown .wrap, .ws-dropdown .secondary-wrap, .ws-dropdown .container {
  background: var(--ws-surface-2) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 6px !important;
  min-height: 32px !important;
  box-shadow: none !important;
}
.ws-dropdown input, .ws-dropdown .single-select, .ws-dropdown .token {
  background: transparent !important;
  color: var(--ws-fg) !important;
  font-size: 13px !important;
  font-weight: 510 !important;
  padding: 5px 10px !important;
  height: auto !important;
}
.ws-dropdown:hover .wrap { border-color: var(--ws-border-strong) !important; }
.ws-dropdown .wrap:focus-within {
  border-color: var(--ws-accent) !important;
  box-shadow: 0 0 0 3px var(--ws-accent-soft) !important;
}

/* ─── Preset pill toggle ───────────────────────────────────────────── */
.ws-preset-group {
  display: inline-flex !important;
  background: var(--ws-surface-2) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 7px !important;
  padding: 2px !important;
  gap: 0 !important;
  width: max-content !important;
  flex-wrap: nowrap !important;
}
.ws-preset-group > * { flex: 0 0 auto !important; }

button.ws-pill {
  background: transparent !important;
  color: var(--ws-fg-muted) !important;
  border: 0 !important;
  border-radius: 5px !important;
  padding: 4px 12px !important;
  min-width: 0 !important;
  min-height: 26px !important;
  font-size: 12px !important;
  font-weight: 510 !important;
  letter-spacing: 0 !important;
  box-shadow: none !important;
  transition: background 120ms ease, color 120ms ease !important;
}
button.ws-pill:hover { color: var(--ws-fg-dim) !important; background: transparent !important; }
button.ws-pill-active {
  background: rgba(94, 132, 255, 0.14) !important;
  color: #ffffff !important;
  /* Drop the outer 0 1px 0 shadow that created a faint horizontal line under
     the segmented control — the sticky header already owns the bottom hairline. */
  box-shadow: inset 0 0 0 1px #5e84ff !important;
}

/* ─── Chrome nav buttons ───────────────────────────────────────────── */
.ws-chrome-actions {
  gap: 4px !important;
  justify-content: flex-end !important;
  flex-wrap: nowrap !important;
}
button.ws-nav-btn {
  background: transparent !important;
  color: var(--ws-fg-dim) !important;
  border: 1px solid transparent !important;
  border-radius: 6px !important;
  padding: 5px 10px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  min-height: 30px !important;
  min-width: 0 !important;
  letter-spacing: 0 !important;
  box-shadow: none !important;
  transition: background 120ms ease, border-color 120ms ease, color 120ms ease !important;
}
button.ws-nav-btn:hover {
  background: var(--ws-surface-2) !important;
  border-color: var(--ws-border) !important;
  color: var(--ws-fg) !important;
}

/* ─── Sidebar ──────────────────────────────────────────────────────── */
#ws-sidebar {
  background: #0a0b0d !important;
  border-right: 1px solid var(--ws-border) !important;
  padding: 16px 12px !important;
  min-height: calc(100vh - 56px) !important;
  flex-shrink: 0 !important;
  gap: 2px !important;
}

.ws-side-heading {
  padding: 6px 10px 4px 10px !important;
  margin-top: 2px !important;
}
.ws-side-heading-divider { margin-top: 18px !important; }
.ws-side-heading-text {
  color: var(--ws-fg-muted) !important;
  font-size: 10.5px !important;
  font-weight: 510 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}

button.ws-side-btn {
  background: transparent !important;
  color: var(--ws-fg-dim) !important;
  border: 0 !important;
  border-radius: 6px !important;
  padding: 6px 10px 6px 22px !important;
  margin: 1px 0 !important;
  width: 100% !important;
  text-align: left !important;
  justify-content: flex-start !important;
  font-size: 13px !important;
  font-weight: 450 !important;
  min-height: 28px !important;
  letter-spacing: -0.005em !important;
  position: relative !important;
  box-shadow: none !important;
  transition: background 120ms ease, color 120ms ease !important;
}
/* Inactive nav rows have no leading glyph — Linear's nav rests on the
   selected-state blue marker alone. */
button.ws-side-btn::before {
  content: "" !important;
  position: absolute !important;
  left: 10px !important; top: 50% !important;
  width: 4px !important; height: 4px !important;
  border-radius: 50% !important;
  background: transparent !important;
  transform: translateY(-50%) !important;
  opacity: 0 !important;
  transition: background 120ms ease, opacity 120ms ease, box-shadow 120ms ease !important;
}
button.ws-side-btn:hover {
  background: var(--ws-surface-2) !important;
  color: var(--ws-fg) !important;
}
/* No `:hover::before` rule — hovering must not override the active dot's
   `::before` (which has lower specificity than :hover would have). The base
   `button.ws-side-btn::before` keeps inactive dots invisible regardless. */
button.ws-side-btn-active {
  background: var(--ws-elev) !important;
  color: var(--ws-fg) !important;
  font-weight: 510 !important;
}
button.ws-side-btn-active::before {
  background: var(--ws-accent) !important;
  opacity: 1 !important;
  box-shadow: 0 0 0 3px rgba(94, 132, 255, 0.18) !important;
}

.ws-side-footer {
  margin-top: auto !important;
  padding: 14px 10px 4px 10px !important;
  border-top: 1px solid var(--ws-border) !important;
  margin-top: 24px !important;
}
.ws-side-footer-row {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  font-size: 10.5px !important;
  color: var(--ws-fg-muted) !important;
  font-family: "Geist Mono", ui-monospace, monospace !important;
}
.ws-side-footer-status { color: #5fa881 !important; }
.ws-side-footer-hint {
  font-size: 10.5px !important;
  color: var(--ws-fg-faint) !important;
  margin-top: 3px !important;
  letter-spacing: -0.005em !important;
}

/* ─── Tab/content area ─────────────────────────────────────────────── */
#ws-content {
  background: #08090a !important;
  padding: 24px 28px !important;
  min-height: calc(100vh - 56px) !important;
}
#ws-content > * { background: transparent !important; }

#ws-content h2 {
  font-size: 22px !important;
  font-weight: 510 !important;
  letter-spacing: -0.018em !important;
  color: var(--ws-fg) !important;
  margin: 0 0 4px 0 !important;
}
#ws-content h3 {
  font-size: 14px !important;
  font-weight: 510 !important;
  letter-spacing: -0.005em !important;
  color: var(--ws-fg) !important;
  margin: 4px 0 !important;
}
#ws-content p, #ws-content li {
  color: var(--ws-fg-dim) !important;
  font-size: 13px !important;
  line-height: 1.55 !important;
}

/* Markdown blocks */
.markdown, .markdown * { background: transparent !important; }

/* Per-tab heading row — replace plain h2 with a leader rule. */
#ws-content div[id^="tab-"] h2 {
  display: flex !important;
  align-items: center !important;
  gap: 14px !important;
  padding-bottom: 14px !important;
  margin: 4px 0 22px 0 !important;
  border-bottom: 1px solid var(--ws-border) !important;
  font-size: 22px !important;
  font-weight: 510 !important;
  letter-spacing: -0.018em !important;
  color: var(--ws-fg) !important;
}
#ws-content div[id^="tab-"] h2::before {
  content: "" !important;
  display: inline-block !important;
  width: 3px !important; height: 18px !important;
  background: var(--ws-accent) !important;
  border-radius: 2px !important;
  flex-shrink: 0 !important;
}

/* ─── Form blocks (inputs / textareas / dropdowns) ─────────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container .input-text,
.gradio-container .scroll-hide {
  background: var(--ws-surface-2) !important;
  color: var(--ws-fg) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 6px !important;
  font-family: "Geist", ui-sans-serif, system-ui, sans-serif !important;
  font-size: 13px !important;
  padding: 8px 10px !important;
  transition: border-color 120ms ease, box-shadow 120ms ease !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
  outline: 0 !important;
  border-color: var(--ws-accent) !important;
  box-shadow: 0 0 0 3px var(--ws-accent-soft) !important;
}

.gradio-container ::placeholder { color: var(--ws-fg-faint) !important; }

/* ─── Card styling (Gradio 6 — .block based) ────────────────────────── */
/* Padded inputs (textbox, slider, dropdown, etc.) get a refined card. */
#ws-content .block.padded:not(.hide-container) {
  background: var(--ws-surface) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 8px !important;
  padding: 14px !important;
  box-shadow: none !important;
}

/* Media blocks (Image, Video, Audio, Gallery) — frame without internal padding. */
#ws-content .block:not(.padded):not(.hide-container) {
  background:
    linear-gradient(180deg, rgba(94,132,255,0.035) 0%, rgba(8,9,10,0) 60%),
    var(--ws-surface) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  position: relative !important;
  min-height: 180px !important;
}
#ws-content .block:not(.padded):not(.hide-container) > .wrap {
  background: transparent !important;
  min-height: 320px !important;
}

/* Floating overlay label for media blocks. */
#ws-content .block:not(.padded):not(.hide-container) > [data-testid="block-label"] {
  position: absolute !important;
  top: 10px !important; left: 12px !important;
  background: rgba(8,9,10,0.72) !important;
  backdrop-filter: blur(6px) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 4px !important;
  padding: 3px 8px !important;
  font-size: 10.5px !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: var(--ws-fg-muted) !important;
  z-index: 5 !important;
  margin: 0 !important;
}

/* The form wrapper (Gradio groups consecutive inputs into a form) — give it spacing only. */
#ws-content .form { background: transparent !important; gap: 12px !important; }

/* Inputs INSIDE accordions: strip the nested card. */
#ws-content [data-testid="accordion-content"] .block.padded {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}

/* Generate button — full-width on tab inputs. The ▶ glyph prefix was dropped
   (it was rendering near-black on electric blue and reading as a hairline). */
#ws-content button.primary, #ws-content button[class*="primary"] {
  width: 100% !important;
  text-align: center !important;
  color: #ffffff !important;
  font-weight: 510 !important;
}
#ws-content button.primary::before, #ws-content button[class*="primary"]::before {
  content: none !important;
}

/* Accordions in tab content are their own card. */
#ws-content .gradio-accordion {
  background: var(--ws-surface) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 8px !important;
  padding: 0 !important;
}
#ws-content .gradio-accordion > .label-wrap {
  padding: 10px 14px !important;
}
#ws-content .gradio-accordion .wrap {
  padding: 0 14px 14px 14px !important;
}

/* Don't card-style nested cards (e.g. accordion children). */
#ws-content .gradio-accordion .gradio-textbox,
#ws-content .gradio-accordion .gradio-slider,
#ws-content .gradio-accordion .gradio-checkbox,
#ws-content .gradio-accordion .gradio-radio,
#ws-content .gradio-accordion .gradio-dropdown {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}

.gap, .ws-stack { gap: 14px !important; }

/* Labels above inputs (Gradio 6 — [data-testid="block-label"]). */
#ws-content [data-testid="block-label"],
#ws-content [data-testid="block-label"] span {
  color: var(--ws-fg-muted) !important;
  font-size: 10.5px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  font-weight: 510 !important;
  margin: 0 0 8px 0 !important;
  font-family: "Geist", ui-sans-serif, system-ui, sans-serif !important;
  background: transparent !important;
  padding: 0 !important;
  border: 0 !important;
}

/* Accordion label — slightly larger, mixed case. */
#ws-content [data-testid="accordion-content"] ~ button,
#ws-content button.label-wrap,
#ws-content .label-wrap {
  background: transparent !important;
}
#ws-content .accordion .label-wrap, #ws-content [class*="accordion"] button.label-wrap {
  font-size: 12px !important;
  font-weight: 510 !important;
  color: var(--ws-fg) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  padding: 12px 14px !important;
}

/* Sliders */
.gradio-container input[type="range"] {
  height: 4px !important;
  background: var(--ws-border-strong) !important;
  border-radius: 999px !important;
  padding: 0 !important;
}
.gradio-container .wrap.svelte-1cl284s .head .min,
.gradio-container .wrap.svelte-1cl284s .head .max { color: var(--ws-fg-muted) !important; font-size: 11px !important; }

/* Accordions */
.accordion, .gradio-accordion {
  background: var(--ws-surface) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 8px !important;
}
.accordion .label-wrap, .gradio-accordion .label-wrap {
  font-size: 12px !important;
  font-weight: 510 !important;
  color: var(--ws-fg) !important;
  padding: 10px 14px !important;
}

/* Radio inputs INSIDE tabs (not chrome) — Linear-style chip list */
.gradio-container .gr-radio, .gradio-container .wrap.gr-radio {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
  gap: 6px !important;
  flex-wrap: wrap !important;
}
.gradio-container .gr-radio label,
.gradio-container .wrap.gr-radio label {
  background: var(--ws-surface-2) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 6px !important;
  padding: 5px 10px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--ws-fg-dim) !important;
  transition: all 120ms ease !important;
}
.gradio-container .gr-radio label:hover { border-color: var(--ws-border-strong) !important; color: var(--ws-fg) !important; }
.gradio-container .gr-radio input[type="radio"] { display: none !important; }
.gradio-container .gr-radio label.selected {
  background: var(--ws-accent-soft) !important;
  border-color: var(--ws-accent-line) !important;
  color: var(--ws-fg) !important;
}

/* Primary "Generate" buttons inside tabs — accent fill, white label. */
button[class*="primary"], button.lg.primary, .gradio-container .primary,
.gr-button-primary, button[variant="primary"] {
  background: var(--ws-accent) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 6px !important;
  font-weight: 510 !important;
  font-size: 13px !important;
  padding: 10px 16px !important;
  min-height: 38px !important;
  letter-spacing: -0.005em !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.35) !important;
  transition: filter 120ms ease, transform 80ms ease !important;
}
button[class*="primary"]:hover { filter: brightness(1.06) !important; }
button[class*="primary"]:active { transform: translateY(1px) !important; }

/* Generic secondary buttons — exclude chrome buttons (ws-pill, ws-side-btn,
   ws-nav-btn) explicitly so the segmented control / sidebar can claim their
   own visual treatment without specificity wars. */
button.secondary:not(.ws-pill):not(.ws-side-btn):not(.ws-nav-btn),
.gradio-container button:not([class*="primary"]):not(.ws-pill):not(.ws-side-btn):not(.ws-nav-btn) {
  background: var(--ws-surface-2) !important;
  color: var(--ws-fg-dim) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  box-shadow: none !important;
}
.gradio-container button:not([class*="primary"]):hover {
  background: var(--ws-elev) !important;
  border-color: var(--ws-border-strong) !important;
  color: var(--ws-fg) !important;
}

/* ─── Warning banner (preserve original semantics, Linear-tinted) ──── */
.warning-banner {
  background: rgba(245, 165, 36, 0.08) !important;
  border: 1px solid rgba(245, 165, 36, 0.25) !important;
  border-left: 3px solid var(--ws-amber) !important;
  border-radius: 6px !important;
  padding: 10px 14px !important;
  color: #f5d18c !important;
  font-size: 12px !important;
}

/* ─── Dev banner (top of page, all viewports) ─────────────────────── */
.ws-dev-banner {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 10px !important;
  padding: 9px 16px !important;
  background: linear-gradient(180deg, #2d2418 0%, #261f15 100%) !important;
  border-bottom: 1px solid #5a4520 !important;
  color: #f1b863 !important;
  font-family: "Geist", "Inter", ui-sans-serif, system-ui, sans-serif !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  letter-spacing: -0.005em !important;
  text-align: center !important;
  line-height: 1.4 !important;
}
.ws-dev-banner .ws-dev-icon {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #f1b863;
  box-shadow: 0 0 0 3px rgba(241,184,99,0.18);
  flex-shrink: 0;
}
.ws-dev-banner b { color: #ffd494 !important; font-weight: 600 !important; }
.ws-dev-banner a {
  color: #ffd494 !important;
  text-decoration: underline !important;
  text-decoration-color: rgba(255,212,148,0.4) !important;
  text-underline-offset: 2px !important;
}
.ws-dev-banner a:hover { text-decoration-color: #ffd494 !important; }
@media (max-width: 767px) {
  .ws-dev-banner { font-size: 11.5px !important; padding: 8px 12px !important; }
}

/* ─── Local backend banner ─────────────────────────────────────────── */
.ws-local-banner {
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 5px 10px !important;
  background: var(--ws-surface-2) !important;
  border: 1px solid var(--ws-border) !important;
  border-radius: 6px !important;
  font-size: 11.5px !important;
  color: var(--ws-fg-dim) !important;
  font-family: "Geist Mono", ui-monospace, monospace !important;
  margin-bottom: 14px !important;
}
.ws-local-banner .dot {
  width: 6px !important; height: 6px !important; border-radius: 50% !important;
  background: #5fa881 !important;
  box-shadow: 0 0 0 3px rgba(95,168,129,0.15) !important;
}

/* ─── Tertiary row spacing ─────────────────────────────────────────── */
#ws-content .row { gap: 10px !important; }
/* Small buttons (Send-to: I2V/VACE/Animate, etc) — readable inline pills. */
#ws-content button.sm, #ws-content button[class*=" sm "], #ws-content button.small, #ws-content button[size="sm"] {
  padding: 6px 12px !important;
  font-size: 11.5px !important;
  font-weight: 500 !important;
  min-height: 30px !important;
  min-width: 60px !important;
}

/* ─── Toasts ───────────────────────────────────────────────────────── */
.toast, .toast-body, .toast-wrap, .gradio-toast {
  background: var(--ws-elev) !important;
  border: 1px solid var(--ws-border-strong) !important;
  border-radius: 8px !important;
  color: var(--ws-fg) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.6), 0 2px 8px rgba(0,0,0,0.4) !important;
}

/* ─── Scrollbar tuning ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1c1e22; border-radius: 999px; border: 2px solid #08090a; }
::-webkit-scrollbar-thumb:hover { background: #2a2d33; }

/* ─── Gradio body row layout glue ──────────────────────────────────── */
#ws-body { gap: 0 !important; align-items: stretch !important; }
#ws-body > * { background: transparent !important; }

/* ─── Mode panel show/hide — driven purely by JS class toggle ──────────
   All 10 panels are mounted with `visible=True` (Gradio never touches
   their display). JS adds `ws-mode-panel-active` to the chosen one. */
.ws-mode-panel {
  display: none !important;
}
.ws-mode-panel.ws-mode-panel-active {
  display: flex !important;  /* gr.Column renders as flex column by default */
  flex-direction: column !important;
}

/* ─── Inline code chips inside mode-title headings ────────────────────
   Backticks were rendering as bordered pills sized to the 22px heading,
   which made S2V/TI2V titles look broken. Render them as plain text. */
#ws-content .mode-title code,
#ws-content h2 code,
#ws-content h2 :is(code, kbd, samp) {
  background: transparent !important;
  border: 0 !important;
  font-size: inherit !important;
  font-family: inherit !important;
  padding: 0 !important;
  color: inherit !important;
  border-radius: 0 !important;
}

/* ─── Gallery (read-only display, not upload zone) ─────────────────────
   `interactive=False` already hides the upload affordance, but suppress
   any lingering "Drop Media Here" overlay defensively. */
.ws-gallery-readonly [data-testid="upload-button"],
.ws-gallery-readonly .upload-container,
.ws-gallery-readonly .upload-button,
.ws-gallery-readonly button.upload,
.ws-gallery-readonly .icon-button-wrapper.upload-button-wrapper {
  display: none !important;
}
.ws-gallery-empty {
  color: var(--ws-fg-muted) !important;
  font-size: 12.5px !important;
  font-style: italic !important;
  padding: 4px 2px 8px 2px !important;
}

/* ─── Preset pill specificity overrides ────────────────────────────────
   gr.Button defaults to variant="secondary", which lands `class="secondary"`
   on the rendered <button>. The generic `button.secondary` rule earlier in
   this stylesheet beats our `button.ws-pill` rule on specificity ties (both
   0,1,1) by source order. We bump specificity with `.gradio-container`
   prefix (0,2,1) so the segmented-control state is unambiguous. */
.gradio-container button.ws-pill {
  background: transparent !important;
  border-color: transparent !important;
  color: var(--ws-fg-muted) !important;
  box-shadow: none !important;
}
.gradio-container button.ws-pill:hover:not(.ws-pill-active) {
  background: transparent !important;
  color: var(--ws-fg-dim) !important;
  border-color: transparent !important;
}
.gradio-container button.ws-pill.ws-pill-active {
  background: rgba(94, 132, 255, 0.18) !important;
  color: #ffffff !important;
  box-shadow: inset 0 0 0 1px #5e84ff !important;
  border-color: #5e84ff !important;
}

/* ─── Hamburger (drawer trigger) ─────────────────────────────────────
   Lives inside `.ws-brand` and is hidden on desktop. Made visible at
   ≤767px below. Styled to match the brand restraint (no fill, just a
   subtle hairline + the icon). */
.ws-hamburger {
  display: none;
  background: transparent;
  border: 1px solid var(--ws-border);
  border-radius: 6px;
  color: var(--ws-fg-dim);
  width: 32px;
  height: 32px;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  padding: 0;
  margin-right: 8px;
  transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
  flex-shrink: 0;
}
.ws-hamburger:hover {
  background: var(--ws-surface-2);
  border-color: var(--ws-border-strong);
  color: var(--ws-fg);
}
.ws-hamburger:active { transform: translateY(1px); }

/* Backdrop covers main content when drawer is open. Injected via JS,
   so it always exists in the DOM but is only interactive when the
   drawer is open. Hidden on desktop unconditionally. */
.ws-sidebar-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  z-index: 150;
  opacity: 0;
  pointer-events: none;
  transition: opacity 200ms ease-out;
  display: none;
}

/* ═══════════════════════════════════════════════════════════════════════
   Responsive: tablet (≤1023px) — extend the Linear aesthetic, narrow
   the chrome, drop secondary affordances. The desktop styles above are
   the source of truth — these rules ONLY override the bits that don't
   work on smaller screens.
   ═══════════════════════════════════════════════════════════════════ */
@media (max-width: 1023px) {
  /* Brand: keep the name, drop the subtitle (frees the brand column). */
  .ws-brand-sub { display: none !important; }
  .ws-chrome-col.ws-brand-col { min-width: 0 !important; }

  /* Sidebar narrows: 248 → 200 px. */
  #ws-sidebar {
    min-width: 200px !important;
    padding: 14px 10px !important;
  }
  button.ws-side-btn { padding: 6px 10px 6px 20px !important; }

  /* Header: tighter gaps, slightly shorter. */
  #ws-header {
    gap: 14px !important;
    padding: 8px 16px !important;
    min-height: 56px !important;
  }

  /* Nav buttons collapse to icon-only via a leading dot marker
     (Linear's restrained "●" pattern). The labels stay readable for
     assistive tech but visually hidden. */
  button.ws-nav-btn {
    padding: 5px 10px !important;
    font-size: 0 !important;
    line-height: 0 !important;
    min-width: 32px !important;
  }
  button.ws-nav-btn::before {
    content: "●" !important;
    display: inline-block !important;
    font-size: 10px !important;
    line-height: 1 !important;
    color: var(--ws-fg-muted) !important;
  }
  button.ws-nav-btn:hover::before { color: var(--ws-fg) !important; }

  /* Tighter tab content padding (32 → 20). */
  #ws-content { padding: 20px 20px !important; }
}

/* ═══════════════════════════════════════════════════════════════════════
   Responsive: mobile (≤767px) — sidebar becomes an overlay drawer,
   header collapses to monogram + hamburger, input/output stack to a
   single column.
   ═══════════════════════════════════════════════════════════════════ */
@media (max-width: 767px) {
  /* ── Header: 48px, brand collapses to monogram only. ───────────── */
  #ws-header {
    min-height: 48px !important;
    padding: 6px 12px !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
    row-gap: 6px !important;
  }
  .ws-brand-text { display: none !important; }
  .ws-brand { gap: 0 !important; }

  /* Hamburger visible on mobile, sits left of brand mark. */
  .ws-hamburger { display: inline-flex !important; }

  /* Brand column shrinks; generation + preset wrap to a second row
     beneath the brand on mobile (header flex-wraps). */
  .ws-chrome-col.ws-brand-col {
    min-width: 0 !important;
    flex: 0 0 auto !important;
  }
  .ws-chrome-col {
    min-width: 0 !important;
    flex: 1 1 auto !important;
  }
  .ws-chrome-right { flex: 0 0 auto !important; }

  /* History/Settings move into the drawer footer; hide the chrome
     versions on mobile. (Existing JS still wires the clicks via the
     sidebar entries.) */
  .ws-chrome-right { display: none !important; }

  /* Generation dropdown + preset pills sit on the second row,
     thumb-reachable. */
  .ws-dropdown .wrap, .ws-dropdown .secondary-wrap, .ws-dropdown .container {
    min-height: 40px !important;
  }
  .ws-dropdown input, .ws-dropdown .single-select, .ws-dropdown .token {
    font-size: 14px !important;
    padding: 8px 12px !important;
  }
  button.ws-pill {
    min-height: 36px !important;
    padding: 7px 16px !important;
    font-size: 13px !important;
  }

  /* ── Sidebar becomes an overlay drawer ─────────────────────────── */
  #ws-sidebar {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 280px !important;
    height: 100vh !important;
    min-height: 100vh !important;
    max-height: 100vh !important;
    z-index: 200 !important;
    transform: translateX(-100%) !important;
    transition: transform 200ms ease-out !important;
    padding: 16px 12px !important;
    background: #0a0b0d !important;
    border-right: 1px solid var(--ws-border) !important;
    overflow-y: auto !important;
    /* Override the desktop `flex-shrink: 0` parent-row behavior — when
       absolute-positioned the row no longer reserves space for us. */
    flex-shrink: 0 !important;
  }
  /* IMPORTANT: Gradio injects a container-scoped copy of every CSS rule
     prefixed with `.gradio-container... .contain`. That means
     `body.X #Y` becomes `.contain body.X #Y`, which never matches because
     `body` is ABOVE `.contain` in the DOM. So drawer-open state lives on
     the sidebar element itself (and on the backdrop element itself),
     which the JS toggles. The `body.ws-sidebar-open` rule outside the
     media query is only used for the unprefixed-sheet scroll lock. */
  #ws-sidebar.ws-open {
    transform: translateX(0) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
  }

  /* Backdrop becomes interactive on mobile when drawer is open. */
  .ws-sidebar-backdrop { display: block !important; }
  .ws-sidebar-backdrop.ws-open {
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  /* Drawer touch targets need to be larger. */
  button.ws-side-btn {
    min-height: 40px !important;
    padding: 10px 10px 10px 24px !important;
    font-size: 14px !important;
  }
  .ws-side-footer { display: none !important; }

  /* ── Main content: stack to single column ─────────────────────── */
  #ws-content {
    padding: 16px !important;
    min-height: calc(100vh - 48px) !important;
  }
  /* The `_two_col` Row in tabs.py uses default flex-row; flip to
     column on mobile. Limit to direct rows inside an active panel
     so we don't break inline rows (e.g. resolution + duration). */
  #ws-content .ws-mode-panel > div > .row,
  #ws-content .ws-mode-panel > .row {
    flex-direction: column !important;
  }
  #ws-content .ws-mode-panel > div > .row > *,
  #ws-content .ws-mode-panel > .row > * {
    width: 100% !important;
    min-width: 0 !important;
  }

  /* Tab heading slightly smaller on mobile. */
  #ws-content div[id^="tab-"] h2 {
    font-size: 19px !important;
    margin: 2px 0 16px 0 !important;
    padding-bottom: 12px !important;
  }

  /* Tighten card padding inside tabs. */
  #ws-content .block.padded:not(.hide-container) {
    padding: 12px !important;
  }

  /* Generate CTA: large + tappable. */
  #ws-content button.primary,
  #ws-content button[class*="primary"] {
    min-height: 48px !important;
    font-size: 14px !important;
    padding: 12px 18px !important;
  }

  /* Inputs inside tabs: bump touch sizing. */
  .gradio-container input[type="text"],
  .gradio-container input[type="number"],
  .gradio-container textarea {
    font-size: 14px !important;
    padding: 10px 12px !important;
  }

  /* Resolution + duration row: keep inline on mobile (was already a
     wrap row in tabs.py), but constrain to two columns. */
  #ws-content .form .row {
    flex-direction: column !important;
    gap: 12px !important;
  }
  #ws-content .form .row > * { width: 100% !important; }

  /* Local-backend banner shrinks. */
  .ws-local-banner {
    font-size: 10.5px !important;
    padding: 5px 8px !important;
  }

  /* "Send to:" row stacks. */
  #ws-content .ws-mode-panel .row .gradio-button.sm {
    min-width: 0 !important;
  }
}

/* iOS body-scroll lock: prevent rubber-band when drawer is open. */
body.ws-sidebar-open { overflow: hidden !important; }
"""


def build() -> gr.Blocks:
    backend = detect()

    with gr.Blocks(
        title="Wan Studio — Linear",
        analytics_enabled=False,
        theme=THEME,
        css=CSS,
    ) as demo:
        # ── Dev banner (top of every page) ───────────────────────────────
        gr.HTML(
            '<div class="ws-dev-banner">'
            '<span class="ws-dev-icon"></span>'
            '<span><b>Wan Studio is in active development.</b> '
            'Please don\'t run inference — every GPU click burns the maintainer\'s '
            'ZeroGPU quota. Follow along at '
            '<a href="https://github.com/techfreakworm/wan-studio" target="_blank" rel="noopener">github.com/techfreakworm/wan-studio</a>.'
            '</span></div>',
            elem_id="ws-dev-banner",
        )

        # ── Header ───────────────────────────────────────────────────────
        header = build_header()

        # ── Main: sidebar + content ──────────────────────────────────────
        with gr.Row(elem_id="ws-body"):
            sidebar = build_sidebar()
            with gr.Column(scale=10, elem_id="ws-content"):
                # Local-backend chip (not on ZeroGPU)
                if not backend.is_zerogpu:
                    gr.HTML(
                        f'<div class="ws-local-banner">'
                        f'<span class="dot"></span>'
                        f'<span><b style="color:var(--ws-fg)">local</b> · '
                        f'{backend.label} · dtype <code>{backend.dtype}</code> · '
                        f'vae <code>{backend.vae_dtype}</code></span>'
                        f'</div>',
                        elem_id="ws-local-banner",
                    )
                tabs = build_all_tabs()

        # ── Tab navigation — 100% client-side ────────────────────────────
        # Round-1 fix: the original cascade (10× gr.update(visible=...) per
        # click) crashed Svelte with effect_update_depth_exceeded. Switching
        # to `gr.Tabs` didn't help — its internal Svelte machinery has the
        # same N-update fan-out when `selected` changes. So we drop ALL
        # Gradio-backed navigation and toggle visibility purely in the DOM:
        #   - Every panel is mounted with `visible=True` and class
        #     `ws-mode-panel`. T2V also has `ws-mode-panel-active`.
        #   - CSS hides `.ws-mode-panel:not(.ws-mode-panel-active)`.
        #   - A small JS listener moves the active class on sidebar clicks.
        # Result: each sidebar click is exactly ZERO Gradio backend calls.
        #
        # Gradio strips <script> from gr.HTML, so we inject via demo.load(js=).
        _NAV_JS = """
        () => {
          if (window.__wsNavBound) { return []; }
          window.__wsNavBound = true;
          var MODE_KEYS = ['t2v','i2v','ti2v','flf2v','v2v','vace','s2v','animate','gallery','settings'];

          // ── Mobile drawer helpers ─────────────────────────────────────
          // Injected backdrop covers main content when drawer is open.
          // IMPORTANT: Must be appended INSIDE `.gradio-container` (or
          // its `.contain` child) because Gradio prefixes CSS rules with
          // `.gradio-container .contain` — anything outside that scope
          // doesn't match the prefixed copy of our rules.
          var backdropParent =
              document.querySelector('.gradio-container .contain') ||
              document.querySelector('.gradio-container') ||
              document.body;
          var backdrop = document.querySelector('.ws-sidebar-backdrop');
          if (!backdrop) {
            backdrop = document.createElement('div');
            backdrop.className = 'ws-sidebar-backdrop';
            backdrop.setAttribute('aria-hidden', 'true');
            backdropParent.appendChild(backdrop);
          }
          // Drawer state lives on the sidebar + backdrop elements directly
          // (Gradio prefixes every CSS selector with `.contain`, which breaks
          // `body.X #Y` patterns since `body` is above `.contain`). We also
          // mirror the state to body.ws-sidebar-open for the unprefixed
          // scroll-lock rule.
          function openDrawer() {
            document.body.classList.add('ws-sidebar-open');
            var sb = document.getElementById('ws-sidebar');
            if (sb) sb.classList.add('ws-open');
            if (backdrop) backdrop.classList.add('ws-open');
          }
          function closeDrawer() {
            document.body.classList.remove('ws-sidebar-open');
            var sb = document.getElementById('ws-sidebar');
            if (sb) sb.classList.remove('ws-open');
            if (backdrop) backdrop.classList.remove('ws-open');
          }
          function isDrawerOpen() {
            var sb = document.getElementById('ws-sidebar');
            return sb && sb.classList.contains('ws-open');
          }
          function isDrawerViewport() { return window.matchMedia('(max-width: 767px)').matches; }

          function setActive(key) {
            if (MODE_KEYS.indexOf(key) === -1) return;
            document.querySelectorAll('.ws-side-btn').forEach(function(b) {
              b.classList.remove('ws-side-btn-active');
            });
            document.querySelectorAll('.ws-mode-panel').forEach(function(p) {
              p.classList.remove('ws-mode-panel-active');
            });
            var sideId = (key === 'gallery') ? 'ws-mode-gallery'
                       : (key === 'settings') ? 'ws-mode-settings'
                       : 'ws-mode-' + key;
            var side = document.getElementById(sideId);
            if (side) side.classList.add('ws-side-btn-active');
            var panel = document.getElementById('tab-' + key);
            if (panel) panel.classList.add('ws-mode-panel-active');
            // On mobile, picking a mode collapses the drawer so the
            // tab content is visible immediately. No-op at ≥768px.
            if (isDrawerViewport()) closeDrawer();
          }
          var sidebar = document.getElementById('ws-sidebar');
          if (sidebar) {
            sidebar.addEventListener('click', function(e) {
              var btn = e.target.closest('.ws-side-btn');
              if (!btn) return;
              var m = (btn.id || '').match(/^ws-mode-(.+)$/);
              if (m) setActive(m[1]);
            });
          }
          var hist = document.getElementById('ws-history-btn');
          var sett = document.getElementById('ws-settings-btn');
          if (hist) hist.addEventListener('click', function(){ setActive('gallery'); });
          if (sett) sett.addEventListener('click', function(){ setActive('settings'); });

          // ── Hamburger + backdrop click bindings ──────────────────────
          var burger = document.getElementById('ws-hamburger');
          if (burger) {
            burger.addEventListener('click', function(e) {
              e.preventDefault();
              if (isDrawerOpen()) { closeDrawer(); } else { openDrawer(); }
            });
          }
          backdrop.addEventListener('click', closeDrawer);
          // Escape key closes the drawer (a11y nicety).
          document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && isDrawerOpen()) {
              closeDrawer();
            }
          });
          // If the viewport grows past mobile while the drawer is open
          // (rotation, devtools toggle), close it so the sticky desktop
          // sidebar takes over cleanly.
          window.addEventListener('resize', function() {
            if (!isDrawerViewport() && isDrawerOpen()) {
              closeDrawer();
            }
          });

          setActive('t2v');
          return [];
        }
        """

        # ── Preset pill toggle (no gr.Radio in chrome) ───────────────────
        def _set_preset(value: str):
            return (
                value,
                gr.update(elem_classes=(
                    ["ws-pill", "ws-pill-active"] if value == "fast" else ["ws-pill"]
                )),
                gr.update(elem_classes=(
                    ["ws-pill", "ws-pill-active"] if value == "quality" else ["ws-pill"]
                )),
            )

        header["preset_fast"].click(
            fn=lambda: _set_preset("fast"),
            outputs=[header["preset_state"], header["preset_fast"], header["preset_quality"]],
        )
        header["preset_quality"].click(
            fn=lambda: _set_preset("quality"),
            outputs=[header["preset_state"], header["preset_fast"], header["preset_quality"]],
        )

        # ── Generate handlers ────────────────────────────────────────────
        # Drive the Generate wiring from HANDLER_REGISTRY: every registered
        # mode wires its Generate button to the per-tier @spaces.GPU entrypoint
        # (large/xlarge by spec.tier); unregistered modes keep the no-op toast
        # until their pipelines land in later waves.
        def _generate_toast():
            gr.Info("Demo mode — Generate disabled while design direction is being chosen.")

        # Common trailing Advanced inputs, shared by every wired mode.
        def _advanced_inputs(tab_in: dict) -> list:
            return [
                tab_in["negative_prompt"],
                tab_in["seed"],
                tab_in["randomize"],
                tab_in["steps"],
                tab_in["cfg"],
                tab_in["cfg_2"],
            ]

        # Per-mode ordered inputs list — must match the runner's unpacking in
        # `_run_<mode>` (i2v leads with the image; both share generation/preset
        # from the header + resolution/duration from the tab, then advanced).
        def _inputs_for(mode: str, tab: dict, hdr: dict) -> list:
            tab_in = tab["inputs"]
            # V2V: (video, generation, preset, prompt, strength, *advanced).
            if mode == "v2v":
                return [
                    tab_in["video"],
                    hdr["generation"],
                    hdr["preset_state"],
                    tab_in["prompt"],
                    tab_in["strength"],
                ] + _advanced_inputs(tab_in)
            # FLF2V: (start_frame, generation, preset, end_uploaded,
            #         end_generated, prompt, *advanced).
            if mode == "flf2v":
                return [
                    tab_in["start_frame"],
                    hdr["generation"],
                    hdr["preset_state"],
                    tab_in["end_frame_uploaded"],
                    tab_in["end_frame_generated"],
                    tab_in["prompt"],
                ] + _advanced_inputs(tab_in)
            # VACE: (submode, generation, preset, source_video, references,
            #        prompt, *advanced). mask_source/mask_input drive the
            #        deferred-mode UX only and are NOT threaded to the runner in v1.
            if mode == "vace":
                return [
                    tab_in["submode"],
                    hdr["generation"],
                    hdr["preset_state"],
                    tab_in["source_video"],
                    tab_in["references"],
                    tab_in["prompt"],
                ] + _advanced_inputs(tab_in)
            # S2V: (reference_image, audio, generation, preset, resolution, prompt,
            #       *advanced). pose_video + duration(markdown) are UI-only in v1.
            if mode == "s2v":
                return [
                    tab_in["reference_image"],
                    tab_in["audio"],
                    hdr["generation"],
                    hdr["preset_state"],
                    tab_in["resolution"],
                    tab_in["prompt"],
                ] + _advanced_inputs(tab_in)
            # ANIMATE: (character, driving, mode, res, prompt, *advanced).
            if mode == "animate":
                return [
                    tab_in["character"],
                    tab_in["driving"],
                    tab_in["mode"],
                    tab_in["res"],
                    tab_in["prompt"],
                ] + _advanced_inputs(tab_in)
            # TI2V: (image, prompt, generation, preset, orientation, *advanced).
            # Fixed-resolution (orientation radio, no resolution/duration sliders).
            if mode == "ti2v":
                return [
                    tab_in["image"],
                    tab_in["prompt"],
                    hdr["generation"],
                    hdr["preset_state"],
                    tab_in["orientation"],
                ] + _advanced_inputs(tab_in)
            lead = [tab_in["image"], tab_in["prompt"]] if mode == "i2v" else [tab_in["prompt"]]
            return lead + [
                hdr["generation"],
                hdr["preset_state"],
                tab_in["resolution"],
                tab_in["duration"],
            ] + _advanced_inputs(tab_in)

        # A mode wires to a real handler iff `_inputs_for` knows its layout
        # (i.e. it's in HANDLER_REGISTRY); otherwise it stays on the no-op toast.
        WIRED = set(HANDLER_REGISTRY)
        for mode, tab in tabs.items():
            tab_in = tab.get("inputs", {})
            if mode in WIRED and "generate" in tab_in:
                spec = HANDLER_REGISTRY[mode]
                entry = generate_xlarge if spec.tier == "xlarge" else generate_large
                tab_in["generate"].click(
                    fn=lambda *a, _m=mode, _entry=entry: _entry(_m, *a),
                    inputs=_inputs_for(mode, tab, header),
                    outputs=tab["outputs"]["video"],
                )
            elif "generate" in tab_in:
                tab_in["generate"].click(fn=_generate_toast, inputs=None, outputs=None)
        # FLF2V has a secondary "Generate end frame" button → real Wan T2I
        # sub-handler (num_frames=1). Bound directly (not via the HANDLER_REGISTRY
        # loop) so it does not double-fire the toast.
        if "inputs" in tabs.get("flf2v", {}) and "generate_end" in tabs["flf2v"]["inputs"]:
            flf2v_in = tabs["flf2v"]["inputs"]
            flf2v_in["generate_end"].click(
                fn=generate_end_frame,
                inputs=[flf2v_in["end_frame_prompt"], header["generation"]],
                outputs=[flf2v_in["end_frame_generated"]],
            )

        # Still referenced by the cfg_2 visibility toggle below.
        t2v_in = tabs["t2v"]["inputs"]
        i2v_in = tabs["i2v"]["inputs"]

        # --- cfg_2 visibility toggle (Wan 2.2 MoE has high/low-noise CFG) ---
        def _toggle_cfg_2(generation: str):
            is_moe = (generation == "wan2.2")
            return [gr.update(visible=is_moe), gr.update(visible=is_moe)]

        header["generation"].change(
            fn=_toggle_cfg_2,
            inputs=[header["generation"]],
            outputs=[t2v_in["cfg_2"], i2v_in["cfg_2"]],
        )
        # Seed initial visibility — default `generation=wan2.2` is MoE so
        # cfg_2 must start visible, but `tabs.py` mounts it `visible=False`.
        demo.load(
            fn=_toggle_cfg_2,
            inputs=[header["generation"]],
            outputs=[t2v_in["cfg_2"], i2v_in["cfg_2"]],
        )

        # ── About-block refresh ──────────────────────────────────────────
        def _refresh_about(generation: str, preset: str) -> str:
            modes = modes_in(generation)  # type: ignore[arg-type]
            return (
                f"**Wan Studio v0.2** — design v2/2 (Linear)\n\n"
                f"- Backend: `{backend.label}` · dtype `{backend.dtype}` · vae `{backend.vae_dtype}`\n"
                f"- Generation: **{generation}**\n"
                f"- Available modes: {', '.join(modes)}\n"
                f"- Preset: **{preset}**\n\n"
                f"_Demo mode — Generate disabled while design direction is being chosen._"
            )

        header["generation"].change(
            fn=_refresh_about,
            inputs=[header["generation"], header["preset_state"]],
            outputs=[tabs["settings"]["about"]],
        )
        header["preset_state"].change(
            fn=_refresh_about,
            inputs=[header["generation"], header["preset_state"]],
            outputs=[tabs["settings"]["about"]],
        )
        demo.load(
            fn=_refresh_about,
            inputs=[header["generation"], header["preset_state"]],
            outputs=[tabs["settings"]["about"]],
        )
        # Bind the client-side nav handler once the DOM is up.
        demo.load(fn=None, inputs=None, outputs=None, js=_NAV_JS)

        # ── Hidden debug endpoint: read back the worker phase trace ──────────
        # The @spaces.GPU worker's stdout is invisible in run logs and it is
        # SIGKILLed on duration overrun, so we trace generation phases to a
        # /tmp file (pipelines/trace.py) and surface them through this non-GPU
        # endpoint (`/_debug_trace`) for latency diagnosis.
        with gr.Row(visible=False):
            _dbg_out = gr.Textbox(label="worker trace")
            _dbg_btn = gr.Button("read trace")

            def _debug_trace() -> str:
                from pipelines.trace import read_trace
                return read_trace(tail=300)

            _dbg_btn.click(fn=_debug_trace, inputs=None, outputs=[_dbg_out],
                           api_name="_debug_trace")

    return demo


def main():
    demo = build()
    # On HF Spaces the public health-check expects port 7860 (gradio default).
    # Locally we use 7863 so we don't clash with other Gradio dev servers.
    default_port = "7860" if os.environ.get("SPACE_ID") else "7863"
    port = int(os.environ.get("WAN_STUDIO_PORT", default_port))
    # ssr_mode=False: Gradio 6's experimental SSR drops the prediction event
    # stream for long @spaces.GPU tasks behind the ZeroGPU queue — the browser
    # shows "Error" within seconds even though the GPU task completes and the
    # API (gradio_client) returns the video fine. Disabling SSR routes results
    # through the reliable client-side path so generated videos actually render.
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
