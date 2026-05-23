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
_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
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

import gradio as gr

from pipelines import modes_in
from ui import build_all_tabs, build_header, build_sidebar, MODE_PILLS
from utils import detect


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
        import time as _t
        key = "wan2.2_t2v_a14b"
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


# ── Preload the default T2V handle into CPU RAM at app startup ─────────
# Worker forks inherit this via copy-on-write so each @spaces.GPU click
# skips the ~120s of disk-to-RAM shard load that was blowing the GPU
# duration budget. Reads from the stitched dir built above — instant,
# no network, no disk pressure.
def _preload_default_t2v_handle() -> None:
    if os.getenv("SPACES_ZERO_GPU") is None:
        return
    try:
        from pipelines.t2v import T2VHandle
        import time as _t
        key = "wan2.2_t2v_a14b"
        print(f"=== PRELOAD T2V handle to CPU: {key} ===", flush=True)
        t0 = _t.time()
        handle = T2VHandle.for_key(key)
        handle.ensure_loaded()  # disk → CPU RAM only (no CUDA touch)
        T2V_HANDLES[key] = handle
        print(f"=== PRELOAD done in {int(_t.time()-t0)}s — handle cached ===", flush=True)
    except Exception as e:
        import traceback
        print(f"=== PRELOAD FAILED: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()


_probe_filesystem()
_stitch_default_model()
_preload_default_t2v_handle()




# ────────────────────────────────────────────────────────────────────────────
# Generate-handler helpers — kept at module scope so the `@spaces.GPU(...)`
# decorator can reference `duration=`/`size=` callables that match the wrapped
# function's signature.  Everything heavy (diffusers, torch, model handles)
# stays lazy: nothing imports diffusers or instantiates a handle at module
# import time, so `from app import build; build()` is cheap.
# ────────────────────────────────────────────────────────────────────────────

# Per-key handle caches.  Populated on first Generate click for that key.
T2V_HANDLES: dict = {}
I2V_HANDLES: dict = {}


def _t2v_key_for(generation: str) -> str:
    """Map generation selector → registry key for T2V."""
    key = "wan2.2_t2v_a14b" if generation == "wan2.2" else "wan2.1_t2v_14b"
    # Local-dev override: allow forcing 1.3B for faster MPS smoke
    local_override = os.getenv("WAN_STUDIO_T2V_LOCAL_KEY")
    if local_override and os.getenv("SPACES_ZERO_GPU") is None:
        key = local_override
    return key


def _i2v_key_for(generation: str, resolution_label: str) -> str:
    """Map (generation, resolution) → registry key for I2V."""
    if generation == "wan2.2":
        key = "wan2.2_i2v_a14b"
    elif "720" in (resolution_label or ""):
        key = "wan2.1_i2v_14b_720p"
    else:
        key = "wan2.1_i2v_14b_480p"
    local_override = os.getenv("WAN_STUDIO_I2V_LOCAL_KEY")
    if local_override and os.getenv("SPACES_ZERO_GPU") is None:
        key = local_override
    return key


def _get_t2v_handle(generation: str):
    """Lazy-load + cache a T2VHandle keyed by registry key."""
    from pipelines.t2v import T2VHandle
    key = _t2v_key_for(generation)
    if key not in T2V_HANDLES:
        T2V_HANDLES[key] = T2VHandle.for_key(key)
    return T2V_HANDLES[key]


def _get_i2v_handle(generation: str, resolution_label: str):
    """Lazy-load + cache an I2VHandle keyed by registry key."""
    from pipelines.i2v import I2VHandle
    key = _i2v_key_for(generation, resolution_label)
    if key not in I2V_HANDLES:
        I2V_HANDLES[key] = I2VHandle.for_key(key)
    return I2V_HANDLES[key]


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

def _get_t2v_duration(prompt, generation, preset_label, resolution_label, duration_s, *args, **kwargs):
    from utils.budget import duration_for
    return duration_for(_t2v_key_for(generation), duration_s=float(duration_s or 3.0))


def _get_i2v_duration(image, prompt, generation, preset_label, resolution_label, duration_s, *args, **kwargs):
    from utils.budget import duration_for
    return duration_for(_i2v_key_for(generation, resolution_label), duration_s=float(duration_s or 3.0))


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


def _build_t2v_handler():
    """Build the decorated T2V Generate handler.  The `@spaces.GPU(...)`
    decorator is fetched at call time so `import spaces` only runs when we're
    actually on ZeroGPU (locally `spaces_gpu_or_noop` is a no-op factory).
    """
    from utils.backend import spaces_gpu_or_noop

    @spaces_gpu_or_noop()(duration=_get_t2v_duration, size="large")
    def generate_t2v(  # noqa: PLR0913 (signature dictated by gradio inputs)
        prompt: str,
        generation: str,
        preset_label: str,
        resolution_label: str,
        duration_s: float,
        negative_prompt: str,
        seed: int,
        randomize: bool,
        steps_override: int,
        cfg_override: float,
        cfg_2_override: float,
        progress=gr.Progress(track_tqdm=False),
    ):
        import random
        import tempfile
        from diffusers.utils import export_to_video

        # Worker-side filesystem + env diagnostics. Logs once per worker
        # fork so we can confirm /tmp/hf_cache is visible from inside the
        # ZeroGPU sandbox.
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

        try:
            if randomize:
                seed = random.randint(0, 2**31 - 1)

            preset = _coerce_preset(preset_label)
            handle = _get_t2v_handle(generation)
            progress(0.05, desc="Configuring preset…")
            preset_kwargs = handle.configure_preset(preset)

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

            height, width = _parse_resolution(resolution_label)
            # Wan VAE temporal patching: num_frames must be 4k+1.
            num_frames = max(17, int(float(duration_s) * 16) // 4 * 4 + 1)

            progress(0.2, desc="Generating frames…")
            frames = handle.generate(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                height=height,
                width=width,
                num_frames=num_frames,
                seed=int(seed),
                preset_kwargs=inference_kwargs,
            )

            progress(0.9, desc="Encoding MP4…")
            fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="wan_t2v_")
            os.close(fd)
            export_to_video(frames, out_path, fps=16)

            if preset_kwargs.fallback_message:
                gr.Info(preset_kwargs.fallback_message, duration=8)
            return out_path
        except gr.Error:
            raise
        except Exception as e:
            _raise_user_error(e)

    return generate_t2v


def _build_i2v_handler():
    """Mirror of `_build_t2v_handler` for image-to-video.  Auto-picks the
    checkpoint by (generation × resolution) and coerces filepath → PIL.Image."""
    from utils.backend import spaces_gpu_or_noop

    @spaces_gpu_or_noop()(duration=_get_i2v_duration, size="large")
    def generate_i2v(
        image,
        prompt: str,
        generation: str,
        preset_label: str,
        resolution_label: str,
        duration_s: float,
        negative_prompt: str,
        seed: int,
        randomize: bool,
        steps_override: int,
        cfg_override: float,
        cfg_2_override: float,
        progress=gr.Progress(track_tqdm=False),
    ):
        import random
        import tempfile
        from diffusers.utils import export_to_video
        from PIL import Image

        if image is None:
            raise gr.Error("Please upload an image.")
        if not prompt or not str(prompt).strip():
            raise gr.Error("Motion prompt is required.")

        try:
            # Coerce filepath → PIL.Image (gr.Image type="pil" already yields a
            # PIL.Image, but tolerate strings for callers that bind type="filepath").
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif hasattr(image, "convert"):
                image = image.convert("RGB")

            if randomize:
                seed = random.randint(0, 2**31 - 1)

            preset = _coerce_preset(preset_label)
            handle = _get_i2v_handle(generation, resolution_label)
            progress(0.05, desc="Configuring preset…")
            preset_kwargs = handle.configure_preset(preset)

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

            # max_area drives `aspect_ratio_resize` — clamp to the picked res.
            h_label, w_label = _parse_resolution(resolution_label)
            max_area = h_label * w_label
            num_frames = max(17, int(float(duration_s) * 16) // 4 * 4 + 1)

            progress(0.2, desc="Generating frames…")
            frames = handle.generate(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                max_area=max_area,
                num_frames=num_frames,
                seed=int(seed),
                preset_kwargs=inference_kwargs,
            )

            progress(0.9, desc="Encoding MP4…")
            fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="wan_i2v_")
            os.close(fd)
            export_to_video(frames, out_path, fps=16)

            if preset_kwargs.fallback_message:
                gr.Info(preset_kwargs.fallback_message, duration=8)
            return out_path
        except gr.Error:
            raise
        except Exception as e:
            _raise_user_error(e)

    return generate_i2v


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

/* Suppress Gradio's status-tracker overlay (demo mode — no real generation). */
#ws-content [data-testid="status-tracker"],
#ws-content .wrap.generating,
#ws-content .wrap.full.generating {
  display: none !important;
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
        # T2V + I2V → real handles (Wave F+G). Other tabs keep the no-op
        # toast until their pipelines land in later waves.
        def _generate_toast():
            gr.Info("Demo mode — Generate disabled while design direction is being chosen.")

        # --- T2V: wire to the real handle ---
        t2v_in = tabs["t2v"]["inputs"]
        t2v_out = tabs["t2v"]["outputs"]
        t2v_in["generate"].click(
            fn=_build_t2v_handler(),
            inputs=[
                t2v_in["prompt"],
                header["generation"],
                header["preset_state"],
                t2v_in["resolution"],
                t2v_in["duration"],
                t2v_in["negative_prompt"],
                t2v_in["seed"],
                t2v_in["randomize"],
                t2v_in["steps"],
                t2v_in["cfg"],
                t2v_in["cfg_2"],
            ],
            outputs=t2v_out["video"],
        )

        # --- I2V: wire to the real handle ---
        i2v_in = tabs["i2v"]["inputs"]
        i2v_out = tabs["i2v"]["outputs"]
        i2v_in["generate"].click(
            fn=_build_i2v_handler(),
            inputs=[
                i2v_in["image"],
                i2v_in["prompt"],
                header["generation"],
                header["preset_state"],
                i2v_in["resolution"],
                i2v_in["duration"],
                i2v_in["negative_prompt"],
                i2v_in["seed"],
                i2v_in["randomize"],
                i2v_in["steps"],
                i2v_in["cfg"],
                i2v_in["cfg_2"],
            ],
            outputs=i2v_out["video"],
        )

        # --- Other tabs stay on the no-op toast ---
        for tab_key in ["ti2v", "flf2v", "v2v", "vace", "s2v", "animate"]:
            gen_btn = tabs[tab_key]["inputs"].get("generate") if "inputs" in tabs[tab_key] else None
            if gen_btn is not None:
                gen_btn.click(fn=_generate_toast, inputs=None, outputs=None)
        # FLF2V has a secondary "Generate end frame" button.
        if "inputs" in tabs.get("flf2v", {}) and "generate_end" in tabs["flf2v"]["inputs"]:
            tabs["flf2v"]["inputs"]["generate_end"].click(fn=_generate_toast, inputs=None, outputs=None)

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

    return demo


def main():
    demo = build()
    # On HF Spaces the public health-check expects port 7860 (gradio default).
    # Locally we use 7863 so we don't clash with other Gradio dev servers.
    default_port = "7860" if os.environ.get("SPACE_ID") else "7863"
    port = int(os.environ.get("WAN_STUDIO_PORT", default_port))
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
