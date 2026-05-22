"""Wan Studio — Gradio entry point (design v2/2: Linear-inspired).

Refined dev-tool dark: warm near-black surface, Geist typography, hairline
borders, restrained accent.

T2V + I2V Generate buttons are wired to the real `pipelines.{t2v,i2v}` handles
(Wave F+G). All other tabs (TI2V, FLF2V, V2V, VACE, S2V, Animate) still fire a
no-op toast until their pipelines are wired in later waves.
"""
from __future__ import annotations

import os

import gradio as gr

from pipelines import modes_in
from ui import build_all_tabs, build_header, build_sidebar, MODE_PILLS
from utils import detect


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


# --- @spaces.GPU(duration=..., size=...) callables.  These MUST share the
# wrapped function's signature (Spaces inspects bound args to call them).

def _get_t2v_duration(prompt, generation, preset_label, resolution_label, duration_s, *args, **kwargs):
    from utils.budget import duration_for
    return duration_for(_t2v_key_for(generation), duration_s=float(duration_s or 3.0))


def _get_t2v_size(prompt, generation, preset_label, resolution_label, duration_s, *args, **kwargs):
    from utils.budget import size_for
    return size_for(_t2v_key_for(generation))


def _get_i2v_duration(image, prompt, generation, preset_label, resolution_label, duration_s, *args, **kwargs):
    from utils.budget import duration_for
    return duration_for(_i2v_key_for(generation, resolution_label), duration_s=float(duration_s or 3.0))


def _get_i2v_size(image, prompt, generation, preset_label, resolution_label, duration_s, *args, **kwargs):
    from utils.budget import size_for
    return size_for(_i2v_key_for(generation, resolution_label))


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

    @spaces_gpu_or_noop()(duration=_get_t2v_duration, size=_get_t2v_size)
    def generate_t2v(
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
        progress=gr.Progress(track_tqdm=True),
    ):
        import random
        import tempfile
        from diffusers.utils import export_to_video

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

    @spaces_gpu_or_noop()(duration=_get_i2v_duration, size=_get_i2v_size)
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
        progress=gr.Progress(track_tqdm=True),
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
"""


def build() -> gr.Blocks:
    backend = detect()

    with gr.Blocks(
        title="Wan Studio — Linear",
        analytics_enabled=False,
    ) as demo:
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
    port = int(os.environ.get("WAN_STUDIO_PORT", "7863"))
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
