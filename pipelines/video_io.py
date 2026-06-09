"""Video/image helpers for video-input modes (V2V) and end-frame modes (FLF2V).

PIL-only (no torchvision — it is intentionally not installed; see #0b).
"""
from __future__ import annotations

from PIL import Image


def center_crop_resize(image: Image.Image, h: int, w: int) -> Image.Image:
    """Resize `image` to COVER (w, h) preserving aspect ratio, then center-crop to (w, h).

    Used for the FLF2V end frame so it matches the first frame's (h, w).
    """
    image = image.convert("RGB")
    target_ar = w / h
    src_ar = image.width / image.height
    if src_ar > target_ar:               # source wider → match height, crop width
        new_h, new_w = h, max(w, int(round(h * src_ar)))
    else:                                 # source taller → match width, crop height
        new_w, new_h = w, max(h, int(round(w / src_ar)))
    resized = image.resize((new_w, new_h))
    left, top = (new_w - w) // 2, (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def decode_video(path: str, pipe, max_area: int) -> tuple[list[Image.Image], int, int]:
    """Decode a video filepath (from gr.Video) into a list of PIL frames resized to the
    VAE/patch grid. Returns (frames, height, width). H/W derive from the FIRST frame's
    aspect ratio, snapped to `vae_scale_factor_spatial * patch_size[1]`, like I2V.
    """
    import numpy as np
    from diffusers.utils import load_video

    raw = load_video(path)               # list[PIL.Image]
    if not raw:
        raise ValueError("could not decode any frames from the input video")
    ar = raw[0].height / raw[0].width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = max(mod, int(round(np.sqrt(max_area * ar))) // mod * mod)
    w = max(mod, int(round(np.sqrt(max_area / ar))) // mod * mod)
    frames = [f.convert("RGB").resize((w, h)) for f in raw]
    return frames, h, w
