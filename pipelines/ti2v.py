"""TI2V — Wan 2.2 TI2V-5B (text+image→video).

Single dense 5B transformer (NOT MoE), with its OWN VAE (AutoencoderKLWan z_dim=48,
16×16×4) — NOT the shared Wan 2.1 VAE (z_dim=16). Image conditioning is done by
VAE-encoding the init image as the first-frame condition (`expand_timesteps=True`),
NOT via a CLIP image encoder (the 5B transformer has image_dim=None). So we build a
`WanImageToVideoPipeline` with `image_encoder=None` + `expand_timesteps=True` — the
diffusers Wan2.2-5B I2V path (pipeline_wan_i2v.py L425/L461/L727).

Only 1280×704 / 704×1280 are supported by this checkpoint.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from PIL import Image

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from utils.backend import detect


def _own_vae(path: str, vae_dtype):
    """Load TI2V-5B's own 16×16×4 VAE (z_dim=48). Apply the same encode-input dtype
    cast as shared.vae() — TI2V VAE-encodes the init image, and a bf16 VAE would hit
    the fp32-input conv mismatch otherwise (see shared.vae for the full rationale)."""
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=vae_dtype)
    vae.enable_tiling()
    vae.enable_slicing()
    _orig_encode = vae.encode

    def _encode_cast_dtype(x, *args, **kwargs):
        if hasattr(x, "to"):
            x = x.to(vae.dtype)
        return _orig_encode(x, *args, **kwargs)

    vae.encode = _encode_cast_dtype
    return vae


def _fit_exact(image: Image.Image, height: int, width: int) -> Image.Image:
    """TI2V-5B only supports 1280×704 / 704×1280 — resize the init image to the EXACT
    target (center-crop to the target AR first so the subject isn't distorted)."""
    tw, th = width, height
    src_ar = image.width / image.height
    tgt_ar = tw / th
    if src_ar > tgt_ar:  # too wide → crop width
        nw = int(round(image.height * tgt_ar))
        left = (image.width - nw) // 2
        image = image.crop((left, 0, left + nw, image.height))
    else:                # too tall → crop height
        nh = int(round(image.width / tgt_ar))
        top = (image.height - nh) // 2
        image = image.crop((0, top, image.width, top + nh))
    return image.resize((tw, th), Image.LANCZOS)


class TI2VHandle(WanModelHandle):
    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)
        transformer = WanTransformer3DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=backend.dtype,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            path,
            transformer=transformer,
            vae=_own_vae(path, backend.vae_dtype),
            text_encoder=shared.text_encoder(),
            image_encoder=None,          # 5B conditions via VAE latent, not CLIP
            torch_dtype=backend.dtype,
        )
        # 5B TI2V uses first-frame-masked timestep expansion. model_index.json carries
        # expand_timesteps=True, but force it on in case the I2V class didn't pick up
        # the WanPipeline-authored config key.
        pipe.register_to_config(expand_timesteps=True)
        return pipe

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int = 704,
        width: int = 1280,
        num_frames: int = 121,
        seed: int = 42,
        preset_kwargs: dict[str, Any],
        step_callback=None,
    ) -> list:
        from pipelines.handle import diffusers_step_callback
        self.ensure_loaded()
        # TI2V-5B is resolution-strict (1280×704 / 704×1280). Snap to the requested
        # exact dims (orientation chosen by caller); center-crop to AR so the subject
        # isn't distorted.
        h, w = height, width
        resized = _fit_exact(image, h, w)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        cb = diffusers_step_callback(step_callback, preset_kwargs.get("num_inference_steps"))
        out = self.pipe(
            image=resized,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=h,
            width=w,
            num_frames=num_frames,
            generator=gen,
            **({"callback_on_step_end": cb} if cb else {}),
            **preset_kwargs,
        )
        return out.frames[0]


from pipelines.handlers import HandlerSpec, register  # noqa: E402


def _ti2v_key_for(generation: str, **_ui) -> str:
    return "wan2.2_ti2v_5b"


register(HandlerSpec(mode="ti2v", handle_cls=TI2VHandle, key_for=_ti2v_key_for, tier="large"))
