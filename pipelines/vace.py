"""VACE — Wan 2.1 control/edit (WanVACEPipeline). Single transformer, Quality-only.

VACE has no CLIP image branch → inject only vae + text_encoder. The sub-mode is
decided by which of video/mask/reference_images is passed (see vace_inputs).
"""
from __future__ import annotations

import os
from typing import Any

import torch
from diffusers import WanVACEPipeline

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from pipelines.handlers import HandlerSpec, register
from utils.backend import detect


class VACEHandle(WanModelHandle):
    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)
        return WanVACEPipeline.from_pretrained(
            path,
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            torch_dtype=backend.dtype,
        )

    def generate(
        self,
        *,
        prompt: str,
        video: list | None = None,
        mask: list | None = None,
        reference_images: list | None = None,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        conditioning_scale: float = 1.0,
        seed: int = 42,
        preset_kwargs: dict,
    ) -> list:
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            video=video,
            mask=mask,
            reference_images=reference_images,
            conditioning_scale=conditioning_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]


def _vace_key_for(generation: str, **_ui) -> str:
    # Wan 2.1 only; default to 14B, allow a local 1.3B override (dev).
    if os.getenv("SPACES_ZERO_GPU") is None:
        override = os.getenv("WAN_STUDIO_VACE_LOCAL_KEY")
        if override:
            return override
    return "wan2.1_vace_14b"


register(HandlerSpec(mode="vace", handle_cls=VACEHandle, key_for=_vace_key_for, tier="large"))
