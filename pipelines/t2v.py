"""T2V pipeline wrapper — Wan 2.1 T2V-1.3B / 14B (single transformer)
and Wan 2.2 T2V-A14B (MoE: transformer + transformer_2).
"""
from __future__ import annotations

from typing import Any

import torch
from diffusers import WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from utils.backend import detect


class T2VHandle(WanModelHandle):
    """Builds WanPipeline. For MoE cards, also loads transformer_2."""

    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)

        if self.card.is_moe:
            # Load both transformers explicitly. transformer_2 is the low-noise expert.
            transformer = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=backend.dtype,
            )
            transformer_2 = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer_2", torch_dtype=backend.dtype,
            )
            pipe = WanPipeline.from_pretrained(
                path,
                transformer=transformer,
                transformer_2=transformer_2,
                vae=shared.vae(),
                text_encoder=shared.text_encoder(),
                torch_dtype=backend.dtype,
            )
        else:
            pipe = WanPipeline.from_pretrained(
                path,
                vae=shared.vae(),
                text_encoder=shared.text_encoder(),
                torch_dtype=backend.dtype,
            )

        pipe.to(backend.device)
        return pipe

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        seed: int = 42,
        preset_kwargs: dict[str, Any],
    ) -> list:
        """Return list of numpy frames. Caller exports via export_to_video."""
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]
