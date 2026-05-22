"""I2V pipeline wrapper — Wan 2.1 I2V-14B-480P/720P (single) and Wan 2.2 I2V-A14B (MoE).

Also handles FLF2V via the same WanImageToVideoPipeline + last_image= kwarg
(used in Phase 2; defined here so the FLF2V handle can subclass).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from PIL import Image

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from utils.backend import detect


def aspect_ratio_resize(
    image: Image.Image,
    pipe: WanImageToVideoPipeline,
    max_area: int,
) -> tuple[Image.Image, int, int]:
    """Resize input image to a multiple of vae_scale_factor_spatial * patch_size[1].

    Returns (resized_image, height, width). Helper from RESEARCH §3.2.
    """
    ar = image.height / image.width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = int(round(np.sqrt(max_area * ar))) // mod * mod
    w = int(round(np.sqrt(max_area / ar))) // mod * mod
    return image.resize((w, h)), h, w


class I2VHandle(WanModelHandle):
    """Builds WanImageToVideoPipeline. Handles MoE for Wan 2.2 A14B."""

    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)

        common_kwargs = dict(
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            image_encoder=shared.image_encoder(),
            torch_dtype=backend.dtype,
        )

        if self.card.is_moe:
            transformer = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=backend.dtype,
            )
            transformer_2 = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer_2", torch_dtype=backend.dtype,
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                path,
                transformer=transformer,
                transformer_2=transformer_2,
                **common_kwargs,
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(path, **common_kwargs)

        # Wan 2.2 MoE I2V carries two 14B transformers + image encoder which
        # won't fit on a `large` ZeroGPU card (48 GB). Use accelerate's model
        # CPU offload to swap modules between CPU and GPU per forward pass.
        # `enable_model_cpu_offload()` is CUDA-only — fall back to plain
        # `.to(device)` on MPS / CPU. Once offload is enabled the pipe MUST
        # NOT be moved with .to() afterwards.
        if self.card.is_moe and backend.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(backend.device)
        return pipe

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str = "",
        max_area: int = 480 * 832,
        num_frames: int = 81,
        seed: int = 42,
        preset_kwargs: dict[str, Any],
    ) -> list:
        self.ensure_loaded()
        resized, h, w = aspect_ratio_resize(image, self.pipe, max_area)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            image=resized,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=h,
            width=w,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]
