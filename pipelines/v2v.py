"""V2V — Wan 2.1 video restyle on the T2V-14B backbone (WanVideoToVideoPipeline)."""
from __future__ import annotations

from typing import Any

import torch
from diffusers import WanVideoToVideoPipeline

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from pipelines.handlers import HandlerSpec, register
from utils.backend import detect


class V2VHandle(WanModelHandle):
    """WanVideoToVideoPipeline on the shared wan2.1-t2v-14b mount. Quality-only."""

    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)
        return WanVideoToVideoPipeline.from_pretrained(
            path,
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            torch_dtype=backend.dtype,
        )

    def generate(
        self,
        video: list,
        prompt: str,
        *,
        negative_prompt: str = "",
        strength: float = 0.7,
        seed: int = 42,
        preset_kwargs: dict,
    ) -> list:
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            video=video,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            strength=strength,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]


def _v2v_key_for(generation: str, **_ui) -> str:
    return "wan2.1_v2v_14b"  # Wan 2.1 only


register(HandlerSpec(mode="v2v", handle_cls=V2VHandle, key_for=_v2v_key_for, tier="large"))
