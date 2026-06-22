"""Animate — Wan 2.2 Animate-14B (character animation from pose+face driving videos).

WanAnimatePipeline. Components: WanAnimateTransformer3DModel + CLIP image_encoder +
image_processor (kept in the bf16 mirror by bf16_plan) + shared UMT5 text_encoder +
shared Wan VAE. The decoupled CV preproc (ViTPose/YOLO → pose_video skeleton frames +
512×512 face crops) is done OFFLINE (scripts/animate_preprocess.py) — this handler just
runs the diffusion on the pre-extracted control videos.

720p self-attn goes through the key-chunked flash patch (pipelines/mps_patches.py also
patches transformer_wan_animate.WanAttnProcessor), so native 720p is correct on MPS.
"""
from __future__ import annotations

from typing import Any

import torch
from diffusers import WanAnimatePipeline
from diffusers.models.transformers.transformer_wan_animate import WanAnimateTransformer3DModel
from transformers import CLIPImageProcessor, CLIPVisionModel
from PIL import Image

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from utils.backend import detect


class AnimateHandle(WanModelHandle):
    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)
        transformer = WanAnimateTransformer3DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=backend.dtype,
        )
        # Animate ships its own CLIP image_encoder (kept in the mirror) — load locally.
        image_encoder = CLIPVisionModel.from_pretrained(
            path, subfolder="image_encoder", torch_dtype=torch.float32,
        )
        image_processor = CLIPImageProcessor.from_pretrained(path, subfolder="image_processor")
        pipe = WanAnimatePipeline.from_pretrained(
            path,
            transformer=transformer,
            image_encoder=image_encoder,
            image_processor=image_processor,
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            torch_dtype=backend.dtype,
        )
        return pipe

    def generate(
        self,
        image: Image.Image,
        pose_video: list,
        face_video: list,
        *,
        prompt: str = "a person, natural motion, cinematic",
        negative_prompt: str = "",
        height: int = 720,
        width: int = 1280,
        segment_frame_length: int = 77,
        seed: int = 42,
        preset_kwargs: dict[str, Any],
        step_callback=None,
    ) -> list:
        from pipelines.handle import diffusers_step_callback
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        cb = diffusers_step_callback(step_callback, preset_kwargs.get("num_inference_steps"))
        out = self.pipe(
            image=image,
            pose_video=pose_video,
            face_video=face_video,
            prompt=prompt or None,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            segment_frame_length=segment_frame_length,
            generator=gen,
            **({"callback_on_step_end": cb} if cb else {}),
            **preset_kwargs,
        )
        return out.frames[0]


from pipelines.handlers import HandlerSpec, register  # noqa: E402


def _animate_key_for(generation: str, **_ui) -> str:
    return "wan2.2_animate_14b"


register(HandlerSpec(mode="animate", handle_cls=AnimateHandle, key_for=_animate_key_for, tier="xlarge"))
