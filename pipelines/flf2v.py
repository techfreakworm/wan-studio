"""FLF2V — Wan 2.1 first-last-frame. WanImageToVideoPipeline + last_image= kwarg.

Subclasses I2VHandle (identical pipeline class + shared-encoder injection); only
generate() differs: it resizes the first frame (aspect_ratio_resize) and
center-crop-resizes the last frame to match, then passes last_image=.
Lightning is BETA (reuses the I2V LoRA, not FLF2V-trained) — UI labels it Beta.
"""
from __future__ import annotations

import torch

from pipelines.i2v import I2VHandle, aspect_ratio_resize
from pipelines.video_io import center_crop_resize


class FLF2VHandle(I2VHandle):
    """720p-locked; max_area fixed at 720*1280."""

    def generate(
        self,
        image,
        last_image,
        prompt: str,
        *,
        negative_prompt: str = "",
        max_area: int = 720 * 1280,
        num_frames: int = 81,
        seed: int = 42,
        preset_kwargs: dict,
    ) -> list:
        self.ensure_loaded()
        first, h, w = aspect_ratio_resize(image, self.pipe, max_area)
        last = center_crop_resize(last_image, h, w)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            image=first,
            last_image=last,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=h,
            width=w,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]


from pipelines.handlers import HandlerSpec, register  # noqa: E402


def _flf2v_key_for(generation: str, **_ui) -> str:
    return "wan2.1_flf2v_14b_720p"  # Wan 2.1 only, 720p


register(HandlerSpec(mode="flf2v", handle_cls=FLF2VHandle, key_for=_flf2v_key_for, tier="large"))  # noqa: E402
