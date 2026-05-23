"""Shared component loaders — load UMT5-XXL text encoder, AutoencoderKLWan VAE, and
CLIP-ViT-H/14 image encoder ONCE at module load. Inject into every pipeline via
`from_pretrained(..., text_encoder=, vae=, image_encoder=)`.

Saves ~15 GB of duplicated weights vs loading per-pipeline. See RESEARCH.md §8.1.

On ZeroGPU we read from the stitched Wan 2.2 T2V dir (mounted weights +
bundled configs) so from_pretrained reads big binaries through the read-only
volume mount (zero disk cost) and small JSONs from the bundled copy
(working around the HF Volume small-file truncation bug). The Wan 2.2 T2V
repo ships with the same UMT5-XXL + AutoencoderKLWan that every Wan pipeline
shares, so loading them from this single stitched dir is correct.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from utils.backend import detect

if TYPE_CHECKING:
    import torch  # noqa: F401


def _wan22_t2v_path() -> str:
    """Stitched Wan 2.2 T2V dir on ZeroGPU; upstream repo as fallback."""
    from pipelines.handle import stitch_local_dir
    from pipelines.registry import BY_KEY

    stitched = stitch_local_dir(BY_KEY["wan2.2_t2v_a14b"])
    if stitched:
        return stitched
    return "Wan-AI/Wan2.1-T2V-14B-Diffusers"


@functools.lru_cache(maxsize=1)
def text_encoder():
    """UMT5-XXL — shared across every Wan pipeline (2.1 + 2.2)."""
    from transformers import UMT5EncoderModel

    backend = detect()
    return UMT5EncoderModel.from_pretrained(
        _wan22_t2v_path(),
        subfolder="text_encoder",
        torch_dtype=backend.dtype,
    )


@functools.lru_cache(maxsize=1)
def vae():
    """AutoencoderKLWan — same class handles Wan 2.1 (8×8×4) and Wan 2.2 (16×16×4) VAEs.

    The Wan 2.2 5B model uses a different VAE config but the same class. For TI2V-5B
    you'd load a separate VAE via `from_pretrained` against that repo's vae subfolder.
    """
    from diffusers import AutoencoderKLWan

    backend = detect()
    instance = AutoencoderKLWan.from_pretrained(
        _wan22_t2v_path(),
        subfolder="vae",
        torch_dtype=backend.vae_dtype,
    )
    # Memory savers — required at higher resolutions on every backend.
    instance.enable_tiling()
    instance.enable_slicing()
    return instance


@functools.lru_cache(maxsize=1)
def image_encoder():
    """CLIP-ViT-H/14 — used by I2V, FLF2V, Animate."""
    import torch
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        subfolder="image_encoder",
        torch_dtype=torch.float32,  # must stay fp32 per diffusers docs
    )
