"""Shared component loaders — load UMT5-XXL text encoder, AutoencoderKLWan VAE, and
CLIP-ViT-H/14 image encoder ONCE at module load. Inject into every pipeline via
`from_pretrained(..., text_encoder=, vae=, image_encoder=)`.

Saves ~15 GB of duplicated weights vs loading per-pipeline. See RESEARCH.md §8.1.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from utils.backend import detect

if TYPE_CHECKING:
    import torch  # noqa: F401

# These imports are heavy and we lazy-load them in functions so just `import
# pipelines.shared` does not pull in diffusers at module import time. That matters
# for tooling that introspects the module graph without wanting to wait on
# torch + diffusers import (~3-5s cold).


@functools.lru_cache(maxsize=1)
def text_encoder():
    """UMT5-XXL — shared across every Wan pipeline (2.1 + 2.2)."""
    import torch
    from transformers import UMT5EncoderModel

    backend = detect()
    return UMT5EncoderModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        subfolder="text_encoder",
        torch_dtype=backend.dtype,
    )


@functools.lru_cache(maxsize=1)
def vae():
    """AutoencoderKLWan — same class handles Wan 2.1 (8×8×4) and Wan 2.2 (16×16×4) VAEs.

    The Wan 2.2 5B model uses a different VAE config but the same class. For TI2V-5B
    you'd load a separate VAE via `from_pretrained` against that repo's vae subfolder.
    """
    import torch
    from diffusers import AutoencoderKLWan

    backend = detect()
    instance = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
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
