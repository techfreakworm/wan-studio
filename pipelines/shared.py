"""Shared component loaders — load UMT5-XXL text encoder, AutoencoderKLWan VAE, and
CLIP-ViT-H/14 image encoder ONCE at module load. Inject into every pipeline via
`from_pretrained(..., text_encoder=, vae=, image_encoder=)`.

Saves ~15 GB of duplicated weights vs loading per-pipeline. See RESEARCH.md §8.1.

On ZeroGPU we prefer the local snapshot dir (set by app.py via
WAN_STUDIO_WAN22_T2V_LOCAL_PATH) so from_pretrained reads off /tmp/hf_cache
instead of triggering snapshot_download in the worker fork. The Wan 2.2 T2V
repo ships with the same UMT5-XXL + AutoencoderKLWan that every Wan pipeline
shares, so this re-uses what's already predownloaded.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING

from utils.backend import detect

if TYPE_CHECKING:
    import torch  # noqa: F401


def _wan22_local_or_upstream() -> str:
    """Local snapshot dir if predownloaded, else upstream Wan-AI repo."""
    local = os.getenv("WAN_STUDIO_WAN22_T2V_LOCAL_PATH")
    if local and Path(local).is_dir():
        return local
    return "Wan-AI/Wan2.1-T2V-14B-Diffusers"


@functools.lru_cache(maxsize=1)
def text_encoder():
    """UMT5-XXL — shared across every Wan pipeline (2.1 + 2.2)."""
    from transformers import UMT5EncoderModel

    backend = detect()
    return UMT5EncoderModel.from_pretrained(
        _wan22_local_or_upstream(),
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
        _wan22_local_or_upstream(),
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
