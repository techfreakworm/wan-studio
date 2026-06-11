"""Shared component loaders — load UMT5-XXL text encoder, AutoencoderKLWan VAE, and
CLIP-ViT-H/14 image encoder ONCE at module load. Inject into every pipeline via
`from_pretrained(..., text_encoder=, vae=, image_encoder=)`.

Saves ~15 GB of duplicated weights vs loading per-pipeline. See RESEARCH.md §8.1.

On ZeroGPU we read these shared encoders from the wan-shared-encoders mount so
from_pretrained reads the weights through the read-only volume mount (zero disk
cost). Locally (no mount) we fall back to the bf16 mirror repo, which downloads
once into the persistent HF cache. The same UMT5-XXL + AutoencoderKLWan + CLIP
encoders are shared across every Wan pipeline (2.1 + 2.2).
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING

from utils.backend import detect

if TYPE_CHECKING:
    import torch  # noqa: F401


SHARED_MOUNT = Path(os.getenv("WAN_STUDIO_MOUNT_ROOT", "/models")) / "wan-shared-encoders"
SHARED_MIRROR_REPO = "techfreakworm/wan-shared-encoders"


def _shared_path() -> str:
    """Resolve the shared-encoders dir.

    ZeroGPU: the wan-shared-encoders mount (fail loud if absent — never silently
    download ~14 GB into /tmp). Local: the bf16 mirror repo id (downloads once to
    the persistent HF cache).
    """
    if SHARED_MOUNT.exists():
        # Stitch: small configs fetched from the repo (the mount truncates
        # sub-1KB JSON like vae/config.json), weights symlinked from the mount.
        from pipelines.handle import stitch_shared_dir
        return stitch_shared_dir() or str(SHARED_MOUNT)
    if os.getenv("SPACES_ZERO_GPU") is not None:
        raise RuntimeError(
            f"wan-shared-encoders mount missing at {SHARED_MOUNT} — check create_space.py manifest"
        )
    return SHARED_MIRROR_REPO


@functools.lru_cache(maxsize=1)
def text_encoder():
    """UMT5-XXL — shared across every Wan pipeline (2.1 + 2.2)."""
    from transformers import UMT5EncoderModel

    backend = detect()
    return UMT5EncoderModel.from_pretrained(
        _shared_path(),
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
        _shared_path(),
        subfolder="vae",
        torch_dtype=backend.vae_dtype,
    )
    # Memory savers — required at higher resolutions on every backend.
    instance.enable_tiling()
    instance.enable_slicing()
    return instance


@functools.lru_cache(maxsize=1)
def image_encoder():
    """CLIP-ViT-H/14 — used by I2V, FLF2V, Animate, S2V."""
    import torch
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(
        _shared_path(),
        subfolder="image_encoder",
        torch_dtype=torch.float32,  # must stay fp32 per diffusers docs
    )


@functools.lru_cache(maxsize=1)
def image_processor():
    """CLIPImageProcessor — required by WanAnimatePipeline (Phase #2)."""
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(_shared_path(), subfolder="image_processor")
