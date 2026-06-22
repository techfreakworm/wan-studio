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

# Local (MPS dev) CLIP source. The bf16 wan-shared-encoders mirror ships only
# text_encoder/ + vae/ — NOT image_encoder/ — so image modes (I2V/FLF2V/Animate/
# S2V) can't load CLIP from it. CLIP-ViT-H/14 is IDENTICAL across every Wan image
# model, so resolve it from an eviction-safe local dir if populated, else straight
# from an upstream I2V repo's image_encoder/ subfolder (downloads ~2.5 GB once).
from pipelines.handle import LOCAL_BF16_ROOT  # noqa: E402
LOCAL_SHARED = LOCAL_BF16_ROOT / "wan-shared-encoders"
CLIP_UPSTREAM_REPO = os.getenv("WAN_STUDIO_CLIP_REPO", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")


def _clip_source() -> str:
    """Resolve a path/repo that carries image_encoder/ + image_processor/.

    ZeroGPU: the shared-encoders mount (via _shared_path). Local: an eviction-safe
    ~/wan-bf16/wan-shared-encoders if it has image_encoder/, else the upstream I2V
    repo (survives a --purge-fp32 of any single converted model)."""
    if os.getenv("SPACES_ZERO_GPU") is not None:
        return _shared_path()
    if (LOCAL_SHARED / "image_encoder").is_dir():
        return str(LOCAL_SHARED)
    return CLIP_UPSTREAM_REPO


def _shared_path() -> str:
    """Resolve the shared-encoders dir.

    ZeroGPU: the wan-shared-encoders mount (fail loud if absent — never silently
    download ~14 GB into /tmp). Local: the bf16 mirror repo id (downloads once to
    the persistent HF cache).
    """
    if SHARED_MOUNT.exists():
        # Stitch: small configs fetched from the repo (the mount truncates
        # sub-1KB JSON like vae/config.json), weights symlinked from the mount.
        from pipelines.handle import stitch_shared_dir, tier2_warm_copy
        stitched = stitch_shared_dir() or str(SHARED_MOUNT)
        # ZeroGPU: back the shared encoders (UMT5-XXL ~11 GB injected into every
        # pipe) with local NVMe too — otherwise pipe.to('cuda') page-faults them
        # from the slow mount on the GPU clock. See pipelines/handle._mount_path.
        if os.getenv("SPACES_ZERO_GPU") is not None and os.getenv("WAN_STUDIO_TIER2", "1") == "1":
            try:
                return tier2_warm_copy("wan-shared-encoders", stitched)
            except Exception as e:
                print(f"=== TIER2 copy failed for shared-encoders ({e}); using mount ===", flush=True)
                return stitched
        return stitched
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

    # ENCODE dtype fix (MPS bf16 VAE). diffusers' Wan pipelines HARDCODE the
    # VAE-encode input to fp32 (pipeline_wan_video2video L635, and the I2V/FLF2V/
    # VACE image/video conditioning paths) because their documented contract is an
    # fp32 VAE. We run a bf16 VAE (cuts 14B decode peak 131.9→92.4GB, pixel-identical
    # on decode), so an fp32 conditioning tensor hits bf16 conv weights →
    # "Input type (float) and bias type (BFloat16) should be the same". Decode is
    # safe (it casts latents to vae.dtype), only ENCODE mismatches. Cast the encode
    # input to the VAE's own dtype so every conditioning mode (v2v/i2v/flf2v/vace)
    # works while keeping the bf16 decode memory win. bf16 keeps fp32's exponent
    # range, so no overflow — same rationale as the decode default.
    _orig_encode = instance.encode

    def _encode_cast_dtype(x, *args, **kwargs):
        if hasattr(x, "to"):
            x = x.to(instance.dtype)
        return _orig_encode(x, *args, **kwargs)

    instance.encode = _encode_cast_dtype
    return instance


@functools.lru_cache(maxsize=1)
def image_encoder():
    """CLIP-ViT-H/14 — used by I2V, FLF2V, Animate, S2V."""
    import torch
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(
        _clip_source(),
        subfolder="image_encoder",
        torch_dtype=torch.float32,  # must stay fp32 per diffusers docs
    )


@functools.lru_cache(maxsize=1)
def image_processor():
    """CLIPImageProcessor — required by WanAnimatePipeline (Phase #2)."""
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(_clip_source(), subfolder="image_processor")
