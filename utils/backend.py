"""Backend detection — device + dtype + ZeroGPU awareness.

Refer to RESEARCH.md §7 for the per-backend loading recipe rationale.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch

Device = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class Backend:
    device: Device
    dtype: torch.dtype
    vae_dtype: torch.dtype
    is_zerogpu: bool
    zerogpu_size: Literal["large", "xlarge"] | None  # None on MPS/CPU
    supports_quant: bool          # FP8 via torchao
    supports_aoti: bool           # spaces.aoti_*
    supports_flash_attn_3: bool

    @property
    def label(self) -> str:
        if self.is_zerogpu:
            return f"ZeroGPU ({self.zerogpu_size})"
        if self.device == "mps":
            return "MPS (Apple Silicon)"
        if self.device == "cuda":
            return "CUDA (self-hosted)"
        return "CPU"


def detect() -> Backend:
    is_zerogpu = os.getenv("SPACES_ZERO_GPU") is not None

    if torch.cuda.is_available():
        device: Device = "cuda"
        dtype = torch.bfloat16
        vae_dtype = torch.float32
        zerogpu_size = (
            "xlarge" if os.getenv("WAN_STUDIO_TIER", "large") == "xlarge" else "large"
        ) if is_zerogpu else None
        return Backend(
            device=device,
            dtype=dtype,
            vae_dtype=vae_dtype,
            is_zerogpu=is_zerogpu,
            zerogpu_size=zerogpu_size,
            supports_quant=True,
            supports_aoti=is_zerogpu,
            supports_flash_attn_3=True,
        )

    if torch.backends.mps.is_available():
        # Wan transformers + the lightx2v Lightning LoRAs are bf16-native; running
        # them in fp16 risks range overflow → NaN → black frames. torch 2.11 MPS
        # bf16 is mature, so we default to bf16 and keep an env escape hatch
        # (WAN_STUDIO_MPS_DTYPE=float16) for A/B testing per model.
        _dtype_map = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        _mps_dtype = _dtype_map.get(os.getenv("WAN_STUDIO_MPS_DTYPE", "bfloat16").lower(),
                                    torch.bfloat16)
        # VAE decode is the MEMORY DRIVER at length. Default bf16: measured to cut the 14B
        # @17f peak 131.9GB→92.4GB (−40GB, off the 137GB ceiling) with PIXEL-IDENTICAL output
        # (sharpness 1203→1209, sat/brightness unchanged, no_nan) — bf16 keeps fp32's exponent
        # range so NO overflow (that's fp16's failure mode), only minor precision. Escape hatch
        # WAN_STUDIO_VAE_DTYPE=float32 if any content/mode shows banding (gate on no_nan check).
        _vae_dtype = _dtype_map.get(os.getenv("WAN_STUDIO_VAE_DTYPE", "bfloat16").lower(),
                                    torch.bfloat16)
        return Backend(
            device="mps",
            dtype=_mps_dtype,
            vae_dtype=_vae_dtype,
            is_zerogpu=False,
            zerogpu_size=None,
            supports_quant=False,      # FP8 crashes Metal
            supports_aoti=False,
            supports_flash_attn_3=False,
        )

    return Backend(
        device="cpu",
        dtype=torch.float32,
        vae_dtype=torch.float32,
        is_zerogpu=False,
        zerogpu_size=None,
        supports_quant=False,
        supports_aoti=False,
        supports_flash_attn_3=False,
    )


def spaces_gpu_or_noop():
    """Returns the `spaces.GPU` decorator if running on ZeroGPU, otherwise a no-op.

    `import spaces` is safe outside ZeroGPU (the decorator is effect-free), but this
    helper keeps the decorator-call site terse and avoids the `spaces` dependency
    erroring on environments where it isn't installed.
    """
    try:
        import spaces  # type: ignore
        return spaces.GPU
    except ImportError:
        def _noop(*_args, **_kwargs):
            def deco(fn):
                return fn
            return deco
        return _noop
