"""ZeroGPU duration budget — per-(mode, generation) tier + duration callable.

Returns the (size, default_seconds) tuple for `@spaces.GPU(duration=callable, size=...)`
and a `duration_for(mode_key, **gen_kwargs)` helper that scales by requested params.

The same `duration_for()` callable is referenced by both:
  - the @spaces.GPU decorator on the inference function
  - the ETA gr.Markdown component in the UI

So display + actual reservation stay in sync.
"""
from __future__ import annotations

from typing import Literal

Size = Literal["large", "xlarge"]


# (size, default_seconds_at_fast_preset)
#
# NOTE on size tier: `xlarge` requires HF PRO+ or Enterprise.
# Phase 1 ships on HF PRO, so every entry is capped to `large` (48 GB) — the
# HF /schedule server returns 422 Unprocessable Entity if a PRO account asks
# for `xlarge`. Wan 2.2 MoE (dual 14B transformer + transformer_2 at bf16 ≈
# 56 GB raw) doesn't fit in 48 GB unaided, so the MoE pipelines must also
# enable `pipe.enable_model_cpu_offload()` (see pipelines/t2v.py +
# pipelines/i2v.py). When this Space upgrades to PRO+, flip the MoE rows back
# to `xlarge` and drop the offload branch.
MODE_BUDGET: dict[str, tuple[Size, int]] = {
    # Wan 2.1 — all single-transformer
    "wan2.1_t2v_1.3b":         ("large",  60),
    "wan2.1_t2v_14b":          ("large",  90),
    "wan2.1_i2v_14b_480p":     ("large",  90),
    "wan2.1_i2v_14b_720p":     ("large", 120),
    "wan2.1_flf2v_14b_720p":   ("large", 150),
    "wan2.1_v2v_14b":          ("large",  90),
    "wan2.1_vace_1.3b":        ("large", 150),
    "wan2.1_vace_14b":         ("large", 180),
    # Wan 2.2 — MoE held at `large` for HF PRO compat; needs model CPU offload.
    "wan2.2_ti2v_5b":          ("large",  60),
    "wan2.2_t2v_a14b":         ("large", 120),
    "wan2.2_i2v_a14b":         ("large", 150),
    "wan2.2_s2v_14b":          ("large", 240),
    "wan2.2_animate_14b":      ("large", 300),
}


def duration_for(mode_key: str, *, duration_s: float = 3.0, steps_override: int | None = None) -> int:
    """Return ZeroGPU seconds to reserve for one generation.

    Scales the per-mode default by requested video duration (longer video = more frames = more denoise time).
    Capped at 500s (RESEARCH §6.2 community practical ceiling).
    """
    if mode_key not in MODE_BUDGET:
        raise KeyError(f"No duration budget defined for mode {mode_key!r}")
    _, default_s = MODE_BUDGET[mode_key]
    # Scale by duration: baseline assumes 3s video; add 30% per extra second
    scaled = default_s * (1.0 + 0.3 * max(0.0, duration_s - 3.0))
    if steps_override:
        # If user overrides steps via Advanced, scale roughly linearly past 4
        scaled = scaled * (steps_override / 4.0)
    return min(500, int(scaled))


def size_for(mode_key: str) -> Size:
    """Return the ZeroGPU size tier for a mode (`large` or `xlarge`)."""
    if mode_key not in MODE_BUDGET:
        raise KeyError(f"No size defined for mode {mode_key!r}")
    return MODE_BUDGET[mode_key][0]
