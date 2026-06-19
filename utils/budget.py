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
    # I2V 14B: first call pays a ~78s cold host→GPU attach of the 28 GB
    # transformer + 13 GB shared encoders (measured via pipelines/trace.py),
    # then 4-step inference. The 90s budget was too tight (attach alone nearly
    # filled it). Headroom set so attach+infer fits with margin; warm repeat
    # calls reuse the GPU-resident handle and finish in ~20-30s. (Reducing the
    # attach itself — staging .to('cuda') into ZeroGPU's free init phase — is a
    # tracked follow-up.)
    "wan2.1_i2v_14b_480p":     ("large", 150),
    "wan2.1_i2v_14b_720p":     ("large", 180),
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


# ── One latency truth source ──────────────────────────────────────────────
# Measured per-(model) cost model: a fixed setup cost (GPU attach / first-step
# component stream) + a per-denoise-step cost (separate at 480p vs 720p). This
# single table feeds ALL of: the UI ETA, the ZeroGPU duration reservation, the
# up-front "won't fit" block, and the preset wall-time claims — so they can't
# drift apart (product-brain D3/D4/D5 unification).
#
# Seeded from trace data (2026-06, pipelines/trace.py): I2V-14B is offloaded so
# its setup is the ~60 s transformer stream and steps are ~13 s; T2V-14B runs
# RESIDENT (no image encoder, fits the slice) so attach is cheap and steps fast.
# REFINE these from real runs — they drive correctness, not just display.
#
#   key: (setup_seconds, per_step_480p, per_step_720p)
#
# IMPORTANT (measured 2026-06-12): "setup" for any 14B is the ~60-70s host→GPU
# move of ~40 GB faulted from the tier-2 local disk — it is NOT cheap even when
# the model runs *resident* (resident only saves per-step streaming, not the
# attach). Under-estimating this aborts the run mid-attach. Refine from real runs.
STEP_COST: dict[str, tuple[float, float, float]] = {
    "wan2.1_t2v_1.3b":         (8.0,   1.6,  6.0),
    # MEASURED live 2026-06-12: cold attach is variable ~102-155s + ~12s fixed
    # (text-encode/VAE); 5.7s/step resident @480p. warm Fast(4)=35s, cold Fast
    # =137-170s, warm Quality(50)=312s, cold ~414s (500s reservation accepted).
    # setup carries margin so a slow cold attach can't abort mid-attach.
    "wan2.1_t2v_14b":          (130.0, 6.0,  14.0),  # resident: big attach, fast steps
    "wan2.1_i2v_14b_480p":     (60.0, 13.0, 13.0),  # offloaded: stream dominates
    "wan2.1_i2v_14b_720p":     (60.0, 18.0, 18.0),
    "wan2.1_flf2v_14b_720p":   (60.0, 18.0, 18.0),
    "wan2.1_v2v_14b":          (10.0, 4.0,  9.0),
    "wan2.1_vace_1.3b":        (4.0,  2.0,  7.0),
    "wan2.1_vace_14b":         (60.0, 13.0, 18.0),
    "wan2.2_ti2v_5b":          (12.0, 3.0,  7.0),
    "wan2.2_t2v_a14b":         (70.0, 15.0, 22.0),  # MoE offload, dual transformer
    "wan2.2_i2v_a14b":         (70.0, 16.0, 24.0),
    "wan2.2_s2v_14b":          (60.0, 16.0, 20.0),
    "wan2.2_animate_14b":      (60.0, 18.0, 24.0),
}

# Practical PRO ZeroGPU reservation ceiling. 90/150/274 s confirmed accepted
# live; probing 500 s to discover the real PRO max and let 14B Quality (50
# steps, ~400 s cold) reserve enough. We stay <= this and BLOCK combinations
# whose estimate exceeds it rather than letting a user wait then abort.
ZEROGPU_DURATION_CAP = 500


def _is_720p(resolution_label: str) -> bool:
    return "720" in (resolution_label or "") or "1280" in (resolution_label or "")


def estimate_seconds(mode_key: str, *, steps: int, resolution_label: str = "",
                     duration_s: float = 3.0) -> int:
    """Best-estimate wall-clock for one generation (for the UI ETA).

    per-step cost scales with frame count (longer clip = bigger latent): baseline
    is a ~3 s clip, +30% per extra second."""
    setup, p480, p720 = STEP_COST.get(mode_key, (30.0, 10.0, 15.0))
    per_step = p720 if _is_720p(resolution_label) else p480
    frame_factor = 1.0 + 0.3 * max(0.0, float(duration_s) - 3.0)
    return int(round(setup + per_step * max(1, int(steps)) * frame_factor))


def reserve_seconds(mode_key: str, *, steps: int, resolution_label: str = "",
                    duration_s: float = 3.0) -> int:
    """ZeroGPU seconds to reserve = estimate + 25% margin, capped."""
    est = estimate_seconds(mode_key, steps=steps, resolution_label=resolution_label,
                           duration_s=duration_s)
    return min(ZEROGPU_DURATION_CAP, int(round(est * 1.25)))


def fits_window(mode_key: str, *, steps: int, resolution_label: str = "",
                duration_s: float = 3.0) -> bool:
    """False when even the bare estimate (no margin) exceeds the cap — the UI
    blocks the Generate up-front with a 'lower res/steps' hint."""
    return estimate_seconds(mode_key, steps=steps, resolution_label=resolution_label,
                            duration_s=duration_s) <= ZEROGPU_DURATION_CAP
