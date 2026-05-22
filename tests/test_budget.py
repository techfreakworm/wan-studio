"""Tests for utils.budget — get_duration() per mode."""
import pytest

from utils.budget import duration_for, MODE_BUDGET
from pipelines.registry import BY_KEY


def test_duration_for_returns_int():
    d = duration_for("wan2.1_t2v_14b", duration_s=3.0)
    assert isinstance(d, int)
    assert d > 0


def test_duration_for_unknown_key_raises():
    with pytest.raises(KeyError):
        duration_for("unknown_mode", duration_s=1.0)


def test_duration_scales_with_video_length():
    """Longer requested video should yield longer GPU reservation."""
    short = duration_for("wan2.1_t2v_14b", duration_s=1.0)
    long = duration_for("wan2.1_t2v_14b", duration_s=5.0)
    assert long > short


def test_duration_capped_at_500():
    """ZeroGPU practical ceiling per RESEARCH §6.2."""
    d = duration_for("wan2.2_animate_14b", duration_s=20.0)
    assert d <= 500


def test_mode_budget_has_all_phase1_modes():
    for key in ("wan2.1_t2v_14b", "wan2.1_i2v_14b_480p", "wan2.1_i2v_14b_720p",
                "wan2.2_t2v_a14b", "wan2.2_i2v_a14b"):
        assert key in MODE_BUDGET, f"missing budget for {key}"


def test_moe_modes_route_to_large_on_pro_tier():
    """Wan 2.2 A14B MoE is held at `large` while this Space is on HF PRO.

    `xlarge` requires HF PRO+ / Enterprise; PRO accounts get a 422 from
    /schedule. MoE pipelines compensate by calling
    `pipe.enable_model_cpu_offload()` to fit in 48 GB (see pipelines/t2v.py
    and pipelines/i2v.py). When the Space upgrades to PRO+, this expectation
    should flip back to `xlarge` and the offload branch should be removed.
    """
    for key in ("wan2.2_t2v_a14b", "wan2.2_i2v_a14b"):
        size, _ = MODE_BUDGET[key]
        assert size == "large", f"{key} must route to large on PRO tier (xlarge → 422)"


def test_small_modes_route_to_large():
    for key in ("wan2.1_t2v_1.3b", "wan2.1_t2v_14b"):
        size, _ = MODE_BUDGET[key]
        assert size == "large"
