"""Tests for pipelines.preset — Fast/Quality resolver + graceful fallback."""
from pipelines.preset import resolve, PresetKwargs
from pipelines.registry import BY_KEY


def test_fast_preset_on_supported_mode():
    """Wan 2.1 T2V-14B has Lightning LoRA → Fast preset uses 4 steps + CFG=1."""
    card = BY_KEY["wan2.1_t2v_14b"]
    kwargs = resolve(card, "fast")

    assert isinstance(kwargs, PresetKwargs)
    assert kwargs.effective_preset == "fast"
    assert kwargs.num_inference_steps == 4
    assert kwargs.guidance_scale == 1.0
    assert kwargs.lora_active is True
    assert kwargs.fallback_message is None


def test_quality_preset_on_supported_mode():
    """Quality preset disables LoRA and uses base steps."""
    card = BY_KEY["wan2.1_t2v_14b"]
    kwargs = resolve(card, "quality")

    assert kwargs.effective_preset == "quality"
    assert kwargs.num_inference_steps == card.quality_steps
    assert kwargs.guidance_scale == card.quality_guidance
    assert kwargs.lora_active is False


def test_fast_preset_falls_back_on_unsupported_mode():
    """Wan 2.1 T2V-1.3B has no Lightning LoRA → Fast falls back to Quality."""
    card = BY_KEY["wan2.1_t2v_1.3b"]
    kwargs = resolve(card, "fast")

    assert kwargs.effective_preset == "quality"
    assert kwargs.num_inference_steps == card.quality_steps
    assert kwargs.lora_active is False
    assert kwargs.fallback_message is not None
    assert "Lightning unavailable" in kwargs.fallback_message


def test_moe_fast_preset_sets_both_guidance_scales():
    """Wan 2.2 T2V-A14B (MoE) → Fast sets guidance_scale_2 too."""
    card = BY_KEY["wan2.2_t2v_a14b"]
    kwargs = resolve(card, "fast")

    assert card.is_moe
    assert kwargs.guidance_scale == 1.0
    assert kwargs.guidance_scale_2 == 1.0


def test_moe_quality_preset_sets_both_guidance_scales():
    card = BY_KEY["wan2.2_t2v_a14b"]
    kwargs = resolve(card, "quality")

    assert kwargs.guidance_scale_2 is not None
    assert kwargs.guidance_scale_2 == card.quality_guidance_2


def test_non_moe_quality_leaves_guidance_2_none():
    """Single-transformer models don't have a low-noise CFG."""
    card = BY_KEY["wan2.1_t2v_14b"]
    kwargs = resolve(card, "quality")
    assert kwargs.guidance_scale_2 is None


def test_moe_t2v_a14b_quality_high_low_cfg_are_distinct():
    """T2V-A14B Quality preset uses distinct high/low CFG per Wan repo config
    (sample_guide_scale=(3.0, 4.0))."""
    card = BY_KEY["wan2.2_t2v_a14b"]
    kwargs = resolve(card, "quality")
    assert kwargs.guidance_scale == 3.0       # high-noise
    assert kwargs.guidance_scale_2 == 4.0     # low-noise
    assert kwargs.guidance_scale != kwargs.guidance_scale_2
