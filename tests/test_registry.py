"""Tests for pipelines.registry — ModelCard catalog consistency."""
import pytest

from pipelines.registry import (
    ALL_MODELS,
    BY_KEY,
    ModelCard,
    WAN_2_1,
    WAN_2_2,
    for_generation,
    for_mode,
    modes_in,
)


def test_no_duplicate_keys():
    keys = [m.key for m in ALL_MODELS]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


def test_by_key_lookup_matches_all_models():
    assert len(BY_KEY) == len(ALL_MODELS)
    for m in ALL_MODELS:
        assert BY_KEY[m.key] is m


def test_wan_2_1_count_matches_research():
    """RESEARCH §2: 7 checkpoint families for Wan 2.1."""
    assert len(WAN_2_1) == 7


def test_wan_2_2_count_matches_research():
    """RESEARCH §2: 5 checkpoint families for Wan 2.2."""
    assert len(WAN_2_2) == 5


def test_every_card_has_required_fields():
    for m in ALL_MODELS:
        assert m.repo
        assert m.size
        assert m.generation in ("wan2.1", "wan2.2")
        assert m.native_fps > 0
        assert m.quality_steps > 0
        assert m.flow_shift > 0


def test_lightning_consistency():
    """If lightning_available, must have either lightning_high_lora set."""
    for m in ALL_MODELS:
        if m.lightning_available:
            assert m.lightning_high_lora, f"{m.key} marked lightning_available but no LoRA path"
            assert m.lightning_steps > 0
            assert m.lightning_guidance == 1.0, "CFG-distilled LoRAs require CFG=1.0"


def test_moe_models_are_wan22_a14b():
    moe = [m for m in ALL_MODELS if m.is_moe]
    assert all(m.generation == "wan2.2" for m in moe)
    assert all("a14b" in m.key.lower() for m in moe)


def test_diffusers_class_present_for_native_modes():
    for m in ALL_MODELS:
        if m.mode in ("s2v", "ti2v"):
            # These vendor upstream wan package — no diffusers class
            assert m.diffusers_class is None, f"{m.key} should have no diffusers_class"
        else:
            assert m.diffusers_class, f"{m.key} missing diffusers_class"


def test_for_generation_filters_correctly():
    assert all(m.generation == "wan2.1" for m in for_generation("wan2.1"))
    assert all(m.generation == "wan2.2" for m in for_generation("wan2.2"))


def test_modes_in_returns_canonical_order():
    modes = modes_in("wan2.1")
    assert modes == [m for m in ["t2v", "i2v", "ti2v", "flf2v", "v2v", "vace", "s2v", "animate"] if m in modes]
