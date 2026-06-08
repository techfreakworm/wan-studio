"""Tests for provisioning.bf16_plan — per-card conversion plan."""
import pytest

from provisioning.bf16_plan import conversion_plan
from pipelines.registry import BY_KEY
from scripts.convert_to_bf16 import _transformer_cls_for


def test_single_transformer_card():
    plan = conversion_plan(BY_KEY["wan2.1_t2v_14b"])
    assert plan.convert_subfolders == ["transformer"]          # bf16
    assert "transformer_2" not in plan.convert_subfolders
    assert "text_encoder" not in plan.keep_subfolders          # lives in shared mirror
    assert "vae" not in plan.keep_subfolders
    assert set(plan.keep_subfolders) >= {"scheduler", "tokenizer"}
    assert plan.keep_files >= {"model_index.json"}


def test_moe_card_converts_both_experts():
    plan = conversion_plan(BY_KEY["wan2.2_t2v_a14b"])
    assert plan.convert_subfolders == ["transformer", "transformer_2"]


def test_animate_keeps_image_processor_and_encoder():
    """Amendment 2: Animate mirror must NOT be stripped of image_processor/encoder."""
    plan = conversion_plan(BY_KEY["wan2.2_animate_14b"])
    assert "transformer" in plan.convert_subfolders
    assert {"image_processor", "image_encoder"} <= set(plan.keep_subfolders)


def test_vendored_cards_have_no_diffusers_plan():
    """S2V / TI2V are vendored (diffusers_class=None) → bf16 conversion deferred to #3."""
    for key in ("wan2.2_s2v_14b", "wan2.2_ti2v_5b"):
        assert conversion_plan(BY_KEY[key]) is None


@pytest.mark.parametrize(
    "card_key,expected_cls",
    [
        # VACE / Animate have distinct transformer classes with extra
        # architecture (vace_*; face/motion encoders) — the regression this
        # commit guards against was loading them through the base class.
        ("wan2.1_vace_14b", "WanVACETransformer3DModel"),
        ("wan2.2_animate_14b", "WanAnimateTransformer3DModel"),
        # Base case: T2V/I2V/FLF2V/V2V all use the plain transformer.
        ("wan2.1_t2v_14b", "WanTransformer3DModel"),
    ],
)
def test_transformer_cls_for_dispatch(card_key, expected_cls):
    assert _transformer_cls_for(BY_KEY[card_key]).__name__ == expected_cls
