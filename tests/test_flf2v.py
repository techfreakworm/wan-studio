"""Tests for FLF2VHandle (no model load)."""
import pipelines  # noqa: F401
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.flf2v import FLF2VHandle
from pipelines.i2v import I2VHandle


def test_flf2v_registered():
    assert "flf2v" in HANDLER_REGISTRY
    spec = HANDLER_REGISTRY["flf2v"]
    assert spec.handle_cls is FLF2VHandle
    assert spec.tier == "large"


def test_flf2v_subclasses_i2v():
    assert issubclass(FLF2VHandle, I2VHandle)


def test_flf2v_key_fixed_720p():
    spec = HANDLER_REGISTRY["flf2v"]
    assert spec.key_for("wan2.1") == "wan2.1_flf2v_14b_720p"
    assert spec.key_for("wan2.2") == "wan2.1_flf2v_14b_720p"  # Wan 2.1 only


def test_flf2v_handle_card():
    h = FLF2VHandle.for_key("wan2.1_flf2v_14b_720p")
    assert h.card.mode == "flf2v"
    assert h.card.requires_image_encoder is True
