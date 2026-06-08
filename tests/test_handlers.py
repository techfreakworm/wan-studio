"""Tests for the HANDLER_REGISTRY plugin pattern."""
import pipelines  # noqa: F401 — triggers self-registration
from pipelines.handlers import HANDLER_REGISTRY, HandlerSpec
from pipelines.t2v import T2VHandle
from pipelines.i2v import I2VHandle


def test_t2v_and_i2v_registered():
    assert "t2v" in HANDLER_REGISTRY
    assert "i2v" in HANDLER_REGISTRY


def test_spec_shape():
    spec = HANDLER_REGISTRY["t2v"]
    assert isinstance(spec, HandlerSpec)
    assert spec.handle_cls is T2VHandle
    assert callable(spec.key_for)         # (generation, **ui) -> registry key
    assert spec.key_for("wan2.2") == "wan2.2_t2v_a14b"


def test_i2v_key_for_resolution():
    spec = HANDLER_REGISTRY["i2v"]
    assert spec.handle_cls is I2VHandle
    assert spec.key_for("wan2.1", resolution_label="832x480 (16:9)") == "wan2.1_i2v_14b_480p"
    assert spec.key_for("wan2.1", resolution_label="1280x720 (16:9)") == "wan2.1_i2v_14b_720p"
