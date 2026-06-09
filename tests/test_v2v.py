"""Tests for V2VHandle (no model load)."""
import pipelines  # noqa: F401 — triggers registration
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.v2v import V2VHandle


def test_v2v_registered_large_tier():
    assert "v2v" in HANDLER_REGISTRY
    spec = HANDLER_REGISTRY["v2v"]
    assert spec.handle_cls is V2VHandle
    assert spec.tier == "large"


def test_v2v_key_is_fixed_wan21():
    # V2V is Wan 2.1 only; key_for ignores generation
    spec = HANDLER_REGISTRY["v2v"]
    assert spec.key_for("wan2.2") == "wan2.1_v2v_14b"
    assert spec.key_for("wan2.1") == "wan2.1_v2v_14b"


def test_v2v_handle_card_and_lazy():
    h = V2VHandle.for_key("wan2.1_v2v_14b")
    assert h.card.mode == "v2v"
    assert h.card.mirror_repo == "techfreakworm/wan2.1-t2v-14b-bf16"  # shared backbone
    assert h.pipe is None
