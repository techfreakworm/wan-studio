"""Tests for VACEHandle (no model load)."""
import os

import pipelines  # noqa: F401
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.vace import VACEHandle


def test_vace_registered_large_quality():
    assert "vace" in HANDLER_REGISTRY
    spec = HANDLER_REGISTRY["vace"]
    assert spec.handle_cls is VACEHandle
    assert spec.tier == "large"


def test_vace_key_default_14b():
    spec = HANDLER_REGISTRY["vace"]
    assert spec.key_for("wan2.1") == "wan2.1_vace_14b"
    assert spec.key_for("wan2.2") == "wan2.1_vace_14b"  # Wan 2.1 only


def test_vace_key_local_override(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.setenv("WAN_STUDIO_VACE_LOCAL_KEY", "wan2.1_vace_1.3b")
    spec = HANDLER_REGISTRY["vace"]
    assert spec.key_for("wan2.1") == "wan2.1_vace_1.3b"


def test_vace_handle_card_quality_only():
    h = VACEHandle.for_key("wan2.1_vace_14b")
    assert h.card.mode == "vace"
    assert h.card.lightning_available is False
    assert h.card.requires_image_encoder is False
    assert h.pipe is None
