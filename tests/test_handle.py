"""Tests for pipelines.handle — WanModelHandle.

Heavy integration (real pipeline build) is in test_smoke_t2v_local.py.
"""
import pytest

from pipelines.handle import WanModelHandle
from pipelines.registry import BY_KEY


def test_for_key_returns_handle_with_correct_card():
    h = WanModelHandle.for_key("wan2.1_t2v_14b")
    assert h.card.key == "wan2.1_t2v_14b"
    assert h.pipe is None  # lazy
    assert h.lora_loaded is False


def test_for_key_unknown_raises():
    with pytest.raises(KeyError):
        WanModelHandle.for_key("not_a_real_key")


def test_unload_to_cpu_is_noop_when_not_loaded():
    h = WanModelHandle.for_key("wan2.1_t2v_14b")
    h.unload_to_cpu()  # should not raise
    assert h.pipe is None


def test_build_pipeline_must_be_overridden():
    h = WanModelHandle(BY_KEY["wan2.1_t2v_14b"])
    with pytest.raises(NotImplementedError):
        h._build_pipeline()
