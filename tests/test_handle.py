"""Tests for pipelines.handle — WanModelHandle.

Heavy integration (real pipeline build) is in test_smoke_t2v_local.py.
"""
from pathlib import Path

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


def test_tier2_warm_copy_symlinks_into_tmp(tmp_path, monkeypatch):
    from pipelines import handle
    src = tmp_path / "stitched"
    (src / "transformer").mkdir(parents=True)
    big = src / "transformer" / "model.safetensors"
    big.write_bytes(b"x" * 1024)
    (src / "config.json").write_text("{}")

    hot_root = tmp_path / "hot"
    monkeypatch.setattr(handle, "TIER2_ROOT", hot_root)

    out = handle.tier2_warm_copy("wan2.1-t2v-14b", str(src))
    assert (Path(out) / "transformer" / "model.safetensors").exists()
    assert (Path(out) / "config.json").exists()
    # idempotent
    assert handle.tier2_warm_copy("wan2.1-t2v-14b", str(src)) == out
