"""Tests for handle._mount_path context-aware resolution."""
import pytest

from pipelines import handle
from pipelines.registry import BY_KEY


def test_stitched_present_returns_stitched(monkeypatch):
    monkeypatch.setattr(handle, "stitch_local_dir", lambda card: "/tmp/wan-stitched/x")
    assert handle._mount_path(BY_KEY["wan2.1_t2v_14b"]) == "/tmp/wan-stitched/x"


def test_local_missing_mount_returns_bf16_mirror(monkeypatch):
    monkeypatch.setattr(handle, "stitch_local_dir", lambda card: None)
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    card = BY_KEY["wan2.1_t2v_14b"]
    assert handle._mount_path(card) == card.mirror_repo  # NOT the upstream fp32 repo


def test_zerogpu_missing_mount_raises(monkeypatch):
    monkeypatch.setattr(handle, "stitch_local_dir", lambda card: None)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    with pytest.raises(RuntimeError, match="mount .* missing"):
        handle._mount_path(BY_KEY["wan2.1_t2v_14b"])
