"""Tests for pipelines.shared — shared-encoder path resolution + image_processor."""
import os
from pathlib import Path

import pytest

from pipelines import shared


def test_shared_path_prefers_mount(monkeypatch, tmp_path):
    """On ZeroGPU with the shared-encoders mount present, use it (tier-2 hot-copy
    layer disabled here so we assert mount-preference directly)."""
    mount = tmp_path / "wan-shared-encoders"
    (mount / "vae").mkdir(parents=True)
    monkeypatch.setattr(shared, "SHARED_MOUNT", mount)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    monkeypatch.setenv("WAN_STUDIO_TIER2", "0")
    assert shared._shared_path() == str(mount)


def test_shared_path_tier2_copies_to_local(monkeypatch, tmp_path):
    """With tier-2 on (default), the shared-encoders dir is hot-copied to local
    disk and that local path is returned — so pipe.to('cuda') faults from NVMe,
    not the slow mount."""
    from pipelines import handle
    mount = tmp_path / "wan-shared-encoders"
    (mount / "vae").mkdir(parents=True)
    (mount / "vae" / "w.bin").write_bytes(b"x" * 16)
    hot = tmp_path / "hot"
    monkeypatch.setattr(shared, "SHARED_MOUNT", mount)
    monkeypatch.setattr(handle, "TIER2_ROOT", hot)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    monkeypatch.setenv("WAN_STUDIO_TIER2", "1")
    out = shared._shared_path()
    assert out == str(hot / "wan-shared-encoders")
    assert (hot / "wan-shared-encoders" / "vae" / "w.bin").read_bytes() == b"x" * 16


def test_shared_path_falls_back_to_mirror_repo(monkeypatch, tmp_path):
    """Locally (no mount), use the bf16 shared-encoders mirror repo id."""
    monkeypatch.setattr(shared, "SHARED_MOUNT", tmp_path / "absent")
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert shared._shared_path() == shared.SHARED_MIRROR_REPO


def test_shared_path_missing_mount_on_zerogpu_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(shared, "SHARED_MOUNT", tmp_path / "absent")
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    with pytest.raises(RuntimeError, match="wan-shared-encoders"):
        shared._shared_path()


def test_image_processor_is_callable():
    assert callable(shared.image_processor)
