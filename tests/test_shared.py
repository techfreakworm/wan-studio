"""Tests for pipelines.shared — shared-encoder path resolution + image_processor."""
import os
from pathlib import Path

import pytest

from pipelines import shared


def test_shared_path_prefers_mount(monkeypatch, tmp_path):
    """On ZeroGPU with the shared-encoders mount present, use it."""
    mount = tmp_path / "wan-shared-encoders"
    (mount / "vae").mkdir(parents=True)
    monkeypatch.setattr(shared, "SHARED_MOUNT", mount)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    assert shared._shared_path() == str(mount)


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
