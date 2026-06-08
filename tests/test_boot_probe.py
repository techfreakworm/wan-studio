"""Tests for app boot-probe mount assertion."""
import pytest


def test_assert_mounts_passes_when_all_present(monkeypatch, tmp_path):
    import app
    for p in __import__("provisioning.manifest", fromlist=["expected_mount_paths"]).expected_mount_paths():
        (tmp_path / p.lstrip("/")).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "MOUNT_ROOT", tmp_path)
    app.assert_expected_mounts()  # should not raise


def test_assert_mounts_raises_on_missing(monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app, "MOUNT_ROOT", tmp_path)  # nothing created
    with pytest.raises(RuntimeError, match="missing mount"):
        app.assert_expected_mounts()
