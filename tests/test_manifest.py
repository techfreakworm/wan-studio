"""Tests for provisioning.manifest — the single source-of-truth mount list."""
from provisioning import manifest
from pipelines.registry import ALL_MODELS, BY_KEY


def test_every_card_has_a_model_mount():
    mounts = {v.mount_path for v in manifest.all_volumes()}
    for m in ALL_MODELS:
        slug = m.key.replace("_", "-")
        # v2v shares the t2v-14b mount → its own slug need not be mounted
        if m.mirror_repo == BY_KEY["wan2.1_t2v_14b"].mirror_repo and m.key != "wan2.1_t2v_14b":
            continue
        assert f"/models/{slug}" in mounts, f"{m.key} not mounted"


def test_shared_preproc_lora_mounts_present():
    mounts = {v.mount_path for v in manifest.all_volumes()}
    assert "/models/wan-shared-encoders" in mounts
    assert "/models/wan-preproc" in mounts
    assert "/models/wan-lightning-loras" in mounts


def test_all_volumes_read_only_models():
    for v in manifest.all_volumes():
        assert v.read_only is True
        assert v.type == "model"


def test_no_duplicate_mount_paths():
    paths = [v.mount_path for v in manifest.all_volumes()]
    assert len(paths) == len(set(paths)), "duplicate mount paths"


def test_expected_mount_paths_helper_matches_volumes():
    assert set(manifest.expected_mount_paths()) == {v.mount_path for v in manifest.all_volumes()}
