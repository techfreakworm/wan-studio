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


def test_create_space_uses_full_manifest():
    """create_space must build its Volume list from manifest.all_volumes() (atomic replace)."""
    import scripts.create_space as cs
    specs = cs.build_volume_specs()          # returns the manifest VolumeSpecs
    from provisioning.manifest import all_volumes
    assert {v.mount_path for v in specs} == {v.mount_path for v in all_volumes()}


def test_every_manifest_source_has_a_creator():
    """Every mounted source must actually be created by a provisioning script.

    The boot probe hard-fails the Space if a mount source repo doesn't exist, so
    every `source` in all_volumes() must be in the set of repos the scripts create:
      (a) the bf16 mirror of every NON-vendored card (created by convert_to_bf16),
      (b) the shared / preproc / lora mirrors,
      (c) the VENDORED_DUPLICATES dest ids (created by duplicate_upstream).
    """
    from provisioning.bf16_plan import conversion_plan
    from scripts.duplicate_upstream import VENDORED_DUPLICATES

    creatable: set[str] = set()
    # (a) convert_to_bf16 creates the mirror for every card with a diffusers plan
    for m in ALL_MODELS:
        if conversion_plan(m) is not None:
            creatable.add(m.mirror_repo)
    # (b) shared / preproc / lora mirrors
    creatable |= {manifest.SHARED_MIRROR, manifest.PREPROC_MIRROR, manifest.LORA_MIRROR}
    # (c) vendored duplicates created by duplicate_upstream
    creatable |= {dest for _upstream, dest in VENDORED_DUPLICATES}

    for v in manifest.all_volumes():
        assert v.source in creatable, f"{v.source} mounted but never created"


# ── Subset-deploy (WAN_STUDIO_MODELS) ──────────────────────────────────────

def test_subset_deploy_filters_to_served_model_plus_shared(monkeypatch):
    monkeypatch.setenv("WAN_STUDIO_MODELS", "wan2.1_t2v_1.3b")
    paths = {v.mount_path for v in manifest.all_volumes()}
    assert paths == {"/models/wan2.1-t2v-1.3b", "/models/wan-shared-encoders"}
    # the 1.3B T2V needs neither preproc nor Lightning LoRAs
    assert "/models/wan-preproc" not in paths
    assert "/models/wan-lightning-loras" not in paths


def test_subset_includes_preproc_and_lora_when_a_served_model_needs_them(monkeypatch):
    monkeypatch.setenv("WAN_STUDIO_MODELS", "wan2.1_vace_14b,wan2.1_t2v_14b")
    paths = {v.mount_path for v in manifest.all_volumes()}
    assert "/models/wan-preproc" in paths           # VACE consumes wan-preproc
    assert "/models/wan-lightning-loras" in paths   # t2v-14b has Lightning


def test_full_deploy_when_env_unset(monkeypatch):
    monkeypatch.delenv("WAN_STUDIO_MODELS", raising=False)
    assert manifest.available_keys() is None
    paths = {v.mount_path for v in manifest.all_volumes()}
    assert {"/models/wan-shared-encoders", "/models/wan-preproc",
            "/models/wan-lightning-loras"} <= paths


def test_resolve_available_key_falls_back_to_served(monkeypatch):
    monkeypatch.setenv("WAN_STUDIO_MODELS", "wan2.1_t2v_1.3b")
    assert manifest.resolve_available_key("wan2.2_t2v_a14b", "t2v") == "wan2.1_t2v_1.3b"
    assert manifest.resolve_available_key("wan2.1_t2v_14b", "t2v") == "wan2.1_t2v_1.3b"
    assert manifest.resolve_available_key("wan2.1_t2v_1.3b", "t2v") == "wan2.1_t2v_1.3b"


def test_resolve_available_key_full_deploy_unchanged(monkeypatch):
    monkeypatch.delenv("WAN_STUDIO_MODELS", raising=False)
    assert manifest.resolve_available_key("wan2.2_t2v_a14b", "t2v") == "wan2.2_t2v_a14b"
