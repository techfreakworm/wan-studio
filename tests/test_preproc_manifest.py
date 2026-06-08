"""Tests for provisioning.preproc_manifest."""
from provisioning import preproc_manifest as pm


def test_shared_encoders_components():
    subs = {c.dest_subfolder for c in pm.SHARED_ENCODERS}
    assert {"text_encoder", "vae", "image_encoder", "image_processor"} <= subs


def test_shared_encoders_dtypes():
    by = {c.dest_subfolder: c for c in pm.SHARED_ENCODERS}
    assert by["text_encoder"].dtype == "bfloat16"   # UMT5
    assert by["vae"].dtype == "float32"             # quality-sensitive
    assert by["image_encoder"].dtype == "float32"   # CLIP


def test_preproc_covers_vace_and_animate():
    names = {a.name for a in pm.PREPROC_ASSETS}
    # VACE subset
    assert {"dwpose", "midas_dpt_hybrid", "raft"} <= names
    # Animate subset (amendment 1: NOT bundled in the diffusers mirror)
    assert {"vitpose_h_wholebody", "yolov10m", "sam2_hiera_large"} <= names


def test_preproc_assets_have_source_and_path():
    for a in pm.PREPROC_ASSETS:
        assert a.source_repo and a.source_path and a.dest_path
