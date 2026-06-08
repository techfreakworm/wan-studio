"""Tests for scripts.duplicate_upstream — pure path logic + dry-run wiring.

The networked download branches are not exercised here; instead the per-asset
source->dest path computation is factored into pure helpers (_preproc_plan /
_preproc_dest) that are unit-tested directly. This is what would have caught the
dest_path-ignored bugs without any network access.
"""
import torch
from safetensors.torch import load_file, save_file

from provisioning.preproc_manifest import PREPROC_ASSETS, PreprocAsset
import scripts.duplicate_upstream as du


# --------------------------------------------------------------------------- #
# _preproc_plan / _preproc_dest — the dest_path contract                       #
# --------------------------------------------------------------------------- #

# Representative staged (upstream repo-relative) files per asset name. These are
# the paths hf_hub_download / snapshot_download reproduce under the stage dir.
_STAGED = {
    "dwpose": ["models/dwpose/dw-ll_ucoco_384.onnx", "models/dwpose/yolox_l.onnx"],
    "midas_dpt_hybrid": ["config.json", "README.md", "pytorch_model.bin"],
    "raft": ["models/raft/raft-things.pth"],
    "vitpose_h_wholebody": ["process_checkpoint/pose2d/vitpose_h_wholebody.onnx"],
    "yolov10m": ["process_checkpoint/det/yolov10m.onnx"],
    "sam2_hiera_large": ["process_checkpoint/sam2/sam2_hiera_large.pt"],
}


def _asset(name):
    return next(a for a in PREPROC_ASSETS if a.name == name)


def test_every_staged_file_lands_under_its_manifest_dest_path():
    """Core regression: NO asset may leak its upstream repo prefix.

    Files must land at the dest_path the manifest documents — glob dest_path is
    a directory (trailing '/'), file dest_path is the full target path.
    """
    for a in PREPROC_ASSETS:
        plan = du._preproc_plan(a)
        for staged in _STAGED[a.name]:
            dest = du._preproc_dest(plan, staged)
            if a.dest_path.endswith("/"):
                # directory contract: dest sits directly under dest_path with
                # the upstream source prefix stripped (no 'models/' / 'process_checkpoint/').
                assert dest.startswith(a.dest_path), (a.name, staged, dest)
                remainder = dest[len(a.dest_path):]
                assert "/" not in remainder, ("nested below dest_path", a.name, dest)
            else:
                # file contract: dest IS the documented path exactly.
                assert dest == a.dest_path, (a.name, staged, dest)


def test_dwpose_strips_upstream_models_prefix():
    plan = du._preproc_plan(_asset("dwpose"))
    assert plan.is_glob
    assert du._preproc_dest(plan, "models/dwpose/yolox_l.onnx") == "vace/dwpose/yolox_l.onnx"
    # The old bug placed it at vace/models/dwpose/yolox_l.onnx.
    assert du._preproc_dest(plan, "models/dwpose/yolox_l.onnx") != "vace/models/dwpose/yolox_l.onnx"


def test_midas_bare_glob_reroots_root_files_under_dest_segment():
    """midas source_path='*' must NOT dump repo root files flat into vace/."""
    plan = du._preproc_plan(_asset("midas_dpt_hybrid"))
    assert plan.is_glob and plan.src_prefix == ""
    assert du._preproc_dest(plan, "config.json") == "vace/midas/config.json"
    # The old bug dropped the 'midas' segment -> 'vace/config.json' (collision).
    assert du._preproc_dest(plan, "config.json") != "vace/config.json"


def test_animate_single_files_drop_process_checkpoint_prefix():
    for name in ("vitpose_h_wholebody", "yolov10m", "sam2_hiera_large"):
        a = _asset(name)
        plan = du._preproc_plan(a)
        assert not plan.is_glob
        dest = du._preproc_dest(plan, _STAGED[name][0])
        assert dest == a.dest_path
        assert not dest.startswith("process_checkpoint/")


def test_file_vs_glob_classification():
    # globs: trailing '*' or a bare single segment.
    assert du._preproc_plan(_asset("dwpose")).is_glob          # models/dwpose/*
    assert du._preproc_plan(_asset("midas_dpt_hybrid")).is_glob  # *
    # single file: has a '/' and no '*'.
    assert not du._preproc_plan(_asset("vitpose_h_wholebody")).is_glob


def test_preproc_plan_no_collisions_across_vace_assets():
    """dwpose / midas / raft all dest under vace/ — verify distinct subdirs."""
    bases = set()
    for name in ("dwpose", "midas_dpt_hybrid", "raft"):
        plan = du._preproc_plan(_asset(name))
        assert plan.dest_base not in bases, ("overlapping dest_base", plan.dest_base)
        bases.add(plan.dest_base)
    assert bases == {"vace/dwpose", "vace/midas", "vace/raft"}


# --------------------------------------------------------------------------- #
# _duplicate_pairs — shared helper used by base + vendored                      #
# --------------------------------------------------------------------------- #

class _FakeApi:
    """Minimal HfApi stand-in recording duplicate_repo calls."""

    def __init__(self, existing=()):
        self._existing = set(existing)
        self.duplicated = []

    def model_info(self, repo_id):
        if repo_id not in self._existing:
            raise RuntimeError("404")
        return object()

    def duplicate_repo(self, from_id, to_id, repo_type):
        self.duplicated.append((from_id, to_id))
        self._existing.add(to_id)


def test_duplicate_pairs_dry_run_makes_no_calls():
    api = _FakeApi()
    du._duplicate_pairs(api, [("up/a", "dst/a"), ("up/b", "dst/b")], dry_run=True)
    assert api.duplicated == []


def test_duplicate_pairs_skips_existing_destinations():
    api = _FakeApi(existing={"dst/a"})
    du._duplicate_pairs(api, [("up/a", "dst/a"), ("up/b", "dst/b")], dry_run=False)
    assert api.duplicated == [("up/b", "dst/b")]  # a already exists -> skipped


def test_base_and_vendored_route_through_shared_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(du, "_duplicate_pairs",
                        lambda api, pairs, dry_run: calls.append((id(pairs), dry_run)))
    du.duplicate_base(object(), dry_run=True)
    du.duplicate_vendored(object(), dry_run=False)
    assert calls == [(id(du.PHASE_1_BASE_DUPLICATES), True),
                     (id(du.VENDORED_DUPLICATES), False)]


# --------------------------------------------------------------------------- #
# _recast_safetensors                                                           #
# --------------------------------------------------------------------------- #

def test_recast_safetensors_recasts_and_copies_verbatim(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    save_file({"w": torch.ones(4, dtype=torch.float32)}, str(src / "model.safetensors"))
    save_file({"b": torch.zeros(2, dtype=torch.float32)}, str(src / "sub" / "extra.safetensors"))
    (src / "config.json").write_text('{"hello": "world"}')

    du._recast_safetensors(src, dst, torch.bfloat16)

    top = load_file(str(dst / "model.safetensors"))
    assert top["w"].dtype == torch.bfloat16
    nested = load_file(str(dst / "sub" / "extra.safetensors"))
    assert nested["b"].dtype == torch.bfloat16
    # non-tensor files copied verbatim, byte-for-byte.
    assert (dst / "config.json").read_text() == '{"hello": "world"}'


# --------------------------------------------------------------------------- #
# dry-run wiring — no network                                                   #
# --------------------------------------------------------------------------- #

class _DryApi:
    def repo_exists(self, *_a, **_k):  # never reached on dry-run
        raise AssertionError("repo_exists must not be called on dry-run")


def test_build_preproc_dry_run_is_network_free(capsys):
    du.build_preproc(_DryApi(), dry_run=True)
    out = capsys.readouterr().out
    for a in PREPROC_ASSETS:
        assert a.dest_path in out


def test_build_shared_encoders_dry_run_is_network_free(capsys):
    du.build_shared_encoders(_DryApi(), dry_run=True)
    out = capsys.readouterr().out
    assert "text_encoder" in out and "vae" in out
