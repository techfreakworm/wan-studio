"""One-shot script — duplicate upstream Wan-AI / lightx2v repos into our account
for Phase 1 resilience. Run BEFORE create_space.py.

Idempotent — skips destinations that already exist.

Phase 1 scope:
  - 5 base model repos (T2V / I2V on Wan 2.1 14B + Wan 2.2 A14B)
  - 3 Lightning LoRA upstream repos consolidated into a single mirror via curated
    file uploads (avoids mirroring all of Kijai/WanVideo_comfy's terabytes)

Usage:
  python scripts/duplicate_upstream.py --dry-run    # print plan
  python scripts/duplicate_upstream.py              # execute
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from provisioning.preproc_manifest import (  # noqa: E402
    PREPROC_ASSETS,
    SHARED_ENCODERS,
    PreprocAsset,
)
from provisioning.manifest import SHARED_MIRROR, PREPROC_MIRROR  # noqa: E402


# Vendored S2V / TI2V repos — mirrored AS-IS (fp32, no dtype conversion here).
# Dest ids MUST match the cards' `mirror_repo` so manifest.all_volumes() mounts
# exactly what this script creates. They carry the `-bf16` suffix despite being
# fp32 mirrors: #3 bf16-converts them in place, keeping the dest id stable.
# (upstream_repo, dest_repo)
VENDORED_DUPLICATES: list[tuple[str, str]] = [
    ("Wan-AI/Wan2.2-S2V-14B",  "techfreakworm/wan2.2-s2v-14b-bf16"),
    ("Wan-AI/Wan2.2-TI2V-5B",  "techfreakworm/wan2.2-ti2v-5b-bf16"),
]


# Base model duplicates — full repo copies.
PHASE_1_BASE_DUPLICATES: list[tuple[str, str]] = [
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers",      "techfreakworm/wan2.1-t2v-14b"),
    ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "techfreakworm/wan2.1-i2v-14b-480p"),
    ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", "techfreakworm/wan2.1-i2v-14b-720p"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",     "techfreakworm/wan2.2-t2v-a14b"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers",     "techfreakworm/wan2.2-i2v-a14b"),
]

# Lightning LoRA files — pulled from various upstream and uploaded into ONE mirror.
# (upstream_repo, upstream_filename, mirror_path_inside_techfreakworm/wan-lightning-loras)
LIGHTNING_FILES: list[tuple[str, str, str]] = [
    # Wan 2.1 T2V-14B (single LoRA — community-recommended rank-128 v2)
    (
        "Kijai/WanVideo_comfy",
        "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
        "wan2.1-t2v-14b/lightning.safetensors",
    ),
    # Wan 2.1 I2V-14B 480P
    (
        "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
        "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        "wan2.1-i2v-14b-480p/lightning.safetensors",
    ),
    # Wan 2.1 I2V-14B 720P (same LoRA file actually works per RESEARCH §5.1; we
    # mirror the 720P-tagged copy for clarity)
    (
        "lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v",
        "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        "wan2.1-i2v-14b-720p/lightning.safetensors",
    ),
    # Wan 2.2 T2V-A14B — paired HIGH + LOW (V2.0 / 250928 — latest stable)
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
        "wan2.2-t2v-a14b/lightning_high.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
        "wan2.2-t2v-a14b/lightning_low.safetensors",
    ),
    # Wan 2.2 I2V-A14B — V1 Seko (no V2 as of May 2026 per RESEARCH §5.1.3)
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
        "wan2.2-i2v-a14b/lightning_high.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
        "wan2.2-i2v-a14b/lightning_low.safetensors",
    ),
]

LIGHTNING_MIRROR = "techfreakworm/wan-lightning-loras"


def _duplicate_pairs(api: HfApi, pairs: list[tuple[str, str]], dry_run: bool) -> None:
    """Mirror each (upstream, dest) repo as-is. Idempotent — skips existing dests."""
    for upstream, dest in pairs:
        if dry_run:
            print(f"  [dry] duplicate_repo({upstream!r} → {dest!r})")
            continue
        try:
            api.model_info(dest)
            print(f"  ✓ already exists: {dest}")
            continue
        except Exception:
            pass
        print(f"  ↻ duplicating {upstream} → {dest}", flush=True)
        api.duplicate_repo(from_id=upstream, to_id=dest, repo_type="model")
        print(f"  ✓ done: https://huggingface.co/{dest}")


def duplicate_base(api: HfApi, dry_run: bool) -> None:
    _duplicate_pairs(api, PHASE_1_BASE_DUPLICATES, dry_run)


def build_lightning_mirror(api: HfApi, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry] create_repo({LIGHTNING_MIRROR!r})")
        for src_repo, src_file, dst_path in LIGHTNING_FILES:
            print(f"    [dry] {src_repo}/{src_file} → {LIGHTNING_MIRROR}/{dst_path}")
        return

    try:
        api.model_info(LIGHTNING_MIRROR)
        print(f"  ✓ mirror repo exists: {LIGHTNING_MIRROR}")
    except Exception:
        api.create_repo(repo_id=LIGHTNING_MIRROR, repo_type="model", private=False)
        print(f"  ✓ created mirror repo: {LIGHTNING_MIRROR}")

    with tempfile.TemporaryDirectory() as tmpdir:
        for src_repo, src_file, dst_path in LIGHTNING_FILES:
            print(f"  ↻ {src_repo}/{src_file}  →  {LIGHTNING_MIRROR}/{dst_path}", flush=True)
            local = hf_hub_download(
                repo_id=src_repo, filename=src_file, cache_dir=tmpdir,
            )
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=dst_path,
                repo_id=LIGHTNING_MIRROR,
                repo_type="model",
                commit_message=f"Mirror {src_file}",
            )
            print(f"  ✓ uploaded {dst_path}")


def duplicate_vendored(api: HfApi, dry_run: bool) -> None:
    """Mirror the vendored S2V / TI2V repos as-is (no dtype conversion)."""
    _duplicate_pairs(api, VENDORED_DUPLICATES, dry_run)


def _recast_safetensors(src: Path, dst: Path, dtype) -> None:
    """Copy `src` → `dst`, recasting every *.safetensors tensor to `dtype`.

    Non-tensor files are copied verbatim.
    """
    import shutil

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        out = dst / item.name
        if item.is_dir():
            _recast_safetensors(item, out, dtype)
        elif item.suffix == ".safetensors":
            # Preserve the safetensors header (e.g. {"format":"pt"}); save_file
            # drops it unless we pass metadata= back through explicitly.
            with safe_open(str(item), framework="pt") as f:
                meta = f.metadata()
            tensors = {k: v.to(dtype) for k, v in load_file(str(item)).items()}
            save_file(tensors, str(out), metadata=meta)
        else:
            shutil.copy2(item, out)


def build_shared_encoders(api: HfApi, *, dry_run: bool) -> None:
    import shutil

    import torch
    from huggingface_hub import snapshot_download

    print(f"[shared] -> {SHARED_MIRROR}")
    if dry_run:
        for c in SHARED_ENCODERS:
            print(f"  {c.source_repo}:{c.source_subfolder} -> {c.dest_subfolder} ({c.dtype})")
        return
    if api.repo_exists(SHARED_MIRROR):
        print(f"  [skip] {SHARED_MIRROR} exists")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for c in SHARED_ENCODERS:
            snapshot_download(c.source_repo, allow_patterns=[f"{c.source_subfolder}/*"], local_dir=out / "_src")
            (out / c.dest_subfolder).parent.mkdir(parents=True, exist_ok=True)
            # text_encoder/vae are dtype-recast; image_* copied as-is fp32
            src = out / "_src" / c.source_subfolder
            if c.dtype == "bfloat16":
                _recast_safetensors(src, out / c.dest_subfolder, torch.bfloat16)
            else:
                shutil.copytree(src, out / c.dest_subfolder, dirs_exist_ok=True)
        api.create_repo(SHARED_MIRROR, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=SHARED_MIRROR, repo_type="model",
                          ignore_patterns=["_src/*"])
    print(f"  [done] {SHARED_MIRROR}")


@dataclass(frozen=True)
class _PreprocPlan:
    """Pure download/placement plan for one PreprocAsset.

    Both hf_hub_download(local_dir=...) and snapshot_download(local_dir=...)
    reproduce the upstream repo-relative tree, so neither honours dest_path on
    its own. This plan captures everything build_preproc needs to relocate the
    staged files into the dest_path layout the manifest documents.

      is_glob       True  -> snapshot_download(allow_patterns=[source_path])
                    False -> hf_hub_download(filename=source_path) (one file)
      allow_pattern source_path passed to snapshot_download (glob only)
      src_prefix    repo-relative dir to strip off each staged file
      dest_base     mirror-relative dir the stripped remainder is placed under
    """
    is_glob: bool
    allow_pattern: str | None
    src_prefix: str
    dest_base: str


def _preproc_plan(asset: PreprocAsset) -> _PreprocPlan:
    """Compute the (pure) download + placement plan for one preproc asset.

    Glob assets (source_path ends in '*' or is a bare '*') download a subtree;
    the leading directory of the glob is stripped and the remainder re-rooted
    under dest_path. File assets name a single file whose dest_path is the full
    target path (dir + filename).
    """
    sp = asset.source_path
    dp = asset.dest_path
    is_glob = sp.endswith("*") or "/" not in sp.rstrip("*")
    if is_glob:
        # 'models/dwpose/*' -> strip 'models/dwpose/'; '*' -> strip nothing.
        src_prefix = sp.rstrip("*").rstrip("/")
        # dest_path for a glob is a directory (manifest writes a trailing '/').
        dest_base = dp.rstrip("/")
        return _PreprocPlan(True, sp, src_prefix, dest_base)
    # Single file: dest_path is dir + filename; strip the source dir, place the
    # bare filename under dest_path's parent so the file lands exactly at dest.
    src_prefix = sp.rsplit("/", 1)[0]
    dest_base = dp.rsplit("/", 1)[0] if "/" in dp else ""
    return _PreprocPlan(False, None, src_prefix, dest_base)


def _preproc_dest(plan: _PreprocPlan, staged_relpath: str) -> str:
    """Map a staged file's repo-relative path to its mirror-relative dest path.

    `staged_relpath` is the path of a downloaded file relative to the staging
    root (i.e. its upstream repo-relative path). The plan's src_prefix is
    stripped and the remainder re-rooted under dest_base.
    """
    rel = staged_relpath
    if plan.src_prefix and rel.startswith(plan.src_prefix + "/"):
        rel = rel[len(plan.src_prefix) + 1:]
    elif plan.src_prefix and rel == plan.src_prefix:
        rel = rel.rsplit("/", 1)[-1]
    return f"{plan.dest_base}/{rel}" if plan.dest_base else rel


def build_preproc(api: HfApi, *, dry_run: bool) -> None:
    import shutil

    from huggingface_hub import snapshot_download

    print(f"[preproc] -> {PREPROC_MIRROR}")
    if dry_run:
        for a in PREPROC_ASSETS:
            print(f"  {a.source_repo}:{a.source_path} -> {a.dest_path}")
        return
    if api.repo_exists(PREPROC_MIRROR):
        print(f"  [skip] {PREPROC_MIRROR} exists")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "_mirror"
        stage = Path(tmp) / "_stage"
        for a in PREPROC_ASSETS:
            plan = _preproc_plan(a)
            asset_stage = stage / a.name
            if plan.is_glob:
                snapshot_download(a.source_repo, allow_patterns=[plan.allow_pattern],
                                  local_dir=asset_stage)
                for f in sorted(p for p in asset_stage.rglob("*") if p.is_file()):
                    rel = f.relative_to(asset_stage).as_posix()
                    if "/.cache/" in f"/{rel}" or rel.startswith(".cache/"):
                        continue  # hf bookkeeping dir
                    dest = out / _preproc_dest(plan, rel)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
            else:
                local = hf_hub_download(a.source_repo, a.source_path,
                                        local_dir=asset_stage, repo_type="model")
                rel = Path(local).relative_to(asset_stage).as_posix()
                dest = out / _preproc_dest(plan, rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, dest)
        api.create_repo(PREPROC_MIRROR, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=PREPROC_MIRROR, repo_type="model")
    print(f"  [done] {PREPROC_MIRROR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    if not args.dry_run:
        print(f"Logged in as: {api.whoami()['name']}")

    print("=== Phase 1 base model duplicates ===")
    duplicate_base(api, args.dry_run)

    print("\n=== Phase 1 Lightning LoRA mirror ===")
    build_lightning_mirror(api, args.dry_run)

    print("\n=== Vendored S2V / TI2V mirrors (as-is) ===")
    duplicate_vendored(api, args.dry_run)

    print("\n=== Shared encoders mirror ===")
    build_shared_encoders(api, dry_run=args.dry_run)

    print("\n=== Preproc mirror ===")
    build_preproc(api, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
