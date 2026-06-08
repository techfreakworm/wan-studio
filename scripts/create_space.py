"""Programmatic Space configuration — sets `space_volumes` for our duplicated mirrors.

Run AFTER scripts/duplicate_upstream.py. The Space repo `techfreakworm/wan-studio` is
ALREADY CREATED (we did this manually for the probe). This script just updates volumes
+ hardware.

`set_space_volumes` REPLACES the entire volume set atomically (program risk R4), so the
volume list is built from the single source-of-truth `provisioning.manifest.all_volumes()`
and ALWAYS passes the COMPLETE set.

Usage:
  python scripts/create_space.py --dry-run
  python scripts/create_space.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provisioning.manifest import all_volumes  # noqa: E402

SPACE_ID = "techfreakworm/wan-studio"


def build_volume_specs():
    return all_volumes()


def apply(space_id: str, *, dry_run: bool) -> None:
    specs = build_volume_specs()
    print(f"[volumes] {len(specs)} mounts -> {space_id}")
    for v in specs:
        print(f"  {v.source} -> {v.mount_path} (ro={v.read_only})")
    if dry_run:
        return
    from huggingface_hub import HfApi, SpaceHardware
    from huggingface_hub.hf_api import Volume
    api = HfApi()
    volumes = [Volume(type=v.type, source=v.source, mount_path=v.mount_path, read_only=v.read_only)
               for v in specs]
    api.set_space_volumes(repo_id=space_id, volumes=volumes)   # REPLACES the whole set
    api.request_space_hardware(repo_id=space_id, hardware=SpaceHardware.ZERO_A10G)
    print("[done] volumes + hardware applied")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--space", default=SPACE_ID)
    args = ap.parse_args()
    apply(args.space, dry_run=args.dry_run)
