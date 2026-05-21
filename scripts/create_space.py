"""Programmatic Space configuration — sets `space_volumes` for our duplicated mirrors.

Run AFTER scripts/duplicate_upstream.py. The Space repo `techfreakworm/wan-studio` is
ALREADY CREATED (we did this manually for the probe). This script just updates volumes
+ hardware.

Volume mount paths inside the Space container:
  /models/wan2.1-t2v-14b
  /models/wan2.1-i2v-14b-480p
  /models/wan2.1-i2v-14b-720p
  /models/wan2.2-t2v-a14b
  /models/wan2.2-i2v-a14b
  /models/wan2.2-lightning      (Lightning LoRA bundle)

Usage:
  python scripts/create_space.py --dry-run
  python scripts/create_space.py
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.hf_api import Volume


SPACE_ID = "techfreakworm/wan-studio"


PHASE_1_VOLUMES = [
    # Base models (one mount per checkpoint — slug convention from pipelines/handle.py:_slug_for)
    Volume(type="model", source="techfreakworm/wan2.1-t2v-14b",
           mount_path="/models/wan2.1-t2v-14b", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.1-i2v-14b-480p",
           mount_path="/models/wan2.1-i2v-14b-480p", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.1-i2v-14b-720p",
           mount_path="/models/wan2.1-i2v-14b-720p", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.2-t2v-a14b",
           mount_path="/models/wan2.2-t2v-a14b", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.2-i2v-a14b",
           mount_path="/models/wan2.2-i2v-a14b", read_only=True),
    # Consolidated Lightning LoRA mirror — single mount, multiple subdirs inside
    Volume(type="model", source="techfreakworm/wan-lightning-loras",
           mount_path="/models/wan-lightning-loras", read_only=True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    print(f"Configuring {SPACE_ID}")
    print(f"  {len(PHASE_1_VOLUMES)} volumes:")
    for v in PHASE_1_VOLUMES:
        print(f"    {v.source}  →  {v.mount_path}")

    if args.dry_run:
        return

    api.set_space_volumes(repo_id=SPACE_ID, volumes=PHASE_1_VOLUMES)
    print("✓ Volumes set")

    api.request_space_hardware(repo_id=SPACE_ID, hardware=SpaceHardware.ZERO_A10G)
    print(f"✓ Hardware requested: {SpaceHardware.ZERO_A10G}")


if __name__ == "__main__":
    sys.exit(main())
