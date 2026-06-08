"""Single source-of-truth for Space volume mounts.

`set_space_volumes` REPLACES the entire volume set atomically (program risk R4),
so create_space.py must always pass the COMPLETE list from here.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipelines.registry import ALL_MODELS

SHARED_MIRROR = "techfreakworm/wan-shared-encoders"
PREPROC_MIRROR = "techfreakworm/wan-preproc"
LORA_MIRROR = "techfreakworm/wan-lightning-loras"


@dataclass(frozen=True)
class VolumeSpec:
    type: str
    source: str        # techfreakworm/<repo>
    mount_path: str     # /models/<slug>
    read_only: bool = True


def _model_volumes() -> list[VolumeSpec]:
    """One read-only mount per distinct model mirror.

    V2V shares the T2V-14B mirror, so dedupe by (source, mount_path): a card
    whose mirror_repo equals another card's is mounted once under the OWNER's
    slug. The owner is the card whose own slug-mirror matches its mirror_repo.
    """
    seen: dict[str, VolumeSpec] = {}
    for m in ALL_MODELS:
        slug = m.key.replace("_", "-")
        own_mirror = f"techfreakworm/{slug}-bf16"
        if m.mirror_repo != own_mirror:
            continue  # shares another card's mirror (e.g. v2v) → not its own mount
        seen[m.mirror_repo] = VolumeSpec("model", m.mirror_repo, f"/models/{slug}")
    return list(seen.values())


def all_volumes() -> list[VolumeSpec]:
    return _model_volumes() + [
        VolumeSpec("model", SHARED_MIRROR, "/models/wan-shared-encoders"),
        VolumeSpec("model", PREPROC_MIRROR, "/models/wan-preproc"),
        VolumeSpec("model", LORA_MIRROR, "/models/wan-lightning-loras"),
    ]


def expected_mount_paths() -> list[str]:
    return [v.mount_path for v in all_volumes()]
