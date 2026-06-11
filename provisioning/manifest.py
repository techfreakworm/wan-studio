"""Single source-of-truth for Space volume mounts.

`set_space_volumes` REPLACES the entire volume set atomically (program risk R4),
so create_space.py must always pass the COMPLETE list from here.

Subset deploys: set the `WAN_STUDIO_MODELS` env var to a comma-separated list of
registry keys to serve only those models (plus the shared-encoders mount, plus
the preproc/lora mounts only when a served model actually needs them). Unset =
full deploy (all models). This lets a staging Space mount just one checkpoint
and boot cleanly, and doubles as the phased-rollout mechanism for production.
"""
from __future__ import annotations

import os
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


def available_keys() -> set[str] | None:
    """Registry keys this deployment serves, or None for a full (all-models) deploy."""
    raw = os.getenv("WAN_STUDIO_MODELS", "").strip()
    if not raw:
        return None
    return {k.strip() for k in raw.split(",") if k.strip()}


def _served_models() -> list:
    avail = available_keys()
    return [m for m in ALL_MODELS if avail is None or m.key in avail]


def resolve_available_key(preferred: str, mode: str) -> str:
    """Pick a registry key that is actually served by this deployment.

    Returns `preferred` if it is served (or this is a full deploy); otherwise the
    first served key for `mode`; otherwise `preferred` unchanged (the caller then
    fails loud on the missing mount). Drives the default preload + the runtime
    key resolution so a subset Space serves whatever it mounted.
    """
    avail = available_keys()
    if avail is None or preferred in avail:
        return preferred
    for m in ALL_MODELS:
        if m.mode == mode and m.key in avail:
            return m.key
    return preferred


def _model_volumes() -> list[VolumeSpec]:
    """One read-only mount per distinct served model mirror.

    V2V shares the T2V-14B mirror, so dedupe by mirror: a card whose mirror_repo
    equals another card's is mounted once under the OWNER's slug (the card whose
    own slug-mirror matches its mirror_repo). Subset deploys keep only served keys.
    """
    avail = available_keys()
    seen: dict[str, VolumeSpec] = {}
    for m in ALL_MODELS:
        if avail is not None and m.key not in avail:
            continue
        slug = m.key.replace("_", "-")
        own_mirror = f"techfreakworm/{slug}-bf16"
        if m.mirror_repo != own_mirror:
            continue  # shares another card's mirror (e.g. v2v) → not its own mount
        seen[m.mirror_repo] = VolumeSpec("model", m.mirror_repo, f"/models/{slug}")
    return list(seen.values())


def all_volumes() -> list[VolumeSpec]:
    avail = available_keys()
    served = _served_models()
    vols = _model_volumes() + [
        VolumeSpec("model", SHARED_MIRROR, "/models/wan-shared-encoders"),
    ]
    # preproc (DWPose/MiDaS/RAFT, ViTPose/YOLO/SAM2) only matters for VACE/Animate
    if avail is None or any(m.mode in ("vace", "animate") for m in served):
        vols.append(VolumeSpec("model", PREPROC_MIRROR, "/models/wan-preproc"))
    # lightning LoRAs only matter when a served model has Lightning
    if avail is None or any(m.lightning_available for m in served):
        vols.append(VolumeSpec("model", LORA_MIRROR, "/models/wan-lightning-loras"))
    return vols


def expected_mount_paths() -> list[str]:
    return [v.mount_path for v in all_volumes()]
