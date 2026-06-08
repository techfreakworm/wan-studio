"""Convert diffusers checkpoints to bf16 transformer-only mirrors.

Designed to run as an HF Job (server-side bandwidth) or locally. For each
non-vendored card: load each convert-subfolder at bf16, save_pretrained, copy
the keep-subfolders/files, push to card.mirror_repo. Idempotent (skip if dest
revision exists). `--dry-run` prints the plan without touching the Hub.

Usage:
  python scripts/convert_to_bf16.py --dry-run
  python scripts/convert_to_bf16.py --only wan2.1_t2v_14b
  python scripts/convert_to_bf16.py            # all non-vendored cards
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.registry import ALL_MODELS, BY_KEY  # noqa: E402
from provisioning.bf16_plan import conversion_plan  # noqa: E402


def _transformer_cls_for(card):
    """Select the transformer class matching the card's diffusers pipeline.

    VACE and Animate transformers are distinct classes with extra architecture
    params/submodules (vace_* for VACE; face/motion encoders for Animate), so
    loading their checkpoints through the base WanTransformer3DModel would fail
    on config mismatch or silently drop those weights. Everything else (T2V/I2V/
    FLF2V/V2V) uses the plain WanTransformer3DModel — consistent with the runtime
    pipelines/t2v.py and pipelines/i2v.py.
    """
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan_vace import (
        WanVACETransformer3DModel,
    )
    from diffusers.models.transformers.transformer_wan_animate import (
        WanAnimateTransformer3DModel,
    )

    return {
        "WanVACEPipeline": WanVACETransformer3DModel,
        "WanAnimatePipeline": WanAnimateTransformer3DModel,
    }.get(card.diffusers_class, WanTransformer3DModel)


def convert_one(card, *, dry_run: bool) -> None:
    plan = conversion_plan(card)
    if plan is None:
        print(f"[skip] {card.key}: vendored (no diffusers plan)")
        return
    print(f"[plan] {card.key} -> {card.mirror_repo}: convert={plan.convert_subfolders} "
          f"keep={plan.keep_subfolders} files={sorted(plan.keep_files)}")
    if dry_run:
        return

    import tempfile
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    transformer_cls = _transformer_cls_for(card)

    api = HfApi()
    if api.repo_exists(card.mirror_repo):
        print(f"[skip] {card.mirror_repo} already exists")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for sub in plan.convert_subfolders:
            t = transformer_cls.from_pretrained(
                card.repo, subfolder=sub, torch_dtype=torch.bfloat16)
            t.save_pretrained(out / sub)
        for sub in plan.keep_subfolders:
            snapshot_download(card.repo, allow_patterns=[f"{sub}/*"], local_dir=out)
        for f in plan.keep_files:
            hf_hub_download(card.repo, f, local_dir=out)
        api.create_repo(card.mirror_repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=card.mirror_repo, repo_type="model")
    print(f"[done] {card.mirror_repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="single card key")
    args = ap.parse_args()
    cards = [BY_KEY[args.only]] if args.only else ALL_MODELS
    for card in cards:
        convert_one(card, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
