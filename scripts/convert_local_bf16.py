#!/usr/bin/env python3
"""Convert an upstream Wan checkpoint to a local bf16 transformer-only mirror.

Option-1 of the push ruling: write bf16 into ~/wan-bf16/<slug>/ (OUTSIDE the HF
cache so the disk-eviction step can't touch it), load from there, DEFER the HF
push. Records a pre-staged push map so a later "push them" is one command.

Per the conversion plan (provisioning/bf16_plan.py): convert transformer(s) to
bf16, keep the small scheduler/tokenizer/model_index + image_processor (and the
large image_encoder only for Animate); text_encoder/vae are dropped (injected at
load from wan-shared-encoders).

Disk discipline: download fp32 → save bf16 → (optionally) delete the fp32 source
cache for this repo, so high-water = one fp32 + kept bf16, not sum-of-all-fp32.

Usage:
  python scripts/convert_local_bf16.py --only wan2.1_vace_1.3b
  python scripts/convert_local_bf16.py --only wan2.2_t2v_a14b --purge-fp32
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for memcheck

from pipelines.registry import BY_KEY, ALL_MODELS  # noqa: E402
from provisioning.bf16_plan import conversion_plan  # noqa: E402
from pipelines.handle import LOCAL_BF16_ROOT, _slug_for  # noqa: E402
from memcheck import SerialLock, free_gb, _TX_BF16_GB  # noqa: E402

PUSH_MAP = LOCAL_BF16_ROOT / "_push_map.json"


def _convert_peak_ram_gb(card) -> float:
    """Conservative peak CPU-RAM estimate (GB) for converting this card.

    `from_pretrained(torch_dtype=bf16)` reads fp32 safetensors (~2x the bf16 size)
    and holds the bf16 output before save_pretrained flushes it. MoE converts the
    two experts SEQUENTIALLY (del+gc between), so peak is ONE transformer's
    fp32-load + its bf16 copy ≈ 3x the per-transformer bf16 footprint. This is a
    HEAVY RAM job — it MUST NOT co-run with an MPS generation (the historical
    parallel-job OS panic), hence SerialLock + this preflight in convert_one.
    """
    tx_bf16 = _TX_BF16_GB.get(card.key, 28.0)
    per_transformer = tx_bf16 / 2.0 if card.is_moe else tx_bf16
    return 3.0 * per_transformer


def _transformer_cls_for(card):
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel
    from diffusers.models.transformers.transformer_wan_animate import WanAnimateTransformer3DModel
    return {
        "WanVACEPipeline": WanVACETransformer3DModel,
        "WanAnimatePipeline": WanAnimateTransformer3DModel,
    }.get(card.diffusers_class, WanTransformer3DModel)


def _record_push(card, local_dir):
    PUSH_MAP.parent.mkdir(parents=True, exist_ok=True)
    m = json.loads(PUSH_MAP.read_text()) if PUSH_MAP.exists() else {}
    m[card.key] = {"local_dir": str(local_dir), "repo": card.mirror_repo}
    PUSH_MAP.write_text(json.dumps(m, indent=2))


def _purge_repo_cache(repo: str):
    """Delete an HF-cache model dir to reclaim the fp32 source after conversion."""
    name = "models--" + repo.replace("/", "--")
    d = Path.home() / ".cache/huggingface/hub" / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"[purge] removed fp32 cache {d}")


def convert_one(card, *, purge_fp32: bool) -> int:
    plan = conversion_plan(card)
    if plan is None:
        print(f"[skip] {card.key}: vendored (no diffusers plan) — needs custom handling")
        return 2
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    slug = _slug_for(card)
    out = LOCAL_BF16_ROOT / slug
    if (out / ".done").exists():
        print(f"[skip] {out} already converted")
        _record_push(card, out)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cls = _transformer_cls_for(card)

    # ── RAM PREFLIGHT (B3) — conversion is a HEAVY CPU-RAM job (fp32 load + bf16
    # copy). Refuse if the machine can't hold it, and NEVER let it co-run with an
    # MPS generation (the parallel-heavy-job OS panic). free_gb() is the true
    # reclaimable estimate; require est_peak + 20 GB headroom.
    est_ram = _convert_peak_ram_gb(card)
    fg = free_gb()
    print(f"[convert] {card.key}: {card.repo} convert={plan.convert_subfolders} "
          f"keep={plan.keep_subfolders} → {out}", flush=True)
    print(f"  [ram-preflight] est peak ~{est_ram:.0f}GB CPU-RAM | free ~{fg:.0f}GB", flush=True)
    if est_ram + 20.0 > fg:
        print(f"  !! REFUSED (convert RAM): need ~{est_ram:.0f}GB + 20GB headroom but only "
              f"~{fg:.0f}GB free. Close other jobs / free RAM and retry.", flush=True)
        return 5

    # SerialLock: a conversion and a generation must never be alive at once.
    with SerialLock():
        for sub in plan.convert_subfolders:
            print(f"  loading {sub} (fp32→bf16)…", flush=True)
            t = cls.from_pretrained(card.repo, subfolder=sub, torch_dtype=torch.bfloat16)
            t.save_pretrained(out / sub)
            del t
            import gc; gc.collect()
        for sub in plan.keep_subfolders:
            snapshot_download(card.repo, allow_patterns=[f"{sub}/*"], local_dir=out)
        for f in plan.keep_files:
            try:
                hf_hub_download(card.repo, f, local_dir=out)
            except Exception as e:
                print(f"  [warn] keep_file {f}: {type(e).__name__}: {e}")

    (out / ".done").touch()
    _record_push(card, out)
    sz = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    print(f"[done] {card.key} → {out} ({sz:.1f} GB bf16). Push deferred (in {PUSH_MAP}).")
    if purge_fp32:
        _purge_repo_cache(card.repo)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single card key")
    ap.add_argument("--purge-fp32", action="store_true", help="delete fp32 source cache after")
    args = ap.parse_args()
    cards = [BY_KEY[args.only]] if args.only else ALL_MODELS
    rc = 0
    for card in cards:
        rc |= convert_one(card, purge_fp32=args.purge_fp32)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
