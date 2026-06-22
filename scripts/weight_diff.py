#!/usr/bin/env python3
"""Compare my bf16 mirror transformer vs the native upstream transformer,
tensor by tensor. If native generates clean but the mirror is neon, the mirror
weights are corrupted — this shows where (shape mismatch, all-zero, transposed,
or wildly different magnitude)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel  # noqa: E402

MIRROR = "techfreakworm/wan2.1-t2v-1.3b-bf16"
NATIVE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def load(repo):
    return WanTransformer3DModel.from_pretrained(repo, subfolder="transformer",
                                                 torch_dtype=torch.float32)


def main():
    print("loading mirror…", flush=True)
    m = dict(load(MIRROR).named_parameters())
    print("loading native…", flush=True)
    n = dict(load(NATIVE).named_parameters())
    keys_m, keys_n = set(m), set(n)
    print(f"mirror params={len(keys_m)} native params={len(keys_n)}")
    print(f"only in mirror: {sorted(keys_m - keys_n)[:5]}")
    print(f"only in native: {sorted(keys_n - keys_m)[:5]}")
    shared = sorted(keys_m & keys_n)
    worst = []
    nshape = 0
    for k in shared:
        a, b = m[k], n[k]
        if a.shape != b.shape:
            nshape += 1
            print(f"SHAPE DIFF {k}: mirror{tuple(a.shape)} native{tuple(b.shape)}")
            continue
        d = (a - b).abs()
        rel = d.mean().item() / (b.abs().mean().item() + 1e-9)
        worst.append((rel, k, d.max().item(), b.abs().mean().item()))
    worst.sort(reverse=True)
    print(f"\nshape mismatches: {nshape}")
    print("top-10 by relative mean abs diff (bf16 rounding should be ~1e-3):")
    for rel, k, mx, base in worst[:10]:
        print(f"  rel={rel:.4f} max={mx:.4f} base|w|={base:.4f}  {k}")
    allrel = [w[0] for w in worst]
    import statistics
    print(f"\nmedian rel diff = {statistics.median(allrel):.5f} (bf16≈0.002 expected)")


if __name__ == "__main__":
    main()
