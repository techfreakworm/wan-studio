#!/usr/bin/env python3
"""If native-vanilla is also neon, the suspect is torch-2.11 MPS attention (the
one transformer op the clean VAE round-trip never exercised). This probe runs
the SAME tiny generation under different SDPA kernels and reports recon
saturation for each — a clean result under one kernel pinpoints the culprit.

Usage: python scripts/attn_probe.py [bf16|float32]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _sat(a):
    af = a.astype(np.float32); mx = af.max(-1); mn = af.min(-1)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def gen_once(pipe, label):
    g = torch.Generator("cpu").manual_seed(42)
    out = pipe(prompt="A red panda eating bamboo on a mossy rock in a sunlit forest",
               negative_prompt="", height=480, width=832, num_frames=9,
               num_inference_steps=12, guidance_scale=5.0, generator=g)
    fr = out.frames[0]
    a = np.asarray(fr[len(fr) // 2])
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
    s = _sat(a)
    print(f"  [{label}] recon_saturation={s:.3f} bright={a.mean():.1f}", flush=True)
    return s


def main():
    # Build via the cached mirror stack (no native download) using the handle, so
    # we test the EXACT pipeline the harness uses.
    from pipelines.t2v import T2VHandle
    h = T2VHandle.for_key("wan2.1_t2v_1.3b")
    h.configure_preset("quality")   # ensure_loaded + .to(mps)
    pipe = h.pipe
    print(f"dtype={next(pipe.transformer.parameters()).dtype} (cached mirror stack)")

    # Baseline (default MPS SDPA).
    gen_once(pipe, "default-sdpa")

    # Force the math kernel — bypasses the fused MPS attention path.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            gen_once(pipe, "math-sdpa")
    except Exception as e:
        print(f"  [math-sdpa] unavailable: {type(e).__name__}: {e}", flush=True)

    # Monkeypatch SDPA to an explicit fp32 softmax implementation.
    import torch.nn.functional as F
    orig = F.scaled_dot_product_attention

    def manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        qf, kf, vf = q.float(), k.float(), v.float()
        sc = scale if scale is not None else 1.0 / (qf.shape[-1] ** 0.5)
        attn = (qf @ kf.transpose(-2, -1)) * sc
        if attn_mask is not None:
            attn = attn + (attn_mask.float() if attn_mask.dtype != torch.bool
                           else attn_mask.logical_not() * -1e9)
        attn = attn.softmax(-1)
        return (attn @ vf).to(q.dtype)

    F.scaled_dot_product_attention = manual_sdpa
    try:
        gen_once(pipe, "manual-fp32-softmax")
    finally:
        F.scaled_dot_product_attention = orig


if __name__ == "__main__":
    main()
