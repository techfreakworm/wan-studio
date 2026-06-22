#!/usr/bin/env python3
"""Decisive attention test: monkeypatch F.scaled_dot_product_attention with an
explicit fp32 softmax implementation (correct signature this time) and compare
recon saturation vs the default MPS SDPA. If the manual path is clean (~0.35)
and default is neon (~0.85), torch-2.11 MPS SDPA is the root cause."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _sat(a):
    af = a.astype(np.float32); mx = af.max(-1); mn = af.min(-1)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


_orig_sdpa = F.scaled_dot_product_attention


def manual_sdpa(query=None, key=None, value=None, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False, **kw):
    q, k, v = query.float(), key.float(), value.float()
    sc = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * sc
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask.float()
    attn = attn.softmax(-1)
    return (attn @ v).to(query.dtype)


def gen(pipe, label):
    g = torch.Generator("cpu").manual_seed(42)
    out = pipe(prompt="A red panda eating bamboo on a mossy rock in a sunlit forest, cinematic, photorealistic",
               negative_prompt="", height=480, width=832, num_frames=13,
               num_inference_steps=10, guidance_scale=5.0, generator=g)
    fr = out.frames[0]
    a = np.asarray(fr[len(fr) // 2])
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
    from PIL import Image
    Image.fromarray(a).save(REPO / f"tests/outputs/sdpa_{label}.png")
    print(f"[{label}] recon_saturation={_sat(a):.3f} bright={a.mean():.1f} -> sdpa_{label}.png", flush=True)


def main():
    from pipelines.t2v import T2VHandle
    h = T2VHandle.for_key("wan2.1_t2v_1.3b")
    h.configure_preset("quality")
    pipe = h.pipe

    print("default MPS SDPA:", flush=True)
    gen(pipe, "default")

    print("manual fp32 softmax (SDPA bypassed):", flush=True)
    F.scaled_dot_product_attention = manual_sdpa
    try:
        gen(pipe, "manual")
    finally:
        F.scaled_dot_product_attention = _orig_sdpa


if __name__ == "__main__":
    main()
