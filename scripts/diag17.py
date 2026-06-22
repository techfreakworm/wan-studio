#!/usr/bin/env python3
"""wan-brain's two discriminators for the 1.3B frame-count neon, run in the CHEAP
LAB (cached 1.3B, MPS) before pivoting:

  A) @17 frames × 3 seeds, default SDPA → deterministic (all broken) vs variance (mixed).
  B) @17 frames, manual-fp32-softmax SDPA (with a call-counter proving the patch is
     actually hit) → manual clean/default neon = long-seq MPS-SDPA bug; both neon =
     transformer temporal-RoPE on the >4-latent path.

sat ~0.38-0.5 = clean; >0.9 = neon/broken (calibrated: panda@13 ≈ 0.81-0.87 vivid-OK,
neon@17 ≈ 0.94)."""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

FRAMES, STEPS = 17, 12
PROMPT = "A red panda eating bamboo on a mossy rock in a sunlit forest, cinematic, photorealistic"


def _sat(a):
    af = a.astype(np.float32); mx = af.max(-1); mn = af.min(-1)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def gen(pipe, seed, tag):
    g = torch.Generator("cpu").manual_seed(seed)
    out = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832,
               num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0, generator=g)
    fr = out.frames[0]; a = np.asarray(fr[len(fr) // 2])
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
    Image.fromarray(a).save(REPO / f"tests/outputs/diag17_{tag}.png")
    print(f"  [{tag}] @{FRAMES}f seed={seed} sat={_sat(a):.3f} bright={a.mean():.1f}", flush=True)


_orig = F.scaled_dot_product_attention
_calls = {"n": 0}
def manual_sdpa(query=None, key=None, value=None, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False, **kw):
    _calls["n"] += 1
    q, k, v = query.float(), key.float(), value.float()
    sc = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * sc
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn + attn_mask.float()
    return (attn.softmax(-1) @ v).to(query.dtype)


def main():
    from pipelines.t2v import T2VHandle
    h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality"); pipe = h.pipe
    print("A) determinism — @17 default SDPA × 3 seeds:", flush=True)
    for s in (42, 7, 123):
        gen(pipe, s, f"default_s{s}")
    print("B) SDPA discriminator — @17 manual fp32 softmax, seed 42:", flush=True)
    F.scaled_dot_product_attention = manual_sdpa
    try:
        gen(pipe, 42, "manual_s42")
    finally:
        F.scaled_dot_product_attention = _orig
    print(f"  manual_sdpa invoked {_calls['n']} times "
          f"(expect ~{30}*{STEPS}=360+ for 30 blocks × {STEPS} steps × 2 CFG; 0 = patch MISSED)",
          flush=True)


if __name__ == "__main__":
    main()
