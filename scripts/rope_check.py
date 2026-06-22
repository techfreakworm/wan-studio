#!/usr/bin/env python3
"""RoPE-table check (wan-brain patch #2) for the manual-also-neon branch.

The Wan transformer's WanRotaryPosEmbed builds per-(frame,height,width) cos/sin
tables from the post-patchify grid. If the >4-latent-frame path corrupts the
tables, even the SHARED positions 0-3 would differ between the 4-latent (clean,
13 frames) and 5-latent (broken, 17 frames) cases. They MUST be byte-identical
for positions 0-3; position 4+ is eyeballed for NaN/wrap.

Builds the rope module directly with dummy latents of each temporal size on MPS.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import torch  # noqa: E402


def rope_tables(rope, latent_T, H=60, W=104, dev="mps"):
    """Call the rope module with dummy hidden_states [B, C, T, H, W]-patchified.
    WanRotaryPosEmbed.forward takes the post-patch-embed hidden_states (B, C, T, Hp, Wp)
    where Hp,Wp are already / patch. We mimic the shape the transformer feeds it."""
    # The transformer calls self.rope(hidden_states) where hidden_states is the
    # patch-embedded tensor [B, C, T, Hp, Wp]. patch_size=(1,2,2) → Hp=H/2, Wp=W/2.
    C = rope.attention_head_dim if hasattr(rope, "attention_head_dim") else 128
    Hp, Wp = H // 2, W // 2
    x = torch.zeros(1, C, latent_T, Hp, Wp, device=dev, dtype=torch.bfloat16)
    cos, sin = rope(x)
    return cos.float().cpu(), sin.float().cpu()


def main():
    from pipelines.t2v import T2VHandle
    h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality")
    rope = h.pipe.transformer.rope
    # 13 px frames → 4 latent; 17 px frames → 5 latent.
    c4, s4 = rope_tables(rope, 4)
    c5, s5 = rope_tables(rope, 5)
    print("cos4 shape", tuple(c4.shape), "cos5 shape", tuple(c5.shape))
    # The token axis is flattened (T*Hp*Wp). To compare temporal positions 0-3 we
    # need the per-frame blocks: tokens are ordered frame-major (ppf outer).
    Hp, Wp = 30, 52
    per_frame = Hp * Wp
    finite5 = bool(torch.isfinite(c5).all() and torch.isfinite(s5).all())
    print("cos5/sin5 finite:", finite5)
    for pos in range(4):
        a = c4[:, pos * per_frame:(pos + 1) * per_frame]
        b = c5[:, pos * per_frame:(pos + 1) * per_frame]
        if a.shape == b.shape:
            d = (a - b).abs().max().item()
            print(f"  temporal pos {pos}: cos max|Δ4-vs-5| = {d:.3e}  ({'IDENTICAL' if d==0 else 'DIFFERS!!'})")
    # position 4 only exists in the 5-frame case — eyeball range
    p4 = c5[:, 4 * per_frame:5 * per_frame]
    print(f"  pos4 (5-frame only): finite={bool(torch.isfinite(p4).all())} "
          f"range[{p4.min().item():.3f},{p4.max().item():.3f}]")


if __name__ == "__main__":
    main()
