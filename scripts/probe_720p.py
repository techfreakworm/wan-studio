#!/usr/bin/env python3
"""Early-abort diagnostic probe for the i2v_720p resolution-specific noise bug.

(wan-brain's hunt) A divergent trajectory corrupts within the first 1-3 denoise
steps, so we do NOT need a full 46-min denoise+decode. This:
  - Stage A: wraps transformer.forward to print shapes/dtype/finiteness of the
    forward inputs (the 36ch concat condition, text embeds, any rotary kwargs) on
    the FIRST call — a misaligned concat / wrong RoPE grid shows with zero denoise.
  - Stage B: runs only --steps steps with output_type='latent' (NO VAE decode) and
    a per-step callback logging latent min/max/mean/std/finite — the divergence
    SIGNATURE: immediate blow-up/NaN/uniform => discrete conditioning/RoPE bug;
    gradual drift => bf16 attention precision over the longer 720p sequence.

Run the failing 720p and the clean 480p back-to-back and diff the signatures.

Usage:
  python scripts/probe_720p.py --mode wan2.1_i2v_14b_720p --image X.png --height 720 --width 1280 --steps 3
  python scripts/probe_720p.py --mode wan2.1_i2v_14b_720p --image X.png --height 480 --width 832 --steps 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def _stat(t, name: str) -> None:
    try:
        f = t.detach().float()
        print(f"   [{name}] shape={tuple(t.shape)} dtype={t.dtype} "
              f"min={f.min().item():.3f} max={f.max().item():.3f} "
              f"mean={f.mean().item():.4f} std={f.std().item():.4f} "
              f"finite={bool(torch.isfinite(f).all().item())}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"   [{name}] <stat err {type(e).__name__}: {e}>", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--fp32-attn", dest="fp32_attn", action="store_true",
                    help="upcast scaled_dot_product_attention to fp32 (precision fix-test)")
    ap.add_argument("--rope-fix", dest="rope_fix", action="store_true",
                    help="apply the mps_patches RoPE strided-assign fix (fix-test)")
    ap.add_argument("--exact-hw", dest="exact_hw", default=None,
                    help="force exact HxW (e.g. 704x1280), bypassing aspect_ratio_resize")
    ap.add_argument("--trace", action="store_true",
                    help="per-block magnitude trace (first forward) to localize the injector")
    args = ap.parse_args()

    if args.rope_fix:
        from pipelines.mps_patches import apply_mps_patches
        ok = apply_mps_patches()
        print(f"=== rope-fix (mps_patches) applied={ok} ===", flush=True)

    if args.fp32_attn:
        # FIX-TEST: force fp32 attention (keep weights bf16). If this kills the 720p
        # step-1 magnitude blowup, the bug is bf16 SDPA precision over the long
        # sequence → fix = resolution-gated fp32 attention.
        import torch.nn.functional as _F
        _orig_sdpa = _F.scaled_dot_product_attention

        def _sdpa_fp32(q, k, v, *a, **kw):
            dt = q.dtype
            out = _orig_sdpa(q.float(), k.float(), v.float(), *a, **kw)
            return out.to(dt)

        _F.scaled_dot_product_attention = _sdpa_fp32
        print("=== fp32-attn MONKEYPATCH active (SDPA upcast to fp32) ===", flush=True)

    from memcheck import assert_safe, SerialLock
    res = "720p" if (args.width >= 1100 or args.height >= 700) else "480p"
    assert_safe(args.mode, args.frames, res)

    with SerialLock():
        import pipelines  # noqa: F401  populate registry/handlers
        from pipelines.registry import BY_KEY
        from pipelines.handlers import HANDLER_REGISTRY
        from pipelines.i2v import aspect_ratio_resize

        card = BY_KEY[args.mode]
        spec = HANDLER_REGISTRY[card.mode]
        handle = spec.handle_cls.for_key(args.mode)
        pk = handle.configure_preset("quality")
        print(f"=== PROBE {args.mode} @ {args.height}x{args.width} steps={args.steps} "
              f"preset=quality g={pk.guidance_scale} ===", flush=True)

        tf = handle.pipe.transformer

        if getattr(args, "trace", False):
            # Per-block magnitude trace: find WHERE the velocity magnitude is injected
            # at the 720p grid. Print each block's output std/max + block0's attn1
            # (self-attn) output, for the FIRST forward only.
            blocks = tf.blocks
            tstate = {"count": 0, "nblocks": len(blocks)}

            def _mk_block_hook(i):
                def _h(mod, inp, out):
                    if tstate["count"] > tstate["nblocks"]:
                        return
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    of = o.float()
                    print(f"   [block{i:02d}] out std={of.std().item():.3f} "
                          f"absmax={of.abs().max().item():.2f}", flush=True)
                    tstate["count"] += 1
                return _h

            for i, b in enumerate(blocks):
                b.register_forward_hook(_mk_block_hook(i))

            def _attn1_hook(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                of = o.float()
                print(f"   [block0.attn1(self)] out std={of.std().item():.3f} "
                      f"absmax={of.abs().max().item():.2f}", flush=True)
            blocks[0].attn1.register_forward_hook(_attn1_hook)
            blocks[0].attn2.register_forward_hook(
                lambda m, i, o: print(f"   [block0.attn2(cross)] absmax="
                                      f"{(o[0] if isinstance(o,(tuple,list)) else o).float().abs().max().item():.2f}",
                                      flush=True))

        orig_forward = tf.forward
        seen = {"n": 0}

        def wrapped(*a, **kw):
            if seen["n"] == 0:
                print("=== FIRST TRANSFORMER FORWARD INPUTS ===", flush=True)
                hs = kw.get("hidden_states", a[0] if a else None)
                if torch.is_tensor(hs):
                    _stat(hs, "hidden_states (noise+cond concat)")
                for k, v in kw.items():
                    if k != "hidden_states" and torch.is_tensor(v):
                        _stat(v, k)
                    elif isinstance(v, (tuple, list)) and v and torch.is_tensor(v[0]):
                        _stat(v[0], f"{k}[0]")
            seen["n"] += 1
            return orig_forward(*a, **kw)

        tf.forward = wrapped

        img = Image.open(args.image).convert("RGB")
        if getattr(args, "exact_hw", None):
            _h, _w = (int(x) for x in args.exact_hw.lower().split("x"))
            resized, h, w = img.resize((_w, _h), Image.LANCZOS), _h, _w
            print(f"[exact-hw] forced {_w}x{_h} (latent {_h // 8}x{_w // 8}, "
                  f"patch {(_h // 8) // 2}x{(_w // 8) // 2})", flush=True)
        else:
            resized, h, w = aspect_ratio_resize(img, handle.pipe, args.height * args.width)
        print(f"resized image -> {resized.size} (h={h} w={w}); latent {h // 8}x{w // 8}", flush=True)

        gen = torch.Generator(device="cpu").manual_seed(42)

        def cb(pipe, step, t, kw):
            lat = kw.get("latents")
            if torch.is_tensor(lat):
                _stat(lat, f"latents@step{step}")
            return kw

        try:
            handle.pipe(
                image=resized, prompt="a red panda eating bamboo", negative_prompt=None,
                height=h, width=w, num_frames=args.frames,
                num_inference_steps=args.steps, guidance_scale=pk.guidance_scale,
                generator=gen, output_type="latent", callback_on_step_end=cb,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[probe] pipe raised: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("PROBE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
