#!/usr/bin/env python3
"""Verify the @33 VAE-decode fix: generate the latent, FREE the transformer (del +
gc + empty_cache — NOT .to('cpu'), which doesn't free unified memory), THEN decode.

The 14B@33 OOM was decode-peak + 40GB resident transformer > 134GB. The decode
ALONE completes (measured). So freeing the transformer before decode unblocks @33.

This is 14B Lightning (CFG-free, no neon) @33 = ~2s real video. If it completes and
the frames are a coherent panda, @33 is unblocked for the working path.
Memory: generation peak ~71GB (14B + activations), decode peak after free ~<94GB. Serial.
"""
from __future__ import annotations
import gc, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from memcheck import SerialLock, free_gb  # noqa: E402

PROMPT = "A red panda eating bamboo on a mossy rock in a sunlit forest, cinematic, photorealistic"
OUT = REPO / "tests" / "outputs"


def sat(a):
    af = a.astype(np.float32); mx = af.max(-1); mn = af.min(-1)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def main() -> int:
    FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    with SerialLock():
        from pipelines.t2v import T2VHandle
        from PIL import Image
        from diffusers.utils import export_to_video
        h = T2VHandle.for_key("wan2.1_t2v_14b")
        pk = h.configure_preset("fast")  # Lightning, g1.0
        pipe = h.pipe
        print(f"preset={pk.effective_preset} lora={pk.lora_active} steps={pk.num_inference_steps} "
              f"| free={free_gb():.0f}GB", flush=True)

        # 1) Generate LATENT only (no internal decode → transformer not needed after).
        g = torch.Generator("cpu").manual_seed(42)
        def cb(p, i, t, kw):
            print(f"  step {i+1}/{pk.num_inference_steps}", flush=True); return kw
        lat = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832, num_frames=FRAMES,
                   num_inference_steps=pk.num_inference_steps, guidance_scale=pk.guidance_scale,
                   generator=g, output_type="latent", callback_on_step_end=cb).frames
        print(f"  latent {tuple(lat.shape)} | free-before-free={free_gb():.0f}GB", flush=True)

        # 2) FREE the transformer (TRUE delete — .to('cpu') keeps it in the unified pool).
        vae = pipe.vae
        pipe.transformer = None
        h.pipe = None
        del pipe, h
        gc.collect()
        try: torch.mps.empty_cache()
        except Exception: pass
        print(f"  transformer freed | free-after-free={free_gb():.0f}GB", flush=True)

        # 3) Decode the latent (now has the full pool).
        lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to("mps", torch.float32)
        ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to("mps", torch.float32)
        z = lat.to("mps", torch.float32) / ls + lm
        with torch.no_grad():
            video = vae.decode(z, return_dict=False)[0]
        # postprocess [B,C,T,H,W] in [-1,1] → uint8 frames
        v = ((video[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to("cpu").numpy().astype("uint8")
        frames = [v[i] for i in range(v.shape[0])]
        print(f"  decoded {len(frames)} frames | free={free_gb():.0f}GB", flush=True)

        # 4) Verify
        mp4 = OUT / "freed_decode_14b_fastFR.mp4"
        export_to_video(frames, str(mp4), fps=16)
        for idx, nm in [(0, "start"), (len(frames)//2, "mid"), (len(frames)-1, "end")]:
            Image.fromarray(frames[idx]).save(OUT / f"freed_decode_14b_fastFR_{nm}.png")
        sats = [round(sat(frames[i]), 3) for i in (0, len(frames)//2, len(frames)-1)]
        mot = float(np.abs(frames[1].astype(np.float32).mean(-1) - frames[0].astype(np.float32).mean(-1)).mean())
        print(f"\n14B-LIGHTNING @33 (freed-decode): sat[s,m,e]={sats} motion≈{mot:.2f} "
              f"frames={len(frames)} → {mp4}", flush=True)
        print("EYEBALL freed_decode_14b_fastFR_mid.png — coherent panda = @33 UNBLOCKED.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
