#!/usr/bin/env python3
"""Isolate the neon-color bug: encode→decode a real image through the shared
AutoencoderKLWan on MPS vs CPU. If MPS corrupts colors but CPU is clean, the
decode is an MPS op bug (independent of the transformer / its dtype)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def roundtrip(device: str, img: Image.Image, T: int = 9):
    from pipelines import shared
    shared.vae.cache_clear()  # fresh instance per device
    vae = shared.vae().to(device).eval()
    x = torch.from_numpy(np.asarray(img)).float() / 127.5 - 1.0   # H,W,C in [-1,1]
    x = x.permute(2, 0, 1)[None, :, None]                          # B,C,1,H,W
    x = x.repeat(1, 1, T, 1, 1).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        rec = vae.decode(z).sample
    arr = ((rec[0, :, T // 2].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5)
    arr = arr.to("cpu").numpy().astype("uint8")
    finite = bool(np.isfinite(rec.float().cpu().numpy()).all())
    return arr, finite, tuple(z.shape)


def sat(a):
    af = a.astype(np.float32); mx = af.max(-1); mn = af.min(-1)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def main():
    out = REPO / "tests" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    img = Image.open(REPO / "tests/assets/job_i2v/i2v_input.png").convert("RGB").resize((832, 480))
    img.save(out / "vae_input.png")
    print(f"input saturation = {sat(np.asarray(img)):.3f}  (T={T} pixel frames)")
    for dev in ["cpu", "mps"]:
        arr, finite, zshape = roundtrip(dev, img, T=T)
        Image.fromarray(arr).save(out / f"vae_rt_{dev}_T{T}.png")
        print(f"[{dev:3s}] latent {zshape} finite={finite} recon_saturation={sat(arr):.3f} "
              f"-> tests/outputs/vae_rt_{dev}_T{T}.png")


if __name__ == "__main__":
    main()
