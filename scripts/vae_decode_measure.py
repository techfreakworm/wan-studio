#!/usr/bin/env python3
"""Measure REAL VAE decode peak (MPS) at 4/6/9 latent frames, TILING ON vs OFF.

Hypothesis: tiling is ON by default (shared.py), and tiled_decode accumulates ALL
~15 spatial tiles' full-temporal output before blending — heavy. Monolithic _decode
streams frame-by-frame (torch.cat of small RGB frames). So DISABLING tiling may use
FAR LESS memory here. This measurement settles it and recalibrates memcheck.

VAE-only (no transformer resident), so even 9-latent (~95GB if tiled) fits 134GB to
measure. Serial-locked, incremental (stops if a step nears the cap).
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from memcheck import SerialLock, free_gb, HARD_CAP_GB  # noqa: E402

H, W = 60, 104  # latent spatial for 480x832


def mps_gb():
    try: return torch.mps.current_allocated_memory() / 1e9
    except Exception: return 0.0


def main() -> int:
    with SerialLock():
        from pipelines import shared
        for tiling in [True, False]:
            shared.vae.cache_clear()
            vae = shared.vae()
            if tiling:
                vae.enable_tiling(); vae.enable_slicing()
            else:
                vae.disable_tiling(); vae.disable_slicing()
            vae = vae.to("mps").eval()
            print(f"\n=== tiling={'ON' if tiling else 'OFF'} ===", flush=True)
            for lf in [4, 6, 9]:
                try: torch.mps.empty_cache()
                except Exception: pass
                # safety: stop if we're already near cap or last step was huge
                if free_gb() < 40:
                    print(f"  lf={lf}: SKIP (free {free_gb():.0f}GB too low)"); continue
                base = mps_gb()
                z = torch.randn(1, vae.config.z_dim, lf, H, W, device="mps", dtype=torch.float32)
                try:
                    with torch.no_grad():
                        out = vae.decode(z).sample
                    peak = mps_gb()
                    print(f"  lf={lf:2d} (≈{(lf-1)*4+1}px): decode peak ≈ {peak-base:.1f}GB "
                          f"out={tuple(out.shape)}", flush=True)
                    del out
                except RuntimeError as e:
                    print(f"  lf={lf:2d}: FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
                del z
                try: torch.mps.empty_cache()
                except Exception: pass
            del vae
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
