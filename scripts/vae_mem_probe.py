#!/usr/bin/env python3
"""Empirically measure REAL MPS memory for VAE decode at increasing latent-frame
counts, to replace the conservative fitted guess in memcheck with measured data.

SAFETY: steps up ONLY within memcheck-safe bounds, measures torch.mps allocation
after each step, and ABORTS before the next (larger) step if the measured peak is
already within ABORT_MARGIN of the hard cap. Serial-locked. Synthetic latent (no
generation) so it isolates the decode cost.

The diffusers AutoencoderKLWan decodes frame-by-frame with a feat_cache (causal
streaming), so the real peak may be much lower than the naive lf^2 fit — this
probe finds the truth without risking a panic.

Run: python scripts/vae_mem_probe.py   (honors the serial lock + per-step abort)
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from memcheck import SerialLock, free_gb, HARD_CAP_GB  # noqa: E402

ABORT_MARGIN_GB = 15.0           # stop before the NEXT step if within this of the cap
LATENT_STEPS = [4, 6, 9]         # 13/21/33 px frames — all individually memcheck-safe
RES = (480, 832)                 # H, W pixels


def mps_alloc_gb() -> float:
    try:
        return torch.mps.current_allocated_memory() / 1e9
    except Exception:
        return 0.0


def main() -> int:
    with SerialLock():
        from pipelines import shared
        vae = shared.vae().to("mps").eval()
        Hl, Wl = RES[0] // 8, RES[1] // 8   # latent spatial (8x compression)
        zdim = vae.config.z_dim
        print(f"VAE z_dim={zdim} latent_spatial={Hl}x{Wl} | cap={HARD_CAP_GB:.0f}GB free={free_gb():.0f}GB")
        results = []
        for lf in LATENT_STEPS:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            # PER-STEP LIVE-FREE GUARD: estimate this step's peak and refuse if it
            # won't fit current reclaimable RAM + compressible margin. Never gamble.
            from memcheck import estimate_peak_gb
            est = estimate_peak_gb("wan2.1_t2v_1.3b", (lf - 1) * 4 + 1, "480p",
                                   decode_chunk_latent=lf)
            fg = free_gb()
            if est["decode_peak_gb"] > fg + 50:
                print(f"  SKIP latent={lf}: est decode {est['decode_peak_gb']}GB > free {fg:.0f}GB+50. "
                      f"Wait for RAM to recover.", flush=True)
                continue
            base = mps_alloc_gb()
            z = torch.randn(1, zdim, lf, Hl, Wl, device="mps", dtype=torch.float32)
            with torch.no_grad():
                out = vae.decode(z).sample
            peak = mps_alloc_gb()
            px_frames = (lf - 1) * 4 + 1
            decode_gb = peak - base
            print(f"  latent={lf:2d} (≈{px_frames}px) → decode peak alloc ≈ {decode_gb:.1f}GB "
                  f"(total {peak:.1f}GB) out_shape={tuple(out.shape)}", flush=True)
            results.append((lf, round(decode_gb, 1), round(peak, 1)))
            del z, out
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            # Abort guard: if measured total is already close to the cap, do NOT try larger.
            if peak > HARD_CAP_GB - ABORT_MARGIN_GB:
                print(f"  ABORT: total {peak:.1f}GB within {ABORT_MARGIN_GB}GB of cap — stop before larger lf", flush=True)
                break
        print("\nMEASURED:", results)
        # Extrapolate decode(lf) ~ a*lf (or a*lf^p) from the measured points for memcheck.
        if len(results) >= 2:
            (l0, d0, _), (l1, d1, _) = results[0], results[-1]
            if d1 > d0 and l1 > l0:
                p = (torch.log(torch.tensor(d1 / max(d0, 0.1))) / torch.log(torch.tensor(l1 / l0))).item()
                print(f"  fit: decode_gb ≈ {d0:.1f} * (lf/{l0})^{p:.2f}  → @13latent ≈ "
                      f"{d0 * (13 / l0) ** p:.0f}GB")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
