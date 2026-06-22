#!/usr/bin/env python3
"""Stopgap test: per-step dynamic-threshold (Imagen-style percentile clamp) on the
latent during sampling, to kill the CFG-at-length oversaturation on 1.3B @21.

If the clamp removes the neon while preserving structure → confirms the radial
runaway (variance over-extrapolation) AND gives CFG modes a working path at length.
Compares: control (no clamp, expect neon ~0.91) vs clamped (expect lower sat + clean).

The clamp: after each step, rescale the latent so its per-channel abs-percentile
stays within the natural range — bounds the variance over-shoot without distorting
structure. Implemented via callback_on_step_end (no pipeline edit).
Memory-safe (1.3B @21 = 32GB), serial-locked.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from memcheck import SerialLock, estimate_peak_gb, HARD_CAP_GB  # noqa: E402

PROMPT = "A red panda eating bamboo on a mossy rock in a sunlit forest, cinematic, photorealistic"


def sat(a):
    af = a.astype(np.float32); mx = af.max(-1); mn = af.min(-1)
    return float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def main() -> int:
    est = estimate_peak_gb("wan2.1_t2v_1.3b", 21, "480p")
    if est["peak_gb"] > HARD_CAP_GB:
        print(f"REFUSED: {est['peak_gb']}GB"); return 4
    with SerialLock():
        from pipelines.t2v import T2VHandle
        from PIL import Image
        h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality")
        pipe = h.pipe

        def run(pctl, tag):
            # pctl=None → control (no clamp). Else per-step dynamic threshold:
            # clamp the latent so its abs-percentile stays at the early-trajectory
            # level (prevents the variance over-shoot that decodes to neon).
            ref = {"s": None}
            def cb(p, step, t, kw):
                if pctl is None:
                    return kw
                lat = kw["latents"]
                s = torch.quantile(lat.abs().flatten().float(), pctl).item()
                if ref["s"] is None:
                    ref["s"] = s  # anchor to first-step percentile
                elif s > ref["s"] * 1.0:
                    lat = lat * (ref["s"] / max(s, 1e-6))
                    kw["latents"] = lat.type_as(kw["latents"])
                return kw
            g = torch.Generator("cpu").manual_seed(42)
            out = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832,
                       num_frames=21, num_inference_steps=20, guidance_scale=5.0,
                       generator=g, callback_on_step_end=cb)
            fr = out.frames[0]; a = np.asarray(fr[len(fr) // 2])
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
            Image.fromarray(a).save(REPO / f"tests/outputs/dynthresh_{tag}.png")
            print(f"  [{tag}] sat={sat(a):.3f} bright={a.mean():.1f}", flush=True)
            try: torch.mps.empty_cache()
            except Exception: pass

        print("=== dynamic-threshold stopgap, 1.3B @21 g5.0 ===", flush=True)
        run(None, "control")          # expect neon ~0.91
        run(0.995, "clamp995")        # 99.5th-percentile anchor
        print("\nINTERPRET: if clampNNN sat << control AND the frame is a coherent panda "
              "(eyeball!) → radial runaway confirmed + working stopgap for CFG modes at length.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
