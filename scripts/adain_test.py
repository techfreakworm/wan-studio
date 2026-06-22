#!/usr/bin/env python3
"""TRUE FINAL rescale variant: per-channel AdaIN toward the conditional (match BOTH
mean+std). Std-only (global φ0.7=0.880, per-channel φ1.0=0.879) left it fully neon;
AdaIN additionally corrects the per-channel MEAN shift = the green/yellow color CAST.
If @21 EYEBALLS as a coherent natural panda → ship CFG modes with AdaIN rescale.
Else rescale is exhausted → PIVOT. 1.3B @21=32GB, serial-locked.
"""
from __future__ import annotations
import os, sys
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
    if estimate_peak_gb("wan2.1_t2v_1.3b", 21, "480p")["peak_gb"] > HARD_CAP_GB:
        print("REFUSED"); return 4
    with SerialLock():
        from pipelines.t2v import T2VHandle
        from PIL import Image
        h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality")
        pipe = h.pipe
        os.environ["WAN_GUIDANCE_RESCALE"] = "1.0"
        os.environ["WAN_GUIDANCE_RESCALE_PERCHANNEL"] = "1"
        os.environ["WAN_GUIDANCE_RESCALE_ADAIN"] = "1"

        def run(frames, tag):
            g = torch.Generator("cpu").manual_seed(42)
            out = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832,
                       num_frames=frames, num_inference_steps=20, guidance_scale=5.0, generator=g)
            fr = out.frames[0]; a = np.asarray(fr[len(fr) // 2])
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
            Image.fromarray(a).save(REPO / f"tests/outputs/adain_{tag}.png")
            print(f"  [{tag}] AdaIN both-moments frames={frames} sat={sat(a):.3f} bright={a.mean():.1f}", flush=True)
            try: torch.mps.empty_cache()
            except Exception: pass

        print("=== AdaIN (mean+std) per-channel rescale, 1.3B g5.0 @21 (ref control 0.926) ===", flush=True)
        run(21, "21")
        run(13, "13")
        print("\nINTERPRET: ship ⟺ @21 EYEBALLS coherent natural panda (no green cast, gradation back), "
              "@13 still clean. Else rescale exhausted → PIVOT to Lightning-only.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
