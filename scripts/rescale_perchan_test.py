#!/usr/bin/env python3
"""Final rescale variant: PER-CHANNEL std-match at φ=1.0 (targets the per-channel
spread = the oversaturation symptom). Global φ=0.7 was only partial (0.926→0.880,
still neon by eye). If per-channel φ=1.0 drops sat@21 toward ~0.76 AND the frame
is a coherent panda (EYEBALL), the fix works; else PIVOT to Lightning-only scorecard.
1.3B @21 = 32GB, serial-locked.
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

        def run(phi, perchan, frames, tag):
            if phi:
                os.environ["WAN_GUIDANCE_RESCALE"] = str(phi)
                os.environ["WAN_GUIDANCE_RESCALE_PERCHANNEL"] = "1" if perchan else "0"
            else:
                os.environ.pop("WAN_GUIDANCE_RESCALE", None)
            g = torch.Generator("cpu").manual_seed(42)
            out = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832,
                       num_frames=frames, num_inference_steps=20, guidance_scale=5.0, generator=g)
            fr = out.frames[0]; a = np.asarray(fr[len(fr) // 2])
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
            Image.fromarray(a).save(REPO / f"tests/outputs/rescalepc_{tag}.png")
            print(f"  [{tag}] phi={phi} perchan={perchan} frames={frames} sat={sat(a):.3f} bright={a.mean():.1f}", flush=True)
            try: torch.mps.empty_cache()
            except Exception: pass

        print("=== per-channel rescale, 1.3B g5.0 @21 (ref: control 0.926, global-phi07 0.880 still neon) ===", flush=True)
        run(1.0, True, 21, "21_pc_phi10")   # per-channel, full
        run(1.0, True, 13, "13_pc_phi10")   # no-harm on the clean short case
        print("\nINTERPRET: clear ⟺ 21_pc_phi10 sat << 0.88 toward ~0.76 AND coherent panda (EYEBALL), "
              "13 still clean. Else PIVOT — rescale variants exhausted.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
