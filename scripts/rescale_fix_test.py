#!/usr/bin/env python3
"""Test guidance_rescale (Lin et al. std-match) as the FIX for CFG-at-length
oversaturation. Uses the WAN_GUIDANCE_RESCALE hook already in pipeline_wan.py.

1.3B, g5.0, 20 steps. Tests @21 (neon regime) control vs φ=0.7, AND @13 (clean
regime) with φ=0.7 to confirm rescale doesn't harm the already-good short case.

STOP CONDITION: sat@21 must drop meaningfully toward ~0.76 AND sat@13 stay clean
AND the frame must be a coherent panda (EYEBALL — saturation alone fooled us before).
Memory-safe (1.3B@21=32GB), serial-locked.
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

        def run(frames, phi, tag):
            if phi:
                os.environ["WAN_GUIDANCE_RESCALE"] = str(phi)
            else:
                os.environ.pop("WAN_GUIDANCE_RESCALE", None)
            g = torch.Generator("cpu").manual_seed(42)
            out = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832,
                       num_frames=frames, num_inference_steps=20, guidance_scale=5.0, generator=g)
            fr = out.frames[0]; a = np.asarray(fr[len(fr) // 2])
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
            Image.fromarray(a).save(REPO / f"tests/outputs/rescalefix_{tag}.png")
            print(f"  [{tag}] frames={frames} phi={phi} sat={sat(a):.3f} bright={a.mean():.1f}", flush=True)
            try: torch.mps.empty_cache()
            except Exception: pass

        print("=== guidance_rescale FIX test, 1.3B g5.0 20 steps ===", flush=True)
        run(21, None, "21_control")   # neon baseline ~0.91
        run(21, 0.7,  "21_phi07")     # the fix
        run(13, 0.7,  "13_phi07")     # must stay clean (no harm to short case)
        print("\nINTERPRET: clear ⟺ 21_phi07 sat << 21_control toward ~0.76, 13_phi07 still clean, "
              "AND both frames coherent pandas (EYEBALL). Then CFG modes ship with rescale@length.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
