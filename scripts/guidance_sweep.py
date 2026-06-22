#!/usr/bin/env python3
"""Find the MAX-clean guidance config at length (wan-brain): 1.3B Quality @21,
30 steps, g ∈ {1.0, 2.0, 3.0}. The optimal working config is the HIGHEST guidance
BELOW the neon onset (g1.0 = no CFG = weak adherence + low contrast; higher g =
better adherence but risks neon). Judge by EYEBALL: sharpness AND prompt-adherence
(red panda? bamboo? forest?), not just saturation. 1.3B@21=57GB, serial-locked.
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
        os.environ.pop("WAN_GUIDANCE_RESCALE", None)
        h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality"); pipe = h.pipe
        print("=== guidance sweep, 1.3B @21, 30 steps (find max-clean) ===", flush=True)
        for gscale in (1.0, 2.0, 3.0):
            g = torch.Generator("cpu").manual_seed(42)
            out = pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832, num_frames=21,
                       num_inference_steps=30, guidance_scale=gscale, generator=g)
            fr = out.frames[0]; a = np.asarray(fr[len(fr) // 2])
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.01 else a.astype("uint8")
            Image.fromarray(a).save(REPO / f"tests/outputs/gsweep_g{gscale}.png")
            print(f"  g={gscale} @21x30: sat={sat(a):.3f} bright={a.mean():.1f} -> gsweep_g{gscale}.png", flush=True)
            try: torch.mps.empty_cache()
            except Exception: pass
        print("\nEYEBALL each gsweep_g*.png: max guidance that is SHARP + ADHERENT (panda/bamboo/forest) "
              "+ NOT neon = the working real-video-at-length config.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
