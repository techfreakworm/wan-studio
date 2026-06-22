#!/usr/bin/env python3
"""Per-step CFG-trajectory probe: log latent magnitude after EACH sampling step
for @13 (clean) vs @21 (neon), 1.3B, g5.0. Localizes WHERE the multi-step blowup
happens. Scalar logging only (no tensor accumulation) — memory-safe, serial-locked.

If @21 latent-max climbs over steps while @13 stays bounded → the blowup is in the
sampling trajectory (sampler × CFG × length), and the divergence step is the clue.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from memcheck import SerialLock, estimate_peak_gb, HARD_CAP_GB  # noqa: E402

PROMPT = "A red panda eating bamboo on a mossy rock in a sunlit forest, cinematic, photorealistic"
STEPS = 20


def main() -> int:
    est = estimate_peak_gb("wan2.1_t2v_1.3b", 21, "480p")
    if est["peak_gb"] > HARD_CAP_GB:
        print(f"REFUSED: {est['peak_gb']}GB > cap"); return 4
    with SerialLock():
        from pipelines.t2v import T2VHandle
        h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality")
        pipe = h.pipe

        def run(frames):
            traj = []
            def cb(p, step, t, kw):
                lat = kw["latents"]
                traj.append((step, round(lat.abs().max().item(), 2), round(lat.std().item(), 3)))
                return kw
            g = torch.Generator("cpu").manual_seed(42)
            pipe(prompt=PROMPT, negative_prompt=None, height=480, width=832,
                 num_frames=frames, num_inference_steps=STEPS, guidance_scale=5.0,
                 generator=g, callback_on_step_end=cb)
            return traj

        print(f"=== per-step latent (max, std), g5.0, {STEPS} steps ===", flush=True)
        for frames, tag in [(13, "CLEAN@13/4lat"), (21, "NEON@21/6lat")]:
            tr = run(frames)
            print(f"\n{tag}:", flush=True)
            for step, mx, sd in tr:
                print(f"  step {step:2d}: max={mx:8.2f}  std={sd:7.3f}", flush=True)
            try: torch.mps.empty_cache()
            except Exception: pass
        print("\nINTERPRET: compare the max-trajectories. If NEON@21 max diverges upward "
              "at some step while CLEAN@13 stays bounded → that step localizes the blowup.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
