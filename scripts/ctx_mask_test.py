#!/usr/bin/env python3
"""Confirm the workflow's root-cause hypothesis: un-masked 512-token zero-padded
text context in cross-attention injects a length-dependent magnitude bias.

DECISIVE single-forward test (memory-safe, no generation, no accumulation):
  encode a real prompt → (a) full 512-padded embeds  (b) cropped-to-real-length.
  Run ONE transformer forward at 4 vs 5 latent frames for each, compare output
  std / max-abs.

PREDICTION if hypothesis correct: with PADDED context, output magnitude grows
4→5 latent; with CROPPED context, 4 and 5 match and both are LOWER. That both
confirms the cause AND validates the crop fix.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from memcheck import SerialLock  # noqa: E402

PROMPT = "A red panda eating bamboo on a mossy rock in a sunlit forest, cinematic, photorealistic"


def main() -> int:
    with SerialLock():
        from pipelines.t2v import T2VHandle
        h = T2VHandle.for_key("wan2.1_t2v_1.3b"); h.configure_preset("quality")
        pipe = h.pipe
        tf = pipe.transformer

        # Encode real prompt the way the pipeline does (512 zero-padded).
        emb, _ = pipe.encode_prompt(prompt=PROMPT, do_classifier_free_guidance=False,
                                    num_videos_per_prompt=1, max_sequence_length=512,
                                    device="mps")
        emb = emb.to("mps", torch.bfloat16)
        # Real token count = last non-all-zero row + 1.
        nonzero = (emb[0].abs().sum(-1) > 1e-6)
        real_len = int(nonzero.nonzero().max().item()) + 1 if nonzero.any() else emb.shape[1]
        print(f"context: padded_len={emb.shape[1]} real_len={real_len} "
              f"(padding rows = {emb.shape[1]-real_len})", flush=True)

        emb_padded = emb                       # full 512
        emb_cropped = emb[:, :real_len].contiguous()   # native-style crop

        def fwd(latentT, context):
            torch.manual_seed(0)
            lat = torch.randn(1, 16, latentT, 60, 104, device="mps", dtype=torch.bfloat16)
            t = torch.tensor([900.0], device="mps", dtype=torch.bfloat16)
            with torch.no_grad():
                out = tf(hidden_states=lat, timestep=t, encoder_hidden_states=context,
                         return_dict=False)[0].float()
            r = (round(out.std().item(), 4), round(out.abs().max().item(), 2))
            del lat, out
            try: torch.mps.empty_cache()
            except Exception: pass
            return r

        print("\n  ctx      4-latent(std,max)   5-latent(std,max)   growth 4→5")
        for name, ctx in [("PADDED512", emb_padded), ("CROPPED", emb_cropped)]:
            s4 = fwd(4, ctx); s5 = fwd(5, ctx)
            growth = round(s5[1] / max(s4[1], 1e-6), 3)
            print(f"  {name:9s} {str(s4):>18s}  {str(s5):>18s}   max×{growth}", flush=True)

        print("\nINTERPRET: if PADDED grows 4→5 but CROPPED stays flat+lower → "
              "un-masked padding IS the length-dependent magnitude source (fix = crop/mask).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
