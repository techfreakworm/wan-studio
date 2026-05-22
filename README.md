---
title: Wan Studio
emoji: 🎬
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "5.49.0"
app_file: app.py
pinned: false
short_description: "Every Wan mode, one clean UI."
python_version: "3.12.12"
startup_duration_timeout: "30m"
preload_from_hub:
  - techfreakworm/wan-lightning-loras
# ZeroGPU hardware is set programmatically by scripts/create_space.py
# (SpaceHardware.ZERO_A10G — empirically the live Blackwell ZeroGPU V2 pool
# as of May 2026).
---

# Wan Studio

Multi-mode Gradio Studio for the Alibaba Wan video diffusion family. Phase 1
ships **Text-to-Video** and **Image-to-Video** on **Wan 2.1** (T2V-14B,
I2V-14B 480P/720P) and **Wan 2.2** (T2V-A14B MoE, I2V-A14B MoE) with two
quality presets:

- **Fast (Lightning)** — 4 steps, CFG = 1.0, official Lightning LoRA loaded
- **Quality** — 30-50 steps, full sampler, no LoRA

Pick generation and preset from the header. Modes live in the left sidebar.

Backed by HF ZeroGPU (Blackwell sm_120). Model weights are mounted read-only
from duplicated mirrors in the maintainer's HF account for resilience against
upstream changes.

## Roadmap

| Phase | Modes | Status |
|---|---|---|
| 1 | T2V, I2V | **in progress** |
| 2 | FLF2V, V2V, TI2V-5B | planned |
| 3 | VACE (depth, pose, sketch, inpaint, outpaint, reference, extension) | planned |
| 4 | Animate (character animation + replacement) | planned |
| 5 | S2V (speech-to-video, audio-driven) | planned |
| 6 | Cross-mode chaining + Gallery + Settings polish | planned |

## Architecture

- **Single Space** (this one) — no multi-Space federation.
- **Volume-mounted model weights** at `/models/<slug>` via `huggingface_hub.Volume`.
- **Lightning LoRA mirror** at `/models/wan-lightning-loras` (HIGH + LOW LoRA pairs for Wan 2.2 MoE; single LoRAs for Wan 2.1).
- **Per-mode handle classes** (`T2VHandle`, `I2VHandle`, etc.) lazy-load on first generate. Shared text-encoder / VAE / image-encoder loaded once at module top.
- **MPS-aware**: same codebase runs locally on Apple Silicon for development (fp16 transformer / fp32 VAE / no quant), and on ZeroGPU Blackwell for production (bf16 / optional torchao FP8 + AOTI).

## Attribution

See [NOTICE.md](NOTICE.md) for Apache 2.0 attribution to Wan-AI, lightx2v, diffusers, and Gradio.

## Maintainer

Mayank Gupta · [@techfreakworm](https://huggingface.co/techfreakworm) · [github.com/techfreakworm/wan-studio](https://github.com/techfreakworm/wan-studio)
