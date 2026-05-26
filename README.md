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
# Volume mounts for model weights are set programmatically by
# scripts/create_space.py (Volume(type="model", source=..., mount_path=...,
# read_only=True) — read-only, served from the maintainer's HF object
# store, zero cost against the 150 GB ephemeral disk cap).
# ZeroGPU hardware is also requested programmatically — SpaceHardware
# .ZERO_A10G, empirically the live Blackwell ZeroGPU V2 pool as of 2026.
---

# Wan Studio

> Every Alibaba Wan video-diffusion mode in one clean Gradio UI — T2V, I2V, TI2V, FLF2V, V2V, VACE, S2V, Animate — backed by HF ZeroGPU.

<p>
  <a href="https://huggingface.co/spaces/techfreakworm/wan-studio"><img alt="Live on Hugging Face Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Wan%20Studio-FFD21F?style=for-the-badge&labelColor=303030"></a>
  <a href="https://github.com/techfreakworm/wan-studio"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-techfreakworm%2Fwan--studio-181717?style=for-the-badge&logo=github&labelColor=303030"></a>
</p>

<p>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <a href="#"><img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white"></a>
  <a href="#"><img alt="Gradio 5.49" src="https://img.shields.io/badge/gradio-5.49-F97316.svg"></a>
  <a href="#"><img alt="ZeroGPU Blackwell" src="https://img.shields.io/badge/ZeroGPU-Blackwell%20sm__120-76B900.svg?logo=nvidia&logoColor=white"></a>
  <a href="#"><img alt="MPS-friendly" src="https://img.shields.io/badge/MPS-friendly-555555.svg?logo=apple&logoColor=white"></a>
</p>

**🔗 Live demo:** https://huggingface.co/spaces/techfreakworm/wan-studio
*(in active development — please don't run inference; it burns the maintainer's ZeroGPU quota)*

---

## What it is

Wan Studio is a single Gradio app that exposes every officially-supported mode of the [Alibaba Wan](https://github.com/Wan-Video) video-diffusion family — Wan 2.1 + Wan 2.2 — through a refined, Linear-inspired UI. Each mode (T2V, I2V, TI2V, FLF2V, V2V, VACE, S2V, Animate) lives in its own sidebar tab with mode-specific inputs and a shared two-preset model:

- **Fast (Lightning)** — 4 steps, CFG = 1.0, official Lightning LoRA loaded
- **Quality** — 30-50 steps, full sampler, no LoRA

Both Wan generations live in the same UI. The header dropdown picks `Wan 2.1` vs `Wan 2.2` and the active mode resolves to the appropriate checkpoint — single-transformer for Wan 2.1 modes, dual-transformer MoE (`transformer` + `transformer_2` paired Lightning LoRAs) for Wan 2.2 A14B modes.

---

## Roadmap

| Phase | Modes | Status |
|---|---|---|
| 1 | T2V, I2V | 🟡 in progress |
| 2 | FLF2V, V2V, TI2V-5B | planned |
| 3 | VACE (depth, pose, sketch, inpaint, outpaint, reference, extension) | planned |
| 4 | Animate (character animation + replacement) | planned |
| 5 | S2V (speech-to-video, audio-driven) | planned |
| 6 | Cross-mode chaining + Gallery + Settings polish | planned |

---

## Architecture

```
                          ┌──────────────────────────┐
   user → HF Space (UI)   │  app.py  (Gradio 5.49)   │
                          │  ─ Linear-themed chrome  │
                          │  ─ JS-only sidebar nav   │
                          │  ─ @spaces.GPU handlers  │
                          └────────────┬─────────────┘
                                       │
        ┌──────────────────────────────┴─────────────────────────────┐
        │                                                            │
┌───────▼──────────┐                                          ┌──────▼────────────┐
│ pipelines/       │                                          │ ui/               │
│  ─ registry.py   │  ModelCard catalog (12 checkpoints)      │  ─ header.py      │
│  ─ handle.py     │  WanModelHandle base + LRU cache         │  ─ sidebar.py     │
│  ─ t2v.py        │  T2VHandle (Wan 2.1 + Wan 2.2 MoE)       │  ─ tabs/*.py      │
│  ─ i2v.py        │  I2VHandle                               │  ─ build_all_*.py │
│  ─ shared.py     │  UMT5 / VAE / CLIP loaded once           └───────────────────┘
│  ─ preset.py     │  Fast vs Quality kwargs resolver
└───────┬──────────┘
        │
        ▼
┌───────────────────┐     ┌──────────────────────────────────────────┐
│ Volume mounts     │     │ HF mirrors (techfreakworm/wan2.*-*)      │
│ /models/<slug>/   │ ←── │ Apache-2.0 duplicates of Wan-AI repos,   │
│  (read-only,      │     │ pinned for resilience against upstream.  │
│   served from HF) │     │ Lightning LoRAs in wan-lightning-loras.  │
└───────────────────┘     └──────────────────────────────────────────┘
```

Design principles:

- **Single Space**, no multi-Space federation. Everything in one container.
- **Volume-mounted weights** via `huggingface_hub.Volume(type="model", read_only=True)` — backed by the maintainer's HF object store, zero ephemeral disk cost.
- **Bundled metadata stitching** (`models_meta/<slug>/`) — works around HF Volume small-file truncation by shipping correct JSONs in the Space repo and symlinking weights from the mount at startup.
- **One handle per (mode, generation)** lazy-loaded on first click; LRU eviction planned for Phase 2+ when more modes go live.
- **Shared encoders** (`UMT5-XXL`, `AutoencoderKLWan`, `CLIP-ViT-H/14`) loaded once via `pipelines/shared.py` and injected into every pipeline — saves ~25 GB of duplicated weights.
- **MPS-friendly**: identical codebase runs locally on Apple Silicon (`fp16` transformer / `fp32` VAE / no quant) and on ZeroGPU Blackwell (`bf16` / optional torchao FP8 / model CPU offload for MoE).

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Web UI | **Gradio 5.49** | ZeroGPU's only first-class SDK; rich video components |
| Diffusion runtime | **diffusers** (latest) | `WanPipeline` + `WanImageToVideoPipeline` first-party support |
| Acceleration | `@spaces.GPU(duration=…, size="large")` | ZeroGPU on-demand H100/Blackwell |
| Model storage | HF Hub model repos + `space_volumes` mounts | Read-only, free, resilient |
| Quantization (optional) | `torchao` FP8 on Blackwell | Halves MoE memory footprint |
| MoE management | `accelerate.enable_model_cpu_offload()` | Two 14B transformers fit on a 48 GB GPU |

---

## Local development (Apple Silicon)

Tested on M5 Max (128 GB unified memory) with macOS 26 / Python 3.12.12.

```bash
git clone https://github.com/techfreakworm/wan-studio.git
cd wan-studio
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the UI (Wan 2.2 T2V default — model auto-downloads on first Generate)
WAN_STUDIO_PORT=7863 python3.12 app.py
# → http://localhost:7863

# Smoke build + tests
python3.12 -c "from app import build; build()"
pytest tests/ -q
```

On MPS the pipelines use `fp16` for transformers and `fp32` for the VAE. No quantization is applied — the M-series GPU doesn't have FP8 tensor cores and quantized kernels are CUDA-specific. The default Wan 2.2 T2V-A14B MoE needs ~56 GB CPU RAM + ~10 GB GPU memory headroom; well within the M5 Max budget.

To force a smaller model for faster local iteration:

```bash
WAN_STUDIO_T2V_LOCAL_KEY=wan2.1_t2v_1.3b WAN_STUDIO_PORT=7863 python3.12 app.py
```

---

## HF Space deployment

Reproducing the Space from scratch:

```bash
# 1. Duplicate the upstream Wan-AI repos into your account + build the LoRA mirror
python3.12 scripts/duplicate_upstream.py --dry-run   # preview
python3.12 scripts/duplicate_upstream.py             # execute (~5 min server-side copy + ~2 min LoRA upload)

# 2. Create the Space + set volume mounts + request ZeroGPU hardware
python3.12 scripts/create_space.py

# 3. Upload code
hf upload <your-username>/wan-studio . --repo-type=space
```

Required HF account capabilities:
- **PRO subscription** (for ZeroGPU `large` 48 GB slice + 10 TB public model storage)
- ~300 GB of model storage will be used by the duplicated Wan-AI mirrors (free under PRO)

Phase 1 currently targets `SpaceHardware.ZERO_A10G`, which empirically resolves to the live Blackwell sm_120 pool (confirmed via a probe app — the "Nvidia H200" badge in the Spaces UI is stale marketing text).

---

## Project layout

```
wan-studio/
├── app.py                          # Gradio entry point + @spaces.GPU handlers
├── pipelines/
│   ├── registry.py                 # ModelCard catalog (12 checkpoints across Wan 2.1/2.2)
│   ├── handle.py                   # WanModelHandle base, mount stitching, LoRA loader
│   ├── t2v.py                      # T2VHandle (Wan 2.1 + Wan 2.2 MoE)
│   ├── i2v.py                      # I2VHandle
│   ├── shared.py                   # Shared text encoder / VAE / image encoder
│   └── preset.py                   # Fast vs Quality preset resolver
├── ui/
│   ├── header.py                   # Brand mark + Generation/Preset chrome
│   ├── sidebar.py                  # 10-mode left rail
│   └── tabs/                       # Per-mode input + output panels
├── utils/
│   ├── backend.py                  # Backend.detect() — MPS vs CUDA vs ZeroGPU
│   └── budget.py                   # ZeroGPU duration callable + size tier per mode
├── models_meta/<slug>/             # Bundled small JSONs (configs, tokenizer)
│   └── wan2.2-t2v-a14b/
├── scripts/
│   ├── duplicate_upstream.py       # Mirror Wan-AI repos into the maintainer's account
│   └── create_space.py             # Programmatic Space configuration
├── tests/                          # 36 unit tests (backend, budget, handle, preset, registry)
├── docs/superpowers/specs/         # Design specs
├── docs/superpowers/plans/         # Implementation plans
└── NOTICE.md                       # Apache 2.0 attribution
```

---

## Acknowledgments

Wan Studio packages and exposes models trained by the [Alibaba Wan-Video team](https://github.com/Wan-Video) under the Apache 2.0 license. Lightning LoRAs are courtesy of [lightx2v](https://github.com/ModelTC/lightx2v) and the [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) community mirror. Built on [diffusers](https://github.com/huggingface/diffusers), [Gradio](https://github.com/gradio-app/gradio), [HF Spaces](https://huggingface.co/spaces), and [HF Volumes](https://huggingface.co/docs/hub/spaces-config-reference). Full attribution in [NOTICE.md](NOTICE.md).

---

## License

[Apache License 2.0](LICENSE) — same as Wan-AI's upstream model releases.

---

## Maintainer

**Mayank Gupta**
🤗 [@techfreakworm on Hugging Face](https://huggingface.co/techfreakworm) · 💻 [@techfreakworm on GitHub](https://github.com/techfreakworm) · 🌐 [mayankgupta.in](https://mayankgupta.in)

Phase 1 in progress. Issues and PRs welcome on [github.com/techfreakworm/wan-studio](https://github.com/techfreakworm/wan-studio).
