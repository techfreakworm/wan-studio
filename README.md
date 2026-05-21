# Wan Studio

Multi-mode Gradio Studio for the Alibaba Wan video diffusion family (Wan 2.1 + Wan 2.2), targeting HF ZeroGPU (Blackwell) and local MPS (Apple Silicon).

See [`RESEARCH.md`](RESEARCH.md) for the full architecture brief, model inventory, mode deep-dive, ZeroGPU integration recipe, Lightning LoRA strategy, UX wireframes, and dependency matrix.

## Status — Phase 0 (UI shell only)

Sidebar + header + per-mode tabs render. **No actual generation yet.** Wireframes for every screen are in [`wireframes/`](wireframes/) (open `wireframes/index.html` for the gallery).

## Quick start (local MPS)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://localhost:7860> in your browser.

## Layout

```
wan-studio/
├── RESEARCH.md            # full architecture brief (1100+ lines)
├── app.py                 # Gradio entry point
├── requirements.txt       # pinned deps (torch 2.8+, diffusers 0.38+, etc.)
├── pipelines/
│   ├── registry.py        # source-of-truth (gen, mode, checkpoint) catalog
│   ├── shared.py          # text-encoder / VAE / image-encoder shared loaders
│   ├── preset.py          # Fast / Quality preset resolver with graceful fallback
│   └── __init__.py
├── ui/
│   ├── header.py          # generation dropdown + Fast/Quality radio + nav icons
│   ├── sidebar.py         # left mode picker
│   ├── tabs.py            # per-mode panels (T2V, I2V, FLF2V, VACE, S2V, Animate, ...)
│   └── __init__.py
├── utils/
│   ├── backend.py         # device + dtype detection + ZeroGPU awareness
│   └── __init__.py
├── tests/                 # pytest stubs (Phase 1+)
├── assets/                # any static files served at /file
├── raw/                   # research artifacts (per-topic deep dives + reference Spaces)
└── wireframes/            # 8 PNG mockups + index.html gallery
```

## Implementation phases

See [`RESEARCH.md` §11](RESEARCH.md#11-impl-plan) — 9 phases, ~3-4 weeks total, MVP in 7-9 days.

Current: **Phase 0 — Scaffold + UI shell.** Next: **Phase 1 — T2V + I2V on Wan 2.1 14B end-to-end.**
