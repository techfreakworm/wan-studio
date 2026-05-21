# Wan Studio — Design Spec

| Field | Value |
|---|---|
| **Date** | 2026-05-21 |
| **Author** | Mayank Gupta |
| **Status** | Approved (post-brainstorm) — ready for implementation plan |
| **Companion docs** | [`RESEARCH.md`](../../../RESEARCH.md) (full architecture brief, 1182 lines) · [`wireframes/index.html`](../../../wireframes/index.html) (8 PNG mockups) · [`raw/`](../../../raw/) (5 per-topic deep-dives) |
| **Brainstorm session** | `.superpowers/brainstorm/31161-1779359863/` |
| **Spec location** | `docs/superpowers/specs/2026-05-21-wan-studio-design.md` |

---

## 1. Summary

Wan Studio is a multi-mode Gradio web application that exposes every officially-supported mode of the Alibaba Wan video diffusion model family (Wan 2.1 + Wan 2.2) through a single clean UI, deployed as a public Hugging Face ZeroGPU Space on the PRO tier (Blackwell hardware).

It targets video-generation users in the Wan community who today have to either run Wan2GP locally (heavy install, broad-but-shallow on Wan, MPS-unfriendly) or hop between single-mode Spaces (fragmented UX, no chaining). The differentiator is **comprehensive Wan coverage in one clean URL** — every mode as a first-class tab, Lightning Fast / Quality preset toggle, cross-mode "Send to" chaining.

---

## 2. Goals

- **G1** — Expose all 7 official Wan modes as first-class tabs: T2V, I2V, FLF2V, V2V, VACE, S2V, Animate (plus TI2V-5B as an additional checkpoint within T2V/I2V).
- **G2** — Cover both open generations (Wan 2.1 and Wan 2.2) via an explicit generation dropdown in the header.
- **G3** — Two-preset model: **Fast** (Lightning LoRA, 4 steps, CFG=1) and **Quality** (no LoRA, 30+ steps, CFG=5). Graceful fallback to Quality for modes without Lightning LoRA coverage (VACE, S2V, Animate, TI2V-5B, T2V-1.3B).
- **G4** — Deploy as a single public ZeroGPU Space on the user's HF PRO account. No multi-Space federation.
- **G5** — Be resilient to upstream changes: own duplicates of all Wan-AI / Kijai / lightx2v repos so upstream deletions or breaking commits don't break the Studio.
- **G6** — Run locally on MPS (M5 Max 128 GB) for dev iteration, primarily against the 1.3B variants.
- **G7** — Cross-mode chaining: "Send to" chips on every output let the user feed a generated video into another mode (T2V → I2V → VACE → Animate) without re-uploading.
- **G8** — Per-mode session gallery + a session-scoped history; a Settings page surfacing model load status and cache controls.

## 3. Non-goals

- **NG1** — Multi-architecture support (Hunyuan, LTX, Flux, Qwen, etc.). Wan-only. Wan2GP already covers that breadth; we're betting on Wan-specific depth.
- **NG2** — Quantization (FP8 / Int8 / GGUF). Memory routing handles Wan 2.2 MoE by escalating to ZeroGPU `xlarge` instead. AOTI compilation is optional, deferred to a post-launch performance pass.
- **NG3** — Cross-session persistence. Gallery is session-scoped. No HF Storage Buckets at launch; can be added later if users ask.
- **NG4** — User authentication / per-user state. Public Space, anonymous traffic, no login.
- **NG5** — Wan 2.5 / 2.6 / 2.7 support. Those are API-only as of May 2026 — no open weights on the Wan-AI HF org. Forward-looking enum slot exists in `pipelines/registry.py` but no implementation.
- **NG6** — Real-time / streaming inference at launch. Use `gr.Video(streaming=True)` only as a post-launch optimization for modes that support it.
- **NG7** — Custom LoRA upload by users. Lightning LoRAs are baked in. Style LoRA support is a future enhancement.
- **NG8** — Heavy preprocessing UX (e.g. drawing masks for VACE). For v1, VACE accepts pre-extracted control videos OR runs a minimal on-server preproc bundle (DWPose + MiDaS + RAFT). The full SAM2 / GroundingDINO / InsightFace stack is out of scope for v1.

---

## 4. Audience

| Persona | Need | What they do on the Space |
|---|---|---|
| **Wan-curious creator** | Wants to try Wan modes without setting up Wan2GP locally | Lands on T2V, types a prompt, hits Generate, watches it work in <30s with Lightning Fast preset |
| **Wan researcher / developer** | Comparing modes, exploring what each one does | Cycles through tabs, uses Quality preset for accurate outputs, reads the per-mode info banners |
| **Storyboarder** | Multi-step narrative video creation | Generates T2V → Send to VACE for control → Send to Animate for character motion → checks Gallery |
| **Lightning-LoRA-curious** | Wants to see Wan 2.2 MoE V2.0 Lightning in action | Toggles Fast preset on T2V-A14B, observes <30s gen time |
| **Mobile visitor** | Quick demo from phone (Twitter / LinkedIn click-through) | Sidebar collapses, single-column layout, can still generate T2V end-to-end |

---

## 5. Locked decisions (from brainstorming)

| # | Decision | Choice | Source |
|---|---|---|---|
| D1 | Build purpose | Public HF Space contribution | Q1 |
| D2 | Positioning | Clean comprehensive Wan UI (focus only on Wan, do UX better than Wan2GP) | Q2 |
| D3 | MVP scope | Full slate at launch (all 7 modes) | Q3 |
| D4 | Generation selector UX | Explicit dropdown in header (Wan 2.1 / Wan 2.2) | Q4 |
| D5 | Brand | "Wan Studio" + indigo accent + utility/descriptive tone | Q5 |
| D6 | Architecture | Single monolithic Space (Approach 1) | post-Q5 |
| D7 | Weight storage strategy | Hybrid: `preload_from_hub` (README YAML) for ~20 GB shared components + `space_volumes` Python API mounts for ~280 GB of per-mode transformers | post-150GB discussion |
| D8 | Upstream repo strategy | Duplicate Wan-AI / Kijai / lightx2v repos to user's own account for resilience; mount duplicates not upstream | user-driven |
| D9 | Memory routing | Small modes → ZeroGPU `large` (48 GB) bf16; Wan 2.2 MoE A14B + Wan 2.2 Animate → `xlarge` (96 GB) bf16; no quantization | RESEARCH §6.3 |
| D10 | Local dev backend | MPS (fp16 transformer / fp32 VAE / no quant) for 1.3B variants and shell iteration | RESEARCH §7 |
| D11 | Theme | Gradio default theme with `primary_hue="indigo"`, dark mode default | RESEARCH §9.8 |

---

## 6. Architecture

### 6.1 High-level

```
                        ┌─────────────────────────────────────────────┐
   user browser ───────►│  mayankgupta/wan-studio (public ZeroGPU)    │
                        │                                             │
                        │  ┌────────┐  ┌──────────────────────────┐  │
                        │  │ Gradio │──│ pipelines/{mode}.py       │  │
                        │  │ Blocks │  │   ├─ WanModelHandle       │  │
                        │  │ UI     │  │   ├─ shared loaders       │  │
                        │  │        │  │   └─ preset.resolve()     │  │
                        │  └────────┘  └──────────┬───────────────┘  │
                        │                         │                    │
                        │  ┌──────────────────────┴──────────────────┐ │
                        │  │ Read-only filesystem mounts             │ │
                        │  │ /models/wan2.1-t2v-1.3b                 │ │
                        │  │ /models/wan2.1-t2v-14b                  │ │
                        │  │ /models/wan2.2-t2v-a14b                 │ │
                        │  │ /models/wan2.2-i2v-a14b                 │ │
                        │  │ ... 8 more                              │ │
                        │  └──────────────────────┬──────────────────┘ │
                        │                         │                    │
                        │  ┌──────────────────────┴──────────────────┐ │
                        │  │ Ephemeral disk (~5 GB used)             │ │
                        │  │  ~/.cache/huggingface/hub/              │ │
                        │  │   ├─ mayankgupta/umt5-xxl (~11 GB)      │ │
                        │  │   ├─ mayankgupta/wan-vae (~3 GB)        │ │
                        │  │   ├─ mayankgupta/clip-vit-h (~1 GB)     │ │
                        │  │   └─ mayankgupta/wan-lightning-loras    │ │
                        │  │      (~5 GB)                             │ │
                        │  └─────────────────────────────────────────┘ │
                        └─────────────────────────────────────────────┘
                                              │
                                              ▼ (volumes mounted from)
                        ┌─────────────────────────────────────────────┐
                        │  User's HF account — duplicate repos        │
                        │  mayankgupta/wan2.1-t2v-1.3b                │
                        │  mayankgupta/wan2.1-t2v-14b                 │
                        │  mayankgupta/wan2.2-t2v-a14b                │
                        │  ... + ~9 more (~300 GB of /10 TB quota)    │
                        └─────────────────────────────────────────────┘
                                              ▲
                                              │ (one-shot duplicate script)
                        ┌─────────────────────────────────────────────┐
                        │  Upstream                                   │
                        │  Wan-AI/Wan2.1-T2V-14B-Diffusers            │
                        │  Wan-AI/Wan2.2-T2V-A14B-Diffusers           │
                        │  Kijai/WanVideo_comfy                       │
                        │  lightx2v/Wan2.2-Lightning                  │
                        └─────────────────────────────────────────────┘
```

### 6.2 Why single Space (not multi-Space federation)

Considered Approach 2 (one Space per mode) and Approach 3 (UI Space + worker Spaces). Both were rejected during brainstorming:
- Multi-Space fragments the UI (Send-to chips become URL navigation, brand dilution)
- Multi-Space consumes 8-10 of the 10 PRO Space slots
- Hybrid worker pattern is overengineered for v1; can be evolved into later if cold-start or quota concentration becomes a real bottleneck

### 6.3 Why hybrid preload + volume mount (not pure preload, not pure mount)

| Mechanism | Size cap | Speed at first call | Used for |
|---|---|---|---|
| `preload_from_hub` (README YAML) | Bound by ephemeral disk (~50 GB documented, possibly higher on PRO ZeroGPU) | Already-warm cache hit | The ~20 GB of components every mode shares: text encoder, VAE, image encoder, all Lightning LoRAs |
| `space_volumes` (Python API) | No documented size cap | Filesystem mount — `from_pretrained()` reads instantly without download | The ~280 GB of mode-specific transformer weights |

Pure `preload_from_hub` won't fit ~300 GB. Pure volume mounts work but the shared components also get filesystem-mounted (slightly slower than ephemeral cache hits). Hybrid is the cleanest split.

### 6.4 Why duplicate upstream (not mount source)

Mounting `Wan-AI/Wan2.1-T2V-14B-Diffusers` directly works but creates a dependency on Wan-AI's continued availability and stability. Duplicating to `mayankgupta/wan2.1-t2v-14b`:

- Storage cost: ~300 GB out of 10 TB PRO quota = 3% of capacity
- License-compliant (Wan-AI is Apache 2.0; add `NOTICE.md` pointing at upstream)
- Studio's `space_volumes` config doesn't change when Wan-AI ships a new version — we promote on our schedule
- Resilient to repo renames, deletions, breaking config changes upstream
- One-time duplication script: `api.duplicate_repo(from_id=..., to_id=...)` for each of ~14 upstream repos

### 6.5 Module layout

```
wan-studio/
├── README.md                       # Public Space landing page + preload_from_hub YAML frontmatter
├── NOTICE.md                       # Apache 2.0 attribution to Wan-AI, Kijai, lightx2v
├── requirements.txt                # torch≥2.8, diffusers≥0.38, transformers≥4.45, spaces≥0.50.2, gradio≥5.4
├── app.py                          # Gradio entry point — builds Blocks, wires routes
├── pipelines/
│   ├── __init__.py
│   ├── registry.py                 # ModelCard catalog (gen × mode × checkpoint × Lightning availability)
│   ├── shared.py                   # text_encoder + vae + image_encoder loaders (functools.lru_cache)
│   ├── preset.py                   # Fast / Quality resolver with graceful fallback
│   ├── handle.py                   # WanModelHandle — one per (mode, gen, size); lazy pipeline build
│   ├── t2v.py                      # WanPipeline wrapper
│   ├── i2v.py                      # WanImageToVideoPipeline wrapper
│   ├── flf2v.py                    # WanImageToVideoPipeline + last_image= wrapper
│   ├── v2v.py                      # WanVideoToVideoPipeline wrapper
│   ├── vace.py                     # WanVACEPipeline wrapper + DWPose/MiDaS/RAFT preproc
│   ├── animate.py                  # WanAnimatePipeline wrapper + ViTPose/YOLOv10/SAM2 preproc
│   ├── s2v.py                      # vendored wan.WanS2V wrapper (upstream package)
│   └── ti2v.py                     # vendored wan.WanTI2V wrapper (upstream package)
├── ui/
│   ├── __init__.py
│   ├── header.py                   # Wan Studio name + Generation dropdown + Preset radio + History/Settings
│   ├── sidebar.py                  # Left mode picker
│   └── tabs.py                     # Per-mode panels — Inputs col + Output col + Send-to chips
├── utils/
│   ├── __init__.py
│   ├── backend.py                  # Device + dtype + ZeroGPU awareness; spaces_gpu_or_noop()
│   ├── budget.py                   # Per-mode get_duration() callable for @spaces.GPU
│   └── gallery.py                  # Session-scoped video gallery (gr.State + /tmp sidecar)
├── scripts/
│   ├── duplicate_upstream.py       # One-shot: api.duplicate_repo for each upstream repo
│   └── create_space.py             # api.create_repo with space_volumes manifest
├── tests/
│   ├── test_registry.py            # Registry consistency (no orphan keys, no missing LoRAs)
│   ├── test_preset.py              # Fast/Quality fallback rules
│   ├── test_backend.py             # Device detection + dtype selection
│   └── test_smoke_t2v.py           # Smoke test on Wan 2.1 T2V-1.3B locally on MPS
├── assets/                          # Static files (favicon, examples thumbnails)
├── docs/superpowers/specs/         # This spec + future specs
├── raw/                             # Per-topic research (do not edit; reference only)
└── wireframes/                      # 8 PNG mockups + index.html gallery
```

---

## 7. Components

### 7.1 `pipelines/registry.py` — model + mode catalog

Single source of truth. One `ModelCard` per (generation, mode, size) tuple. Fields include: HF repo path (mirror name after duplication), parameter count, modality, native resolutions, frames default, FPS, Diffusers pipeline class (or `None` for vendored upstream), MoE flag, Lightning availability + HIGH/LOW LoRA file paths, ZeroGPU duration + size. Already drafted in the Phase-0 scaffold.

### 7.2 `pipelines/shared.py` — shared component loaders

Three `@functools.lru_cache(maxsize=1)` functions: `text_encoder()`, `vae()`, `image_encoder()`. Lazy-imported diffusers/transformers inside the function to avoid 3-5s `import` cost at module load. Returns objects in their respective dtypes (UMT5 in `Backend.dtype`, VAE always fp32, CLIP always fp32). VAE has `enable_tiling()` + `enable_slicing()` toggled on.

### 7.3 `pipelines/handle.py` — WanModelHandle

One `WanModelHandle` per `(mode, generation, size)`. Encapsulates:
- `ensure_loaded()` — lazy build of the mode-appropriate pipeline class (`WanPipeline` for T2V, `WanImageToVideoPipeline` for I2V/FLF2V, `WanVACEPipeline` for VACE, `WanVideoToVideoPipeline` for V2V, `WanAnimatePipeline` for Animate, vendored `wan.WanS2V` / `wan.WanTI2V` for S2V/TI2V-5B) from the mounted path; loads Lightning LoRA if available; never re-runs once built
- `configure_preset(preset)` — toggles LoRA via `set_adapters(...)` or `disable_lora()`; returns the kwargs dict to splat into `pipe(...)`
- `generate(**kwargs)` — the `@spaces.GPU(duration=get_duration, size=...)`-decorated call site
- `unload_to_cpu()` — when the user switches modes, the active handle's transformer moves to CPU + `torch.cuda.empty_cache()`

A module-level dict `ACTIVE_HANDLE: dict[str, WanModelHandle | None]` holds at-most-one warmed handle per Space process.

### 7.4 `pipelines/preset.py` — preset resolver

`resolve(card: ModelCard, requested: Preset) -> PresetKwargs`. If user picks Fast but the card has `lightning_available=False`, falls back to Quality with a `fallback_message` populated. Caller surfaces the message via `gr.Info(...)`.

### 7.5 `ui/header.py`, `ui/sidebar.py`, `ui/tabs.py`

Gradio Blocks composition. Each tab function returns a `dict` of component handles for downstream wiring. `tabs.build_all_tabs()` returns `{mode_key: {tab, inputs, outputs}}`. The sidebar mode buttons wire `.click` handlers that flip `gr.update(visible=...)` on each tab's outer column.

### 7.6 `utils/backend.py` — Backend detection

Returns a frozen `Backend` dataclass with `device`, `dtype`, `vae_dtype`, `is_zerogpu`, `zerogpu_size`, `supports_quant`, `supports_aoti`, `supports_flash_attn_3`. `spaces_gpu_or_noop()` returns the real `spaces.GPU` decorator on ZeroGPU and a no-op on MPS.

### 7.7 `utils/budget.py` — duration callable factory

Per RESEARCH §8.3. `get_duration(mode_key, **gen_kwargs)` returns the ZeroGPU duration budget based on requested resolution + duration + preset. Same callable is referenced by both the `@spaces.GPU(duration=callable, size=...)` decorator AND the ETA `gr.Markdown` element in the UI — display + reservation stay in sync.

### 7.8 `utils/gallery.py` — session gallery

Stores `(video_path, mode, prompt, params)` tuples in `gr.State`. Maxsize 24 per session. On select, repopulates the active mode's inputs. No cross-session persistence (Space sandbox blows away `/tmp`).

### 7.9 `scripts/duplicate_upstream.py`

One-shot script run **once before Space deploy**. For each of:
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`
- `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`
- `Wan-AI/Wan2.1-VACE-1.3B-diffusers`
- `Wan-AI/Wan2.1-VACE-14B-diffusers`
- `Wan-AI/Wan2.2-TI2V-5B`
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- `Wan-AI/Wan2.2-S2V-14B`
- `Wan-AI/Wan2.2-Animate-14B-Diffusers`
- `Kijai/WanVideo_comfy` (selective: only Lightx2v + Wan22-Lightning sub-paths to keep mirror small)
- `lightx2v/Wan2.2-Lightning` (full mirror — primary Lightning source)

Calls `api.duplicate_repo(from_id=upstream, to_id=f"mayankgupta/{slug}")`. Logs each duplication. Idempotent (skip if dest exists with same commit SHA).

### 7.10 `scripts/create_space.py`

Programmatic Space creation:

```python
api.create_repo(
    repo_id="mayankgupta/wan-studio",
    repo_type="space",
    space_sdk="gradio",
    # NOTE: ZeroGPU (Blackwell) hardware enum string is not documented in the public
    # `SpaceHardware` reference as of May 2026. Configure ZeroGPU hardware via the
    # web UI post-creation, or via `api.request_space_hardware(...)` once we confirm
    # the correct enum value (suspected `SpaceHardware.ZERO_A10G` legacy alias for
    # the new Blackwell pool — verify empirically before scripting).
    space_volumes=[
        Volume(type="model", source=f"mayankgupta/{slug}", mount_path=f"/models/{slug}", read_only=True)
        for slug in WAN_MIRROR_SLUGS
    ],
)
```

---

## 8. Data flow — a single T2V generation request

```
User in browser
   │
   │ types prompt + clicks Generate (Fast preset, Wan 2.2 generation selected)
   ▼
Gradio websocket → app.py @spaces.GPU(duration=get_duration, size="xlarge") handler
   │
   │ 1. resolve(card, "fast") → PresetKwargs(steps=4, cfg=1.0, cfg_2=1.0, lora_active=True)
   │ 2. handle = ACTIVE_HANDLE.get("t2v_2.2_a14b") or WanModelHandle(...)
   │ 3. handle.ensure_loaded():
   │      pipe = WanPipeline.from_pretrained(
   │          "/models/wan2.2-t2v-a14b",  ← read-only volume mount
   │          vae=shared.vae(),
   │          text_encoder=shared.text_encoder(),
   │          torch_dtype=Backend.dtype,
   │      )
   │      pipe.transformer_2 = WanTransformer3DModel.from_pretrained(
   │          "/models/wan2.2-t2v-a14b", subfolder="transformer_2", ...)
   │      pipe.to("cuda")
   │      pipe.load_lora_weights("/models/wan-lightning-loras",
   │          weight_name=".../HIGH_..._rank128.safetensors",
   │          adapter_name="lightning_high")
   │      pipe.load_lora_weights("/models/wan-lightning-loras",
   │          weight_name=".../LOW_..._rank64.safetensors",
   │          adapter_name="lightning_low",
   │          load_into_transformer_2=True)
   │ 4. handle.configure_preset("fast"):
   │      pipe.set_adapters(["lightning_high","lightning_low"], [1.0, 1.0])
   │ 5. frames = pipe(prompt=..., num_inference_steps=4, guidance_scale=1.0,
   │                  guidance_scale_2=1.0, num_frames=81).frames[0]
   │ 6. video_path = export_to_video(frames, fps=16)
   │ 7. append (video_path, mode, prompt, params) to gr.State gallery
   ▼
Return video_path to gr.Video(autoplay=True, loop=True)
   │
   ▼
Browser plays video; "Send to: I2V VACE Animate" chips become enabled
```

---

## 9. Per-mode specifications

### 9.1 T2V (Wan 2.1 1.3B / 14B, Wan 2.2 A14B MoE)

| Field | Wan 2.1 1.3B | Wan 2.1 14B | Wan 2.2 A14B (MoE) |
|---|---|---|---|
| Mount path | `/models/wan2.1-t2v-1.3b` | `/models/wan2.1-t2v-14b` | `/models/wan2.2-t2v-a14b` |
| Pipeline class | `WanPipeline` | `WanPipeline` | `WanPipeline` (with `transformer_2`) |
| Default rez × frames × fps | 832×480 × 81 × 16 | 1280×720 × 81 × 16 | 1280×720 × 81 × 24 |
| Fast preset | ❌ no LoRA → fall back to Quality | ✅ `lightx2v_T2V_14B_cfg_step_distill_v2_rank128` | ✅ `Wan22-Lightning V2.0` HIGH+LOW pair |
| Quality steps / CFG | 50 / 5.0 | 50 / 5.0 | 40 / 3.0+4.0 |
| ZeroGPU size | `large` | `large` | `xlarge` |
| ZeroGPU duration | 60 | 90 | 120 |

### 9.2 I2V (Wan 2.1 14B 480P/720P, Wan 2.2 A14B MoE)

| Field | Wan 2.1 480P | Wan 2.1 720P | Wan 2.2 A14B |
|---|---|---|---|
| Mount path | `/models/wan2.1-i2v-14b-480p` | `/models/wan2.1-i2v-14b-720p` | `/models/wan2.2-i2v-a14b` |
| Pipeline class | `WanImageToVideoPipeline` | `WanImageToVideoPipeline` | `WanImageToVideoPipeline` (with `transformer_2`) |
| Image encoder | shared CLIPVisionModel fp32 | same | same |
| Fast preset | ✅ Lightx2v I2V LoRA | same | ⚠️ Only V1 Seko; hybrid trick (use Wan 2.1 LoRA on Wan 2.2) exposed as Advanced toggle |
| Quality steps / CFG | 40 / 5.0 | 40 / 5.0 | 40 / 3.5+3.5 |
| flow_shift | 3.0 | 5.0 | 8.0 |
| ZeroGPU size / duration | `large` / 90 | `large` / 120 | `xlarge` / 150 |

### 9.3 FLF2V (Wan 2.1 only)

Uses `WanImageToVideoPipeline` with `last_image=` kwarg. Mount: `/models/wan2.1-flf2v-14b-720p`. Lightning is empirical via I2V LoRA — flag as "Beta — uses I2V LoRA, not officially tested" in the UI. Default CFG 5.5. ZeroGPU `large` / 150s. End-frame slot has nested Upload/Generate tabs (Generate calls Wan T2I to synthesize the end frame from a prompt).

### 9.4 V2V (Wan 2.1, restyle)

`WanVideoToVideoPipeline` on the Wan 2.1 T2V-14B backbone (no separate checkpoint). Mount: shares `/models/wan2.1-t2v-14b`. Strength slider 0.1-1.0 (default 0.7). Quality preset only. ZeroGPU `large` / 90s.

### 9.5 VACE (Wan 2.1 only — Wan 2.2 has no VACE checkpoint)

`WanVACEPipeline`. Two mounts: `/models/wan2.1-vace-1.3b` and `/models/wan2.1-vace-14b`. Sub-mode radio at top of inputs column with 9 options: Depth / Pose / Sketch / Flow / **Inpaint (default)** / Outpaint / Reference / Extension / Animate-Anything. Visibility of secondary controls (mask source, reference image gallery) is mode-conditional via `.change()` handlers.

Preprocessing: ship the lightweight subset only — DWPose (Wholebody), MiDaS (dpt_hybrid), RAFT. Total ~1 GB extra in preload. **Skip SAM2 / GroundingDINO / InsightFace for v1** — those sub-modes (label/caption inpaint, mask-track, anything-* variants) require the user to upload pre-extracted control videos. Surface this in the sub-mode info banner.

VACE never gets Lightning — hard-disable Fast preset, show toast on switch.

ZeroGPU `large` / 150s (1.3B) or 180s (14B).

### 9.6 S2V (Wan 2.2, not in diffusers)

Vendor upstream `wan` package (clone `Wan-Video/Wan2.2` and import its `wan` module). Wrap `wan.WanS2V(config=WAN_CONFIGS['s2v-14B'], checkpoint_dir="/models/wan2.2-s2v-14b").generate(image, audio, prompt, ...)`. The wav2vec2-large-xlsr-53-english audio encoder is bundled inside the same repo so the mount surfaces it at `/models/wan2.2-s2v-14b/wav2vec2-large-xlsr-53-english/`.

Inputs panel: reference image + audio (upload OR mic) + optional pose video + prompt + resolution dropdown. Duration is read-only — driven by audio length. Quality preset only. ZeroGPU `large` / 240s.

### 9.7 Wan-Animate (Wan 2.2)

`WanAnimatePipeline`. Mount `/models/wan2.2-animate-14b`. Mode radio: Character Swap / Pose Retarget / Replacement. Resolution radio: Low 480p / Medium 720p. Quality preset only.

Preprocessing required: ViTPose-H Wholebody, YOLOv10-Medium, SAM2 Hiera Large — total ~2 GB. These are in `Wan-AI/Wan2.2-Animate-14B/process_checkpoint/` subdir so they're part of the mounted volume — no separate preload needed.

Multi-segment stitching is native via `segment_frame_length=77` + `prev_segment_conditioning_frames=1`. Output column has an open `gr.Accordion("🎭 Processing outputs")` exposing pose / face / bg / mask intermediate videos.

ZeroGPU `xlarge` / 300s. Yellow info banner: "Pose+face preproc runs on CPU before GPU (~30s extra)".

### 9.8 TI2V-5B (Wan 2.2, not in diffusers)

Vendor upstream `wan` package — `wan.WanTI2V(config=WAN_CONFIGS['ti2v-5B'], checkpoint_dir="/models/wan2.2-ti2v-5b").generate(...)`. Inputs: prompt + optional image + orientation radio (Landscape 1280×704 / Portrait 704×1280 only).

Folded into the T2V or I2V tab as an additional generation option when generation == Wan 2.2 (decision: keep separate "TI2V" sidebar entry per the original wireframes, since it's mechanically different — vendored upstream not diffusers).

Quality preset only. ZeroGPU `large` / 60s.

---

## 10. UI specification

Wireframes are the canonical source: [`wireframes/index.html`](../../../wireframes/index.html). The 8 PNGs (`w1_shell_t2v.png` through `w8_gallery.png`) are 1:1 with the implementation target.

### 10.1 Layout

- Top header (`gr.Row`, sticky): `◉ Wan Studio` wordmark · `Generation: [Wan 2.2 ▾]` dropdown · `Preset: ◉Fast ○Quality` radio · `History` icon button · `Settings` icon button.
- Left sidebar (`gr.Sidebar(position="left", width=260, open=True)`): seven mode buttons (T2V / I2V / TI2V / FLF2V / V2V / VACE / S2V / Animate) + divider + Gallery / Settings entries. Active mode highlighted with indigo left-border accent. Modes unavailable in the selected generation are greyed but not hidden.
- Main area: two-column `gr.Row`, Inputs scale=2 / Output scale=3. Each mode's tab is a `gr.Column(visible=...)` toggled by sidebar `.click` handlers (only one visible at a time).
- Examples row below the two-col area for each mode (`gr.Examples(..., cache_examples=False, cache_mode="lazy")`).

### 10.2 Component standards

| Element | Component | Notes |
|---|---|---|
| Prompt | `gr.Textbox(lines=4)` + sibling "✨ Enhance Prompt" button (calls Qwen3 small via diffusers/transformers if budget allows; otherwise no-op v1) | |
| Negative prompt | `gr.Textbox(lines=2)` inside `gr.Accordion("Advanced", open=False)` | Pre-fill the official Wan Chinese boilerplate |
| Image input | `gr.Image(type="pil", sources=["upload", "clipboard"], image_mode="RGB")` | Clipboard paste is the killer feature |
| Audio (S2V) | `gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav")` | Mic optional |
| Driving video (Animate) | `gr.Video(sources=["upload"], include_audio=False)` | Audio stripped to avoid duplicate streams |
| Control gallery (VACE) | `gr.Gallery(columns=3, rows=2, allow_preview=True, sources=["upload"])` | Auto-suggests target H/W on first upload |
| Duration | `gr.Slider(0.5, 8.0, value=2.0, step=0.1, label="Duration (s)")` | Read-only for S2V (audio-driven) |
| Resolution | `gr.Dropdown(["1280x720 (16:9)", "720x1280 (9:16)", "960x960 (1:1)", "832x480 (16:9)", "480x832 (9:16)"])` | Aspect-labeled per HunyuanVideo Space pattern |
| Steps (Advanced) | `gr.Slider(1, 50)` | Default = preset default; user can override |
| CFG (Wan 2.2 dual) | Two `gr.Slider(0, 10, step=0.1)` — high-noise / low-noise | Second slider `visible=True` only when generation == 2.2 |
| Seed | `gr.Slider(0, 2**31-1, step=1)` + `gr.Checkbox("Randomize", value=True)` | |
| Generate | `gr.Button("Generate", variant="primary", size="lg")` | Full-width bottom of input column |
| ETA | `gr.Markdown("⌚ ZeroGPU reservation: ~Ns")` | Updates on `.change()` of params via the same `get_duration()` used by `@spaces.GPU` |
| Output video | `gr.Video(autoplay=True, loop=True, show_download_button=True, interactive=False)` | |
| Send-to chips | `gr.Row` of `gr.Button(size="sm")` — one per applicable next-mode | `.click` copies video path into target tab's input slot and switches sidebar |
| Gallery | `gr.Gallery(columns=4, height=520, allow_preview=True)` in the Gallery tab; session-scoped | |

### 10.3 Theme

`gr.themes.Default(primary_hue="indigo", neutral_hue="slate")`. Dark mode by default. Custom CSS:

```css
#wan-studio-header { padding: 8px 16px; border-bottom: 1px solid #2a2a2a; }
#wan-studio-sidebar { padding: 12px 8px; border-right: 1px solid #2a2a2a; min-height: 80vh; }
#wan-studio-sidebar button { width: 100%; text-align: left; margin-bottom: 4px; }
.warning-banner { background: #443811; border-left: 3px solid #d4a23b; padding: 8px 12px; border-radius: 4px; }
```

### 10.4 Mobile (<768 px viewport)

Sidebar collapses to a `≡` hamburger; tapping it slides the sidebar over the main column. Input/Output two-col stacks vertically (Inputs on top, Output below).

---

## 11. Error handling

| Failure | Detection | UX |
|---|---|---|
| User picks Fast for VACE/S2V/Animate/TI2V-5B | `preset.resolve()` returns `effective_preset="quality"` + fallback_message | `gr.Info(fallback_message)` toast; UI radio stays on Fast but params reflect Quality |
| `@spaces.GPU` quota exhausted | Caught at decorator level by ZeroGPU | `gr.Error("Daily ZeroGPU quota exceeded. Upgrade to PRO for 40 min/day, or wait until reset.")` |
| Volume mount missing at boot | `Path("/models/...").exists() == False` | App startup fails fast with clear `RuntimeError` in build logs |
| Wan 2.2 MoE on `large` OOM | `torch.cuda.OutOfMemoryError` | Re-raise with hint: "This mode requires xlarge tier — check `MODE_BUDGET` config" |
| Wan-Animate preproc CPU OOM | `MemoryError` during ViTPose batch | Reduce `motion_encode_batch_size` and retry once; surface error if still fails |
| Diffusers pipeline class missing | `ImportError` | "diffusers version is too old — required >=0.38.0" |
| Lightning LoRA missing in mount | `FileNotFoundError` from `load_lora_weights` | Fall back to Quality preset with toast |
| User closes tab mid-gen | ZeroGPU subprocess keeps running, costs quota | `callback_on_step_end` polls `progress.is_canceled`; returns early if set. Surfaces `gr.Warning("Generation aborted by client")` |

---

## 12. Testing strategy

### 12.1 Unit tests (`tests/`)

| File | Coverage |
|---|---|
| `test_registry.py` | All ModelCards have valid `repo`, `mode`, `generation`, `size`. Lightning availability flags match presence of HIGH/LOW LoRA paths. No duplicate `key`s. |
| `test_preset.py` | `resolve(card, "fast")` falls back to Quality with toast when `card.lightning_available=False`. Wan 2.2 MoE cards get `guidance_scale_2` set. |
| `test_backend.py` | `detect()` returns MPS on Apple Silicon, CUDA on cards, CPU otherwise. `spaces_gpu_or_noop()` returns no-op outside ZeroGPU. |

### 12.2 Smoke tests

`tests/test_smoke_t2v.py` — Wan 2.1 T2V-1.3B end-to-end on MPS. Generate 16-frame video at 480p, assert MP4 file exists + non-zero bytes + ffprobe metadata correct. Runs in CI on macOS-latest with self-hosted M-series runner (or skipped if `pytest.importorskip("torch.backends.mps")` fails).

### 12.3 Manual ZeroGPU integration tests

Pre-deploy checklist documented in `docs/deploy-checklist.md`:
1. Duplicate script runs cleanly (idempotent)
2. Space builds without timeout (30 min default startup_duration)
3. T2V Fast preset on Wan 2.1 14B completes <30s on `large`
4. T2V Fast preset on Wan 2.2 A14B completes <60s on `xlarge`
5. Animate Quality preset completes <300s on `xlarge`
6. VACE pose sub-mode with pre-extracted DWPose video completes
7. S2V with a 12-second WAV completes (vendored `wan` package working)
8. TI2V-5B at 1280×704 completes
9. Send-to chip from T2V → I2V passes the video correctly
10. Mobile viewport (390×844) renders without horizontal scroll

---

## 13. Deployment

### 13.1 One-time setup

```bash
# 1. Duplicate upstream repos to your account
python scripts/duplicate_upstream.py --dry-run    # preview
python scripts/duplicate_upstream.py              # execute (~15 min)

# 2. Create the Space with volume mounts
python scripts/create_space.py                    # uses huggingface_hub.HfApi
                                                  # API + Volume manifest

# 3. Push code
hf upload mayankgupta/wan-studio . --commit-message "Initial deploy"
```

### 13.2 README YAML frontmatter

```yaml
---
title: Wan Studio
emoji: 🎬
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: "5.4.0"
app_file: app.py
pinned: false
short_description: "Every Wan mode, one clean UI."
python_version: "3.12.12"
startup_duration_timeout: "30m"
preload_from_hub:
  - mayankgupta/umt5-xxl
  - mayankgupta/wan-vae
  - mayankgupta/clip-vit-h
  - mayankgupta/wan-lightning-loras
# ZeroGPU hardware: not set via `suggested_hardware` because the Blackwell-era enum
# string is not documented in HF's public docs as of May 2026. Hardware is set
# post-creation via the Space's web UI (Settings → Hardware → Zero).
---
```

### 13.3 CI / CD

- GitHub Actions workflow `.github/workflows/test.yml`: ruff + pytest on PRs
- Manual deploy via `hf upload` after CI green
- No auto-deploy (don't want every commit to bounce the Space)

---

## 14. Risks & open questions

| Risk | Severity | Mitigation |
|---|---|---|
| 150 GB volume mount cap exists but undocumented; would force trimming | Medium | Empirical test on first Space deploy. Fallback: ship only Wan 2.2 modes + Wan 2.1 small variants at launch; add big Wan 2.1 14B variants later |
| Vendored `wan` package has install / dep conflicts with diffusers ≥0.38 | Medium | Pin upstream `wan` commit SHA; test in local venv before push |
| Wan-Animate preproc weights (~2 GB) exhaust CPU RAM on ZeroGPU | Low | Run preproc with `torch.no_grad()` + `del intermediate` aggressively; cap `motion_encode_batch_size` |
| ZeroGPU subprocess can't be preempted on tab close → quota burn | Low | Implement soft-cancel via `callback_on_step_end` + `progress.is_canceled` |
| Wan 2.2 I2V Lightning V2 doesn't ship before public launch → V1-only quality regression | Low | Surface hybrid-trick toggle in Advanced (reuse Wan 2.1 lightx2v I2V LoRA on Wan 2.2 pipeline) |
| Wan-AI / Kijai pushes breaking config to original repos after our duplication | Low | Duplicates are immutable; we promote new versions manually after testing |
| Public Space gets brigaded / hammered, quota burns through fast | Low | Standard HF rate-limiting + per-user quota handles it; PRO daily 40 min is shared across visitors |

---

## 15. Out of scope (recap)

- LoRA upload by users · multi-architecture (Hunyuan/LTX/Flux) · cross-session persistence · auth · Wan 2.5/2.6/2.7 weights · real-time streaming · the full VACE annotator stack (SAM2/GroundingDINO/InsightFace)

---

## 16. Phase rollout (informational — full phase plan is the writing-plans deliverable next)

| Phase | Scope | Estimated duration |
|---|---|---|
| Phase 0 | Scaffold (already partially done) | 1-2 days |
| Phase 1 | T2V + I2V on Wan 2.1 14B + Wan 2.2 A14B with Fast/Quality preset; first ZeroGPU deploy | 4-5 days |
| Phase 2 | FLF2V + V2V + TI2V-5B (vendored `wan`) | 3-4 days |
| Phase 3 | VACE (lightweight preproc subset) | 4 days |
| Phase 4 | Wan-Animate (full preproc bundle) | 4 days |
| Phase 5 | S2V (vendored `wan`) | 3 days |
| Phase 6 | Cross-mode Send-to chains + Gallery + Settings page | 2 days |
| Phase 7 | Polish: mobile, examples, theme tuning, error UX | 2-3 days |
| Phase 8 | Public launch on `mayankgupta/wan-studio` | 1 day |

**Total: ~24-28 working days** to a polished public launch. Phase 1 deploy is publishable as a private/draft Space for iterative review.

---

## 17. References

- [`RESEARCH.md`](../../../RESEARCH.md) — full architecture brief (1182 lines, sections cited inline above as RESEARCH §X)
- [`raw/01_model_inventory.md`](../../../raw/01_model_inventory.md) — Wan model family catalog
- [`raw/02_modes_deep_dive.md`](../../../raw/02_modes_deep_dive.md) — Per-mode inputs/outputs/quirks
- [`raw/03_zerogpu_diffusers.md`](../../../raw/03_zerogpu_diffusers.md) — ZeroGPU runtime + diffusers loading
- [`raw/04_lightning_loras.md`](../../../raw/04_lightning_loras.md) — Lightning LoRA family + MoE dual-LoRA pattern
- [`raw/05_ux_patterns.md`](../../../raw/05_ux_patterns.md) — Reference Space study + UX architecture
- [`wireframes/index.html`](../../../wireframes/index.html) — Gallery of 8 PNG wireframes + montage
- HF Spaces config reference: <https://huggingface.co/docs/hub/spaces-config-reference>
- HF Spaces ZeroGPU: <https://huggingface.co/docs/hub/spaces-zerogpu>
- HF Spaces storage / volume mounts: <https://huggingface.co/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space>
