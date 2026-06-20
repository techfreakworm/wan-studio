# Wan Video Diffusion Model Family — Inventory

**Compiled:** 2026-05-21
**Scope:** Every officially released checkpoint from the `Wan-AI` organization on Hugging Face, plus official Alibaba forks. Third-party FP8/GGUF quantizations are listed only in the appendix.
**Source of truth:** [Wan-AI HF org](https://huggingface.co/Wan-AI), [Wan-Video GitHub](https://github.com/Wan-Video), [diffusers Wan docs](https://huggingface.co/docs/diffusers/en/api/pipelines/wan).

> **Key finding:** As of May 2026, the Wan-AI HF org holds **23 model repos** spanning two open generations: **Wan 2.1 (Feb–May 2025)** and **Wan 2.2 (Jul–Nov 2025)**. Wan 2.5, 2.6, and 2.7 — although announced and live on Alibaba Cloud Model Studio / Together AI — **do NOT yet have open weights on the Wan-AI HF org as of today**. Third-party blog claims that Wan 2.5 is "Apache 2.0 with weights on Hugging Face" do not match the live Wan-AI org listing.

---

## Summary Table

| Repo Path | Params | Modality | Resolution × Frames | Native dtype | Diffusers Class | Min Diffusers |
|---|---|---|---|---|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 1.3B | T2V | 480p × 81f | bf16 | `WanPipeline` | 0.33.0 |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 14B | T2V | 480p / 720p × 81f | bf16 | `WanPipeline` | 0.33.0 |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 14B | I2V | 480p × 81f | bf16 | `WanImageToVideoPipeline` | 0.33.0 |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 14B | I2V | 720p × 81f | bf16 | `WanImageToVideoPipeline` | 0.33.0 |
| `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` | 14B | FLF2V | 720p × 81f | bf16 | `WanImageToVideoPipeline` (via `last_image`) | 0.34.0 |
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | 1.3B | VACE (T2V/I2V/V2V/control/mask) | 480p × 81f | bf16 | `WanVACEPipeline` | 0.34.0 |
| `Wan-AI/Wan2.1-VACE-14B-diffusers` | 14B | VACE | 480p / 720p × 81f | bf16 | `WanVACEPipeline` | 0.34.0 |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 5B dense | TI2V (T2V + I2V unified) | 720p × ~121f @ 24fps | bf16 (VAE: fp32) | `WanPipeline` | 0.35.0 |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 27B total / 14B active (MoE) | T2V | 480p / 720p × 81f @ 24fps | bf16 (VAE: fp32) | `WanPipeline` (two transformers) | 0.35.0 |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 27B total / 14B active (MoE) | I2V | 480p / 720p × 81f @ 24fps | bf16 (VAE: fp32) | `WanImageToVideoPipeline` (two transformers) | 0.35.0 |
| `Wan-AI/Wan2.2-S2V-14B` | 14B | S2V (speech/audio-driven) | 480p / 720p × var. @ 24fps | bf16 | not yet integrated (native repo only) | — |
| `Wan-AI/Wan2.2-Animate-14B-Diffusers` | 14B dense | Character animation / replacement | 720p × 77f/segment @ 30fps | bf16 (VAE: fp32) | `WanAnimatePipeline` | 0.36.0+ |

**Companion non-diffusers repos** (native PyTorch weights, same checkpoints, no `-Diffusers` suffix):
`Wan2.1-T2V-1.3B`, `Wan2.1-T2V-14B`, `Wan2.1-I2V-14B-480P`, `Wan2.1-I2V-14B-720P`, `Wan2.1-FLF2V-14B-720P`, `Wan2.1-VACE-1.3B`, `Wan2.1-VACE-14B`, `Wan2.2-TI2V-5B`, `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`, `Wan2.2-S2V-14B`, `Wan2.2-Animate-14B`. These are used by the official `generate.py` in the Wan-Video repos.

---

## Wan 2.1 (February – May 2025)

The "original" open release. All Wan 2.1 checkpoints share:
- **Architecture:** Flow Matching Diffusion Transformer (DiT), single dense (no MoE).
- **VAE:** **Wan-VAE** — proprietary 3D causal VAE, `AutoencoderKLWan` in diffusers.
- **Text encoder:** `UMT5-XXL` (multilingual T5 variant from `google/umt5-xxl`), wrapped as `UMT5EncoderModel`.
- **License:** Apache 2.0.
- **Frame count convention:** `num_frames = 4*k + 1` (e.g. 81 frames = 5s @ 16 fps for T2V, also default for VACE/FLF2V).

### Wan2.1-T2V-1.3B

- **HF repos:** `Wan-AI/Wan2.1-T2V-1.3B` and `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B))
- **Params:** 1.3B
- **Modality:** Text-to-Video
- **Resolution × frames:** 480p (832×480) × 81 frames (≈5s @ 16fps); 720p experimental, not officially recommended.
- **Native dtype:** bfloat16
- **Release:** 25 Feb 2025 (native), 4 Apr 2025 (diffusers fork)
- **License:** Apache 2.0
- **Architecture notes:** DiT, dim=1536, 12 heads, 30 layers, FFN=8960, input/output=16 (latent channels). Single dense (no MoE).
- **VAE:** Wan-VAE (`AutoencoderKLWan`)
- **Diffusers integration:** `WanPipeline` — landed in diffusers **0.33.0** (Apr 9 2025). VRAM ~8.19 GB on RTX 4090 with optimizations.

### Wan2.1-T2V-14B

- **HF repos:** `Wan-AI/Wan2.1-T2V-14B` and `Wan-AI/Wan2.1-T2V-14B-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B))
- **Params:** 14B
- **Modality:** Text-to-Video
- **Resolution × frames:** 480p **and** 720p (1280×720), 81 frames @ 16 fps
- **Native dtype:** bfloat16
- **Release:** 12 Mar 2025 (native), 4 Apr 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** DiT, dim=5120, 40 heads, 40 layers, FFN=13824, input/output=16. Single dense.
- **VAE:** Wan-VAE (`AutoencoderKLWan`)
- **Diffusers integration:** `WanPipeline`, available since diffusers 0.33.0. Recommended flow_shift: 5.0 for 720p, 3.0 for 480p.

### Wan2.1-I2V-14B-480P

- **HF repos:** `Wan-AI/Wan2.1-I2V-14B-480P` and `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P))
- **Params:** 14B
- **Modality:** Image-to-Video
- **Resolution × frames:** 480p × 81 frames
- **Native dtype:** bfloat16
- **Release:** 26 Feb 2025
- **License:** Apache 2.0
- **Architecture notes:** Same DiT topology as T2V-14B (dim=5120, 40/40). Adds a **CLIP image encoder** (`openai/clip-vit-huge-patch14`, wrapped as `CLIPVisionModel`) for the conditioning image.
- **VAE:** Wan-VAE
- **Diffusers integration:** `WanImageToVideoPipeline` (since 0.33.0).

### Wan2.1-I2V-14B-720P

- **HF repos:** `Wan-AI/Wan2.1-I2V-14B-720P` and `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P))
- **Params:** 14B
- **Modality:** Image-to-Video
- **Resolution × frames:** 720p (1280×720) × 81 frames
- **Native dtype:** bfloat16
- **Release:** 26 Feb 2025
- **License:** Apache 2.0
- **Architecture notes:** Identical to I2V-14B-480P, retrained for 720p. CLIP-H/14 image encoder.
- **VAE:** Wan-VAE
- **Diffusers integration:** `WanImageToVideoPipeline`.

### Wan2.1-FLF2V-14B-720P

- **HF repos:** `Wan-AI/Wan2.1-FLF2V-14B-720P` and `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P))
- **Params:** 14B
- **Modality:** First-Last-Frame to Video (interpolates between two anchor frames)
- **Resolution × frames:** 720p × 81 frames
- **Native dtype:** bfloat16
- **Release:** 17 Apr 2025 (native), 22 Apr 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Same DiT as I2V-14B-720P, but trained with both first and last frame as conditioning. Trained primarily on Chinese text-video pairs; Chinese prompts recommended. CLIP image encoder used.
- **VAE:** Wan-VAE
- **Diffusers integration:** Reuses `WanImageToVideoPipeline` — pass `last_image=` parameter alongside `image=`. Available 0.34.0+.

### Wan2.1-VACE-1.3B

- **HF repos:** `Wan-AI/Wan2.1-VACE-1.3B` and `Wan-AI/Wan2.1-VACE-1.3B-diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B))
- **Params:** 1.3B
- **Modality:** Unified "Video All-in-one Creation and Editing" — supports T2V, R2V (reference-to-video), V2V editing (depth/pose/sketch/flow/grayscale/scribble/layout control), MV2V (masked V2V), inpainting/outpainting, subject-to-video.
- **Resolution × frames:** 480p × 81 frames
- **Native dtype:** bfloat16
- **Release:** 19 May 2025 (native), 6 Jun 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Same DiT topology as T2V-1.3B (dim=1536, 12/30) with added VACE control-injection layers. Built on `WanVACETransformer3DModel`.
- **VAE:** Wan-VAE
- **Diffusers integration:** `WanVACEPipeline` (0.34.0+). API uses `video=`, `mask=` (black = condition, white = generate), `reference_images=`. Default flow_shift = 3.0 for 480p.

### Wan2.1-VACE-14B

- **HF repos:** `Wan-AI/Wan2.1-VACE-14B` and `Wan-AI/Wan2.1-VACE-14B-diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B))
- **Params:** 14B
- **Modality:** Same VACE multi-task suite as 1.3B
- **Resolution × frames:** 480p and 720p × 81 frames
- **Native dtype:** bfloat16
- **Release:** 19 May 2025 (native), 6 Jun 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Same DiT topology as T2V-14B (dim=5120, 40/40) with VACE control branches.
- **VAE:** Wan-VAE
- **Diffusers integration:** `WanVACEPipeline`. Recommended flow_shift = 5.0 for 720p, 3.0 for 480p.

---

## Wan 2.2 (July – November 2025)

The big architectural upgrade. Headline change: **MoE for the 14B variants** (T2V, I2V) and a **new high-compression Wan2.2-VAE** for the 5B variant.

### Shared properties across Wan 2.2

- **License:** Apache 2.0 across all 5 model families.
- **Native dtype:** **bfloat16** for the transformers and text encoder; **float32 strongly recommended for the VAE** for decode quality.
- **Text encoder:** UMT5-XXL (same as 2.1).
- **VAE:**
  - **A14B (MoE) variants** continue to use the Wan 2.1 VAE (8×8×4 compression).
  - **TI2V-5B** uses the new **Wan2.2-VAE with 16×16×4 (T×H×W) compression**, plus a patchification layer giving 4×32×32 effective compression (64× total). Same `AutoencoderKLWan` class in diffusers — the difference is in config, not class.
- **Default frames:** 81 (matches Wan 2.1) for A14B; ~121 frames (5s @ 24fps) for TI2V-5B.
- **Training data:** +65.6% more images and +83.2% more videos vs Wan 2.1.

### Wan 2.2 MoE architecture (read carefully — this is the key change)

For the **A14B** variants (T2V-A14B and I2V-A14B), the model uses a **two-expert MoE across denoising timesteps**, not across tokens:

- **High-noise expert** (`transformer` in diffusers, ~14B params) — active during early/high-noise denoising steps. Handles layout and global composition.
- **Low-noise expert** (`transformer_2` in diffusers, ~14B params) — active during later/low-noise denoising steps. Refines details.
- **Switching boundary:** The diffusers `model_index.json` for `Wan2.2-T2V-A14B-Diffusers` has **`boundary_ratio: 0.875`**. This means `transformer` (high-noise) handles timesteps `t >= 0.875 * num_train_timesteps`, and `transformer_2` (low-noise) handles `t < 0.875 * num_train_timesteps`. Wan's paper describes the boundary as the timestep where SNR equals `0.5 * SNR_min`; the empirical value is 0.875 for Wan 2.2.
- **Total / active parameters:** 27B total parameter footprint on disk, but only ~14B active per forward pass (the active expert).
- **"A14B" naming:** "A14B" = "Active 14B" — emphasizes ~14B active per step despite the ~27B total.
- **Two `guidance_scale` knobs in diffusers:** `WanPipeline.__call__()` accepts both `guidance_scale` (for the high-noise transformer) and `guidance_scale_2` (for `transformer_2`). When `boundary_ratio` is set and `guidance_scale_2 is None`, it defaults to the same as `guidance_scale`.
- **LoRA loading:** By default, `load_lora_weights()` only loads into `transformer`. Pass `load_into_transformer_2=True` to load into the low-noise denoiser as well.

### Wan2.2-TI2V-5B

- **HF repos:** `Wan-AI/Wan2.2-TI2V-5B` and `Wan-AI/Wan2.2-TI2V-5B-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B))
- **Params:** 5B **dense** (NOT MoE — this is the consumer-grade variant)
- **Modality:** Unified Text-to-Video **and** Image-to-Video in one checkpoint (TI2V = Text/Image-to-Video).
- **Resolution × frames:** 720p (1280×704 or 704×1280) × ~121 frames @ 24 fps (≈5s)
- **Native dtype:** bf16 (VAE in fp32)
- **Release:** 7 Aug 2025 (native), 9 Aug 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Single 5B dense `WanTransformer3DModel`, NO MoE. The "TI2V" naming reflects unified text+image conditioning.
- **VAE:** **Wan2.2-VAE** (the new high-compression one) — 16×16×4 compression ratio, achieving 64× total compression with the patch layer. This is what enables 720p @ 24fps on a single RTX 4090 (~24 GB VRAM) in <9 minutes.
- **Diffusers integration:** `WanPipeline` (handles both T2V and I2V via optional image input). Min diffusers: 0.35.0 (where Wan 2.2 support landed).

### Wan2.2-T2V-A14B (MoE)

- **HF repos:** `Wan-AI/Wan2.2-T2V-A14B` and `Wan-AI/Wan2.2-T2V-A14B-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B))
- **Params:** ~27B on disk / 14B active per step (MoE, 2 experts)
- **Modality:** Text-to-Video
- **Resolution × frames:** 480p and 720p × 81 frames @ 24fps (≈5s @ 16fps, or shorter @ 24fps)
- **Native dtype:** bf16 transformers, fp32 VAE recommended
- **Release:** 7 Aug 2025 (native), 9 Aug 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Dual `WanTransformer3DModel` (transformer + transformer_2). `boundary_ratio: 0.875`. SNR-threshold switching.
- **VAE:** Wan-VAE (carried over from 2.1).
- **Diffusers integration:** `WanPipeline`. `model_index.json` includes both `transformer` and `transformer_2` subfolders. **Requires diffusers ≥ 0.35.0** (Wan 2.2 MoE support landed in the dev branch leading to 0.35.0; at first release some setups required `pip install git+https://github.com/huggingface/diffusers`).

### Wan2.2-I2V-A14B (MoE)

- **HF repos:** `Wan-AI/Wan2.2-I2V-A14B` and `Wan-AI/Wan2.2-I2V-A14B-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B))
- **Params:** ~27B on disk / 14B active per step (MoE)
- **Modality:** Image-to-Video
- **Resolution × frames:** 480p and 720p × 81 frames @ 24fps
- **Native dtype:** bf16 transformers, fp32 VAE
- **Release:** 7 Aug 2025 (native), 9 Aug 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Same dual-transformer MoE design as T2V-A14B, with image conditioning branch (CLIP-H/14 image encoder).
- **VAE:** Wan-VAE.
- **Diffusers integration:** `WanImageToVideoPipeline` with the same `boundary_ratio: 0.875` two-stage scheme. Min diffusers 0.35.0.

### Wan2.2-S2V-14B

- **HF repo:** `Wan-AI/Wan2.2-S2V-14B` ([HF](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B))
- **Params:** 14B (the model card mentions MoE-style architecture similar to A14B; whether it's true 27B/14B MoE or single 14B dense is ambiguous from the card alone — see Open Questions)
- **Modality:** Speech-to-Video / audio-driven cinematic video. Inputs: reference image + driving audio (+ optional pose video, optional text prompt).
- **Resolution × frames:** 480p and 720p × variable (follows audio length) @ 24 fps
- **Native dtype:** bf16
- **Release:** 17 Sep 2025 (HF upload; the model was first announced 26 Aug 2025)
- **License:** Apache 2.0
- **Architecture notes:** Uses **wav2vec2** as the audio encoder (per HF tags). Supports lip-sync editing and pose-conditioned audio-driven generation.
- **VAE:** Wan-VAE (the 2.1-style VAE, not the new 16×16×4 2.2 VAE).
- **Diffusers integration:** **NOT YET INTEGRATED** as of May 2026 — no `-Diffusers` variant; use the native `generate.py` from `Wan-Video/Wan2.2` repo.

### Wan2.2-Animate-14B

- **HF repos:** `Wan-AI/Wan2.2-Animate-14B` and `Wan-AI/Wan2.2-Animate-14B-Diffusers` ([HF](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B))
- **Params:** 14B **dense** (NOT MoE; built on the I2V-A14B base architecture but trained as a single dense checkpoint for character animation)
- **Modality:** Two-mode unified pipeline:
  1. **Animation mode** (default) — animate a character image to match motion+expression from reference pose+face videos.
  2. **Replacement mode** — replace the character in a background video with the supplied character image while preserving the scene; uses a Relighting LoRA.
- **Resolution × frames:** 720p (1280×720) × 77 frames per segment @ 30fps (segments chained for longer outputs)
- **Native dtype:** bf16 transformer, fp32 VAE recommended
- **Release:** 5 Nov 2025 (native), 13 Nov 2025 (diffusers)
- **License:** Apache 2.0
- **Architecture notes:** Single `WanAnimateTransformer3DModel`, no high/low expert split. Uses skeleton keypoints (pose video) for body motion and implicit facial features (face video) for expressions. Default CFG disabled (`guidance_scale=1.0`); can be enabled for finer prompt/face control. SAM2 and pose-detection preprocessing required.
- **VAE:** Wan-VAE (`AutoencoderKLWan`).
- **Diffusers integration:** `WanAnimatePipeline` (introduced in PR around diffusers 0.36 dev; documented for v0.38.0). API accepts `image`, `pose_video`, `face_video`, plus optional `background_video` + `mask_video` for replacement mode.

---

## Wan 2.5 / 2.6 / 2.7 — Announced but **NOT open-weight** as of 2026-05-21

These generations have been publicly announced and are accessible commercially via APIs, but **no weights have been published to the `Wan-AI` Hugging Face organization** as of the date of this report. The Wan-AI HF org listing currently stops at the Wan 2.2 family.

> **Important:** Several third-party blog posts (e.g. mindstudio.ai, cliprise.app) describe Wan 2.5 / 2.7 as "open source Apache 2.0 with weights on Hugging Face." This claim is **NOT verifiable** against the Wan-AI org listing or the Wan-Video/Wan2.x GitHub repos as of today. Treat these as **API-only** until weights actually land at `huggingface.co/Wan-AI/Wan2.5-*` etc.

### Wan 2.5 (Sep 2025 announce)
- **Where to use:** Alibaba Cloud Model Studio API, WaveSpeed AI, Together AI.
- **Headline features:** Joint **video + audio** generation in one pass (voice, ambient sound). 1080p output. 720p and 1080p resolutions, 5–10s duration.
- **Param count:** Not officially published. Third-party reports suggest similar MoE A14B-class footprint, but unverifiable.
- **Open weights on HF?** **No.**

### Wan 2.6 (Dec 2025 announce)
- **Model variants announced:** Wan2.6-T2V, Wan2.6-I2V, **Wan2.6-R2V** (reference-to-video with voice cloning — appearance + voice from a reference clip), Wan2.6-T2I, Wan2.6-image.
- **Headline features:** Multi-shot storytelling, multi-person dialogue, up to 15s duration, expand character + voice clone.
- **Open weights on HF?** **No.** Access via Alibaba Cloud Model Studio and `wan.video` only. Alibaba Cloud blog: "Users can access and deploy the models through Model Studio—Alibaba Cloud's AI development platform—and Wan's official website" (no mention of HF weights release).

### Wan 2.7 (Apr 1–6 2026 announce)
- **Model variants announced:** Wan 2.7 T2V, Wan 2.7 I2V, Wan 2.7 R2V (with voice cloning), Wan 2.7 VideoEdit (instruction-based editing).
- **Headline features:** 1080p native, 2–15 seconds, multi-shot narrative control, 5000-char prompt window, 9-grid multi-image input, first/last frame control.
- **Param count (reported):** 27B total / 14B active (MoE) — same family scale as Wan 2.2 A14B.
- **License (reported):** Apache 2.0.
- **Open weights on HF?** **Not yet** at the `Wan-AI` org. Several blogs claim "weights on Hugging Face" but no actual repos exist under `Wan-AI` as of 2026-05-21. Together AI hosts inference endpoints `Wan-AI/wan2.7-t2v`, `Wan-AI/wan2.7-i2v`, `Wan-AI/wan2.7-r2v`, `Wan-AI/wan2.7-videoedit` (these are **API endpoint identifiers**, not HF repos).
- **Implication for Studio app:** Plan around **Wan 2.2** as the latest open weights. Wan 2.7 / Wan 3.0 should be treated as future-but-not-yet-open work.

### Wan 3.0 (pre-announced, no release date)
- **Reported targets:** 60B parameters, 4K resolution, 30s generation, Apache 2.0, mid-2026.
- **Source:** Multiple third-party blogs cite Alibaba confirmation, not yet on official Alibaba channels with weights.

---

## Diffusers Integration Story (Summary)

Officially supported in `diffusers`:

| Pipeline class | Covers | Min version | Key kwargs |
|---|---|---|---|
| `WanPipeline` | Wan 2.1 T2V (1.3B, 14B), Wan 2.2 T2V-A14B, Wan 2.2 TI2V-5B (used for both T2V and I2V on TI2V-5B) | 0.33.0 (Wan 2.1); 0.35.0 (Wan 2.2 MoE) | `transformer`, `transformer_2` (optional, MoE), `boundary_ratio` (MoE switch), `guidance_scale`, `guidance_scale_2` (MoE low-noise stage), `num_frames` (default 81) |
| `WanImageToVideoPipeline` | Wan 2.1 I2V (480P/720P), Wan 2.1 FLF2V (`last_image=`), Wan 2.2 I2V-A14B | 0.33.0; 0.34.0 for FLF2V; 0.35.0 for 2.2 MoE | `image`, `last_image` (FLF2V), `image_embeds`, `transformer_2` + `boundary_ratio` (MoE 2.2) |
| `WanVACEPipeline` | Wan 2.1 VACE 1.3B and 14B | 0.34.0 | `video`, `mask`, `reference_images`, `conditioning_scale` |
| `WanVideoToVideoPipeline` | Generic V2V refinement on Wan 2.1 T2V backbone | 0.34.0 | `video`, `strength` |
| `WanAnimatePipeline` | Wan 2.2 Animate-14B (character animation + replacement) | 0.36.0+ (documented in 0.38.0) | `image`, `pose_video`, `face_video`, `background_video`, `mask_video`, `mode={"animate","replace"}`, `segment_frame_length` (default 77), `prev_segment_conditioning_frames` |

**Models / VAEs / Transformers exposed:**

- `WanTransformer3DModel` — used by T2V, I2V, TI2V, S2V (when integration lands).
- `WanVACETransformer3DModel` — VACE-specific transformer with control branches.
- `WanAnimateTransformer3DModel` — Animate-specific transformer.
- `AutoencoderKLWan` — **same class for both Wan-VAE (2.1) and Wan2.2-VAE**. The compression difference (8×8×4 vs 16×16×4) is in the `config.json`, not a new Python class. Recommended `torch_dtype=torch.float32` for decode quality.

**LoRA:** Both Wan 2.1 and 2.2 support LoRAs via `load_lora_weights()` (the `WanLoraLoaderMixin`). For Wan 2.2 MoE checkpoints, pass `load_into_transformer_2=True` to apply LoRA to the low-noise denoiser.

**Single-file loading:** `WanTransformer3DModel` and `AutoencoderKLWan` both support `from_single_file()` — useful for loading Kijai's repackaged safetensors.

**Lightning / LightX2V speed LoRAs:** Both 2.1 and 2.2 support [Kijai LightX2V LoRAs](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v) for step reduction. Wan 2.2 usage is slightly more involved (see diffusers PR #12040).

---

## Appendix A — Notable Third-Party Forks (NOT in main inventory)

These are NOT official `Wan-AI` releases but are widely used. Listed here for reference only; the user has explicitly opted out of quantized variants in the main inventory.

### Alibaba PAI "Fun" family — official Alibaba (different team)
Released by the `alibaba-pai` org (Alibaba's Platform of AI team), not the Wan-AI team. These are fine-tunes/extensions of the Wan base models for control-conditioned video generation.

- `alibaba-pai/Wan2.1-Fun-1.3B-Control` — depth/pose/canny control
- `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP` — image-to-video inpainting
- `alibaba-pai/Wan2.1-Fun-V1.1-14B-InP`
- `alibaba-pai/Wan2.2-Fun-A14B-InP` — A14B inpainting variant
- `alibaba-pai/Wan2.2-Fun-A14B-Control` — A14B control variant
- `alibaba-pai/Wan2.2-Fun-5B-InP`
- `alibaba-pai/Wan2.2-Fun-5B-Control`
- `alibaba-pai/Wan2.2-Fun-5B-Control-Camera`
- `alibaba-pai/Wan2.2-VACE-Fun-A14B` — VACE-style multi-task wrapper for A14B

These are full-precision (bf16) and Apache 2.0, but they are explicitly NOT in the Wan-AI canonical lineage. Useful if you need extra control modalities (e.g. camera-conditioned video) on top of the base Wan 2.2 MoE.

### Kijai community quantizations (FP8 / repackaged safetensors)
- `Kijai/WanVideo_comfy` — repackaged single-file safetensors for ComfyUI
- `Kijai/WanVideo_comfy_fp8_scaled` — **FP8 e4m3fn scaled** variants of Wan 2.1, Wan 2.2 T2V/I2V/Animate
  - Covers e.g. `Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors`
- `Kijai/WanVideo_comfy/Lightx2v/` — LightX2V step-reduction LoRAs

### City96 GGUF quants
- Various `city96/Wan2.*-GGUF` repos (Q2..Q8) for llama.cpp-style inference on CPU/low-VRAM GPUs.

### Comfy-Org repackaged
- `Comfy-Org/Wan_2.1_ComfyUI_repackaged` — official ComfyUI-friendly single-file repacks (referenced in diffusers docs for `from_single_file()` examples).

These third-party forks are explicitly out of scope for the main Studio inventory per user constraint (full-precision bf16 only).

---

## Open Questions / Ambiguities

1. **Wan2.2-S2V-14B MoE status.** The model card states "14B parameters" but also references the same SNR-based expert switching as the A14B family. Whether S2V is a **single 14B dense** transformer or a **dual-expert MoE** like the A14B variants is ambiguous from the card alone. The HF org listing reports it under modality tag `image-to-video` (which is also odd). To verify, would need to inspect `Wan-AI/Wan2.2-S2V-14B/model_index.json` or the safetensors structure for `transformer_2`. Working assumption: **single 14B dense** with the S2V branch added.

2. **Wan2.2-S2V diffusers integration.** No `-Diffusers` repo exists. No `WanSpeechToVideoPipeline` in the diffusers docs as of 0.38.0. If the user wants S2V in the Studio app, would need to invoke the native `generate.py` from `Wan-Video/Wan2.2`, OR wait for a future diffusers integration.

3. **Wan2.2-VAE diffusers class.** Both the 2.1 Wan-VAE and the new 2.2-VAE (16×16×4) load via the same `AutoencoderKLWan` class — only the `config.json` differs (compression ratios, channel counts). This is verified from the diffusers source. There is **no** separate `AutoencoderKLWan2` or `AutoencoderKLWan22` class. This is good news for the Studio app: one VAE wrapper handles both generations.

4. **Boundary ratio precision.** `Wan-AI/Wan2.2-T2V-A14B-Diffusers/model_index.json` reports `boundary_ratio: 0.875`. The Wan 2.2 paper describes the switch as "at SNR = 0.5 × SNR_min". The diffusers value is the empirical, hard-coded `0.875` — this is what's actually used at inference. For the Studio app, hardcode 0.875 unless we have reason to vary.

5. **Wan 2.5 / 2.7 weight release timeline.** Some third-party blogs claim "weights on Hugging Face under Apache 2.0", but the live Wan-AI org listing does not contain any 2.5/2.6/2.7 repos as of 2026-05-21. The Wan-Video GitHub org lists only `Wan2.1` and `Wan2.2` repos (no `Wan2.5`/`Wan2.7` repo). For Studio planning, assume **Wan 2.2 is the latest open generation** and revisit if/when `Wan-AI/Wan2.5-*` or `Wan-AI/Wan2.7-*` repos materialize.

6. **Frame count for I2V on Wan 2.1.** The HF model cards for `Wan2.1-I2V-14B-480P` and `Wan2.1-I2V-14B-720P` do not explicitly state 81 frames in the prose, but the diffusers integration example and the `num_frames=81` default in `WanImageToVideoPipeline.__call__` confirm 81 frames. Confirmed.

7. **TI2V-5B frame default.** The model card mentions "5 seconds at 720P 24fps." 5s × 24fps = 120 frames, but Wan uses the `4k+1` rule, so the actual default is likely **121 frames**. To confirm precisely, inspect the pipeline default for the `Wan2.2-TI2V-5B-Diffusers` model. Working assumption: 121 frames.

---

## Source URLs

**Wan-AI org and model cards**
- https://huggingface.co/Wan-AI
- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
- https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
- https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
- https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P
- https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
- https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
- https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/main/model_index.json (boundary_ratio = 0.875)

**Diffusers docs and source**
- https://huggingface.co/docs/diffusers/en/api/pipelines/wan
- https://huggingface.co/docs/diffusers/en/api/models/autoencoder_kl_wan
- https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan
- https://github.com/huggingface/diffusers/releases (0.33.0 = Apr 9 2025 with Wan 2.1)

**Wan-Video GitHub**
- https://github.com/Wan-Video
- https://github.com/Wan-Video/Wan2.1
- https://github.com/Wan-Video/Wan2.2

**Wan 2.5 / 2.6 / 2.7 announcements (API-only, no open weights)**
- https://www.alibabacloud.com/blog/alibaba-unveils-wan2-6-series-enabling-everyone-to-star-in-videos_602742
- https://www.cliprise.app/news/wan-2-7-video-release
- https://wavespeed.ai/landing/wan-2.5
- https://www.mindstudio.ai/blog/what-is-wan-2-5-video-open-source (claims weights on HF — UNVERIFIED against live org)

**Notable third-party forks (appendix only)**
- https://huggingface.co/alibaba-pai (Wan2.x-Fun control/InP variants)
- https://huggingface.co/Kijai/WanVideo_comfy
- https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled
- https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
