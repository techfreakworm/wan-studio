# Wan Studio — Local MPS Acceptance Ledger

One source of truth for "all 13 modes work locally with real, frame-verified video."
A mode is **DONE** only when every column is ✅. Metrics gates: not_black, has_contrast,
has_detail, has_motion, not_static, not_pure_noise (saturation only catches pure noise — it
CANNOT separate vivid-correct from broken), no_nan — AND VISUAL inspection of start/mid/end
frames confirms a coherent scene (the real arbiter; eyeball in both directions).

Compute dtype: **bf16** on MPS (validated clean; see dtype lock). Weights stored bf16 in
`~/wan-bf16/<slug>/` (outside HF cache, eviction-safe). Pushes DEFERRED (wan-brain ruling).

DONE = all of: Loads, Infers, Frames✓ (eyeball start/mid/end coherent), Metrics, Adheres
(conditioning honored), In-app (Playwright in the Gradio UI). Frame counts are the static-safe
memcheck envelope (wan-brain ruling #8): 1.3B=25f, 14B@480p=17f, 14B@720p=13f, MoE=9f→13f, ti2v=13f.

| # | Mode key | Handler | bf16 | Loads | Infers | Frames✓ | Metrics | Adheres | In-app | Notes |
|---|----------|:------:|:----:|:-----:|:------:|:------:|:------:|:------:|:------:|-------|
| 1 | wan2.1_t2v_1.3b | ✅ | ✅ | ✅ | ✅ | ✅@13f | ✅ | ✅ prompt | ⬜ | CLEAN red panda @13f (FlowMatchEuler fix). |
| 2 | wan2.1_t2v_14b | ✅ | ✅ | ✅ | ✅ | ✅@17f | ✅ | ✅ prompt | ✅ Playwright | **2026-06-21 PASS** Lightning fast 4-step euler shift5 @17f/480p: sharp 1209, sat 0.393, motion 3.5, coherent photoreal red panda. peak 92.4GB. **IN-APP**: Gradio Fast/1s → wan_t2v_*.mp4 832×480 playing in UI (Playwright + DOM-confirmed). |
| 3 | wan2.1_v2v_14b | ✅ | reuse #2 | ✅ | ✅ | ✅@17f | ✅ | ✅ source | ⬜ | **2026-06-21 PASS** Quality 16-step strength0.6 @17f/480p: sharp 1092, sat 0.413, motion 3.7. Output follows source clip structure+motion (red panda on rock) with restyle applied. Peak 91.3GB. **FIXED bf16-VAE-encode bug (shared.vae encode-input cast) — applies to all encode modes.** |
| 4 | wan2.1_i2v_14b_480p | ✅ | ✅ | ✅ | ✅ | ✅@17f | ✅ | ✅ frame0≈input | ⬜ | **2026-06-21 PASS** Quality 16-step @17f/480p, input=t2v panda frame: sharp 1201, sat 0.397, motion 4.2. frame[0]≈input panda, end shows head/paw motion. Peak 99.5GB. CLIP via local B1 + encode fix. |
| 5 | wan2.1_vace_1.3b | ✅ | ✅ | ✅ | ✅ | ✅@25f | ✅ | ✅ reference | ⬜ | **2026-06-21 PASS** Quality 30-step @25f/480p, reference=panda frame, prompt=forest branch: motion 5.2, peak 84.6GB. Output = red panda (reference subject) on branch in forest (follows prompt). Soft (483 sharp, 1.3B weak) but coherent+on-subject. |
| 6 | wan2.2_ti2v_5b | ✅ new | ✅ | ✅ | ✅ | ✅@13f | ✅ | ✅ SSIM 1.0 | ⬜ | **2026-06-21 PASS** new ti2v.py (WanImageToVideoPipeline + expand_timesteps + own z48 VAE + image_encoder=None). Run A panda-init SSIM 0.904, Run B car-init SSIM **1.000** (control vs panda 0.384) → conditioning PROVEN wired, not silent-T2V. 13f/720p Quality 20-step, ~12s/step, peak 86.7GB. |
| 7 | wan2.1_vace_14b | ✅ | ✅ | ✅ | ✅ | ✅@17f | ✅ | ✅ reference | ⬜ | **2026-06-21 PASS** Quality 20-step @17f/480p, reference=panda: sharp 1152 (much better than 1.3B), motion 4.7, peak 101GB. Red panda (reference) coherent + motion. |
| 8 | wan2.1_i2v_14b_720p | ✅ | ✅ | ✅ | ✅ | ✅@13f | ✅ | ✅ frame0≈input | ⬜ | **2026-06-22 PASS @native 720p** after the MPS-SDPA-long-key fix (mps_patches key-chunked flash). Quality 16-step @13f/704×1280: sharp 690, autocorr **0.871** (vs 0.17 noise), sat 0.39, motion 8.2, peak 129GB driver. Clean photoreal red panda, frame[0]≈input. **FIX: fused MPS SDPA returns wrong output over ~14k keys → key-chunked online-softmax (pipelines/mps_patches.py).** |
| 9 | wan2.1_flf2v_14b_720p | ✅ | ✅ | ✅ | ✅ | ✅@13f | ✅ | ✅ first&last | ⬜ | **2026-06-22 PASS @native 720p** (key-chunked flash fix). Quality 16-step @13f/704×1280, first=panda-start/last=panda-end: autocorr 0.885, sharp 374, sat 0.40, motion 4.5, peak 129GB driver. frame[0]≈first, frame[-1]≈last, coherent morph. Lightning DISABLED (no bf16 720p LoRA exists) → Quality-only. |
| 10 | wan2.2_t2v_a14b | ✅ | ✅ | ✅ | ✅ | ✅@9f | ✅ | ✅ prompt | ⬜ | **2026-06-21 PASS** MoE 4-step Lightning (Seko-V1 high→transformer/low→transformer_2, both adapters active), 9f/480p: sharp 743, autocorr 0.698 (has_structure), motion 5.1, peak 98GB driver (~81 real, safe). Coherent red panda walking in forest. OS-panic-class — clean at 9f. |
| 11 | wan2.2_i2v_a14b | ✅ | ✅ | ✅ | ✅ | ✅@9f | ✅ | ✅ frame0≈input | ⬜ | **2026-06-22 PASS** MoE 4-step Lightning (Seko-V1 high/low), 9f/480p, panda init: sharp 1337, autocorr 0.838, peak 101GB driver (~real lower, safe). Clean red panda, frame[0]≈input + motion. Both A14B MoE modes done. |
| 12 | wan2.2_animate_14b | ✅ new | ✅ | ✅ | ✅ | ✅@13f | ✅ | ✅ identity+motion | ⬜ | **2026-06-22 PASS @native 720p**. new animate.py (WanAnimatePipeline) + real ViTPose/YOLO preproc (scripts/animate_preprocess.py → pose skeleton + face crops) + chunked-flash (mps_patches patches animate transformer too) + PYTORCH_ENABLE_MPS_FALLBACK=1 (linalg_qr op-gap). Quality 16-step @13f, autocorr 0.836, peak 135GB driver. IDENTITY (blue-elf char preserved) + MOTION (follows driving pose). |
| 13 | wan2.2_s2v_14b | ✅ new | ✅ bf16 | ✅ | ✅ | ✅@17f | ✅ | ✅ identity+audio | ⬜ | **2026-06-22 PASS @480p (512×704)**. Upstream `wan.WanS2V` ported to MPS (NOT in diffusers): scripts/s2v_smoke.py vendors wan + shims — device cuda→mps factory (handles `\|` unions + isinstance), float64→fp32 downcast, autocast→mps bf16 + **2 fp32 islands** (time_embedding, head; MPS autocast has no fp32), flash_attention replaced in all 4 namespaces (q_scale+k_lens+chunked, MHA no-GQA), wav2vec2 on CPU @16kHz mono, bf16 VAE. 16-step UniPC: autocorr **0.903**, sharp 124, motion 8.2, all gates PASS. EYEBALL: cat-in-sunglasses, IDENTITY preserved + AUDIO-driven mouth (closed→open→partial). Mem: per-forward + pre-decode empty_cache + offload_model=True; peak ~110GB at decode (89-frame VAE; tight but completes). ⚠️ runs via MPS bf16-autocast — explicit-bf16 rewrite staged as known-good fallback (autocast op-coverage is torch-version-dependent). Some autoregressive color drift over frames (not banding). |

Legend: ✅ done · ⏳ in progress · ⬜ todo · — not started · reuse/cached = satisfied via another row.

## Harness self-test (gate before trusting any PASS) — ✅ PASS 2026-06-22 (scripts/harness_selftest.py)
- [x] all-black clip → FAILS (not_black, has_contrast, has_detail, has_motion, has_structure)
- [x] frozen/duplicate-frame clip → FAILS (has_motion, has_structure)
- [x] NaN render → FAILS (not_black + cascade)
- [x] neon under-denoised noise → FAILS (has_structure, not_pure_noise) — sat 0.98
- [x] (control) good_clip + vivid_correct → PASS. Gate proven trustworthy → all PASS verdicts credible.

## Tier-3 feasibility spikes (DONE 2026-06-20)
- [x] **ti2v_5b — FEASIBLE (low risk).** `Wan-AI/Wan2.2-TI2V-5B-Diffusers` exists as diffusers
      `WanPipeline` (dual transformer+transformer_2, boundary_ratio/expand_timesteps, OWN vae
      16×16×4). Registry `diffusers_class=None` is STALE → set to WanPipeline. Needs its own VAE
      (not shared). New ti2v.py handler.
- [x] **animate_14b — FEASIBLE (medium).** `Wan-AI/Wan2.2-Animate-14B-Diffusers` = diffusers
      `WanAnimatePipeline`. __call__ takes PRE-EXTRACTED pose_video/face_video/background_video/
      mask_video → the ViTPose/YOLO/SAM2 CV chain is DECOUPLED preprocessing (CPU or synthetic),
      NOT part of the diffusion. Diffusion runs on MPS like the others. Component a/b/c/d:
      WanAnimatePipeline=(a) MPS; ViTPose/YOLO/SAM2=(c) CPU preprocess stages producing video
      inputs; none are (d) hard blockers. New animate.py handler + a preproc/synthetic-control step.
- [ ] **s2v_14b — HARDEST.** `Wan-AI/Wan2.2-S2V-14B` has NO diffusers model_index (404). Needs
      upstream `wan` package + audio (wav2vec2). MPS coverage unknown — spike the `wan` install next.

## Findings / lessons (durable)
- **Local MPS + bf16 WORKS.** Wan 2.1 1.3B produces clean, coherent video on MPS in bf16.
  Backend dtype switched float16→bf16 (utils/backend.py, env WAN_STUDIO_MPS_DTYPE).
- **Saturation is NOT a coherence signal.** A vivid-correct frame (orange panda on green,
  sat ~0.86) out-saturates a broken one (psychedelic q30, sat ~0.74). The metric gate was a
  false-positive factory; relaxed to catch only pure-noise (≥0.93). VISUAL inspection of
  start/mid/end PNGs is the real arbiter — eyeball every mode, never trust the number alone.
- **Eliminated as causes of the "neon":** dtype (bf16==fp32), MPS SDPA (manual softmax bypass
  identical), VAE round-trip (clean), text-encoder (finite emb), scheduler, transformer config,
  mirror weights (healthy), RoPE strided-assign, negative prompt (WAN_NEG==empty both clean).
- **Real residual:** frame-count artifact (13 clean, 17/25 broken) — VAE 3D-conv or temporal
  RoPE at >4 latent frames suspected. f49 (12 latent frames) test in progress.
- **hf_xet installed** for fast/large HF downloads (was missing → LFS path slow + lock stalls).
- **_lora_repo_for** fixed for local: consolidated mirror repo (techfreakworm/wan-lightning-loras)
  resolves weight_name subpaths locally.
