# Wan Studio — All-Modes Local MPS Bring-Up (Design Spec)

**Date:** 2026-06-21
**Author:** Mayank (via Claude Code lead + wan-brain proxy review)
**Status:** Awaiting wan-brain sign-off → writing-plans → execute

## 1. Goal & Hard Constraints

Make **all 13 Wan modes** produce real, frame-verified video **locally on the M5 Max
(128 GB unified, MPS)**, exercised end-to-end **through the Gradio app and confirmed via
Playwright MCP**. Autonomous, do-not-stop.

Hard constraints (from Mayank's goal):
- **No OOM, ever.** memcheck calculation BEFORE every test. The OS-panic vector (parallel
  heavy MPS jobs / oversized VAE decode) is real and has restarted this Mac before.
- **Local only — DO NOT PUSH.** Convert fp32→bf16 into `~/wan-bf16/<slug>/` (outside HF
  cache). Pre-stage `_push_map.json` (records intent only). Never push without explicit OK.
- **Disk:** if space runs short, clear non-Wan HF cache (Mayank will re-download). If still
  short → **STOP immediately**.
- **Verification:** an mp4 is not enough. Extract start/mid/end frames via ffmpeg, measure
  quality metrics, **and eyeball** + verify the output actually honors its conditioning input.
- **codex CLI** for any input images needed (i2v/flf2v/animate/vace-ref/ti2v init).
- Design brainstormed and **approved by wan-brain** before implementing.

## 2. Verified Current State (2026-06-21)

- **Env:** torch 2.11.0/MPS, diffusers 0.38.0, transformers 5.9.0, 137 GB RAM. Disk: 176 GB
  free; HF cache 483 GB (~350 GB removable non-Wan).
- **diffusers Wan classes present:** WanPipeline, WanImageToVideoPipeline, WanVACEPipeline,
  WanVideoToVideoPipeline, WanAnimatePipeline, AutoencoderKLWan. **Absent: any S2V pipeline.**
- **Handlers registered:** t2v, i2v, flf2v, v2v, vace. **Missing: ti2v, s2v, animate.**
- **bf16 weights ready (cached):** t2v_1.3b, t2v_14b, shared-encoders (text_encoder+vae only).
  v2v reuses t2v_14b. **Lightning local:** wan2.1-t2v-14b (consolidated repo);
  lightx2v/Wan2.2-Lightning (MoE high/low) cached.
- **Scheduler fix already shipped** (commit 935847b): FlowMatchEuler (not UniPC), preset-aware
  shift, `enable_lora()` before `set_adapters()`. Lifted the old ≤13f scare; 14B Lightning
  @17f previously PASS.
- **memcheck.py** covers all 13 keys; `assert_safe` HARD-gates on the conservative static
  `estimate_peak_gb` (cap 92.4 GB) + a reclaimable-free check; `SerialLock` present.

### Confirmed blockers (must fix before the affected mode)
- **B1 — CLIP image encoder missing.** `wan-shared-encoders` has only `text_encoder/`+`vae/`;
  `shared.image_encoder()` loads `CLIPVisionModel` from `subfolder="image_encoder"` which does
  not exist. **Every image mode (i2v/flf2v/animate/s2v) fails to load until fixed.**
- **B2 — ti2v_5b registry stale.** `diffusers_class=None` → `conversion_plan()` returns None →
  conversion skips it. Must set `diffusers_class="WanPipeline"` and `repo` to the `-Diffusers`
  id. ti2v has its **own 16×16×4 VAE** (not the shared Wan VAE) → new `ti2v.py` loads that VAE
  from the ti2v repo; memcheck generic `_VAE_GB=0.5` may need calibration for it.
- **B3 — convert script is unguarded.** `convert_local_bf16.py` loads a full fp32 transformer
  into RAM but does **not** take `SerialLock` or do a RAM preflight → a conversion run
  concurrent with a generation is exactly the parallel-heavy-job panic vector.

## 3. Architecture — the per-mode spine

Every mode goes through the same pipeline (differences captured per-row in §8):

1. **Pre-reqs** — registry/handler/wiring fixes for this mode (one-time).
2. **Acquire weights** — if bf16 mirror not in `~/wan-bf16/`, download fp32 → convert bf16
   (`convert_local_bf16.py --only KEY --purge-fp32`, now guarded per B3) → purge fp32.
   Fetch Lightning LoRA if the mode needs Fast preset.
3. **memcheck preflight** — `assert_safe(KEY, frames, res)` must pass; pick the **largest
   frame count the static gate permits** (§4). Re-check free disk + free RAM here too.
4. **Generate** — via `scripts/local_verify.py --mode KEY ...` under `SerialLock`, on the
   FlowMatchEuler path, at the chosen preset (Quality for non-MoE proof; Lightning for MoE).
   Save via `pipelines/video_io.save_video` (never `export_to_video`).
5. **Verify** — ffmpeg start/mid/end PNGs + quality metrics + **eyeball** + **conditioning
   adherence** (§6).
6. **Ledger** — update `ACCEPTANCE_LEDGER.md` (with the 2 new columns).
7. **In-app** — drive the live Gradio app via Playwright MCP, run the mode, screenshot the
   produced video in-UI.

## 4. Memory safety (THE hard gate) — run *within* the existing gate

Decision: **do not loosen the safety gate.** `assert_safe` (static `estimate_peak_gb` ≤ 92.4 GB)
stays the authority. Each mode runs at the **largest frame count that passes the static gate**;
no risky re-calibration is required to get every mode working. Per-mode first-run targets
(static peak in GB shown; all ≤ 92.4):

| Model class | Res | First-run frames | static peak | Notes |
|---|---|---|---|---|
| 1.3B (t2v_1.3b, vace_1.3b) | 480p | 25f (7 latent) | ~73 | t2v_1.3b already DONE @13f |
| 14B dense @480p (t2v_14b, v2v_14b, i2v_480p, vace_14b, s2v_14b) | 480p | 17f (5 latent) | ~70–72 | 25f=98 → refused |
| 14B @720p (i2v_720p, flf2v_720p, animate_14b) | 720p | 13f (4 latent) | ~86 | 17f=111 → refused; 720p is danger zone #2 |
| A14B MoE (t2v_a14b, i2v_a14b) | 480p | **9f → measure → 13f** | 87 / 89 | 17f=100 → refused; ≤13f hard ceiling; danger zone #1 |
| ti2v_5b | 720p | 13f (4 latent) | ~68 | own VAE; recalibrate memcheck after first measure |

**Calibration loop (sanctioned):** after each first successful run, record the **measured**
driver + real peak; only authorize a higher frame count if `measured × 1.3 < HARD_CAP`. The
HARD ceiling is never raised. Note: the app.py runtime guard may gate on the driver model
(720p driver ≈138 GB > 128) — verify at the i2v_720p step that the in-app guard permits 13f;
if it blocks a genuinely-safe run, switch the **app guard** (not assert_safe) to the
vm_stat-calibrated REAL model. Real memory for these runs is ~68–89 GB — safe on 128 GB.

## 5. Concurrency model (under SerialLock)

- **Strictly serial, one at a time, all under `SerialLock`:** any generation, any **conversion**
  (loads full fp32 into RAM), any VAE-decode/probe, any `local_verify` run.
- **May overlap a running serial job:** pure **download** (prefetch next model's fp32 — streams
  to disk, low RAM) and **purge**. Guardrails: prefetch must not fill disk mid-gen, and must not
  be so large it evicts the page cache the live gen needs.
- **B3 fix is prerequisite:** wrap conversion in `SerialLock` + free-RAM preflight before any
  overlap is allowed. Default safe mode = fully serial; download-overlap is the opt-in speedup.

## 6. Acceptance bar — DONE = all five

1. memcheck-safe and ran **without OOM**.
2. Metrics PASS (not_black, has_contrast, has_detail, has_motion, not_static, not_pure_noise,
   no_nan).
3. **Eyeball** start/mid/end frames → coherent scene (the real arbiter; saturation lies both
   ways).
4. **Conditioning adherence** — cheap SSIM/MSE-vs-input or eyeball: i2v frame[0]≈input image;
   flf2v frame[0]≈first & frame[-1]≈last; v2v follows source structure/motion; vace follows
   control; ti2v honors text+init; animate follows driving pose/face; s2v tracks audio.
5. **In-app via Playwright** — run the mode in the live Gradio app, screenshot the produced
   video in-UI.

Add two columns to `ACCEPTANCE_LEDGER.md`: **"conditioning adheres"** and **"in-app (Playwright)"**.
`local_verify` alone = "inference-verified", NOT done.

## 7. Disk & weight-source strategy

- **Clear non-Wan HF cache up front (mandatory).** Worst transient = one MoE fp32 (~125 GB) +
  its bf16 output (~56 GB) ≈ 181 GB > 176 GB free. Whole-campaign kept bf16 ≈ 334 GB + 125 GB
  transient ≈ 459 GB. **Target ≥ 470 GB free before starting.** Clear the big non-Wan repos
  (FLUX.2 60G, LTX 45+44+39G, flux2 33G, FLUX.1 22G, Qwen 16G, z_image 11+7.5G, …). Keep
  `~/wan-bf16` (outside cache). Re-check free disk in every per-mode memcheck step. Budget extra
  for S2V upstream `wan` weights (not in the diffusers accounting).
- **MoE weight source = official fp32 diffusers → convert → purge.** Do NOT use GGUF (quantized;
  FP8/quant crashes Metal) or ComfyUI single-file (needs ComfyUI→diffusers key remap;
  silent-garbage risk) — ComfyUI is plan-B only if a diffusers download proves infeasible.
- **MoE sequencing (disk-safe):** clear → convert t2v_a14b (transformer then transformer_2,
  free each after save) → purge its fp32 → THEN i2v_a14b. **Never both MoE fp32 on disk at once.**

## 8. Per-mode work plan (execution order)

Order (de-risks new-handler pattern early; danger zones late):
`t2v_14b → v2v_14b → i2v_480p → vace_1.3b → ti2v_5b → vace_14b → i2v_720p → flf2v_720p →
t2v_a14b → i2v_a14b → animate_14b → s2v_14b (spike, last)`. (t2v_1.3b already DONE.)

| # | Mode | New code | Weights | Lightning | Frames | Special |
|---|------|----------|---------|-----------|--------|---------|
| 1 | t2v_14b | — | bf16 cached | t2v-14b (local) | 17f/480p | re-confirm prod path |
| 2 | v2v_14b | — | reuse #1 | — (Quality) | 17f/480p | needs source clip (dogfood a t2v output) |
| 3 | i2v_14b_480p | — | dl→bf16 | optional | 17f/480p | **B1: acquire CLIP image_encoder first**; codex input image |
| 4 | vace_1.3b | — | dl→bf16 | — (Quality) | 25f/480p | reference + control inputs |
| 5 | ti2v_5b | **ti2v.py** + B2 fix | dl→bf16 | — (Quality) | 13f/720p | own 16×16×4 VAE; text+init image |
| 6 | vace_14b | — | dl→bf16 | — (Quality) | 17f/480p | |
| 7 | i2v_14b_720p | — | dl→bf16 | optional | 13f/720p | app-guard check (§4) |
| 8 | flf2v_14b_720p | — | dl→bf16 | optional (Kijai) | 13f/720p | first+last frame inputs |
| 9 | t2v_a14b | — | dl→bf16 (MoE) | lightx2v Wan2.2 high/low | 9f→13f/480p | MoE; load_into_transformer_2 |
| 10 | i2v_a14b | — | dl→bf16 (MoE) | lightx2v Wan2.2 high/low | 9f→13f/480p | MoE + CLIP; worst memory profile |
| 11 | animate_14b | **animate.py** | dl→bf16 (keeps image_encoder) | — | 13f/720p | pre-extracted pose/face/bg/mask (CPU/synthetic preproc) |
| 12 | s2v_14b | **s2v.py** | dl→bf16 (vendored) | — | 17f/480p | §9 feasibility ladder |

## 9. S2V feasibility ladder (the crux — time-boxed, carved out)

S2V is **not in diffusers**; upstream `wan` uses flash_attn (CUDA) + wav2vec2. But it has an
SDPA fallback (`--attention sdpa`, auto for pre-RTX40xx) and `requirements_s2v.txt` has **no
hard-CUDA deps**. Ladder, most→least likely to satisfy "in-app via Gradio":

1. **Upstream `wan` on MPS — SPIKE FIRST (~½ day box).** Gating question: does
   `wan/modules/attention.py` hard-import flash_attn or guard it with an SDPA fallback /
   `--attn` selector? If a non-flash path exists → device cuda→mps, wav2vec2 on CPU/MPS, reuse
   our bf16 transformer + shared VAE.
2. **Manual diffusers-style S2V loop** — port the S2V transformer forward + audio cross-attn
   adapter onto AutoencoderKLWan + FlowMatchEuler. Real fallback if upstream is CUDA-locked.
3. **CPU execution** of upstream (no flash_attn on CPU) — proves the mode; 14B-on-CPU is brutally
   slow; last-resort "it runs".
4. **ComfyUI-MPS** — only to PROVE the weights run on Metal (de-risks path 2); does NOT satisfy
   Playwright-in-Gradio, so not the deliverable.

**Carve-out:** S2V must not block the other 12. If 1–3 all fail in-app within the time-box,
deliver S2V as "weights+wav2vec verified on Metal via reference; in-app integration BLOCKED on
flash_attn/CUDA coupling" and escalate to Mayank for a scope call.

## 10. Prerequisite fixes (blocking, do before affected modes)

- **B1:** acquire CLIP-ViT-H image_encoder into the local shared path (from a Wan I2V
  `-Diffusers` repo's `image_encoder/`); ensure `shared.image_encoder()` resolves it locally.
  Before mode #3.
- **B2:** registry ti2v_5b `diffusers_class="WanPipeline"`, `repo` → `-Diffusers` id; new
  `ti2v.py` loads the ti2v repo's own VAE; calibrate memcheck VAE term. Before mode #5.
- **B3:** guard `convert_local_bf16.py` with `SerialLock` + free-RAM preflight. Before any
  conversion.
- New handler modules `ti2v.py`, `animate.py`, `s2v.py` + `register()` + `app.py`
  (_MODE_RUNNERS/HANDLER_REGISTRY) + `local_verify` wiring as their modes come up.

## 11. wan-brain decision ledger (rulings folded in)

1. Push: **local only** (two emphatic DO-NOT-PUSH override the lone push line; reversible).
2. Disk: **clear non-Wan up front, target ≥470 GB**; surgical clearing risks stopping mid-run.
3. MoE source: **official fp32 diffusers → convert → purge**; no GGUF; ComfyUI plan-B only.
4. S2V: **time-boxed spike + carve-out** (§9).
5. A14B: **480p only, Lightning, 9f→measure→13f**, ≤13f hard ceiling, 720p MoE out.
6. Concurrency: **compute strictly serial; only download/purge may overlap**; fix B3 first.
7. Acceptance: **5 conditions** incl. conditioning-adherence + in-app Playwright (§6).
8. memcheck gate: **run within the existing static gate** (it already permits every mode at a
   low frame count); calibrate upward only from measured peaks.

## 12. Risks & open items

- **A14B MoE** is the OS-panic class (≥56 GB resident + CLIP, no offload). Start 9f, measure,
  never parallel. Highest care.
- **720p trio** is danger zone #2; expect static-gate refusals above 13f (that's the gate
  working).
- **S2V** may not reach the in-app bar on MPS (flash_attn coupling) — known risk, carved out.
- **app.py runtime guard** may over-refuse 720p (driver model) — switch that guard (not
  assert_safe) to the REAL model if it blocks a safe run.
- **Conditioning false-pass** — a coherent video that ignores its input is the new "saturation
  trap"; adherence check is mandatory.
