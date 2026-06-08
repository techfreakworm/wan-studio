# Wan Studio — Program Architecture & Risk Register (all 12 modes, all phases)

| Field | Value |
|---|---|
| **Date** | 2026-06-08 |
| **Author** | Mayank Gupta |
| **Status** | For review — umbrella doc above the per-phase specs/plans |
| **Purpose** | Capture cross-cutting architecture + every known risk across the whole program **up front**, so phased implementation hits no architectural surprises. |
| **Companion** | [`#0 foundation spec`](./2026-06-08-wan-studio-storage-latency-foundation-design.md) · [`original design`](./2026-05-21-wan-studio-design.md) · [`RESEARCH.md`](../../../RESEARCH.md) |
| **Method** | Synthesized from 7 parallel architecture+risk analyses (per-mode + cross-cutting + execution), each grounded in code + diffusers/`wan` source + mid-2026 web docs. |

> **How to read this:** §1 is the phase map. §2 is the per-mode architecture table. §3 is the cross-cutting decisions that bind all phases. **§4 is the consolidated risk register — the centrepiece.** §5 is the parallel-execution plan. §6 lists empirical spikes that must run before committing. §7 lists amendments this analysis forces onto the already-approved #0 spec.

---

## 1. Program overview & phase map

Goal: ONE HF ZeroGPU PRO Space serving all 12 Wan checkpoints e2e. The dependency graph is a **strict diamond** — #0 is the single serial gate; once frozen, the mode phases are largely independent; #4 integrates last.

```
                 ┌─────────────────────────── #0 FOUNDATION (serial gate) ───────────────────────────┐
                 │ bf16 mirrors · mounts · context-path · LRU · shared encoders · tier-2 · HANDLER_REGISTRY │
                 └───────────────┬───────────────┬───────────────┬───────────────┬─────────────────────┘
                                 │               │               │               │
                      ┌──────────▼──┐   ┌────────▼────────┐  ┌───▼──────────┐  ┌─▼────────────┐
                      │ #0.5 AOTI   │   │ #1 diffusers    │  │ #2 Animate   │  │ #3 vendored  │   (parallel
                      │ (compile)   │   │ VACE/FLF2V/V2V  │  │ (diffusers)  │  │ S2V/TI2V-5B  │    after #0)
                      └─────────────┘   └────────┬────────┘  └──────┬───────┘  └──────┬───────┘
                                                 └────────────┬─────┴─────────────────┘
                                                        ┌─────▼─────┐
                                                        │ #4 polish │  Send-to · Gallery · Settings (LAST)
                                                        └───────────┘
```

**Status reality check:** T2V + I2V are the only modes with working handlers today. "#2 Animate done" from the analysis means *its architecture was analysed* — there is **no `pipelines/animate.py`** in the repo. All of #1/#2/#3/#4 are unbuilt.

---

## 2. Per-mode architecture (all 12 checkpoints)

| Mode / checkpoint | Pipeline | Phase | Vendored? | Preproc | ZeroGPU tier | bf16 mirror | Net-new code |
|---|---|---|---|---|---|---|---|
| **T2V** 2.1-1.3B/14B, 2.2-A14B(MoE) | `WanPipeline` | wired | diffusers | none | large / **xlarge**(MoE) | ✅ | — (done) |
| **I2V** 2.1-480p/720p, 2.2-A14B(MoE) | `WanImageToVideoPipeline` | wired | diffusers | CLIP | large / **xlarge**(MoE) | ✅ | — (done) |
| **VACE** 2.1-1.3B/14B | `WanVACEPipeline` ✅verified | #1 | diffusers | DWPose+MiDaS+RAFT (~1GB) | large | ✅ | VACEHandle, video→PIL decode, mask builder, 3 annotator wrappers |
| **FLF2V** 2.1-14B | `WanImageToVideoPipeline`+`last_image=` ✅verified | #1 | diffusers | CLIP + center_crop | large | ✅ | subclass I2VHandle, end-frame T2I |
| **V2V** on 2.1-T2V-14B | `WanVideoToVideoPipeline` ✅**exists** (no workaround) | #1 | diffusers | `load_video` decode | large | ✅(shared mount) | V2VHandle, **registry card missing** |
| **Animate** 2.2-14B | `WanAnimatePipeline` ✅verified v0.36 | #2 | diffusers (GPU) + **vendored preproc** | ViTPose-H + YOLOv10m + SAM2 (~2GB) | **xlarge** | ✅ (keep image_processor/) | AnimateHandle, CPU preproc module, 5-output wiring |
| **TI2V-5B** 2.2 | `wan.WanTI2V` (vendored) | #3 | **VENDORED** | none, **own 16×16×4 VAE** | large | ⚠️ vendored-format | VendoredWanHandle, wan vendoring |
| **S2V-14B** 2.2 | `wan.WanS2V` (vendored) | #3 | **VENDORED** | wav2vec2 (rides S2V mount) | large(tight)/xlarge | ⚠️ vendored-format | VendoredWanHandle, audio path |

Key verified facts: VACE/FLF2V/Animate are 100% diffusers-native (no vendoring). **V2V's `WanVideoToVideoPipeline` is a real shipped class** — the feared custom workaround is unnecessary. S2V/TI2V-5B are `diffusers_class=None` → genuinely require vendoring the upstream `wan` package.

---

## 3. Cross-cutting architecture decisions

### 3.1 Freeze the #0 contract first (the serial gate)
Every handler calls `registry.acquire(key)`, reuses `shared.{vae,text_encoder,image_encoder}()`, and resolves `_mount_path(card)`. Until that interface is merged + tagged (`v0-foundation`), no phase branches. The immutable interface = `WanModelHandle` subclass contract (`_build_pipeline`/`generate`) + `ModelRegistry.acquire` signature + `ModelCard.mirror_repo`.

### 3.2 HANDLER_REGISTRY plugin pattern (enables parallel teams)
Four files are edited by **every** phase: `pipelines/registry.py`, `pipelines/__init__.py`, `app.py`, `requirements.txt`. That shared-file surface — not pipeline logic — is the entire merge-conflict story. **Decision:** introduce a `HANDLER_REGISTRY: dict[mode, HandlerSpec]` that each `pipelines/<mode>.py` registers into; `__init__.py` and the `app.py` Generate-wiring loop iterate the registry instead of hard-coding per-mode `_build_X_handler()` + `.click()` blocks. Each phase then adds **one new file + one append-only stanza**, not edits to a shared dispatch block. This refactor lands as the last step of #0 (or a thin pre-#1 step).

### 3.3 Vendored `wan` integration (#3) — the program's #1 risk
S2V/TI2V are not in diffusers; they need Alibaba's upstream `wan` package whose `generate()` bypasses `from_pretrained`. Its `requirements.txt` pins **`transformers>=4.49.0,<=4.51.3`, `numpy<2`, `flash_attn`** — colliding head-on with our `transformers 5.9 / numpy 2 / no-flash-attn` stack. **You cannot `pip install` it into the shared env.** Decision — **vendor-narrow**:
1. `git clone Wan-Video/Wan2.2` at a pinned SHA into `third_party/wan2.2/` (submodule or vendored), add only `wan/` to `sys.path`, import **only** the S2V/TI2V model + VAE + audio modules — never its diffusers/transformers shims. Install with `--no-deps`.
2. Monkeypatch `flash_attn`→`torch.nn.functional.scaled_dot_product_attention` (the wan code has a fallback branch; force it). Mandatory on ZeroGPU Blackwell *and* MPS.
3. Run against the installed transformers 5.9 / numpy 2 and **shim the 2–3 broken call sites** (wav2vec2 + UMT5 tokenizer loaders).
4. **Fallback if shimming is too invasive:** subprocess isolation — a separate venv with the old pins, invoked from inside `@spaces.GPU`, passing `checkpoint_dir` + inputs over disk. Heavier (loses CoW preload + in-process LRU, eats cold-import time) but fully decouples the conflict.
5. A `VendoredWanHandle(WanModelHandle)` keeps app.py orchestration uniform but overrides `_build_pipeline` to construct `wan.WanS2V/WanTI2V(checkpoint_dir=_mount_path(card))`; it reuses #0's mount-resolution (must return a **directory**, not a repo id), LRU, tier-2 cache, and `@spaces.GPU` wiring — but **not** shared-encoder injection (the wan model builds its own UMT5/VAE/wav2vec2).

> The exact upper-bound (`<=4.51.3`) vs loose-lower-bound reading differed between analyses → treated as **High/uncertain**; the §6 import-smoke spike decides vendor-narrow-shim vs subprocess.

### 3.4 Mounts: one atomic manifest + cap fallback
`set_space_volumes` **replaces the entire volume set atomically** — a per-phase mount edit silently unmounts everything else. **Decision:** keep the full manifest as one source-of-truth list in `create_space.py`; always re-pass the complete set; add a boot probe asserting every `/models/<slug>` exists. The **per-Space mount cap is undocumented**; we want ~15 mounts (12 model + shared + preproc + loras). If capped → consolidate related checkpoints into fewer repos with per-checkpoint subfolders (collapses 12 model mounts toward ~3–4). **Must be probed on staging before finalizing (§6).**

### 3.5 `@spaces.GPU` size is a static literal → one handler per tier
Size cannot be a callable (serializes → HTTP 422). T2V/I2V-MoE, Animate, and possibly S2V want `xlarge`; the rest want `large`. **Decision:** two statically-decorated entrypoints (`generate_large`, `generate_xlarge`); route by mode via `MODE_BUDGET`. `duration` stays a callable.

### 3.6 `shared.py` extensions (additive, land in #0)
- Add `shared.image_processor()` (`CLIPImageProcessor`) — `WanAnimatePipeline` needs it and shared.py currently injects vae/text_encoder/image_encoder only.
- **TI2V-5B needs the Wan 2.2 16×16×4 VAE**, different from the shared 8×8×4 VAE → never inject `shared.vae()` into TI2V; the vendored model loads its own (auto-avoided since it bypasses injection).

### 3.7 Preproc provisioning (corrected)
The Diffusers Animate mirror **does not** contain `process_checkpoint/` (verified file listing) — spec §9.7's "no separate preload" is **wrong**. Provisioning reality:
- **VACE**: DWPose+MiDaS+RAFT (~1GB) → dedicated `techfreakworm/wan-preproc` mirror.
- **Animate**: ViTPose-H + YOLOv10m + SAM2 (~2GB) → **must also go into `wan-preproc`** (not bundled in the Diffusers mirror).
- **S2V**: wav2vec2 (~1.26GB) → **rides the S2V as-is mount** (it's inside the original repo, and S2V is mirrored as-is).
- All preproc is ONNX/CPU; runs in the main process **before** `@spaces.GPU` so it doesn't burn GPU duration.

### 3.8 Gallery/Settings persistence (#4) needs a read-write Storage Bucket
`/tmp` is ephemeral, mounts are read-only. #4 attaches a read-write **Storage Bucket** (Xet/S3, ~$4–5/mo for 300GB, survives restart) for gallery outputs + settings JSON. Not provisioned by #0.

---

## 4. Consolidated risk register

Severity × phase it bites × mitigation. **Critical/High first.**

| # | Sev | Phase | Risk | Mitigation |
|---|---|---|---|---|
| R1 | **Critical** | #3 | upstream `wan` pins `transformers≤4.51.3` / `numpy<2` / `flash_attn` — incompatible with our 5.9 / numpy 2 / no-flash | vendor-narrow + `--no-deps`, monkeypatch flash_attn→SDPA, shim wav2vec2/UMT5 call sites; **subprocess-isolated old-pin venv as fallback** (§3.3) |
| R2 | **High** | #0 | #0 contract not frozen before teams branch → cascading rebase churn | merge + tag `v0-foundation`; publish the handle/registry interface as immutable before any phase starts |
| R3 | **High** | all | 4 shared files (registry/`__init__`/app.py/requirements) edited by every team → merge hell | HANDLER_REGISTRY plugin pattern (§3.2); per-mode files; one integrator merges in order |
| R4 | **High** | #0/all | `set_space_volumes` atomic-replace → a per-phase mount edit unmounts everything | single source-of-truth manifest, always re-pass full set, boot probe asserts each mount |
| R5 | **High** | #0/#1/#3 | undocumented per-Space mount cap vs ~15 desired mounts | probe on staging (§6); fallback = consolidate into multi-subfolder repos |
| R6 | **High** | #2/#3 | `@spaces.GPU` size must be static literal; Animate/S2V want xlarge, rest large | one decorated handler per tier; route via MODE_BUDGET; duration stays callable |
| R7 | **High** | #2 | Animate preproc weights are **NOT** in the Diffusers mirror (spec §9.7 wrong) | provision ViTPose/YOLO/SAM2 into `wan-preproc`; fail-loud assert on ZeroGPU; local snapshot_download fallback |
| R8 | **High** | #2 | diffusers ships no Animate pose/face/SAM2 preproc (only accepts pre-extracted `List[PIL]`) | vendor the narrow upstream `wan/modules/animate/preprocess` CPU code into `pipelines/animate_preprocess/`; raw onnxruntime, no mmpose/ultralytics |
| R8b | **High** | #2 | Animate preproc CPU RAM/OOM (SAM2+ViTPose+frames + CoW-resident bf16 transformer) | sequential del+gc per preproc model; cap `motion_encode_batch_size`, reduce-and-retry-once; frames→/tmp not RAM; `torch.no_grad()` |
| R9 | **High** | #0/#2 | transformer-only bf16 strip would drop `image_processor/`/`scheduler/`/`tokenizer/` that Animate (and maybe VACE) `from_pretrained` needs | keep those subfolders for Animate's mirror; add `shared.image_processor()`; smoke-test each newer pipeline from a transformer-only stitched dir before mass conversion |
| R10 | **High** | #3 | S2V-14B ~43GB tight on 48GB large at 720p | vendored model's native `offload_model=True`/`convert_model_dtype` (not accelerate); or xlarge |
| R11 | **High** | #3 | TI2V-5B needs its own 16×16×4 VAE; `shared.vae()` is an lru_cache singleton | never inject shared.vae() into TI2V; let the vendored model load its own |
| R12 | Med | #3 | wan code written for transformers 4.x may call removed APIs under 5.9 | import smoke-test spike (§6); monkeypatch broken call sites; keep S2V/TI2V fp32-mirrored until verified |
| R13 | Med | #3 | vendored checkpoints load via wan's own state_dict reader → #0's diffusers `convert_to_bf16.py` doesn't cover them | #3 ships a vendored-path bf16 converter or load-time `.to(bf16)`; validate the wan loader tolerates a bf16 state_dict before mass-convert |
| R14 | Med | #1 | no `gr.Video→list[PIL]` decoder exists (T2V/I2V handle single images); VACE/V2V need it | add a `utils` decoder (`diffusers.utils.load_video`/`av`), resize to vae_scale×patch multiple |
| R15 | Med | #1 | **no `v2v` ModelCard exists** in registry → KeyError | add `wan2.1_v2v_14b` card (repo = T2V-14B, `WanVideoToVideoPipeline`, Quality-only, flow_shift 5.0/3.0) + budget entry, mapped to the shared t2v-14b mount |
| R16 | Med | #1 | VACE `video`/`mask` are full-length `list[PIL]`; length/size mismatch corrupts output | reuse the `prepare_video_and_mask` gray-fill helper from the diffusers docstring; assert `len(video)==len(mask)==num_frames` |
| R17 | Med | #1 | VACE deferred sub-modes (Track/Label/Caption/Animate-Anything, Reference-Face) need SAM2/GroundingDINO/InsightFace — out of v1 | gate those mask-sources: require user-uploaded pre-extracted control video + info banner, else clear `gr.Error`; ship Depth/Pose/Sketch/Flow/Inpaint/Outpaint/Reference-Object/Extension |
| R18 | Med | all | app.py replaces 6 stubbed Generate clicks via a toast loop with inputs=None — new handlers must REPLACE not append the `.click()` | delete each tab_key from the loop as its phase lands; bind one fresh `.click()` |
| R19 | Med | #2/#4 | Animate handler returns a **5-tuple** (video + pose/face/bg/mask previews); T2V/I2V template returns one path | dedicated `outputs=[...]` of 5; None placeholders for animate/retarget modes (bg/mask only in replace) |
| R20 | Med | #2 | spaces 0.50.2 — does running heavy CPU preproc in the main handler then invoking a separate `@spaces.GPU` fn keep preproc time off the GPU duration budget? | two-phase handler probe on the real ZeroGPU box (§6) |
| R21 | Med | #1 | FLF2V end-frame "Generate" loads a 2nd 14B transformer (T2V) inside an FLF2V session → fights the ≤1-warm LRU | run T2I as a short load→gen→unload `@spaces.GPU`, or pre-generate end frame; or use 1.3B for the T2I |
| R22 | Med | #2/#3 | tier-2 /tmp hot-copy (~56GB MoE / 14B + bundled preproc) can crowd 150GB | LRU-evict prior hot copy on switch, cap to one model, skip-on-ENOSPC; exclude bundled preproc from the copy |
| R23 | Med | #4 | Gallery/Settings can't persist on ephemeral /tmp + read-only mounts | attach a read-write Storage Bucket (§3.8); net-new in #4 |
| R24 | Med | #1 | VACE-14B at 480p needs flow_shift=3.0 but card hardcodes 5.0 | pick flow_shift from (card, chosen resolution), not blindly from `card.flow_shift` |
| R25 | Low | #1 | VACE-14B ≈44GB "very tight" on 48GB large at 720p/81f | keep VAE tiling/slicing (already on), cap default frames, OOM→gr.Error; consider 480p cap or xlarge for 14B@720p |
| R26 | Low | #1 | FLF2V Lightning is empirical (reuses I2V LoRA, not trained for FLF2V) | label Fast "Beta" in UI; default Quality CFG 5.5 (advanced cfg defaults to 1.0 — must override) |
| R27 | Low | #3 | HF Volume small-JSON truncation may hit the vendored loader's (non-diffusers) config files | verify which small files the wan loader reads; extend #0 stitch/models_meta if hit |
| R28 | Low | #3 | S2V MoE-vs-dense ambiguous | inspect S2V `model_index.json` for `transformer_2`; assume single-dense |
| R29 | Low | #0.5 | AOTI artifacts wrap `_build_pipeline`; divergent handler shapes break uniform caching | keep all `_build_pipeline` signatures identical; land #0.5 after #1/#3 to cover all modes |
| R30 | Low | #3 | optional CosyVoice TTS bundle (hydra/lightning/modelscope/decord/onnxruntime) is a 2nd dep-conflict surface | exclude TTS from v1; require user-supplied audio for S2V |

---

## 5. Execution & parallelization plan

**Sequence:** `#0 (serial) → freeze v0-foundation → [#0.5, #1, #2, #3] in parallel → #4 last`.

**Recommended teams (git-worktree isolated, after #0 freeze):**
- **Team-A / #1 diffusers** — owns `pipelines/{vace,flf2v,v2v}.py` + the video decode helpers + VACE preproc wrappers. Split internally: **#1a** FLF2V+V2V (trivial I2VHandle/WanVideoToVideoPipeline reuse, ~half session) and **#1b** VACE (9 sub-modes + 3 preproc + conditional UI — full session).
- **Team-B / #3 vendored** — owns `pipelines/{s2v,ti2v}.py` + `third_party/wan2.2/` vendoring + dep-pinning + vendored bf16 conversion. **Own worktree AND own throwaway venv** for the dep-conflict spike. Highest isolation + debugging load.
- **Team-C / #2 Animate** — owns `pipelines/animate.py` (GPU) + `pipelines/animate_preprocess/` (CPU). Can run against a stub preproc returning fixture pose/face frames while the real preproc is built.
- **Team-D / #0.5 + #4-scaffold** — AOTI wrap + Gallery/Settings UI scaffolding against stubs. Send-to chip wiring integrates LAST.

**Integration:** one integrator merges in order **#1 → #2 → #3 → #4**, resolving the 4 shared files (which the HANDLER_REGISTRY pattern reduces to append-only stanzas). Merge #3 early-ish if you want dep conflicts surfaced sooner.

**Where context-exhaustion is real:** #1b VACE and #3 vendored (package + transformers-5.x debugging + bf16 conversion) — give each a dedicated session. **Not real:** #1a, #2-GPU-half, #4, #0.5.

**Each phase = its own spec → plan → execute cycle.** This umbrella doc is the shared risk/architecture reference they all cite.

---

## 6. Empirical spikes (must run before committing the design they gate)

1. **Mount cap** — mount all ~15 volumes on `wan-studio-staging`, observe (gates §3.4 consolidation).
2. **Vendored import smoke** — in an isolated venv, `import` the narrow `wan` S2V/TI2V modules under transformers 5.9 + numpy 2 + no-flash; record exact breakages (decides vendor-narrow-shim vs subprocess, R1/R12).
3. **Transformer-only tolerance** — `from_pretrained` each newer class (VACE, Animate, V2V) from a transformer-only stitched dir with injected kwargs (gates the bf16 mass-conversion + R9).
4. **spaces 0.50.2 CPU-then-GPU** — confirm CPU preproc in the main handler then a separate `@spaces.GPU` fn keeps preproc off the duration budget (R20).
5. **Vendored bf16 round-trip** — does the wan loader load a bf16-resaved checkpoint (R13)?
6. **Preproc sizes** — byte-measure ViTPose/YOLO/SAM2/DWPose/MiDaS/RAFT to confirm `wan-preproc` fits under 150GB alongside everything.

---

## 7. Amendments this analysis forces onto the approved #0 spec

These refine #0 — to be folded into the #0 implementation plan (flagged here so #0 isn't built wrong):

1. **wan-preproc must include Animate's ViTPose/YOLO/SAM2** (not just VACE's DWPose/MiDaS/RAFT) — they are NOT in the Diffusers Animate mirror. (R7)
2. **Do not transformer-only-strip the Animate mirror** — keep `image_processor/`, `scheduler/`, `tokenizer/`, `image_encoder/`. Verify VACE/V2V tolerate the strip. (R9)
3. **Add `shared.image_processor()`** (CLIPImageProcessor). (§3.6)
4. **Add a `wan2.1_v2v_14b` ModelCard** + budget entry, mapped to the shared t2v-14b mount. (R15)
5. **Fold the HANDLER_REGISTRY refactor into #0** (or a thin pre-#1 step) so downstream phases are append-only. (§3.2)
6. **Retrofit the existing `T2V_HANDLES`/`I2V_HANDLES` app.py dicts onto `ModelRegistry.acquire`** so all phases inherit the LRU pattern uniformly. (§3.1)
7. **Mount manifest is one atomic source-of-truth list**; add a boot probe. (R4)
8. **Two per-tier `@spaces.GPU` entrypoints** (large/xlarge). (R6)

---

## 8. Open questions (carried per-phase)
- Exact `wan.WanS2V/WanTI2V.generate` signatures (read from pinned source in #3).
- Whether spaces 0.50.2 keeps CPU-preproc off the GPU budget (#2 probe).
- Per-Space mount cap (staging probe).
- Whether the wan loader honors bf16 (#3 spike).
- S2V MoE-vs-dense.
- Gallery Storage-Bucket write visibility from forked `@spaces.GPU` workers (#4).
