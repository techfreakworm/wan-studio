# Wan Studio — Storage + Latency + Lifecycle Foundation (Sub-project #0) — Design Spec

| Field | Value |
|---|---|
| **Date** | 2026-06-08 |
| **Author** | Mayank Gupta |
| **Status** | Approved (post-brainstorm) — ready for implementation plan |
| **Scope** | Sub-project #0 of the "all-12-modes" program. Foundation only. |
| **Companion docs** | [`2026-05-21-wan-studio-design.md`](./2026-05-21-wan-studio-design.md) (original design) · [`RESEARCH.md`](../../../RESEARCH.md) · [`project_wan_studio_constraints.md`](memory) (firsthand HF/ZeroGPU constraints) |

---

## 1. Summary

Wan Studio must eventually serve **all ~12 Wan checkpoints** (Wan 2.1 + 2.2: T2V, I2V, TI2V, FLF2V, V2V, VACE, S2V, Animate) in one HF ZeroGPU Space. The collective weight set is ~300 GB fp32 — far larger than any single memory or disk budget. The current build hit two pains: (a) trouble "persisting" that many GB, and (b) re-downloading models at inference, slowly.

This sub-project builds the **storage + latency + lifecycle foundation** that makes the full mode set viable and never re-downloads weights. It is **not** mode-wiring — only T2V + I2V remain functionally wired this pass; VACE/FLF2V/V2V/Animate/S2V/TI2V pipelines are follow-on sub-projects (#1–#4) built *on top of* this foundation. Provisioning (mirror + mount + preproc weights) for all 12 is done here so those follow-ons are pure pipeline code.

The design rests on one principle: **persist-once, lazy-load-one, evict-on-switch.** Weights live permanently in the maintainer's bf16 HF model repos, attached to the Space as read-only mounts (zero ephemeral-disk cost, never re-downloaded). Shared encoders load once at module scope (forks inherit copy-on-write). Exactly one transformer family is warm at a time, evicted on mode switch.

---

## 2. Problem diagnosis (what's actually wrong today)

Confirmed by code audit + current-2026 HF storage research (this session):

- **Category error in the original worry.** HF *does* allow 300 GB — as **model repos** (10 TB public quota on PRO). It does *not* allow it on the **Space ephemeral disk** (~150 GB cap, firsthand "storage limit exceeded (150G)"). Weights belong in repos, mounted — never on the Space disk.
- **The existing `space_volumes` mount approach is already the correct "never re-download" mechanism** — a read-only mount (hf-mount FUSE/NFS over Xet) is re-attached, not re-pulled, on restart, and doesn't count against the 150 GB cap. The felt pain was *not* re-downloading per se; it was two bugs and one inefficiency:
  1. **Only 5 of ~12 repos are mirrored + mounted** (`scripts/duplicate_upstream.py` `PHASE_1_BASE_DUPLICATES`). VACE/S2V/Animate/TI2V/FLF2V/1.3B have no mirror → `pipelines/handle.py:_mount_path` silently falls back to `card.repo` (upstream fp32) → `from_pretrained` **downloads into ephemeral `/tmp`**, re-downloading every cold start and able to blow the 150 GB cap.
  2. **`pipelines/shared.py:image_encoder()` always downloads CLIP** from upstream (no mount path) — every first I2V/FLF2V/Animate use re-fetches it.
  3. **Mirrors are fp32** (`duplicate_repo` is byte-for-byte). The A14B experts are 57 GB each fp32; the pipeline casts to bf16 at load anyway, so fp32 is pure waste — doubling storage, mount-read time, and any fallback download.
- **Fast transfer is disabled** — `hf_transfer`/`hf_xet` not installed and `HF_HUB_DISABLE_XET=1` is set (`app.py`), so any download uses the slowest plain-HTTP path.
- **"Load all at startup" is physically impossible** — ~300 GB ≫ 96 GB VRAM (`xlarge`) ≫ ~192 GB CPU RAM ≫ 150 GB disk. The only viable model is persist-once + lazy-load-one + evict.

---

## 3. Goals / Non-goals

### Goals (G)
- **G1** — No weight is ever re-downloaded once provisioned. One-time load at Space boot for the default model; one-time mount-read on first use of any other mode.
- **G2** — All ~12 checkpoints + shared encoders + per-mode preproc weights are mirrored (bf16 where applicable) and mounted, so no mode hits the silent download fallback.
- **G3** — Halve storage + load latency by storing transformers bf16 (zero runtime quality change — pipeline runs bf16/fp16 anyway).
- **G4** — One warmed transformer family resident at a time, with explicit LRU eviction on mode switch; shared encoders loaded once at module scope and inherited by ZeroGPU forks (copy-on-write).
- **G5** — Preserve and improve **local M5 Max parity**: identical handle/LRU code; mounts no-op → bf16-mirror download to the persistent local HF cache; Space-only optimizations degrade to no-ops.
- **G6** — Fail loudly on ZeroGPU when an expected mount is missing (never silently download 100+ GB into `/tmp`).

### Non-goals (NG)
- **NG1** — Wiring VACE/FLF2V/V2V/Animate/S2V/TI2V pipeline handlers (sub-projects #1–#4). Only their *provisioning* is done here.
- **NG2** — AOTI compile-artifact caching (item G). Deferred to sub-project **#0.5** as a fast-follow once the storage+LRU core is proven live. AOTI is ZeroGPU-only and only cuts compile time, orthogonal to the storage problem.
- **NG3** — bf16 conversion of the *vendored non-diffusers* S2V/TI2V checkpoints (handled in #3); foundation mirrors those as-is.
- **NG4** — Cross-mode Send-to / Gallery / Settings (sub-project #4).
- **NG5** — FP8/torchao quantization (CUDA-only; out of scope everywhere here).

---

## 4. Locked decisions (from brainstorming)

| # | Decision | Choice |
|---|---|---|
| D1 | Program scope | All 12 modes eventually; **foundation first**, modes as phased follow-ons #1–#4 |
| D2 | Persistence mechanism | Read-only HF Volume mounts of bf16 model repos (the GA "never re-download" path); not `/data` (deprecated for new Spaces) |
| D3 | Storage dtype | Transformers **bf16** (halves storage + load); VAE stays **fp32** (quality-sensitive, small); UMT5 already bf16; CLIP fp32 |
| D4 | bf16 conversion host | **HF server-side** (HF Job / throwaway big-disk Space) — all ~450 GB transfer stays in HF's datacenter |
| D5 | Shared-encoder delivery **[CONFIRM A ✓]** | **Dedicated mounted mirror** (`wan-shared-encoders`), uniform with everything else; *not* `preload_from_hub` (avoids HF_HOME/permission wrinkles) |
| D6 | AOTI **[CONFIRM B ✓]** | Deferred to sub-project **#0.5** (fast-follow), not in foundation v1 |
| D7 | Test/deploy loop **[CONFIRM C ✓]** | Validate on a throwaway **`wan-studio-staging`** Space, then promote to the live Space (honors "never push while testing on HF") |
| D8 | Lifecycle | persist-once + lazy-load-one + in-process LRU evict-on-switch; shared encoders at module scope (fork CoW) |
| D9 | Memory routing | Wan 2.1 single-transformer → `large` (48 GB); Wan 2.2 A14B MoE → `enable_model_cpu_offload()` on `large`, or `xlarge` if upgraded. (Unchanged from prior; foundation does not alter this.) |

---

## 5. Architecture

```
PROVISIONING (one-time, server-side HF Job)        RUNTIME (every Space boot)
┌────────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ upstream fp32 (Wan-AI/*, Kijai, lightx) │        │ mount /models/<slug>  (read-only, 0   │
│   │  diffusers: load → save bf16        │        │   ephemeral-disk cost, re-mounted not │
│   ▼                                      │  ───►  │   re-pulled across restart)           │
│ techfreakworm/<slug>-bf16  (~150 GB)     │        │ stitch: symlink weights + copy        │
│ techfreakworm/wan-shared-encoders        │        │   bundled JSON (truncation fix)       │
│ techfreakworm/wan-preproc (DWPose/SAM2…) │        │ module scope: shared encoders (CoW)   │
│ techfreakworm/wan-lightning-loras        │        │ LRU: ≤1 warm transformer, evict       │
└────────────────────────────────────────┘        │ tier-2: hot model shards mount→/tmp   │
                                                    └──────────────────────────────────────┘
LOCAL M5 Max: same handle/LRU code. No /models mount → download bf16 mirror into
persistent ~/.cache (warm forever on SSD). AOTI/stitch/tier-2 no-op via backend detect.
```

**Storage tiers (current-2026, ranked):**

| Mechanism | Re-downloads? | Counts vs 150 GB disk | Read speed | Use for |
|---|---|---|---|---|
| Volume mount (RO, `type=model`) | Never (re-mounted) | No | Slow 1st read, warm after | All transformers + shared + preproc + LoRAs |
| Storage Bucket (RW) + `HF_HOME` | Never (persists) | No | Object-store | `HF_HOME` cache so incidental downloads survive restart (belt-and-braces) |
| Download → `/tmp` | **Every cold start** | Yes | Slow | ❌ the bug — eliminate |
| Persistent `/data` | — | — | SSD | ❌ deprecated for new Spaces — do not use |

---

## 6. Components

### 6.1 Provisioning (offline)

**`scripts/convert_to_bf16.py` (new)** — HF-Job-runnable. For each diffusers checkpoint:
- Convert `transformer` (+ `transformer_2` on MoE) to bf16 via `save_pretrained(torch_dtype=torch.bfloat16)`.
- **Transformer-only mirrors.** The bf16 mirror ships **only** the `transformer`/`transformer_2` subfolders + `model_index.json` + `scheduler/` + `tokenizer/` — **not** `text_encoder/` or `vae/`. Those shared components live once in `wan-shared-encoders` and are injected at runtime (`from_pretrained(..., text_encoder=, vae=)`), which the current code already always does. This avoids duplicating the ~11 GB UMT5 across all 12 mirrors (~130 GB saved) and keeps the total well under 150 GB. **VAE stored fp32, UMT5 bf16, CLIP fp32 — all in `wan-shared-encoders`.**
- Push to `techfreakworm/<slug>-bf16`. Idempotent (skip if dest exists at expected revision).
- Designed to run as an HF Job (server-side bandwidth). Falls back to local execution.

**`scripts/duplicate_upstream.py` (modified)** — now also:
- Builds `techfreakworm/wan-shared-encoders` (UMT5-XXL bf16 + Wan-VAE fp32 + CLIP-ViT-H fp32).
- Builds `techfreakworm/wan-preproc` (DWPose/MiDaS/RAFT, ViTPose/YOLOv10/SAM2, wav2vec2) — *staged now, consumed by #1–#3*.
- Mirrors the vendored S2V + TI2V-5B checkpoints **as-is** (bf16 conversion deferred to #3).
- Keeps the consolidated `wan-lightning-loras` mirror.

**`scripts/create_space.py` (modified)** — mount manifest extended to **all 12 bf16 mirrors + shared-encoders + preproc + lightning-loras** at `/models/<slug>`.

### 6.2 Runtime path resolution & lifecycle

**`pipelines/registry.py` (modified)** — each `ModelCard` gains a `mirror_repo` (e.g. `techfreakworm/wan2.2-t2v-a14b-bf16`) distinct from upstream `repo`. Slug convention unchanged.

**`pipelines/handle.py` (modified):**
- `_mount_path(card)` becomes context-aware:
  - stitched mount present → return it (unchanged happy path).
  - mount absent **on ZeroGPU** → `raise RuntimeError(...)` (fail loud; no `/tmp` download).
  - mount absent **locally** → return `card.mirror_repo` (download the bf16 mirror, *not* upstream fp32) into the persistent HF cache.
- **LRU registry** — a module-level `ModelRegistry` holding ≤1 warmed transformer family. `acquire(key)` evicts the current (`del` transformer(s) + `gc.collect()` + `torch.cuda.empty_cache()`) before building the next. Shared encoders are *never* evicted (loaded once, module scope).

**`pipelines/shared.py` (modified):**
- `text_encoder()` / `vae()` / `image_encoder()` all read from the mounted/stitched `wan-shared-encoders` dir (or its bf16/fp32 mirror locally). **Kills the always-upstream CLIP download.**

**`app.py` (modified):**
- Drop `HF_HUB_DISABLE_XET=1`; rely on `hf_xet`. Keep `HF_HUB_CACHE=/tmp/hf_cache` redirect (optionally point at a Storage Bucket mount for cross-restart persistence — belt-and-braces).
- Startup preload unchanged in spirit (default Wan2.2-T2V handle → CPU RAM), now from the bf16 mount.

**`requirements.txt` (modified)** — add `hf_xet`, `hf_transfer`.

### 6.3 Latency layer

**Tier-2 warm cache** — on first use of a mode whose model > ~10 GB (i.e. larger than hf-mount's default local chunk cache), copy that checkpoint's shards mount→`/tmp/wan-hot/<slug>` once; subsequent reads in the same boot are local-disk speed and shared by forked workers. LRU-bounded under the 150 GB cap (evict the previous hot model's `/tmp` copy on switch). Re-paid on each cold boot (`/tmp` ephemeral) — acceptable since weights are never *re-downloaded*, only re-read from the mount.

---

## 7. Data flow — a mode switch (Wan 2.2 T2V → Wan 2.1 VACE-14B, on the Space)

```
User clicks VACE (already provisioned + mounted; handler wired in #1, but path/LRU is foundation)
  │
  │ 1. registry.acquire("wan2.1_vace_14b")
  │ 2.   evict current: del t2v transformer(+_2); gc; empty_cache()
  │ 3.   _mount_path → /tmp/wan-stitched/wan2.1-vace-14b (symlinks into RO mount)
  │ 4.   tier-2: copy vace-14b shards mount→/tmp/wan-hot/... (first use only)
  │ 5.   build pipeline from stitched/hot dir, inject shared encoders (already resident)
  │ 6.   ensure_cuda_attached() inside @spaces.GPU
  ▼
Warm. Repeat VACE use = instant. Switching back to T2V re-incurs steps 2–6 for T2V.
```

No network download at any step (mounts present). Shared encoders never reloaded.

---

## 8. Error handling

| Failure | Detection | Behavior |
|---|---|---|
| Expected mount missing (ZeroGPU) | `_mount_path` stitched dir None + `is_zerogpu` | `raise RuntimeError("mount /models/<slug> missing — check create_space.py manifest")` — fail fast, no `/tmp` download |
| Mount missing (local) | same + not ZeroGPU | download `card.mirror_repo` (bf16) to persistent cache; log it |
| bf16 mirror not yet provisioned | `from_pretrained` 404 | clear error naming the missing `techfreakworm/<slug>-bf16` repo |
| A14B OOM on `large` | `OutOfMemoryError` | hint to enable offload / request `xlarge` (unchanged) |
| `/tmp` full during tier-2 copy | `OSError ENOSPC` | skip tier-2 (read straight from mount), warn; do not crash |

---

## 9. Testing strategy

**Unit (local, MPS — no live Space needed):**
- `test_path_resolution.py` — ZeroGPU-missing-mount raises; local-missing-mount returns `mirror_repo`; stitched-present returns stitched.
- `test_lru.py` — `acquire` evicts prior transformer, calls `gc`/`empty_cache`, never evicts shared encoders; ≤1 family resident.
- `test_registry.py` (extend) — every card has a `mirror_repo`; manifest in `create_space.py` covers all mounted slugs; no orphan slugs.
- `test_bf16.py` — conversion target dtype assertions (transformer bf16, VAE fp32).
- **`test_smoke_t2v_local.py`** — recreate the deleted 1.3B MPS smoke (forced via `WAN_STUDIO_T2V_LOCAL_KEY`); end-to-end, asserts MP4 written.

**Integration (live staging Space):**
- Boot probe: all expected mounts present (extend `_probe_filesystem`).
- Per-checkpoint first-load timing logged (validate "one mount-read, then warm").
- Default T2V Fast generate < target; mode switch T2V↔I2V loads + evicts cleanly.

---

## 10. Deployment

1. Run `convert_to_bf16.py` as an HF Job → bf16 mirrors.
2. Run `duplicate_upstream.py` → shared-encoders + preproc + loras + vendored as-is.
3. Run `create_space.py` against **`techfreakworm/wan-studio-staging`** → mount manifest + ZeroGPU hardware.
4. `hf upload` code to staging; validate §9 integration checks.
5. Promote: apply the same manifest to the live `techfreakworm/wan-studio` and upload.

---

## 11. Sequencing

- **#0 (this spec)** — storage + latency + lifecycle foundation; provisioning for all 12; T2V/I2V stay wired.
- **#0.5** — AOTI compile-artifact caching (ZeroGPU-only, fast-follow).
- **#1** — diffusers single-transformer modes: VACE (1.3B/14B + DWPose/MiDaS/RAFT), FLF2V, V2V.
- **#2** — Animate (ViTPose/YOLOv10/SAM2 + multi-segment).
- **#3** — vendored S2V + TI2V-5B (`wan` package; bf16-convert these here).
- **#4** — cross-mode Send-to + Gallery + Settings + examples.

---

## 12. Risks & open questions

| Risk | Severity | Mitigation |
|---|---|---|
| Max number of Volume mounts per Space unknown (mounting 12+ + shared + preproc + loras) | Medium | Verify the per-Space volume limit before finalizing the manifest; if capped, consolidate related checkpoints into fewer repos (one mount, subfolders) |
| HF Job availability/quota for the conversion | Low | Fallback: throwaway big-disk CPU Space, or local M5 Max model-by-model |
| bf16 vs fp32 numerical drift | Low | Pipeline already casts to bf16/fp16 at load — storing bf16 is a no-op for output quality; spot-check one T2V output bf16-mirror vs fp32-upstream |
| hf-mount truncation bug status in mid-2026 unverified | Low | Keep the stitch (ship correct JSONs in `models_meta/`) regardless — it's cheap insurance |
| Transformer-only mirror: diffusers must tolerate absent `text_encoder/`/`vae/` subfolders when those are injected as kwargs | Medium | Verify `WanPipeline.from_pretrained` loads from a transformer-only dir with injected `text_encoder=`/`vae=` before mass conversion; if not, ship a stub config or keep the subfolders |
| Tier-2 `/tmp` copy + A14B (~56 GB hot transformers) crowds 150 GB cap | Medium | LRU-evict prior hot copy on switch; cap tier-2 to one model; skip-on-ENOSPC. (Shared encoders are RAM-resident from module scope, not tier-2 copied.) |
| Vendored S2V/TI2V mounted fp32-as-is until #3 | Low | They're only *provisioned* here, not loaded; bf16-convert in #3 |

---

## 13. Out of scope (recap)
Mode pipeline wiring (#1–#4) · AOTI (#0.5) · vendored-checkpoint bf16 conversion (#3) · Send-to/Gallery/Settings (#4) · FP8/quantization · `/data` persistent storage.
