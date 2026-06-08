# Wan Studio Foundation #0a — Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Tasks 1–6 are code+TDD (locally testable). Tasks 7–8 are OPS — they run real HF operations against the maintainer's account and are executed deliberately, not auto-run.**

**Goal:** Provision everything the #0b runtime loads — bf16 transformer-only mirrors of all 12 checkpoints, a `wan-shared-encoders` mirror, a `wan-preproc` mirror (VACE + Animate preproc), and an atomic `create_space.py` mount manifest + boot probe — then deploy to a staging Space and run the gating spikes.

**Architecture:** Pure-Python *plan/manifest* builders (unit-tested locally) wrap thin HF-I/O *execute* layers (dry-run-able). The bf16 conversion runs server-side as an HF Job. Mirrors are transformer-only (shared encoders centralized once → ~130 GB saved). `create_space.py` re-passes the COMPLETE volume set every time (`set_space_volumes` replaces atomically).

**Tech Stack:** Python 3.12 · huggingface_hub 1.x (`HfApi`, `Volume`, `run_job`/Jobs) · diffusers 0.38 · safetensors · pytest 9

**References:** [`#0 spec §6.1 + §14`](../specs/2026-06-08-wan-studio-storage-latency-foundation-design.md) · [`program architecture & risks`](../specs/2026-06-08-wan-studio-program-architecture-and-risks.md) (R4, R5, R7) · [`#0b runtime plan`](./2026-06-09-wan-studio-foundation-0b-runtime.md) (already executed)

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `provisioning/__init__.py` | **create** | package marker |
| `provisioning/manifest.py` | **create** | single source-of-truth: slugs, mount paths, mirror map, shared/preproc/lora mounts; pure data + builders |
| `provisioning/bf16_plan.py` | **create** | per-card "what to convert / keep / skip" plan (pure logic) |
| `provisioning/preproc_manifest.py` | **create** | wan-preproc + wan-shared-encoders contents (repo, subpath, dtype) — pure data |
| `scripts/convert_to_bf16.py` | **create** | HF-Job-runnable: execute the bf16 plan per checkpoint (dry-run-able) |
| `scripts/duplicate_upstream.py` | modify | build shared-encoders + wan-preproc mirrors + vendored-as-is, from the manifests |
| `scripts/create_space.py` | modify | atomic `set_space_volumes` from `manifest.all_volumes()` + hardware + dry-run |
| `app.py` | modify (`_probe_filesystem`) | assert every expected mount exists on ZeroGPU (fail loud) |
| `tests/test_manifest.py` | **create** | registry↔manifest consistency (every card slug mounted; no orphans) |
| `tests/test_bf16_plan.py` | **create** | conversion plan per card (transformer-only; Animate keeps image_processor) |
| `tests/test_preproc_manifest.py` | **create** | preproc/shared manifests cover VACE+Animate+shared encoders |

**Conventions:** slug = `card.key.replace("_","-")`; model mount `/models/<slug>`; mirror `techfreakworm/<slug>-bf16` (or the card's explicit `mirror_repo`). Shared mount `/models/wan-shared-encoders`, preproc `/models/wan-preproc`, loras `/models/wan-lightning-loras`.

---

## Task 1: Single-source mount manifest + consistency test

**Files:**
- Create: `provisioning/__init__.py`, `provisioning/manifest.py`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manifest.py`:

```python
"""Tests for provisioning.manifest — the single source-of-truth mount list."""
from provisioning import manifest
from pipelines.registry import ALL_MODELS, BY_KEY


def test_every_card_has_a_model_mount():
    mounts = {v.mount_path for v in manifest.all_volumes()}
    for m in ALL_MODELS:
        slug = m.key.replace("_", "-")
        # v2v shares the t2v-14b mount → its own slug need not be mounted
        if m.mirror_repo == BY_KEY["wan2.1_t2v_14b"].mirror_repo and m.key != "wan2.1_t2v_14b":
            continue
        assert f"/models/{slug}" in mounts, f"{m.key} not mounted"


def test_shared_preproc_lora_mounts_present():
    mounts = {v.mount_path for v in manifest.all_volumes()}
    assert "/models/wan-shared-encoders" in mounts
    assert "/models/wan-preproc" in mounts
    assert "/models/wan-lightning-loras" in mounts


def test_all_volumes_read_only_models():
    for v in manifest.all_volumes():
        assert v.read_only is True
        assert v.type == "model"


def test_no_duplicate_mount_paths():
    paths = [v.mount_path for v in manifest.all_volumes()]
    assert len(paths) == len(set(paths)), "duplicate mount paths"


def test_expected_mount_paths_helper_matches_volumes():
    assert set(manifest.expected_mount_paths()) == {v.mount_path for v in manifest.all_volumes()}
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_manifest.py -v`
Expected: FAIL — no `provisioning.manifest`.

- [ ] **Step 3: Implement `provisioning/manifest.py`**

```python
"""Single source-of-truth for Space volume mounts.

`set_space_volumes` REPLACES the entire volume set atomically (program risk R4),
so create_space.py must always pass the COMPLETE list from here.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipelines.registry import ALL_MODELS, BY_KEY

SHARED_MIRROR = "techfreakworm/wan-shared-encoders"
PREPROC_MIRROR = "techfreakworm/wan-preproc"
LORA_MIRROR = "techfreakworm/wan-lightning-loras"


@dataclass(frozen=True)
class VolumeSpec:
    type: str
    source: str        # techfreakworm/<repo>
    mount_path: str     # /models/<slug>
    read_only: bool = True


def _model_volumes() -> list[VolumeSpec]:
    """One read-only mount per distinct model mirror.

    V2V shares the T2V-14B mirror, so dedupe by (source, mount_path): a card
    whose mirror_repo equals another card's is mounted once under the OWNER's
    slug. The owner is the card whose own slug-mirror matches its mirror_repo.
    """
    seen: dict[str, VolumeSpec] = {}
    for m in ALL_MODELS:
        slug = m.key.replace("_", "-")
        own_mirror = f"techfreakworm/{slug}-bf16"
        if m.mirror_repo != own_mirror:
            continue  # shares another card's mirror (e.g. v2v) → not its own mount
        seen[m.mirror_repo] = VolumeSpec("model", m.mirror_repo, f"/models/{slug}")
    return list(seen.values())


def all_volumes() -> list[VolumeSpec]:
    return _model_volumes() + [
        VolumeSpec("model", SHARED_MIRROR, "/models/wan-shared-encoders"),
        VolumeSpec("model", PREPROC_MIRROR, "/models/wan-preproc"),
        VolumeSpec("model", LORA_MIRROR, "/models/wan-lightning-loras"),
    ]


def expected_mount_paths() -> list[str]:
    return [v.mount_path for v in all_volumes()]
```

Create empty `provisioning/__init__.py`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_manifest.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add provisioning/__init__.py provisioning/manifest.py tests/test_manifest.py
git commit -m "Add single-source mount manifest + consistency tests"
```

---

## Task 2: bf16 conversion plan (pure logic) + converter

**Files:**
- Create: `provisioning/bf16_plan.py`, `scripts/convert_to_bf16.py`
- Test: `tests/test_bf16_plan.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bf16_plan.py`:

```python
"""Tests for provisioning.bf16_plan — per-card conversion plan."""
from provisioning.bf16_plan import conversion_plan
from pipelines.registry import BY_KEY


def test_single_transformer_card():
    plan = conversion_plan(BY_KEY["wan2.1_t2v_14b"])
    assert plan.convert_subfolders == ["transformer"]          # bf16
    assert "transformer_2" not in plan.convert_subfolders
    assert "text_encoder" not in plan.keep_subfolders          # lives in shared mirror
    assert "vae" not in plan.keep_subfolders
    assert set(plan.keep_subfolders) >= {"scheduler", "tokenizer"}
    assert plan.keep_files >= {"model_index.json"}


def test_moe_card_converts_both_experts():
    plan = conversion_plan(BY_KEY["wan2.2_t2v_a14b"])
    assert plan.convert_subfolders == ["transformer", "transformer_2"]


def test_animate_keeps_image_processor_and_encoder():
    """Amendment 2: Animate mirror must NOT be stripped of image_processor/encoder."""
    plan = conversion_plan(BY_KEY["wan2.2_animate_14b"])
    assert "transformer" in plan.convert_subfolders
    assert {"image_processor", "image_encoder"} <= set(plan.keep_subfolders)


def test_vendored_cards_have_no_diffusers_plan():
    """S2V / TI2V are vendored (diffusers_class=None) → bf16 conversion deferred to #3."""
    for key in ("wan2.2_s2v_14b", "wan2.2_ti2v_5b"):
        assert conversion_plan(BY_KEY[key]) is None
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_bf16_plan.py -v`
Expected: FAIL — no `provisioning.bf16_plan`.

- [ ] **Step 3: Implement `provisioning/bf16_plan.py`**

```python
"""Per-card bf16 conversion plan.

Transformer-only mirrors: convert transformer(s) to bf16, keep the small
config/scheduler/tokenizer + model_index.json, DROP text_encoder/ and vae/
(they live once in wan-shared-encoders and are injected at load). Animate is
the exception — it also keeps image_processor/ and image_encoder/ (amendment 2).
Vendored S2V/TI2V (diffusers_class=None) return None — handled in #3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pipelines.registry import ModelCard

_BASE_KEEP = {"scheduler", "tokenizer"}
_BASE_FILES = {"model_index.json"}


@dataclass(frozen=True)
class ConversionPlan:
    card_key: str
    convert_subfolders: list[str]        # → bf16 via save_pretrained(torch_dtype=bf16)
    keep_subfolders: list[str]           # copied as-is (small)
    keep_files: set[str] = field(default_factory=lambda: set(_BASE_FILES))


def conversion_plan(card: ModelCard) -> ConversionPlan | None:
    if card.diffusers_class is None:
        return None  # vendored — deferred to #3
    convert = ["transformer"] + (["transformer_2"] if card.is_moe else [])
    keep = set(_BASE_KEEP)
    if card.mode == "animate":
        keep |= {"image_processor", "image_encoder"}
    return ConversionPlan(card.key, convert, sorted(keep))
```

- [ ] **Step 4: Run plan tests**

Run: `.venv/bin/pytest tests/test_bf16_plan.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Write `scripts/convert_to_bf16.py` (executes the plan; dry-run-able)**

```python
"""Convert diffusers checkpoints to bf16 transformer-only mirrors.

Designed to run as an HF Job (server-side bandwidth) or locally. For each
non-vendored card: load each convert-subfolder at bf16, save_pretrained, copy
the keep-subfolders/files, push to card.mirror_repo. Idempotent (skip if dest
revision exists). `--dry-run` prints the plan without touching the Hub.

Usage:
  python scripts/convert_to_bf16.py --dry-run
  python scripts/convert_to_bf16.py --only wan2.1_t2v_14b
  python scripts/convert_to_bf16.py            # all non-vendored cards
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.registry import ALL_MODELS, BY_KEY  # noqa: E402
from provisioning.bf16_plan import conversion_plan  # noqa: E402


def convert_one(card, *, dry_run: bool) -> None:
    plan = conversion_plan(card)
    if plan is None:
        print(f"[skip] {card.key}: vendored (no diffusers plan)")
        return
    print(f"[plan] {card.key} -> {card.mirror_repo}: convert={plan.convert_subfolders} "
          f"keep={plan.keep_subfolders} files={sorted(plan.keep_files)}")
    if dry_run:
        return

    import tempfile
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    api = HfApi()
    if api.repo_exists(card.mirror_repo):
        print(f"[skip] {card.mirror_repo} already exists")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for sub in plan.convert_subfolders:
            t = WanTransformer3DModel.from_pretrained(
                card.repo, subfolder=sub, torch_dtype=torch.bfloat16)
            t.save_pretrained(out / sub)
        for sub in plan.keep_subfolders:
            d = WanTransformer3DModel  # placeholder import guard; use snapshot for subfolders
            from huggingface_hub import snapshot_download
            snapshot_download(card.repo, allow_patterns=[f"{sub}/*"], local_dir=out)
        for f in plan.keep_files:
            hf_hub_download(card.repo, f, local_dir=out)
        api.create_repo(card.mirror_repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=card.mirror_repo, repo_type="model")
    print(f"[done] {card.mirror_repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="single card key")
    args = ap.parse_args()
    cards = [BY_KEY[args.only]] if args.only else ALL_MODELS
    for card in cards:
        convert_one(card, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the dry-run prints a sane plan (no Hub access)**

Run: `.venv/bin/python scripts/convert_to_bf16.py --dry-run`
Expected: prints `[plan]` lines for the 10 diffusers cards and `[skip] ... vendored` for s2v/ti2v. No network call, no error.

- [ ] **Step 7: Commit**

```bash
git add provisioning/bf16_plan.py scripts/convert_to_bf16.py tests/test_bf16_plan.py
git commit -m "Add bf16 conversion plan (transformer-only; Animate keeps processors) + converter"
```

---

## Task 3: Shared-encoders + preproc mirror manifests

**Files:**
- Create: `provisioning/preproc_manifest.py`
- Test: `tests/test_preproc_manifest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preproc_manifest.py`:

```python
"""Tests for provisioning.preproc_manifest."""
from provisioning import preproc_manifest as pm


def test_shared_encoders_components():
    subs = {c.dest_subfolder for c in pm.SHARED_ENCODERS}
    assert {"text_encoder", "vae", "image_encoder", "image_processor"} <= subs


def test_shared_encoders_dtypes():
    by = {c.dest_subfolder: c for c in pm.SHARED_ENCODERS}
    assert by["text_encoder"].dtype == "bfloat16"   # UMT5
    assert by["vae"].dtype == "float32"             # quality-sensitive
    assert by["image_encoder"].dtype == "float32"   # CLIP


def test_preproc_covers_vace_and_animate():
    names = {a.name for a in pm.PREPROC_ASSETS}
    # VACE subset
    assert {"dwpose", "midas_dpt_hybrid", "raft"} <= names
    # Animate subset (amendment 1: NOT bundled in the diffusers mirror)
    assert {"vitpose_h_wholebody", "yolov10m", "sam2_hiera_large"} <= names


def test_preproc_assets_have_source_and_path():
    for a in pm.PREPROC_ASSETS:
        assert a.source_repo and a.source_path and a.dest_path
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_preproc_manifest.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `provisioning/preproc_manifest.py`**

```python
"""Contents of the wan-shared-encoders and wan-preproc mirrors.

These are DATA manifests consumed by duplicate_upstream.py. Source paths are
the upstream repos; dest paths are inside the maintainer's mirror repos.
Amendment 1: Animate's ViTPose/YOLO/SAM2 live ONLY in the non-Diffusers
Wan-AI/Wan2.2-Animate-14B repo (the Diffusers mirror has no process_checkpoint/),
so they are provisioned here, not assumed bundled.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedComponent:
    source_repo: str
    source_subfolder: str
    dest_subfolder: str
    dtype: str  # "bfloat16" | "float32"


@dataclass(frozen=True)
class PreprocAsset:
    name: str
    source_repo: str
    source_path: str   # file or glob in the source repo
    dest_path: str     # path inside techfreakworm/wan-preproc


SHARED_ENCODERS = [
    SharedComponent("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "text_encoder", "text_encoder", "bfloat16"),
    SharedComponent("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "vae", "vae", "float32"),
    SharedComponent("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "image_encoder", "image_encoder", "float32"),
    SharedComponent("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "image_processor", "image_processor", "float32"),
]

PREPROC_ASSETS = [
    # VACE lightweight subset (~1 GB)
    PreprocAsset("dwpose", "ali-vilab/VACE", "models/dwpose/*", "vace/dwpose/"),
    PreprocAsset("midas_dpt_hybrid", "Intel/dpt-hybrid-midas", "*", "vace/midas/"),
    PreprocAsset("raft", "ali-vilab/VACE", "models/raft/*", "vace/raft/"),
    # Animate (~2 GB) — amendment 1
    PreprocAsset("vitpose_h_wholebody", "Wan-AI/Wan2.2-Animate-14B",
                 "process_checkpoint/pose2d/vitpose_h_wholebody.onnx", "animate/pose2d/vitpose_h_wholebody.onnx"),
    PreprocAsset("yolov10m", "Wan-AI/Wan2.2-Animate-14B",
                 "process_checkpoint/det/yolov10m.onnx", "animate/det/yolov10m.onnx"),
    PreprocAsset("sam2_hiera_large", "Wan-AI/Wan2.2-Animate-14B",
                 "process_checkpoint/sam2/sam2_hiera_large.pt", "animate/sam2/sam2_hiera_large.pt"),
]
```

> **Verify-at-execution:** the exact upstream `source_path`s (especially `ali-vilab/VACE` model subpaths and the Animate `process_checkpoint/` layout) must be confirmed against the live repos during Task 7's dry-run before the real fetch. Adjust paths inline if the upstream layout differs.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_preproc_manifest.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add provisioning/preproc_manifest.py tests/test_preproc_manifest.py
git commit -m "Add shared-encoders + wan-preproc mirror manifests (incl. Animate preproc)"
```

---

## Task 4: `duplicate_upstream.py` — build shared + preproc mirrors

**Files:**
- Modify: `scripts/duplicate_upstream.py`

> Read the current file first; it already mirrors base repos + the lightning LoRA bundle. Add two builders driven by the Task-3 manifests, and mirror the vendored S2V/TI2V repos as-is. Keep everything `--dry-run`-able.

- [ ] **Step 1: Add the shared-encoders builder**

Append to `scripts/duplicate_upstream.py` (adapt imports to the file's existing `api`/`HfApi` usage):

```python
from provisioning.preproc_manifest import SHARED_ENCODERS, PREPROC_ASSETS
from provisioning.manifest import SHARED_MIRROR, PREPROC_MIRROR


def build_shared_encoders(api, *, dry_run: bool) -> None:
    import tempfile, torch
    from pathlib import Path
    from huggingface_hub import snapshot_download
    print(f"[shared] -> {SHARED_MIRROR}")
    if dry_run:
        for c in SHARED_ENCODERS:
            print(f"  {c.source_repo}:{c.source_subfolder} -> {c.dest_subfolder} ({c.dtype})")
        return
    if api.repo_exists(SHARED_MIRROR):
        print(f"  [skip] {SHARED_MIRROR} exists"); return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for c in SHARED_ENCODERS:
            snapshot_download(c.source_repo, allow_patterns=[f"{c.source_subfolder}/*"], local_dir=out / "_src")
            (out / c.dest_subfolder).parent.mkdir(parents=True, exist_ok=True)
            # text_encoder/vae are dtype-recast; image_* copied as-is fp32
            src = out / "_src" / c.source_subfolder
            if c.dtype == "bfloat16":
                _recast_safetensors(src, out / c.dest_subfolder, torch.bfloat16)
            else:
                import shutil; shutil.copytree(src, out / c.dest_subfolder, dirs_exist_ok=True)
        api.create_repo(SHARED_MIRROR, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=SHARED_MIRROR, repo_type="model",
                          ignore_patterns=["_src/*"])
    print(f"  [done] {SHARED_MIRROR}")
```

(`_recast_safetensors(src, dst, dtype)` = load each `*.safetensors` in `src` with `safetensors.torch.load_file`, `.to(dtype)` each tensor, `save_file` to `dst`, and copy non-tensor files. Write this helper in the same file.)

- [ ] **Step 2: Add the preproc builder**

```python
def build_preproc(api, *, dry_run: bool) -> None:
    from pathlib import Path
    import tempfile
    from huggingface_hub import hf_hub_download, snapshot_download
    print(f"[preproc] -> {PREPROC_MIRROR}")
    if dry_run:
        for a in PREPROC_ASSETS:
            print(f"  {a.source_repo}:{a.source_path} -> {a.dest_path}")
        return
    if api.repo_exists(PREPROC_MIRROR):
        print(f"  [skip] {PREPROC_MIRROR} exists"); return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for a in PREPROC_ASSETS:
            dest = out / a.dest_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if a.source_path.endswith("*") or "/" not in a.source_path.rstrip("*"):
                snapshot_download(a.source_repo, allow_patterns=[a.source_path], local_dir=dest.parent)
            else:
                hf_hub_download(a.source_repo, a.source_path, local_dir=out, repo_type="model")
        api.create_repo(PREPROC_MIRROR, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=PREPROC_MIRROR, repo_type="model")
    print(f"  [done] {PREPROC_MIRROR}")
```

- [ ] **Step 3: Mirror vendored S2V/TI2V as-is + wire into `main()`**

In the file's `main()`, add the vendored duplicates (as-is, no conversion) and call the new builders. Vendored repos to `api.duplicate_repo` (or snapshot+upload): `Wan-AI/Wan2.2-S2V-14B → techfreakworm/wan2.2-s2v-14b`, `Wan-AI/Wan2.2-TI2V-5B → techfreakworm/wan2.2-ti2v-5b`. Then `build_shared_encoders(api, dry_run=args.dry_run)` and `build_preproc(api, dry_run=args.dry_run)`.

- [ ] **Step 4: Dry-run verify**

Run: `.venv/bin/python scripts/duplicate_upstream.py --dry-run`
Expected: prints the shared-encoder component list, the preproc asset list, and the vendored as-is mirrors. No network, no error.

- [ ] **Step 5: Commit**

```bash
git add scripts/duplicate_upstream.py
git commit -m "duplicate_upstream: build shared-encoders + wan-preproc mirrors; vendored as-is"
```

---

## Task 5: `create_space.py` — atomic manifest + hardware + dry-run

**Files:**
- Modify: `scripts/create_space.py`
- Test: `tests/test_manifest.py` (extend)

> Read the current file; it already calls `set_space_volumes` + `request_space_hardware`. Replace its hardcoded `PHASE_1_VOLUMES` with `manifest.all_volumes()` and make it always pass the COMPLETE set.

- [ ] **Step 1: Write the failing test (extend `tests/test_manifest.py`)**

```python
def test_create_space_uses_full_manifest():
    """create_space must build its Volume list from manifest.all_volumes() (atomic replace)."""
    import scripts.create_space as cs
    specs = cs.build_volume_specs()          # returns the manifest VolumeSpecs
    from provisioning.manifest import all_volumes
    assert {v.mount_path for v in specs} == {v.mount_path for v in all_volumes()}
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_manifest.py::test_create_space_uses_full_manifest -v`
Expected: FAIL — no `build_volume_specs`.

- [ ] **Step 3: Refactor `scripts/create_space.py`**

Add a pure `build_volume_specs()` returning `manifest.all_volumes()`, and convert to `huggingface_hub.Volume` only inside the execute path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provisioning.manifest import all_volumes  # noqa: E402

SPACE_ID = "techfreakworm/wan-studio"

def build_volume_specs():
    return all_volumes()

def apply(space_id: str, *, dry_run: bool) -> None:
    specs = build_volume_specs()
    print(f"[volumes] {len(specs)} mounts -> {space_id}")
    for v in specs:
        print(f"  {v.source} -> {v.mount_path} (ro={v.read_only})")
    if dry_run:
        return
    from huggingface_hub import HfApi, Volume, SpaceHardware
    api = HfApi()
    volumes = [Volume(type=v.type, source=v.source, mount_path=v.mount_path, read_only=v.read_only)
               for v in specs]
    api.set_space_volumes(repo_id=space_id, volumes=volumes)   # REPLACES the whole set
    api.request_space_hardware(repo_id=space_id, hardware=SpaceHardware.ZERO_A10G)
    print("[done] volumes + hardware applied")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--space", default=SPACE_ID)
    apply(ap.parse_args().space, dry_run=ap.parse_args().dry_run)
```

- [ ] **Step 4: Run test + dry-run**

Run: `.venv/bin/pytest tests/test_manifest.py -v && .venv/bin/python scripts/create_space.py --dry-run --space techfreakworm/wan-studio-staging`
Expected: test PASS; dry-run prints the full mount list (no Hub call).

- [ ] **Step 5: Commit**

```bash
git add scripts/create_space.py tests/test_manifest.py
git commit -m "create_space: atomic volume manifest from single source + dry-run"
```

---

## Task 6: Boot probe — assert mounts exist on ZeroGPU

**Files:**
- Modify: `app.py` (`_probe_filesystem`)
- Test: `tests/test_boot_probe.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_boot_probe.py`:

```python
"""Tests for app boot-probe mount assertion."""
import pytest


def test_assert_mounts_passes_when_all_present(monkeypatch, tmp_path):
    import app
    for p in __import__("provisioning.manifest", fromlist=["expected_mount_paths"]).expected_mount_paths():
        (tmp_path / p.lstrip("/")).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "MOUNT_ROOT", tmp_path)
    app.assert_expected_mounts()  # should not raise


def test_assert_mounts_raises_on_missing(monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app, "MOUNT_ROOT", tmp_path)  # nothing created
    with pytest.raises(RuntimeError, match="missing mount"):
        app.assert_expected_mounts()
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_boot_probe.py -v`
Expected: FAIL — no `assert_expected_mounts`/`MOUNT_ROOT`.

- [ ] **Step 3: Implement in `app.py`**

Add near the top (after the cache redirect), and call from `_probe_filesystem` when on ZeroGPU:

```python
from pathlib import Path as _Path
MOUNT_ROOT = _Path("/")

def assert_expected_mounts() -> None:
    """Fail loud at boot if any expected /models/<slug> mount is missing."""
    from provisioning.manifest import expected_mount_paths
    missing = [p for p in expected_mount_paths()
               if not (MOUNT_ROOT / p.lstrip("/")).exists()]
    if missing:
        raise RuntimeError(f"missing mount(s): {missing} — check create_space.py manifest")
```

In `_probe_filesystem()` (which already runs only on ZeroGPU), call `assert_expected_mounts()` at the end (wrapped so the FS-probe logging still prints first).

- [ ] **Step 4: Run tests + full suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_boot_probe.py
git commit -m "Boot probe: fail loud if an expected mount is missing on ZeroGPU"
```

---

## Task 7 (OPS): Provision the bf16 mirrors + shared/preproc, against staging

> **This runs real HF operations on the maintainer's account.** Do dry-runs first; execute deliberately. Requires `hf auth login` with write scope and HF PRO.

- [ ] **Step 1: Dry-run everything**

```bash
.venv/bin/python scripts/convert_to_bf16.py --dry-run
.venv/bin/python scripts/duplicate_upstream.py --dry-run
.venv/bin/python scripts/create_space.py --dry-run --space techfreakworm/wan-studio-staging
```
Confirm the printed plans/paths look right; fix any upstream-path drift in `preproc_manifest.py` (the verify-at-execution note in Task 3).

- [ ] **Step 2: Convert one small checkpoint first (cheap canary)**

```bash
.venv/bin/python scripts/convert_to_bf16.py --only wan2.1_t2v_1.3b
```
Expected: `techfreakworm/wan2.1-t2v-1.3b-bf16` created. Verify on the Hub that it has a bf16 `transformer/` + `scheduler/` + `tokenizer/` + `model_index.json` and NO `text_encoder/`/`vae/`.

- [ ] **Step 3: Run the full conversion as an HF Job (server-side)**

Submit `scripts/convert_to_bf16.py` (no `--only`) as an HF Job so the ~450 GB transfer stays in HF's datacenter. (Use `hf jobs run` with a CPU flavor + the repo as context, or run on a throwaway big-disk CPU Space.) Then `build_shared_encoders` + `build_preproc`:

```bash
.venv/bin/python scripts/duplicate_upstream.py        # builds shared-encoders + wan-preproc + vendored as-is
```

- [ ] **Step 4: Create the staging Space + mount manifest**

```bash
.venv/bin/python scripts/create_space.py --space techfreakworm/wan-studio-staging
hf upload techfreakworm/wan-studio-staging . --repo-type=space
```

- [ ] **Step 5: Verify boot**

Open the staging Space logs; confirm `assert_expected_mounts()` passes (no "missing mount") and the default Wan2.2-T2V handle preloads. (No commit — this is an ops step.)

---

## Task 8 (OPS): Run the gating spikes; wire tier-2 if validated

> Per program doc §6. These results gate later design decisions.

- [ ] **Step 1: Mount-cap spike (program R5)**

With all ~15 mounts in the manifest, confirm the staging Space attaches them all (logs show every `/models/<slug>`). If some fail to mount → consolidate related checkpoints into fewer multi-subfolder repos and update `manifest.py` + `_model_volumes`.

- [ ] **Step 2: Transformer-only `from_pretrained` tolerance spike (program R9)**

On staging, exercise a T2V Fast generate (loads the transformer-only bf16 mirror with injected shared encoders). Confirm `WanPipeline.from_pretrained` loads cleanly from a dir with no `text_encoder/`/`vae/`. (VACE/Animate get the same check in their phases.)

- [ ] **Step 3: Wire `tier2_warm_copy` (deferred from #0b) + validate**

Now that mirrors load on a real ZeroGPU box, wire `tier2_warm_copy(_slug_for(card), stitched_dir)` into the load path (e.g. in `handle._build_pipeline` on ZeroGPU only, guarded by an approx-size threshold + ENOSPC skip), measure first-vs-warm read time, and confirm the 150 GB cap holds with LRU eviction. Commit the wiring once validated on staging.

```bash
git add pipelines/handle.py
git commit -m "Wire tier-2 warm-copy into the load path (validated on staging)"
```

- [ ] **Step 4: Promote to live**

Once staging is green: `python scripts/create_space.py --space techfreakworm/wan-studio` then `hf upload techfreakworm/wan-studio . --repo-type=space`.

---

## Self-review checklist (completed)

- **Spec coverage:** bf16 transformer-only conversion (T2 ✓), Animate keeps image_processor/encoder (T2 ✓ amendment 2), shared-encoders mirror w/ correct dtypes (T3 ✓ amendment 3), wan-preproc incl. Animate ViTPose/YOLO/SAM2 (T3 ✓ amendment 1), vendored S2V/TI2V as-is (T4 ✓), atomic single-source manifest (T1/T5 ✓ amendment 7 / R4), boot probe (T6 ✓), mount-cap + transformer-only spikes (T8 ✓ §6), tier-2 wiring deferred-then-done (T8 ✓ closing #0b Bug 2). **Out of scope (other plans):** HANDLER_REGISTRY/per-tier handlers (done in #0b), vendored bf16 conversion (#3), Gallery bucket (#4).
- **Placeholder scan:** none — every code step has real code. Two explicit verify-at-execution flags (upstream preproc paths in T3; the `_recast_safetensors` helper noted in T4 Step 1) are called out, not hidden TODOs.
- **Type consistency:** `manifest.all_volumes()/expected_mount_paths()` (T1) used in T5/T6; `conversion_plan(card) -> ConversionPlan|None` (T2) used in `convert_to_bf16` (T2); `SHARED_ENCODERS`/`PREPROC_ASSETS` (T3) consumed in T4; `build_volume_specs()` (T5) tested in T5; `assert_expected_mounts()`/`MOUNT_ROOT` (T6) tested in T6.

---

## Execution handoff

Tasks 1–6 are local code+TDD (subagent-driven, like #0b). Tasks 7–8 are OPS — execute interactively when ready (HF auth + PRO required), starting with the 1.3B canary before the full HF-Job conversion.
