# Wan Studio Foundation #0b — Runtime Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runtime loading/lifecycle core of the Wan Studio storage foundation — context-aware model-path resolution (never silently downloads on ZeroGPU), an in-process LRU that keeps ≤1 transformer warm, a centralized shared-encoder mount, a tier-2 `/tmp` warm cache, and a `HANDLER_REGISTRY` plugin pattern so every later mode phase is append-only.

**Architecture:** Pure-Python changes to `pipelines/` + `utils/` + `app.py`, all unit-testable on Apple Silicon MPS without the live Space. Weights are resolved through a `mirror_repo` (bf16) field on each `ModelCard`; on ZeroGPU a missing mount fails loud, locally it downloads the bf16 mirror to the persistent HF cache. A module-level `ModelRegistry.acquire(key)` evicts the previously-warm transformer before building the next. Shared encoders (UMT5/VAE/CLIP + image_processor) load once at module scope. `t2v`/`i2v` register into a `HANDLER_REGISTRY` that drives `app.py`'s Generate wiring.

**Tech Stack:** Python 3.12 · diffusers 0.38 · transformers 5.9 · torch 2.11 · gradio 6.14 · spaces 0.50.2 · pytest 9 · MPS (local) / ZeroGPU bf16 (Space)

**References:** [`#0 spec`](../specs/2026-06-08-wan-studio-storage-latency-foundation-design.md) (incl. §14 amendments) · [`program architecture & risks`](../specs/2026-06-08-wan-studio-program-architecture-and-risks.md)

**Companion plan (separate):** `#0a Provisioning` (bf16 conversion + mirrors + mount manifest) — executed at deploy time.

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `requirements.txt` | modify | add `hf_xet`, `hf_transfer` |
| `app.py` | modify (L17-25, L141-..., wiring loop) | drop `HF_HUB_DISABLE_XET`; per-tier `@spaces.GPU` entrypoints; iterate `HANDLER_REGISTRY`; route handle caching through `ModelRegistry` |
| `pipelines/registry.py` | modify | add `mirror_repo` field + populate all cards; add `wan2.1_v2v_14b` card |
| `pipelines/shared.py` | modify | resolve shared encoders from the `wan-shared-encoders` mount; add `image_processor()` |
| `pipelines/handle.py` | modify | context-aware `_mount_path` (fail-loud ZeroGPU / `mirror_repo` local); `tier2_warm_copy`; `ModelRegistry` LRU |
| `pipelines/handlers.py` | **create** | `HandlerSpec` + `HANDLER_REGISTRY` + `register()` |
| `pipelines/__init__.py` | modify | import the per-mode modules so they self-register; export new symbols |
| `tests/test_registry.py` | modify | `mirror_repo` + v2v card consistency |
| `tests/test_shared.py` | **create** | shared-encoder path resolver + image_processor |
| `tests/test_path_resolution.py` | **create** | ZeroGPU-raise / local-mirror / stitched-present |
| `tests/test_lru.py` | **create** | `ModelRegistry.acquire` eviction semantics |
| `tests/test_handlers.py` | **create** | `HANDLER_REGISTRY` registration |

**Slug convention (unchanged):** `card.key.replace("_","-")` → mount `/models/<slug>`. The bf16 mirror repo is `techfreakworm/<slug>-bf16` (shared-encoders mirror is `techfreakworm/wan-shared-encoders`).

---

## Task 1: Dependencies + enable Xet transfer

**Files:**
- Modify: `requirements.txt`
- Modify: `app.py:12-25` (the cache-redirect block)

- [ ] **Step 1: Add transfer deps to `requirements.txt`**

Append under the `# HF infrastructure` group:

```
hf_xet>=1.0           # Xet chunk-dedup + parallel transfer for >1GB safetensors
hf_transfer>=0.1.8    # accelerated HTTP fallback (HF_HUB_ENABLE_HF_TRANSFER=1)
```

- [ ] **Step 2: Stop disabling Xet in `app.py`**

In the cache-redirect block (`app.py:17-21`), delete the `HF_HUB_DISABLE_XET` line so Xet is used:

```python
import os as _os
_os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache")
_os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
_os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
```

(Remove the `_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")` line entirely.)

- [ ] **Step 3: Install + verify**

Run: `.venv/bin/pip install -r requirements.txt && .venv/bin/python -c "import hf_xet, hf_transfer; print('xet', hf_xet.__version__)"`
Expected: prints a version, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt app.py
git commit -m "Enable Xet + hf_transfer; stop disabling Xet"
```

---

## Task 2: Add `mirror_repo` to the registry + the missing V2V card

**Files:**
- Modify: `pipelines/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
def test_every_card_has_mirror_repo():
    """bf16 mirror repo id, distinct from the upstream fp32 `repo`."""
    for m in ALL_MODELS:
        assert m.mirror_repo, f"{m.key} missing mirror_repo"
        assert m.mirror_repo.startswith("techfreakworm/"), m.mirror_repo
        assert m.mirror_repo != m.repo


def test_mirror_repo_slug_matches_key():
    for m in ALL_MODELS:
        slug = m.key.replace("_", "-")
        assert m.mirror_repo == f"techfreakworm/{slug}-bf16", m.mirror_repo


def test_v2v_card_exists():
    card = BY_KEY["wan2.1_v2v_14b"]
    assert card.mode == "v2v"
    assert card.diffusers_class == "WanVideoToVideoPipeline"
    assert card.requires_image_encoder is False
    assert card.lightning_available is False


def test_v2v_shares_t2v_14b_weights():
    """V2V runs on the T2V-14B backbone; its mirror points at t2v-14b."""
    card = BY_KEY["wan2.1_v2v_14b"]
    assert card.mirror_repo == "techfreakworm/wan2.1-t2v-14b-bf16"
```

Note: `test_v2v_shares_t2v_14b_weights` deliberately expects `mirror_repo` to NOT follow the key-slug rule for V2V (it reuses T2V-14B weights). This makes `test_mirror_repo_slug_matches_key` need an exemption — see Step 3.

- [ ] **Step 2: Run, observe failures**

Run: `.venv/bin/pytest tests/test_registry.py -k "mirror or v2v" -v`
Expected: FAIL — `ModelCard` has no `mirror_repo`, no `wan2.1_v2v_14b` key.

- [ ] **Step 3: Add the field + a helper + the V2V card**

In `pipelines/registry.py`, add `mirror_repo` to the dataclass (after `repo`):

```python
    repo: str                             # upstream HF repo path (fp32)
    mirror_repo: str = ""                 # bf16 mirror (techfreakworm/<slug>-bf16); defaulted post-init
```

Add a `__post_init__` to default it from the slug (frozen dataclass → use `object.__setattr__`):

```python
    def __post_init__(self):
        if not self.mirror_repo:
            slug = self.key.replace("_", "-")
            object.__setattr__(self, "mirror_repo", f"techfreakworm/{slug}-bf16")
```

Append the V2V card to `WAN_2_1` (it reuses the T2V-14B weights, so set `mirror_repo` explicitly to the t2v-14b mirror):

```python
    ModelCard(
        key="wan2.1_v2v_14b",
        generation="wan2.1", mode="v2v",
        repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        mirror_repo="techfreakworm/wan2.1-t2v-14b-bf16",  # shares the T2V-14B backbone
        size="14B",
        native_resolutions=("480p", "720p"), native_fps=16, frames_default=81,
        diffusers_class="WanVideoToVideoPipeline",
        is_moe=False, requires_image_encoder=False,
        lightning_available=False,
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=40, quality_guidance=5.0, flow_shift=5.0,  # 720p; 3.0 for 480p
        zerogpu_duration=90,
        notes="Restyle on the T2V-14B backbone (WanVideoToVideoPipeline). Quality-only.",
    ),
```

Relax `test_mirror_repo_slug_matches_key` to exempt shared-weight cards — update the test:

```python
def test_mirror_repo_slug_matches_key():
    SHARED_WEIGHT = {"wan2.1_v2v_14b": "techfreakworm/wan2.1-t2v-14b-bf16"}
    for m in ALL_MODELS:
        if m.key in SHARED_WEIGHT:
            assert m.mirror_repo == SHARED_WEIGHT[m.key]
            continue
        assert m.mirror_repo == f"techfreakworm/{m.key.replace('_','-')}-bf16"
```

- [ ] **Step 4: Run all registry tests**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: PASS (the existing `test_wan_2_1_count_matches_research` expects 7 → now 8 with V2V; update that assertion to `== 8` and its docstring).

- [ ] **Step 5: Commit**

```bash
git add pipelines/registry.py tests/test_registry.py
git commit -m "Add mirror_repo (bf16) to ModelCard; add wan2.1_v2v_14b card"
```

---

## Task 3: Centralize shared encoders on the mount + add `image_processor()`

**Files:**
- Modify: `pipelines/shared.py`
- Test: `tests/test_shared.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shared.py`:

```python
"""Tests for pipelines.shared — shared-encoder path resolution + image_processor."""
import os
from pathlib import Path

import pytest

from pipelines import shared


def test_shared_path_prefers_mount(monkeypatch, tmp_path):
    """On ZeroGPU with the shared-encoders mount present, use it."""
    mount = tmp_path / "wan-shared-encoders"
    (mount / "vae").mkdir(parents=True)
    monkeypatch.setattr(shared, "SHARED_MOUNT", mount)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    assert shared._shared_path() == str(mount)


def test_shared_path_falls_back_to_mirror_repo(monkeypatch, tmp_path):
    """Locally (no mount), use the bf16 shared-encoders mirror repo id."""
    monkeypatch.setattr(shared, "SHARED_MOUNT", tmp_path / "absent")
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert shared._shared_path() == shared.SHARED_MIRROR_REPO


def test_shared_path_missing_mount_on_zerogpu_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(shared, "SHARED_MOUNT", tmp_path / "absent")
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    with pytest.raises(RuntimeError, match="wan-shared-encoders"):
        shared._shared_path()


def test_image_processor_is_callable():
    assert callable(shared.image_processor)
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_shared.py -v`
Expected: FAIL — `shared` has no `SHARED_MOUNT`/`_shared_path`/`image_processor`.

- [ ] **Step 3: Implement in `pipelines/shared.py`**

Replace the `_wan22_t2v_path()` helper with a shared-encoders resolver and update the loaders. Add at the top (after imports):

```python
import os
from pathlib import Path

SHARED_MOUNT = Path(os.getenv("WAN_STUDIO_MOUNT_ROOT", "/models")) / "wan-shared-encoders"
SHARED_MIRROR_REPO = "techfreakworm/wan-shared-encoders"


def _shared_path() -> str:
    """Resolve the shared-encoders dir.

    ZeroGPU: the wan-shared-encoders mount (fail loud if absent — never silently
    download ~14 GB into /tmp). Local: the bf16 mirror repo id (downloads once to
    the persistent HF cache).
    """
    if SHARED_MOUNT.exists():
        return str(SHARED_MOUNT)
    if os.getenv("SPACES_ZERO_GPU") is not None:
        raise RuntimeError(
            f"shared-encoders mount missing at {SHARED_MOUNT} — check create_space.py manifest"
        )
    return SHARED_MIRROR_REPO
```

Point `text_encoder()` and `vae()` at `_shared_path()` (replace the `_wan22_t2v_path()` calls), and add `image_processor()`:

```python
@functools.lru_cache(maxsize=1)
def image_encoder():
    """CLIP-ViT-H/14 — used by I2V, FLF2V, Animate, S2V."""
    import torch
    from transformers import CLIPVisionModel
    return CLIPVisionModel.from_pretrained(
        _shared_path(), subfolder="image_encoder", torch_dtype=torch.float32,
    )


@functools.lru_cache(maxsize=1)
def image_processor():
    """CLIPImageProcessor — required by WanAnimatePipeline (Phase #2)."""
    from transformers import CLIPImageProcessor
    return CLIPImageProcessor.from_pretrained(_shared_path(), subfolder="image_processor")
```

Delete the old `_wan22_t2v_path()` function and its hardcoded `Wan-AI/...` fallbacks.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_shared.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pipelines/shared.py tests/test_shared.py
git commit -m "Centralize shared encoders on wan-shared-encoders mount; add image_processor()"
```

---

## Task 4: Context-aware `_mount_path` (fail-loud on ZeroGPU, bf16 mirror locally)

**Files:**
- Modify: `pipelines/handle.py:111-122` (`_mount_path`)
- Test: `tests/test_path_resolution.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_path_resolution.py`:

```python
"""Tests for handle._mount_path context-aware resolution."""
import pytest

from pipelines import handle
from pipelines.registry import BY_KEY


def test_stitched_present_returns_stitched(monkeypatch):
    monkeypatch.setattr(handle, "stitch_local_dir", lambda card: "/tmp/wan-stitched/x")
    assert handle._mount_path(BY_KEY["wan2.1_t2v_14b"]) == "/tmp/wan-stitched/x"


def test_local_missing_mount_returns_bf16_mirror(monkeypatch):
    monkeypatch.setattr(handle, "stitch_local_dir", lambda card: None)
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    card = BY_KEY["wan2.1_t2v_14b"]
    assert handle._mount_path(card) == card.mirror_repo  # NOT the upstream fp32 repo


def test_zerogpu_missing_mount_raises(monkeypatch):
    monkeypatch.setattr(handle, "stitch_local_dir", lambda card: None)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    with pytest.raises(RuntimeError, match="mount .* missing"):
        handle._mount_path(BY_KEY["wan2.1_t2v_14b"])
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_path_resolution.py -v`
Expected: FAIL — current `_mount_path` returns `card.repo` (upstream), never raises.

- [ ] **Step 3: Rewrite `_mount_path` in `pipelines/handle.py`**

```python
def _mount_path(card: ModelCard) -> str:
    """Resolve where the checkpoint lives for from_pretrained().

    Priority:
      1. Stitched local dir (mount + bundled metadata) — zero disk on ZeroGPU.
      2. ZeroGPU + no mount → RAISE (never silently download fp32 into /tmp).
      3. Local + no mount → the bf16 mirror repo (downloads once to persistent cache).
    """
    stitched = stitch_local_dir(card)
    if stitched:
        return stitched
    if os.getenv("SPACES_ZERO_GPU") is not None:
        raise RuntimeError(
            f"mount /models/{_slug_for(card)} missing — check create_space.py manifest"
        )
    return card.mirror_repo
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_path_resolution.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pipelines/handle.py tests/test_path_resolution.py
git commit -m "Context-aware _mount_path: fail-loud on ZeroGPU, bf16 mirror locally"
```

---

## Task 5: Tier-2 `/tmp` warm cache for hot model shards

**Files:**
- Modify: `pipelines/handle.py` (add `tier2_warm_copy` + a constant)
- Test: `tests/test_handle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_handle.py`:

```python
def test_tier2_warm_copy_symlinks_into_tmp(tmp_path, monkeypatch):
    from pipelines import handle
    src = tmp_path / "stitched"
    (src / "transformer").mkdir(parents=True)
    big = src / "transformer" / "model.safetensors"
    big.write_bytes(b"x" * 1024)
    (src / "config.json").write_text("{}")

    hot_root = tmp_path / "hot"
    monkeypatch.setattr(handle, "TIER2_ROOT", hot_root)

    out = handle.tier2_warm_copy("wan2.1-t2v-14b", str(src))
    assert (Path(out) / "transformer" / "model.safetensors").exists()
    assert (Path(out) / "config.json").exists()
    # idempotent
    assert handle.tier2_warm_copy("wan2.1-t2v-14b", str(src)) == out
```

(Ensure `from pathlib import Path` is imported in the test file.)

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_handle.py::test_tier2_warm_copy_symlinks_into_tmp -v`
Expected: FAIL — no `tier2_warm_copy`.

- [ ] **Step 3: Implement in `pipelines/handle.py`**

```python
TIER2_ROOT = Path("/tmp/wan-hot")


def tier2_warm_copy(slug: str, src_dir: str) -> str:
    """Copy a stitched/mounted checkpoint dir into local /tmp once.

    The first read of a >10 GB model over the HF mount is slow (network
    page-faults); copying to local disk makes repeat reads (and forked
    workers) fast. Idempotent via a marker. Caller is responsible for
    LRU-evicting prior hot copies to stay under the 150 GB disk cap.
    """
    src = Path(src_dir)
    dst = TIER2_ROOT / slug
    marker = dst / ".wan_hot_done"
    if marker.exists():
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(f, target)  # resolves symlinks → real local bytes
    marker.touch()
    return str(dst)


def tier2_evict(keep_slug: str) -> None:
    """Remove every hot copy except keep_slug (LRU bound = 1 model)."""
    if not TIER2_ROOT.exists():
        return
    for child in TIER2_ROOT.iterdir():
        if child.is_dir() and child.name != keep_slug:
            shutil.rmtree(child, ignore_errors=True)
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/pytest tests/test_handle.py::test_tier2_warm_copy_symlinks_into_tmp -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipelines/handle.py tests/test_handle.py
git commit -m "Add tier-2 /tmp warm cache (copy + LRU evict)"
```

---

## Task 6: `ModelRegistry` — in-process LRU, ≤1 warm transformer

**Files:**
- Modify: `pipelines/handle.py` (add `ModelRegistry` + module singleton)
- Test: `tests/test_lru.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lru.py`:

```python
"""Tests for handle.ModelRegistry — one-warm-transformer LRU."""
from pipelines.handle import ModelRegistry, WanModelHandle
from pipelines.registry import BY_KEY


class _FakeHandle(WanModelHandle):
    def __init__(self, card):
        super().__init__(card)
        self.unloaded = False

    def ensure_loaded(self):
        self.pipe = object()  # pretend built

    def unload_to_cpu(self):
        self.unloaded = True


def test_acquire_builds_and_caches():
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    h1 = reg.acquire("wan2.1_t2v_14b")
    assert reg.acquire("wan2.1_t2v_14b") is h1  # same key → same warm handle


def test_acquire_evicts_previous_on_switch():
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    h1 = reg.acquire("wan2.1_t2v_14b")
    h2 = reg.acquire("wan2.1_i2v_14b_480p")
    assert h1.unloaded is True          # previous transformer evicted
    assert h2.unloaded is False
    assert reg.warm_key == "wan2.1_i2v_14b_480p"


def test_acquire_unknown_raises():
    import pytest
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    with pytest.raises(KeyError):
        reg.acquire("nope")
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_lru.py -v`
Expected: FAIL — no `ModelRegistry`.

- [ ] **Step 3: Implement in `pipelines/handle.py`**

```python
class ModelRegistry:
    """Holds at-most-one warmed handle. Switching keys evicts the prior one.

    factory(key) -> WanModelHandle builds a fresh handle for a registry key
    (injected so tests can stub it; production passes the HANDLER_REGISTRY
    builder).
    """

    def __init__(self, factory):
        self._factory = factory
        self._handles: dict[str, WanModelHandle] = {}
        self.warm_key: str | None = None

    def acquire(self, key: str) -> WanModelHandle:
        if key not in BY_KEY:
            raise KeyError(f"Unknown model key: {key!r}")
        if self.warm_key == key and key in self._handles:
            return self._handles[key]
        if self.warm_key is not None and self.warm_key != key:
            prev = self._handles.get(self.warm_key)
            if prev is not None:
                prev.unload_to_cpu()
                tier2_evict(_slug_for(prev.card))
        handle = self._handles.get(key) or self._factory(key)
        handle.ensure_loaded()
        self._handles[key] = handle
        self.warm_key = key
        return handle
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_lru.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pipelines/handle.py tests/test_lru.py
git commit -m "Add ModelRegistry LRU (one warm transformer, evict on switch)"
```

---

## Task 7: `HANDLER_REGISTRY` plugin pattern

**Files:**
- Create: `pipelines/handlers.py`
- Modify: `pipelines/t2v.py`, `pipelines/i2v.py` (self-register)
- Modify: `pipelines/__init__.py` (import modules so they register)
- Test: `tests/test_handlers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handlers.py`:

```python
"""Tests for the HANDLER_REGISTRY plugin pattern."""
import pipelines  # noqa: F401 — triggers self-registration
from pipelines.handlers import HANDLER_REGISTRY, HandlerSpec
from pipelines.t2v import T2VHandle
from pipelines.i2v import I2VHandle


def test_t2v_and_i2v_registered():
    assert "t2v" in HANDLER_REGISTRY
    assert "i2v" in HANDLER_REGISTRY


def test_spec_shape():
    spec = HANDLER_REGISTRY["t2v"]
    assert isinstance(spec, HandlerSpec)
    assert spec.handle_cls is T2VHandle
    assert callable(spec.key_for)         # (generation, **ui) -> registry key
    assert spec.key_for("wan2.2") == "wan2.2_t2v_a14b"


def test_i2v_key_for_resolution():
    spec = HANDLER_REGISTRY["i2v"]
    assert spec.key_for("wan2.1", resolution_label="832x480 (16:9)") == "wan2.1_i2v_14b_480p"
    assert spec.key_for("wan2.1", resolution_label="1280x720 (16:9)") == "wan2.1_i2v_14b_720p"
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_handlers.py -v`
Expected: FAIL — no `pipelines.handlers`.

- [ ] **Step 3: Create `pipelines/handlers.py`**

```python
"""HANDLER_REGISTRY — per-mode plugin registration.

Each mode module (t2v.py, i2v.py, vace.py, ...) calls register(...) at import
time. app.py and __init__.py iterate this registry instead of hard-coding
per-mode wiring, so later phases are append-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HandlerSpec:
    mode: str
    handle_cls: type            # WanModelHandle subclass
    key_for: Callable[..., str] # (generation, **ui_kwargs) -> registry key
    tier: str = "large"         # "large" | "xlarge" — @spaces.GPU size literal


HANDLER_REGISTRY: dict[str, HandlerSpec] = {}


def register(spec: HandlerSpec) -> None:
    HANDLER_REGISTRY[spec.mode] = spec
```

- [ ] **Step 4: Self-register in `pipelines/t2v.py`**

At the bottom of `pipelines/t2v.py`:

```python
from pipelines.handlers import HandlerSpec, register


def _t2v_key_for(generation: str, **_ui) -> str:
    return "wan2.2_t2v_a14b" if generation == "wan2.2" else "wan2.1_t2v_14b"


register(HandlerSpec(mode="t2v", handle_cls=T2VHandle, key_for=_t2v_key_for, tier="large"))
```

- [ ] **Step 5: Self-register in `pipelines/i2v.py`**

At the bottom of `pipelines/i2v.py`:

```python
from pipelines.handlers import HandlerSpec, register


def _i2v_key_for(generation: str, *, resolution_label: str = "", **_ui) -> str:
    if generation == "wan2.2":
        return "wan2.2_i2v_a14b"
    return "wan2.1_i2v_14b_720p" if "720" in resolution_label else "wan2.1_i2v_14b_480p"


register(HandlerSpec(mode="i2v", handle_cls=I2VHandle, key_for=_i2v_key_for, tier="large"))
```

- [ ] **Step 6: Ensure registration runs — `pipelines/__init__.py`**

Confirm `pipelines/__init__.py` imports `t2v` and `i2v` (it already imports `T2VHandle`/`I2VHandle`). Add an explicit re-export:

```python
from pipelines.handlers import HANDLER_REGISTRY, HandlerSpec, register  # noqa: F401
```

and append `"HANDLER_REGISTRY", "HandlerSpec", "register"` to `__all__`.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/pytest tests/test_handlers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add pipelines/handlers.py pipelines/t2v.py pipelines/i2v.py pipelines/__init__.py tests/test_handlers.py
git commit -m "Add HANDLER_REGISTRY plugin pattern; t2v/i2v self-register"
```

---

## Task 8: Route app.py through `ModelRegistry` + per-tier `@spaces.GPU` entrypoints

**Files:**
- Modify: `app.py` (handle caches → `ModelRegistry`; two per-tier entrypoints; Generate wiring iterates `HANDLER_REGISTRY`)
- Test: manual smoke (`from app import build; build()`)

> This task refactors the *wiring* without changing T2V/I2V behaviour. The two existing `_build_t2v_handler`/`_build_i2v_handler` closures are replaced by one generic builder driven by `HandlerSpec`, decorated once per tier.

- [ ] **Step 1: Replace the per-mode handle caches with one `ModelRegistry`**

In `app.py`, delete `T2V_HANDLES`/`I2V_HANDLES` and the `_get_t2v_handle`/`_get_i2v_handle`/`_t2v_key_for`/`_i2v_key_for` helpers (now in the handler specs). Add:

```python
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.handle import ModelRegistry, _slug_for

def _build_handle(key: str):
    from pipelines.registry import BY_KEY
    spec = next(s for s in HANDLER_REGISTRY.values()
                if s.key_for(BY_KEY[key].generation) == key or key.startswith(f"{BY_KEY[key].generation}_{s.mode}"))
    return spec.handle_cls.for_key(key)

REGISTRY = ModelRegistry(factory=_build_handle)
```

(Keep `_preload_default_t2v_handle` but have it call `REGISTRY.acquire("wan2.2_t2v_a14b")` instead of populating a dict.)

- [ ] **Step 2: Write one generic generate fn per tier**

```python
def _run(spec, generation, preset_label, resolution_label, duration_s, gen_inputs):
    """Shared body: resolve key, acquire (LRU), configure preset, generate, export."""
    import random, tempfile
    from diffusers.utils import export_to_video
    key = spec.key_for(generation, resolution_label=resolution_label)
    handle = REGISTRY.acquire(key)
    preset = _coerce_preset(preset_label)
    pk = handle.configure_preset(preset)
    frames = handle.generate(preset_kwargs=_merge_kwargs(pk, gen_inputs), **gen_inputs)
    path = tempfile.mkstemp(suffix=".mp4", prefix=f"wan_{spec.mode}_")[1]
    export_to_video(frames, path, fps=16)
    return path, pk.fallback_message
```

Then two decorated entrypoints (size MUST be a literal):

```python
@spaces_gpu_or_noop()(duration=_get_duration, size="large")
def generate_large(mode, *args):
    return _run(HANDLER_REGISTRY[mode], *args)

@spaces_gpu_or_noop()(duration=_get_duration, size="xlarge")
def generate_xlarge(mode, *args):
    return _run(HANDLER_REGISTRY[mode], *args)
```

- [ ] **Step 3: Drive Generate wiring from the registry in `build()`**

Replace the per-mode `.click()` blocks AND the `_generate_toast` loop with one loop over `HANDLER_REGISTRY`; modes not in the registry keep the toast:

```python
WIRED = set(HANDLER_REGISTRY)
for mode, tab in tabs.items():
    if mode in WIRED:
        spec = HANDLER_REGISTRY[mode]
        entry = generate_xlarge if spec.tier == "xlarge" else generate_large
        tab["inputs"]["generate"].click(
            fn=lambda *a, _m=mode: entry(_m, *a),
            inputs=_inputs_for(mode, tab, header),
            outputs=[tab["outputs"]["video"]],
        )
    elif "generate" in tab.get("inputs", {}):
        tab["inputs"]["generate"].click(fn=_generate_toast, inputs=None, outputs=None)
```

- [ ] **Step 4: Smoke-build (no GPU, no model load)**

Run: `.venv/bin/python -c "from app import build; demo = build(); print('built', type(demo).__name__)"`
Expected: prints `built Blocks` with no exception (handlers registered, wiring iterates registry).

- [ ] **Step 5: Run the full unit suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: all PASS (existing + new).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "Route Generate wiring through HANDLER_REGISTRY + ModelRegistry; per-tier @spaces.GPU entrypoints"
```

---

## Task 9: Local MPS smoke test (recreate the deleted one)

**Files:**
- Create: `tests/test_smoke_t2v_local.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_smoke_t2v_local.py`:

```python
"""Local MPS smoke — Wan 2.1 T2V-1.3B end-to-end via the foundation path.

Downloads the bf16 mirror (or upstream fallback) on first run. Skipped off MPS.
"""
from pathlib import Path
import pytest, torch

pytestmark = [
    pytest.mark.slow, pytest.mark.mps,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"),
]


def test_t2v_1_3b_smoke():
    from diffusers.utils import export_to_video
    from pipelines.handle import ModelRegistry, _slug_for
    from pipelines.t2v import T2VHandle
    from pipelines.registry import BY_KEY

    reg = ModelRegistry(factory=lambda k: T2VHandle.for_key(k))
    handle = reg.acquire("wan2.1_t2v_1.3b")
    pk = handle.configure_preset("fast")        # 1.3B has no Lightning → falls back to Quality
    frames = handle.generate(
        prompt="a red panda eating bamboo, daylight",
        negative_prompt="blurry, static",
        height=480, width=832, num_frames=17, seed=42,
        preset_kwargs={"num_inference_steps": 8, "guidance_scale": pk.guidance_scale},
    )
    out = Path("tests/outputs/smoke_t2v_1.3b.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out), fps=16)
    assert out.stat().st_size > 10_000
```

- [ ] **Step 2: Run it (LONG — first run downloads ~16 GB: 1.3B bf16 + shared UMT5)**

Run: `.venv/bin/pytest tests/test_smoke_t2v_local.py -v -s`
Expected on M5 Max: completes, writes a non-empty MP4. (If `techfreakworm/wan2.1-t2v-1.3b-bf16` doesn't exist yet — 0a not run — temporarily set the card's `mirror_repo` to the upstream fp32 repo to validate the path, or run after 0a.)

- [ ] **Step 3: Commit (the .mp4 is gitignored)**

```bash
git add tests/test_smoke_t2v_local.py
git commit -m "Recreate MPS smoke test on Wan 2.1 T2V-1.3B via foundation path"
```

---

## Self-review checklist (completed)

- **Spec coverage:** mirror_repo (T2 ✓), v2v card (T2 ✓, amendment 4), shared-encoder mount + image_processor (T3 ✓, amendments 3), context path fail-loud/local (T4 ✓), tier-2 cache (T5 ✓, §6.3), ModelRegistry LRU (T6 ✓, amendment 6), HANDLER_REGISTRY (T7 ✓, amendment 5), per-tier handlers (T8 ✓, amendment 8), xet (T1 ✓). **Deferred to 0a:** bf16 conversion, mount manifest, boot probe, wan-preproc provisioning (amendments 1,2,7). **Deferred to phases:** stitch already exists (unchanged).
- **Placeholder scan:** none — every code step shows real code; `_merge_kwargs`/`_get_duration`/`_inputs_for` are existing app.py helpers reused (verify names at T8 against current app.py; rename in-plan if they differ).
- **Type consistency:** `ModelCard.mirror_repo` (T2) used in T4/T9; `HandlerSpec.key_for(generation, **ui)` (T7) called in T8; `ModelRegistry(factory=...)` (T6) constructed in T8/T9; `tier2_evict(slug)` (T5) called in T6.

> **Open verify-at-execution note:** T8's `_inputs_for`/`_merge_kwargs`/`_get_duration` reference existing app.py helpers; the executing agent must confirm exact names in the current `app.py` (`_build_t2v_handler` body) and adapt. This is wiring glue, not new logic.

---

## Execution handoff

After this plan: run **0a Provisioning** (separate plan) to create the bf16 mirrors + mounts, then deploy to `wan-studio-staging` and run the §6 spikes (mount cap, transformer-only tolerance).
