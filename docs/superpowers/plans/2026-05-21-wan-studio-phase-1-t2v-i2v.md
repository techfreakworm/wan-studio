# Wan Studio — Phase 1: T2V + I2V on Wan 2.1 + Wan 2.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Wan Studio v0.1 to `techfreakworm/wan-studio` HF ZeroGPU Space with working T2V + I2V tabs, two-preset (Fast Lightning + Quality) toggle, and the explicit Wan 2.1 / Wan 2.2 generation dropdown — all end-to-end on the live Blackwell ZeroGPU MIG slices we just verified.

**Architecture:** Single monolithic Gradio Space. Lightning LoRAs loaded eagerly + toggled at runtime via `set_adapters` / `disable_lora`. Memory routing: small modes → `large` (48 GB MIG 2g) bf16; Wan 2.2 MoE A14B → `xlarge` (96 GB MIG 4g) bf16, no quantization. Duplicated Wan-AI / Kijai repos mounted via `space_volumes` for resilience.

**Tech Stack:** Python 3.12 · PyTorch 2.8+ · diffusers 0.38+ · transformers 4.45+ · Gradio 5.49+ · `spaces` 0.50.2+ · huggingface_hub 1.x

**Companion docs:**
- Design spec: [`docs/superpowers/specs/2026-05-21-wan-studio-design.md`](../specs/2026-05-21-wan-studio-design.md)
- Architecture brief: [`RESEARCH.md`](../../../RESEARCH.md)
- Wireframes: [`wireframes/index.html`](../../../wireframes/index.html)

**Phase 1 deliverable:** Live public Space at `https://huggingface.co/spaces/techfreakworm/wan-studio` with:
- T2V working on Wan 2.1 T2V-14B + Wan 2.2 T2V-A14B (MoE)
- I2V working on Wan 2.1 I2V-14B-480P/720P + Wan 2.2 I2V-A14B (MoE)
- Fast preset (Lightning LoRA, 4 steps, CFG=1.0) for all 4
- Quality preset (no LoRA, 40-50 steps) for all 4
- Generation dropdown + Preset radio in header, sidebar with T2V/I2V active and other modes greyed out
- Smoke-tested locally on MPS (1.3B variants) before deploy

**Out of scope for Phase 1** (covered in later phase plans):
- FLF2V, V2V, VACE, S2V, Animate, TI2V-5B modes
- Cross-mode Send-to chips
- Gallery + Settings pages
- Mobile responsiveness pass
- Examples curation (basic stubs OK)

---

## File structure for Phase 1

| Path | Status | Responsibility |
|---|---|---|
| `requirements.txt` | exists, needs minor update | Pinned deps |
| `app.py` | exists (Phase-0 scaffold), needs Phase-1 wiring | Gradio entry, tab routing, header/sidebar wiring |
| `utils/backend.py` | exists | Backend detection (MPS / ZeroGPU large / xlarge) |
| `utils/__init__.py` | exists | Re-exports |
| `utils/budget.py` | **new** | Per-mode `get_duration(mode_key, **kwargs)` for `@spaces.GPU(duration=callable)` |
| `pipelines/registry.py` | exists (Phase-0 scaffold) | ModelCard catalog. Phase 1 only uses 5 entries; rest stay defined but unused |
| `pipelines/shared.py` | exists | UMT5 / VAE / CLIP shared loaders. Phase 1 needs all three |
| `pipelines/preset.py` | exists | Fast/Quality resolver. Add tests in Phase 1 |
| `pipelines/handle.py` | **new** | `WanModelHandle` — lazy pipeline build, preset toggle, LoRA load |
| `pipelines/t2v.py` | **new** | T2V-specific wrapper around `WanPipeline` |
| `pipelines/i2v.py` | **new** | I2V-specific wrapper around `WanImageToVideoPipeline` |
| `pipelines/__init__.py` | exists | Re-exports |
| `ui/header.py` | exists | Already wires Generation dropdown + Preset radio |
| `ui/sidebar.py` | exists | Mode buttons |
| `ui/tabs.py` | exists, needs Phase-1 wiring | T2V + I2V tabs need `Generate.click()` → pipeline call |
| `ui/__init__.py` | exists | Re-exports |
| `scripts/duplicate_upstream.py` | **new** | One-shot duplication of 4 base + 1 LoRA mirror repos |
| `scripts/create_space.py` | **new** | `api.create_repo(space_volumes=[...])` + hardware request |
| `tests/test_backend.py` | **new** | Device detection unit tests |
| `tests/test_registry.py` | **new** | Registry consistency unit tests |
| `tests/test_preset.py` | **new** | Fast/Quality resolver unit tests |
| `tests/test_smoke_t2v_local.py` | **new** | Local MPS smoke test on Wan 2.1 T2V-1.3B |
| `tests/conftest.py` | **new** | pytest config (`PYTHONPATH=.` etc.) |
| `pytest.ini` | **new** | pytest configuration |
| `.gitignore` | **new** | Standard Python + HF cache + .venv |
| `NOTICE.md` | **new** | Apache 2.0 attribution to upstream Wan-AI / Kijai / lightx2v |
| `README.md` | exists, needs Phase-1 update | Public Space landing page + `preload_from_hub` YAML frontmatter |

---

## Task 1: Initialize git repo and commit existing scaffold

**Files:**
- Create: `/Users/techfreakworm/Projects/llm/wan-studio/.gitignore`
- Existing scaffold files already in place: `RESEARCH.md`, `requirements.txt`, `app.py`, `pipelines/*`, `ui/*`, `utils/*`, etc.

- [ ] **Step 1: Create `.gitignore`**

Write to `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Virtual environment
.venv/
venv/
ENV/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Local dev artifacts
/tmp/
.superpowers/
*.log

# HF cache (local dev only — Space mounts handle prod)
.cache/

# Output videos from local smoke tests
/tests/outputs/
*.mp4

# Codex generated images (local-only)
~/.codex/
```

- [ ] **Step 2: Initialize git and stage**

Run from `/Users/techfreakworm/Projects/llm/wan-studio/`:

```bash
cd /Users/techfreakworm/Projects/llm/wan-studio
git init
git config user.email "techfreakworm@gmail.com"
git config user.name "Mayank Gupta"
git add .gitignore RESEARCH.md README.md requirements.txt
git add app.py pipelines/ ui/ utils/ tests/__init__.py
git add docs/ wireframes/index.html wireframes/all_wireframes.png
git add wireframes/w*.png
```

Do NOT add `raw/`, `.superpowers/`, or `wireframes/_codex_*`. Verify with `git status`.

- [ ] **Step 3: Initial commit**

```bash
git commit -m "Initial scaffold: research, design spec, Phase-0 module skeletons, wireframes"
```

Expected output: `[main (root-commit) <sha>] Initial scaffold...` with file count >20.

---

## Task 2: Set up Python 3.12 venv and install dependencies

**Files:**
- Modify: `requirements.txt` (bump versions per Phase 1 needs)

- [ ] **Step 1: Update `requirements.txt`**

Replace the entire content with:

```
# Wan Studio — Phase 1 pinned dependencies
# Verified empirically against the ZeroGPU Blackwell MIG environment:
#   torch 2.11.0+cu130 / Python 3.12.12 / ZEROGPU_V2=true

# Core ML
torch>=2.8.0,<2.12          # sm_120 Blackwell support, ZeroGPU runtime upper-bound
diffusers>=0.38.0           # load_into_transformer_2 for Wan 2.2 MoE LoRA (PR #12074)
transformers>=4.45          # UMT5EncoderModel, CLIPVisionModel
accelerate>=0.34
peft>=0.13

# HF infrastructure
huggingface_hub>=1.0,<2.0   # v1.x API — note HfFolder removed
spaces>=0.50.2              # ZeroGPU @spaces.GPU + AOTI API
gradio>=5.49.0              # HfFolder import compat fix landed here

# Quantization (ZeroGPU-only, guarded — not installed locally on MPS)
torchao>=0.7

# Video / image I/O
opencv-python-headless>=4.10
imageio>=2.36
imageio-ffmpeg>=0.5
Pillow>=11.0
numpy>=2.0
einops>=0.8

# Audio (S2V mode — preload for parity even though Phase 1 doesn't use it)
librosa>=0.10
soundfile>=0.13

# Dev tooling
pytest>=8.0
ruff>=0.6
```

- [ ] **Step 2: Create venv + install**

```bash
cd /Users/techfreakworm/Projects/llm/wan-studio
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: install succeeds; `pip list` shows torch, diffusers, gradio, spaces all at the pinned versions.

- [ ] **Step 3: Sanity-check imports**

```bash
.venv/bin/python -c "import torch, diffusers, transformers, gradio, spaces; print('torch', torch.__version__, 'diffusers', diffusers.__version__, 'gradio', gradio.__version__)"
```

Expected output (versions may differ slightly):
```
torch 2.8.0
diffusers 0.38.x
gradio 5.49.x
```

If any import fails, fix the version pin before continuing.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Pin Phase-1 dependencies; bump gradio for hf_hub v1 compat"
```

---

## Task 3: Add pytest config and conftest

**Files:**
- Create: `pytest.ini`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers --tb=short
markers =
    slow: tests that take >10 seconds (real pipeline runs)
    mps: tests that require Apple Silicon MPS backend
    cuda: tests that require CUDA backend
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Pytest config for Wan Studio.

Adds the project root to sys.path so tests can `from pipelines import ...`
without an installed package.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Smoke-run pytest**

```bash
.venv/bin/pytest tests/ -v
```

Expected: `no tests ran in <time>` with 0 errors (no test files yet, but conftest must load cleanly).

- [ ] **Step 4: Commit**

```bash
git add pytest.ini tests/conftest.py
git commit -m "Add pytest config + conftest with project-root sys.path"
```

---

## Task 4: Write `tests/test_backend.py` for backend detection

**Files:**
- Create: `tests/test_backend.py`
- Reference: `utils/backend.py` (already exists)

- [ ] **Step 1: Write failing tests**

```python
"""Tests for utils.backend — device detection + dtype selection."""
import torch
from utils.backend import detect, Backend, spaces_gpu_or_noop


def test_detect_returns_backend_instance():
    backend = detect()
    assert isinstance(backend, Backend)


def test_detect_device_is_one_of_known():
    backend = detect()
    assert backend.device in ("cuda", "mps", "cpu")


def test_detect_vae_dtype_is_always_float32():
    """VAE must stay fp32 on every backend per RESEARCH §7.2."""
    backend = detect()
    assert backend.vae_dtype == torch.float32


def test_detect_mps_uses_float16_transformer():
    backend = detect()
    if backend.device == "mps":
        assert backend.dtype == torch.float16, (
            "MPS bf16 is patchy as of mid-2026; transformer must be fp16"
        )


def test_detect_cuda_uses_bfloat16():
    backend = detect()
    if backend.device == "cuda":
        assert backend.dtype == torch.bfloat16


def test_zerogpu_flag_false_outside_space(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    backend = detect()
    assert backend.is_zerogpu is False


def test_spaces_gpu_decorator_is_noop_offline():
    """Outside ZeroGPU, the decorator must not modify the function."""
    deco = spaces_gpu_or_noop()

    @deco(duration=60)
    def my_fn(x):
        return x * 2

    assert my_fn(21) == 42


def test_backend_label_is_human_readable():
    backend = detect()
    assert isinstance(backend.label, str)
    assert len(backend.label) > 0
```

- [ ] **Step 2: Run tests, observe pass (backend.py already exists)**

```bash
.venv/bin/pytest tests/test_backend.py -v
```

Expected: all 8 tests PASS on MPS (your M5 Max).

- [ ] **Step 3: If any fail, patch `utils/backend.py` until green**

The most likely failure is `test_spaces_gpu_decorator_is_noop_offline` if `spaces_gpu_or_noop()` was implemented differently from what the test expects. Read the current implementation, adjust either the test or the function.

- [ ] **Step 4: Commit**

```bash
git add tests/test_backend.py
git commit -m "Add backend detection tests (device, dtype, ZeroGPU flag, decorator noop)"
```

---

## Task 5: Write `tests/test_registry.py` for model registry

**Files:**
- Create: `tests/test_registry.py`
- Reference: `pipelines/registry.py` (already exists)

- [ ] **Step 1: Write registry consistency tests**

```python
"""Tests for pipelines.registry — ModelCard catalog consistency."""
import pytest

from pipelines.registry import (
    ALL_MODELS,
    BY_KEY,
    ModelCard,
    WAN_2_1,
    WAN_2_2,
    for_generation,
    for_mode,
    modes_in,
)


def test_no_duplicate_keys():
    keys = [m.key for m in ALL_MODELS]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


def test_by_key_lookup_matches_all_models():
    assert len(BY_KEY) == len(ALL_MODELS)
    for m in ALL_MODELS:
        assert BY_KEY[m.key] is m


def test_wan_2_1_count_matches_research():
    """RESEARCH §2: 7 checkpoint families for Wan 2.1."""
    assert len(WAN_2_1) == 7


def test_wan_2_2_count_matches_research():
    """RESEARCH §2: 5 checkpoint families for Wan 2.2."""
    assert len(WAN_2_2) == 5


def test_every_card_has_required_fields():
    for m in ALL_MODELS:
        assert m.repo
        assert m.size
        assert m.generation in ("wan2.1", "wan2.2")
        assert m.native_fps > 0
        assert m.quality_steps > 0
        assert m.flow_shift > 0


def test_lightning_consistency():
    """If lightning_available, must have either lightning_high_lora set."""
    for m in ALL_MODELS:
        if m.lightning_available:
            assert m.lightning_high_lora, f"{m.key} marked lightning_available but no LoRA path"
            assert m.lightning_steps > 0
            assert m.lightning_guidance == 1.0, "CFG-distilled LoRAs require CFG=1.0"


def test_moe_models_are_wan22_a14b():
    moe = [m for m in ALL_MODELS if m.is_moe]
    assert all(m.generation == "wan2.2" for m in moe)
    assert all("a14b" in m.key.lower() for m in moe)


def test_diffusers_class_present_for_native_modes():
    for m in ALL_MODELS:
        if m.mode in ("s2v", "ti2v"):
            # These vendor upstream wan package — no diffusers class
            assert m.diffusers_class is None, f"{m.key} should have no diffusers_class"
        else:
            assert m.diffusers_class, f"{m.key} missing diffusers_class"


def test_for_generation_filters_correctly():
    assert all(m.generation == "wan2.1" for m in for_generation("wan2.1"))
    assert all(m.generation == "wan2.2" for m in for_generation("wan2.2"))


def test_modes_in_returns_canonical_order():
    modes = modes_in("wan2.1")
    assert modes == [m for m in ["t2v", "i2v", "ti2v", "flf2v", "v2v", "vace", "s2v", "animate"] if m in modes]
```

- [ ] **Step 2: Run, observe**

```bash
.venv/bin/pytest tests/test_registry.py -v
```

If `test_wan_2_1_count_matches_research` fails (count != 7), the existing registry might miss a checkpoint. The 7 Wan 2.1 entries are: T2V-1.3B, T2V-14B, I2V-14B-480P, I2V-14B-720P, FLF2V-14B-720P, VACE-1.3B, VACE-14B.

If `test_wan_2_2_count_matches_research` fails (count != 5), the 5 Wan 2.2 entries are: TI2V-5B, T2V-A14B, I2V-A14B, S2V-14B, Animate-14B.

- [ ] **Step 3: Fix registry if any test fails**

The Phase-0 scaffold's `pipelines/registry.py` already has 7 + 5 entries. If a test fails, read the registry and patch — likely a missing field on one entry.

- [ ] **Step 4: Commit**

```bash
git add tests/test_registry.py
git commit -m "Add registry consistency tests (counts, fields, lightning, MoE)"
```

---

## Task 6: Write `tests/test_preset.py` for Fast/Quality resolver

**Files:**
- Create: `tests/test_preset.py`
- Reference: `pipelines/preset.py` (already exists)

- [ ] **Step 1: Write resolver tests**

```python
"""Tests for pipelines.preset — Fast/Quality resolver + graceful fallback."""
from pipelines.preset import resolve, PresetKwargs
from pipelines.registry import BY_KEY


def test_fast_preset_on_supported_mode():
    """Wan 2.1 T2V-14B has Lightning LoRA → Fast preset uses 4 steps + CFG=1."""
    card = BY_KEY["wan2.1_t2v_14b"]
    kwargs = resolve(card, "fast")

    assert isinstance(kwargs, PresetKwargs)
    assert kwargs.effective_preset == "fast"
    assert kwargs.num_inference_steps == 4
    assert kwargs.guidance_scale == 1.0
    assert kwargs.lora_active is True
    assert kwargs.fallback_message is None


def test_quality_preset_on_supported_mode():
    """Quality preset disables LoRA and uses base steps."""
    card = BY_KEY["wan2.1_t2v_14b"]
    kwargs = resolve(card, "quality")

    assert kwargs.effective_preset == "quality"
    assert kwargs.num_inference_steps == card.quality_steps
    assert kwargs.guidance_scale == card.quality_guidance
    assert kwargs.lora_active is False


def test_fast_preset_falls_back_on_unsupported_mode():
    """Wan 2.1 T2V-1.3B has no Lightning LoRA → Fast falls back to Quality."""
    card = BY_KEY["wan2.1_t2v_1.3b"]
    kwargs = resolve(card, "fast")

    assert kwargs.effective_preset == "quality"
    assert kwargs.num_inference_steps == card.quality_steps
    assert kwargs.lora_active is False
    assert kwargs.fallback_message is not None
    assert "Lightning unavailable" in kwargs.fallback_message


def test_moe_fast_preset_sets_both_guidance_scales():
    """Wan 2.2 T2V-A14B (MoE) → Fast sets guidance_scale_2 too."""
    card = BY_KEY["wan2.2_t2v_a14b"]
    kwargs = resolve(card, "fast")

    assert card.is_moe
    assert kwargs.guidance_scale == 1.0
    assert kwargs.guidance_scale_2 == 1.0


def test_moe_quality_preset_sets_both_guidance_scales():
    card = BY_KEY["wan2.2_t2v_a14b"]
    kwargs = resolve(card, "quality")

    assert kwargs.guidance_scale_2 is not None
    assert kwargs.guidance_scale_2 == card.quality_guidance


def test_non_moe_quality_leaves_guidance_2_none():
    """Single-transformer models don't have a low-noise CFG."""
    card = BY_KEY["wan2.1_t2v_14b"]
    kwargs = resolve(card, "quality")
    assert kwargs.guidance_scale_2 is None
```

- [ ] **Step 2: Run**

```bash
.venv/bin/pytest tests/test_preset.py -v
```

Expected: all 6 PASS. If a Wan 2.2 MoE quality test fails because `quality_guidance` is the high-noise value but `guidance_scale_2` should be the low-noise value, patch `pipelines/preset.py` to read `card.lightning_guidance` for the low-noise stage at quality preset — wait, actually the per-mode quality CFG comes from the registry. The registry's `quality_guidance` is the high-noise value. For Wan 2.2 MoE we need a separate `quality_guidance_2`. **Add `quality_guidance_2: float | None = None` to `ModelCard`** if missing, and populate it for MoE entries (T2V-A14B: 4.0; I2V-A14B: 3.5). Update `preset.py` to use it.

- [ ] **Step 3: Fix preset.py + registry if needed**

If the test for MoE quality CFG_2 fails, add to `ModelCard`:

```python
@dataclass(frozen=True)
class ModelCard:
    # ... existing fields ...
    quality_guidance_2: float | None = None  # MoE low-noise stage; None for non-MoE
```

And in `pipelines/preset.py`:

```python
def resolve(card: ModelCard, requested: Preset) -> PresetKwargs:
    # ... existing fallback logic ...

    if requested == "fast":
        return PresetKwargs(
            num_inference_steps=card.lightning_steps,
            guidance_scale=card.lightning_guidance,
            guidance_scale_2=card.lightning_guidance if card.is_moe else None,
            flow_shift=card.flow_shift,
            lora_active=True,
            effective_preset="fast",
            fallback_message=None,
        )

    # quality
    return PresetKwargs(
        num_inference_steps=card.quality_steps,
        guidance_scale=card.quality_guidance,
        guidance_scale_2=card.quality_guidance_2 if card.is_moe else None,
        flow_shift=card.flow_shift,
        lora_active=False,
        effective_preset="quality",
        fallback_message=None,
    )
```

Then update `WAN_2_2` MoE entries in `pipelines/registry.py` to set `quality_guidance_2=4.0` for T2V-A14B and `quality_guidance_2=3.5` for I2V-A14B.

- [ ] **Step 4: Commit**

```bash
git add tests/test_preset.py pipelines/preset.py pipelines/registry.py
git commit -m "Add preset resolver tests; add quality_guidance_2 for MoE low-noise stage"
```

---

## Task 7: Write `utils/budget.py` — ZeroGPU duration callable

**Files:**
- Create: `utils/budget.py`
- Create: `tests/test_budget.py`
- Modify: `utils/__init__.py` (re-export)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_budget.py
"""Tests for utils.budget — get_duration() per mode."""
import pytest

from utils.budget import duration_for, MODE_BUDGET
from pipelines.registry import BY_KEY


def test_duration_for_returns_int():
    d = duration_for("wan2.1_t2v_14b", duration_s=3.0)
    assert isinstance(d, int)
    assert d > 0


def test_duration_for_unknown_key_raises():
    with pytest.raises(KeyError):
        duration_for("unknown_mode", duration_s=1.0)


def test_duration_scales_with_video_length():
    """Longer requested video should yield longer GPU reservation."""
    short = duration_for("wan2.1_t2v_14b", duration_s=1.0)
    long = duration_for("wan2.1_t2v_14b", duration_s=5.0)
    assert long > short


def test_duration_capped_at_500():
    """ZeroGPU practical ceiling per RESEARCH §6.2."""
    d = duration_for("wan2.2_animate_14b", duration_s=20.0)
    assert d <= 500


def test_mode_budget_has_all_phase1_modes():
    for key in ("wan2.1_t2v_14b", "wan2.1_i2v_14b_480p", "wan2.1_i2v_14b_720p",
                "wan2.2_t2v_a14b", "wan2.2_i2v_a14b"):
        assert key in MODE_BUDGET, f"missing budget for {key}"


def test_moe_modes_route_to_xlarge():
    """Wan 2.2 A14B MoE must use xlarge per RESEARCH §6.3."""
    for key in ("wan2.2_t2v_a14b", "wan2.2_i2v_a14b"):
        size, _ = MODE_BUDGET[key]
        assert size == "xlarge", f"{key} should route to xlarge"


def test_small_modes_route_to_large():
    for key in ("wan2.1_t2v_1.3b", "wan2.1_t2v_14b"):
        size, _ = MODE_BUDGET[key]
        assert size == "large"
```

- [ ] **Step 2: Write `utils/budget.py`**

```python
"""ZeroGPU duration budget — per-(mode, generation) tier + duration callable.

Returns the (size, default_seconds) tuple for `@spaces.GPU(duration=callable, size=...)`
and a `duration_for(mode_key, **gen_kwargs)` helper that scales by requested params.

The same `duration_for()` callable is referenced by both:
  - the @spaces.GPU decorator on the inference function
  - the ETA gr.Markdown component in the UI

So display + actual reservation stay in sync.
"""
from __future__ import annotations

from typing import Literal

Size = Literal["large", "xlarge"]


# (size, default_seconds_at_fast_preset)
MODE_BUDGET: dict[str, tuple[Size, int]] = {
    # Wan 2.1 — all single-transformer
    "wan2.1_t2v_1.3b":         ("large",  60),
    "wan2.1_t2v_14b":          ("large",  90),
    "wan2.1_i2v_14b_480p":     ("large",  90),
    "wan2.1_i2v_14b_720p":     ("large", 120),
    "wan2.1_flf2v_14b_720p":   ("large", 150),
    "wan2.1_vace_1.3b":        ("large", 150),
    "wan2.1_vace_14b":         ("large", 180),
    # Wan 2.2 — MoE goes xlarge for bf16 fit
    "wan2.2_ti2v_5b":          ("large",  60),
    "wan2.2_t2v_a14b":         ("xlarge", 120),
    "wan2.2_i2v_a14b":         ("xlarge", 150),
    "wan2.2_s2v_14b":          ("large", 240),
    "wan2.2_animate_14b":      ("xlarge", 300),
}


def duration_for(mode_key: str, *, duration_s: float = 3.0, steps_override: int | None = None) -> int:
    """Return ZeroGPU seconds to reserve for one generation.

    Scales the per-mode default by requested video duration (longer video = more frames = more denoise time).
    Capped at 500s (RESEARCH §6.2 community practical ceiling).
    """
    if mode_key not in MODE_BUDGET:
        raise KeyError(f"No duration budget defined for mode {mode_key!r}")
    _, default_s = MODE_BUDGET[mode_key]
    # Scale by duration: baseline assumes 3s video; add 30% per extra second
    scaled = default_s * (1.0 + 0.3 * max(0.0, duration_s - 3.0))
    if steps_override:
        # If user overrides steps via Advanced, scale roughly linearly past 4
        scaled = scaled * (steps_override / 4.0)
    return min(500, int(scaled))


def size_for(mode_key: str) -> Size:
    """Return the ZeroGPU size tier for a mode (`large` or `xlarge`)."""
    if mode_key not in MODE_BUDGET:
        raise KeyError(f"No size defined for mode {mode_key!r}")
    return MODE_BUDGET[mode_key][0]
```

- [ ] **Step 3: Update `utils/__init__.py`**

Replace with:

```python
from utils.backend import Backend, detect, spaces_gpu_or_noop
from utils.budget import MODE_BUDGET, duration_for, size_for

__all__ = [
    "Backend",
    "detect",
    "spaces_gpu_or_noop",
    "MODE_BUDGET",
    "duration_for",
    "size_for",
]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_budget.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/budget.py utils/__init__.py tests/test_budget.py
git commit -m "Add ZeroGPU duration budget table + duration_for() callable"
```

---

## Task 8: Write `pipelines/handle.py` — WanModelHandle base class

**Files:**
- Create: `pipelines/handle.py`
- Create: `tests/test_handle.py` (lightweight tests; full integration in Task 11)

- [ ] **Step 1: Write `pipelines/handle.py`**

```python
"""WanModelHandle — one handle per (mode, generation, size) tuple.

Lifecycle:
  1. ensure_loaded()   — lazy-build the pipeline from the mounted path, attach LoRA if available
  2. configure_preset()— toggle Fast/Quality via set_adapters() / disable_lora(); return inference kwargs
  3. generate()        — call the pipeline (called from within @spaces.GPU)
  4. unload_to_cpu()   — move transformers to CPU + empty_cache() when switching modes

Only ONE handle's pipeline lives on GPU at a time (managed by app.py orchestrator).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from pipelines.preset import Preset, PresetKwargs, resolve
from pipelines.registry import BY_KEY, ModelCard
from utils.backend import detect


# Path on the deployed Space where mounted volumes appear:
#   /models/wan2.1-t2v-14b/, /models/wan2.2-t2v-a14b/, etc.
# Locally (MPS dev), fall back to standard HF cache.
SPACE_MOUNT_ROOT = Path(os.getenv("WAN_STUDIO_MOUNT_ROOT", "/models"))


def _slug_for(card: ModelCard) -> str:
    """Compute the directory slug used for the duplicated mirror.

    Convention: underscores → dashes, dots preserved. Examples:
      wan2.1_t2v_14b   → wan2.1-t2v-14b   (mirror: techfreakworm/wan2.1-t2v-14b)
      wan2.2_i2v_a14b  → wan2.2-i2v-a14b
    This MUST match Volume(mount_path=...) in scripts/create_space.py.
    """
    return card.key.replace("_", "-")


def _mount_path(card: ModelCard) -> str:
    """Resolve where the checkpoint lives for from_pretrained().

    On ZeroGPU: the duplicated mirror is mounted under /models/<slug>.
    Locally: fall back to the upstream HF repo (downloads on demand to hub cache).
    """
    backend = detect()
    if backend.is_zerogpu:
        candidate = SPACE_MOUNT_ROOT / _slug_for(card)
        if candidate.exists():
            return str(candidate)
    # Locally: upstream repo path triggers cached download via huggingface_hub
    return card.repo


# Path to the mounted Lightning LoRA bundle on ZeroGPU.
LIGHTNING_MIRROR_MOUNT = SPACE_MOUNT_ROOT / "wan-lightning-loras"
LIGHTNING_MIRROR_REPO = "techfreakworm/wan-lightning-loras"


def _lora_repo_for(card: ModelCard) -> str:
    """Resolve where to load Lightning LoRA weights from.

    On ZeroGPU: read from the mounted /models/wan-lightning-loras consolidated mirror.
    Locally: fall back to whatever `card.lightning_lora_repo` points at upstream.
    """
    backend = detect()
    if backend.is_zerogpu and LIGHTNING_MIRROR_MOUNT.exists():
        return str(LIGHTNING_MIRROR_MOUNT)
    assert card.lightning_lora_repo, f"{card.key} missing lightning_lora_repo upstream fallback"
    return card.lightning_lora_repo


class WanModelHandle:
    """Wraps a single (mode, generation, size) combo.

    Concrete pipeline construction is delegated to subclasses by mode
    (T2VHandle, I2VHandle, etc.) via `_build_pipeline()`.
    """

    def __init__(self, card: ModelCard):
        self.card = card
        self.pipe: Any = None  # set in ensure_loaded
        self.lora_loaded: bool = False
        self.current_preset: Preset | None = None

    @classmethod
    def for_key(cls, key: str) -> "WanModelHandle":
        """Look up a card by key and return a fresh handle."""
        if key not in BY_KEY:
            raise KeyError(f"Unknown model key: {key!r}")
        return cls(BY_KEY[key])

    def ensure_loaded(self) -> None:
        """Build the pipeline + attach Lightning LoRA if available. Idempotent."""
        if self.pipe is not None:
            return
        self.pipe = self._build_pipeline()
        self._configure_scheduler()
        if self.card.lightning_available:
            self._load_lightning_lora()
            self.lora_loaded = True

    def configure_preset(self, preset: Preset) -> PresetKwargs:
        """Apply Fast/Quality preset; return inference kwargs."""
        self.ensure_loaded()
        kwargs = resolve(self.card, preset)

        if not self.lora_loaded:
            self.current_preset = kwargs.effective_preset
            return kwargs

        if kwargs.effective_preset == "fast":
            if self.card.is_moe:
                self.pipe.set_adapters(["lightning_high", "lightning_low"], [1.0, 1.0])
            else:
                self.pipe.set_adapters(["lightning"], [1.0])
        else:
            self.pipe.disable_lora()

        self.current_preset = kwargs.effective_preset
        return kwargs

    def unload_to_cpu(self) -> None:
        """Move transformers off GPU. Called when switching active mode."""
        if self.pipe is None:
            return
        if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            self.pipe.transformer.to("cpu")
        if hasattr(self.pipe, "transformer_2") and self.pipe.transformer_2 is not None:
            self.pipe.transformer_2.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Mode-specific overrides (implemented in subclasses) ---

    def _build_pipeline(self) -> Any:
        raise NotImplementedError("Subclass must implement _build_pipeline()")

    def _configure_scheduler(self) -> None:
        """Set UniPCMultistepScheduler with the mode's flow_shift. Override if needed."""
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            flow_shift=self.card.flow_shift,
        )

    def _load_lightning_lora(self) -> None:
        """Attach Lightning LoRA(s) to the transformer(s).

        Wan 2.1 (single transformer): one LoRA call.
        Wan 2.2 MoE: two LoRA calls — HIGH onto transformer, LOW onto transformer_2 with
        `load_into_transformer_2=True`.

        Resolves the LoRA source via `_lora_repo_for()` so the same code path works
        for both ZeroGPU (mounted mirror) and local dev (upstream hub).
        """
        if not self.card.lightning_available:
            return

        lora_repo = _lora_repo_for(self.card)

        # HIGH-noise / single-transformer LoRA
        self.pipe.load_lora_weights(
            lora_repo,
            weight_name=self.card.lightning_high_lora,
            adapter_name="lightning_high" if self.card.is_moe else "lightning",
        )

        if self.card.is_moe:
            assert self.card.lightning_low_lora, "MoE card missing low-noise LoRA path"
            self.pipe.load_lora_weights(
                lora_repo,
                weight_name=self.card.lightning_low_lora,
                adapter_name="lightning_low",
                load_into_transformer_2=True,  # diffusers PR #12074
            )
```

- [ ] **Step 2: Write lightweight tests**

```python
# tests/test_handle.py
"""Tests for pipelines.handle — WanModelHandle.

Heavy integration (real pipeline build) is in test_smoke_t2v_local.py.
"""
import pytest

from pipelines.handle import WanModelHandle
from pipelines.registry import BY_KEY


def test_for_key_returns_handle_with_correct_card():
    h = WanModelHandle.for_key("wan2.1_t2v_14b")
    assert h.card.key == "wan2.1_t2v_14b"
    assert h.pipe is None  # lazy
    assert h.lora_loaded is False


def test_for_key_unknown_raises():
    with pytest.raises(KeyError):
        WanModelHandle.for_key("not_a_real_key")


def test_unload_to_cpu_is_noop_when_not_loaded():
    h = WanModelHandle.for_key("wan2.1_t2v_14b")
    h.unload_to_cpu()  # should not raise
    assert h.pipe is None


def test_build_pipeline_must_be_overridden():
    h = WanModelHandle(BY_KEY["wan2.1_t2v_14b"])
    with pytest.raises(NotImplementedError):
        h._build_pipeline()
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/pytest tests/test_handle.py -v
```

Expected: 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add pipelines/handle.py tests/test_handle.py
git commit -m "Add WanModelHandle base class (lazy pipeline + preset toggle + LoRA)"
```

---

## Task 9: Write `pipelines/t2v.py` — T2V handle subclass

**Files:**
- Create: `pipelines/t2v.py`
- Modify: `pipelines/__init__.py` (re-export)

- [ ] **Step 1: Write `pipelines/t2v.py`**

```python
"""T2V pipeline wrapper — Wan 2.1 T2V-1.3B / 14B (single transformer)
and Wan 2.2 T2V-A14B (MoE: transformer + transformer_2).
"""
from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from pipelines.handle import WanModelHandle, _mount_path
from pipelines import shared
from utils.backend import detect


class T2VHandle(WanModelHandle):
    """Builds WanPipeline. For MoE cards, also loads transformer_2."""

    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)

        if self.card.is_moe:
            # Load both transformers explicitly. transformer_2 is the low-noise expert.
            transformer = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=backend.dtype,
            )
            transformer_2 = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer_2", torch_dtype=backend.dtype,
            )
            pipe = WanPipeline.from_pretrained(
                path,
                transformer=transformer,
                transformer_2=transformer_2,
                vae=shared.vae(),
                text_encoder=shared.text_encoder(),
                torch_dtype=backend.dtype,
            )
        else:
            pipe = WanPipeline.from_pretrained(
                path,
                vae=shared.vae(),
                text_encoder=shared.text_encoder(),
                torch_dtype=backend.dtype,
            )

        pipe.to(backend.device)
        return pipe

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        seed: int = 42,
        preset_kwargs: dict[str, Any],
    ) -> list:
        """Return list of numpy frames. Caller exports via export_to_video."""
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]
```

- [ ] **Step 2: Update `pipelines/__init__.py`**

Add to the imports + `__all__`:

```python
from pipelines.handle import WanModelHandle
from pipelines.t2v import T2VHandle

__all__ = [
    # ... existing exports ...
    "WanModelHandle",
    "T2VHandle",
]
```

- [ ] **Step 3: Smoke-import test**

```bash
.venv/bin/python -c "from pipelines.t2v import T2VHandle; h = T2VHandle.for_key('wan2.1_t2v_1.3b'); print('handle for', h.card.key, 'mode=', h.card.mode)"
```

Expected: `handle for wan2.1_t2v_1.3b mode= t2v`. (Does NOT trigger pipeline build.)

- [ ] **Step 4: Commit**

```bash
git add pipelines/t2v.py pipelines/__init__.py
git commit -m "Add T2VHandle — single-transformer and MoE WanPipeline construction"
```

---

## Task 10: Write `pipelines/i2v.py` — I2V handle subclass

**Files:**
- Create: `pipelines/i2v.py`
- Modify: `pipelines/__init__.py` (re-export)

- [ ] **Step 1: Write `pipelines/i2v.py`**

```python
"""I2V pipeline wrapper — Wan 2.1 I2V-14B-480P/720P (single) and Wan 2.2 I2V-A14B (MoE).

Also handles FLF2V via the same WanImageToVideoPipeline + last_image= kwarg
(used in Phase 2; defined here so the FLF2V handle can subclass).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from PIL import Image

from pipelines.handle import WanModelHandle, _mount_path
from pipelines import shared
from utils.backend import detect


def aspect_ratio_resize(image: Image.Image, pipe: WanImageToVideoPipeline, max_area: int) -> tuple[Image.Image, int, int]:
    """Resize input image to a multiple of vae_scale_factor_spatial * patch_size[1].

    Returns (resized_image, height, width). Helper from RESEARCH §3.2.
    """
    ar = image.height / image.width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = int(round(np.sqrt(max_area * ar))) // mod * mod
    w = int(round(np.sqrt(max_area / ar))) // mod * mod
    return image.resize((w, h)), h, w


class I2VHandle(WanModelHandle):
    """Builds WanImageToVideoPipeline. Handles MoE for Wan 2.2 A14B."""

    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)

        common_kwargs = dict(
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            image_encoder=shared.image_encoder(),
            torch_dtype=backend.dtype,
        )

        if self.card.is_moe:
            transformer = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=backend.dtype,
            )
            transformer_2 = WanTransformer3DModel.from_pretrained(
                path, subfolder="transformer_2", torch_dtype=backend.dtype,
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                path,
                transformer=transformer,
                transformer_2=transformer_2,
                **common_kwargs,
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(path, **common_kwargs)

        pipe.to(backend.device)
        return pipe

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str = "",
        max_area: int = 480 * 832,
        num_frames: int = 81,
        seed: int = 42,
        preset_kwargs: dict[str, Any],
    ) -> list:
        self.ensure_loaded()
        resized, h, w = aspect_ratio_resize(image, self.pipe, max_area)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            image=resized,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=h,
            width=w,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]
```

- [ ] **Step 2: Update `pipelines/__init__.py`**

Add:

```python
from pipelines.i2v import I2VHandle, aspect_ratio_resize

__all__ = [
    # ... existing ...
    "I2VHandle",
    "aspect_ratio_resize",
]
```

- [ ] **Step 3: Smoke-import test**

```bash
.venv/bin/python -c "from pipelines.i2v import I2VHandle; print(I2VHandle.for_key('wan2.1_i2v_14b_480p').card.key)"
```

Expected: `wan2.1_i2v_14b_480p`.

- [ ] **Step 4: Commit**

```bash
git add pipelines/i2v.py pipelines/__init__.py
git commit -m "Add I2VHandle — single/MoE WanImageToVideoPipeline + aspect_ratio_resize"
```

---

## Task 11: Write local MPS smoke test `tests/test_smoke_t2v_local.py`

**Files:**
- Create: `tests/test_smoke_t2v_local.py`

- [ ] **Step 1: Write the smoke test**

```python
"""Local MPS smoke test — Wan 2.1 T2V-1.3B end-to-end.

Skipped unless run on Apple Silicon. Downloads ~3 GB of weights to HF cache on first run.
"""
import shutil
from pathlib import Path

import pytest
import torch

from pipelines.t2v import T2VHandle
from pipelines.preset import resolve
from pipelines.registry import BY_KEY


pytestmark = [
    pytest.mark.slow,
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="Requires Apple Silicon MPS backend"
    ),
]

OUTPUT_DIR = Path("tests/outputs")


def test_wan_2_1_t2v_1_3b_smoke():
    """Generate a 16-frame video at 480p; verify MP4 written."""
    from diffusers.utils import export_to_video

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "smoke_t2v_1.3b.mp4"
    if out_path.exists():
        out_path.unlink()

    handle = T2VHandle.for_key("wan2.1_t2v_1.3b")
    card = BY_KEY["wan2.1_t2v_1.3b"]

    # 1.3B has no Lightning — resolve will fall back to Quality
    preset_kwargs = handle.configure_preset("fast")
    assert preset_kwargs.effective_preset == "quality"  # fallback

    # Override steps to 8 instead of 50 for smoke speed
    inference_kwargs = {
        "num_inference_steps": 8,
        "guidance_scale": preset_kwargs.guidance_scale,
    }
    if preset_kwargs.guidance_scale_2 is not None:
        inference_kwargs["guidance_scale_2"] = preset_kwargs.guidance_scale_2

    frames = handle.generate(
        prompt="A red panda eating bamboo, photorealistic, daylight",
        negative_prompt="static, blurred, low quality",
        height=480,
        width=832,
        num_frames=17,  # 4k+1 minimum
        seed=42,
        preset_kwargs=inference_kwargs,
    )

    assert len(frames) == 17
    export_to_video(frames, str(out_path), fps=16)
    assert out_path.exists()
    assert out_path.stat().st_size > 10_000, "MP4 looks empty"
```

- [ ] **Step 2: Run the smoke test (LONG — first run downloads ~3 GB)**

```bash
.venv/bin/pytest tests/test_smoke_t2v_local.py -v -s
```

Expected on M5 Max MPS: first run takes ~5-10 min for weight download + ~3-5 min for 8-step generation. Subsequent runs: just the generation time.

If it OOMs locally, reduce `num_frames` from 17 → 9, or comment out `image_encoder` from `shared.py` for T2V tests (T2V doesn't use it).

- [ ] **Step 3: Inspect the output video**

```bash
open tests/outputs/smoke_t2v_1.3b.mp4
```

Should show a recognizable (if noisy at 8 steps) red panda.

- [ ] **Step 4: Commit (do NOT commit the .mp4 — it's gitignored)**

```bash
git add tests/test_smoke_t2v_local.py
git commit -m "Add MPS smoke test on Wan 2.1 T2V-1.3B end-to-end"
```

---

## Task 12: Wire Generate button in T2V tab → T2VHandle

**Files:**
- Modify: `ui/tabs.py` (T2V section)
- Modify: `app.py` (handler registration)

- [ ] **Step 1: Refactor `ui/tabs.py` build_t2v_tab to return component handles cleanly**

Open `ui/tabs.py`. The existing `build_t2v_tab` returns a dict with `inputs` and `outputs`. Confirm it has at minimum: `prompt`, `resolution`, `duration`, `negative_prompt`, `seed`, `randomize`, `steps`, `cfg`, `cfg_2`, `generate` (button), `video` (output), `eta` (markdown). If any are missing, add them in the existing layout. Do not change function signatures.

- [ ] **Step 2: Add the click handler in `app.py`**

In `app.py`, after the `_show_only` wiring and before `demo.load(...)`, add:

```python
# ---------------------------------------------------------------------------
# T2V: Generate handler
# ---------------------------------------------------------------------------
from pipelines.t2v import T2VHandle
from pipelines.handle import WanModelHandle
from utils.budget import duration_for, size_for
from utils.backend import spaces_gpu_or_noop

# Module-level handle cache. Only one active on GPU at a time.
T2V_HANDLES: dict[str, T2VHandle] = {}

def _get_t2v_handle(generation: str) -> T2VHandle:
    key = "wan2.2_t2v_a14b" if generation == "wan2.2" else "wan2.1_t2v_14b"
    if key not in T2V_HANDLES:
        T2V_HANDLES[key] = T2VHandle.for_key(key)
    return T2V_HANDLES[key]


def _parse_resolution(label: str) -> tuple[int, int]:
    """'1280x720 (16:9)' → (720, 1280) as (height, width)."""
    rez = label.split(" ")[0]
    w, h = (int(x) for x in rez.split("x"))
    return h, w


@spaces_gpu_or_noop()(duration=lambda *a, **kw: duration_for(_get_t2v_handle(kw.get("generation", "wan2.2")).card.key, duration_s=kw.get("duration_s", 3.0)),
                       size=lambda *a, **kw: size_for(_get_t2v_handle(kw.get("generation", "wan2.2")).card.key))
def generate_t2v(
    prompt: str,
    generation: str,
    preset_label: str,
    resolution_label: str,
    duration_s: float,
    negative_prompt: str,
    seed: int,
    randomize: bool,
    steps_override: int,
    cfg_override: float,
    cfg_2_override: float,
    progress=gr.Progress(track_tqdm=True),
):
    """T2V Generate click handler. Returns (video_path, info_md)."""
    import random
    from diffusers.utils import export_to_video

    if randomize:
        seed = random.randint(0, 2**31 - 1)

    preset = "fast" if preset_label.startswith("Fast") else "quality"
    handle = _get_t2v_handle(generation)
    progress(0.05, desc="Configuring preset...")
    preset_kwargs = handle.configure_preset(preset)

    # Build inference kwargs from preset + overrides
    inference_kwargs = {
        "num_inference_steps": steps_override if steps_override > 0 else preset_kwargs.num_inference_steps,
        "guidance_scale": cfg_override if cfg_override > 0 else preset_kwargs.guidance_scale,
    }
    if preset_kwargs.guidance_scale_2 is not None:
        inference_kwargs["guidance_scale_2"] = cfg_2_override if cfg_2_override > 0 else preset_kwargs.guidance_scale_2

    height, width = _parse_resolution(resolution_label)
    num_frames = max(17, int(duration_s * 16) // 4 * 4 + 1)  # respect 4k+1

    progress(0.2, desc="Generating frames...")
    frames = handle.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height, width=width, num_frames=num_frames,
        seed=seed, preset_kwargs=inference_kwargs,
    )

    progress(0.9, desc="Encoding video...")
    import tempfile
    fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="wan_t2v_")
    os.close(fd)
    export_to_video(frames, out_path, fps=16)

    info = f"**Generated** — {len(frames)} frames @ {width}×{height} · preset={preset_kwargs.effective_preset} · seed={seed}"
    if preset_kwargs.fallback_message:
        info = f"⚠️ {preset_kwargs.fallback_message}\n\n{info}"
    return out_path, info


# Wire the button (after build_all_tabs in the existing app.py)
t2v_inputs = tabs["t2v"]["inputs"]
t2v_outputs = tabs["t2v"]["outputs"]

t2v_inputs["generate"].click(
    fn=generate_t2v,
    inputs=[
        t2v_inputs["prompt"],
        header["generation"],
        header["preset"],
        t2v_inputs["resolution"],
        t2v_inputs["duration"],
        t2v_inputs["negative_prompt"],
        t2v_inputs["seed"],
        t2v_inputs["randomize"],
        t2v_inputs["steps"],
        t2v_inputs["cfg"],
        t2v_inputs["cfg_2"],
    ],
    outputs=[t2v_outputs["video"], t2v_outputs["eta"]],
)
```

Add `import os` and `import gradio as gr` at the top of `app.py` if not present.

- [ ] **Step 3: Smoke-test the app loads locally**

```bash
.venv/bin/python app.py
```

Expected: Gradio launches at `http://localhost:7860`. Open in browser. Click T2V mode in sidebar. Type "a cat", hit Generate. **Locally on MPS, this will use Wan 2.1 T2V-14B which is 14B → too big for 128 GB MPS without offload. Expect OOM or slow.**

For MPS local dev, instead set `WAN_STUDIO_T2V_LOCAL_KEY=wan2.1_t2v_1.3b` and have `_get_t2v_handle` honor it. Add at the top of `_get_t2v_handle`:

```python
local_override = os.getenv("WAN_STUDIO_T2V_LOCAL_KEY")
if local_override and not detect().is_zerogpu:
    if local_override not in T2V_HANDLES:
        T2V_HANDLES[local_override] = T2VHandle.for_key(local_override)
    return T2V_HANDLES[local_override]
```

(Add `from utils.backend import detect` if not already imported.)

Re-run with env var:

```bash
WAN_STUDIO_T2V_LOCAL_KEY=wan2.1_t2v_1.3b .venv/bin/python app.py
```

Generate a 1-second video at 480×832, observe the video appears in the output column. Should take 3-8 min on M5 Max.

- [ ] **Step 4: Commit**

```bash
git add app.py ui/tabs.py
git commit -m "Wire T2V Generate handler — preset-aware, generation-aware, MPS-friendly env override"
```

---

## Task 13: Wire Generate button in I2V tab → I2VHandle

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add I2V handler in `app.py`**

After the T2V handler block:

```python
# ---------------------------------------------------------------------------
# I2V: Generate handler
# ---------------------------------------------------------------------------
from pipelines.i2v import I2VHandle
from PIL import Image

I2V_HANDLES: dict[str, I2VHandle] = {}

def _get_i2v_handle(generation: str, resolution_label: str) -> I2VHandle:
    if generation == "wan2.2":
        key = "wan2.2_i2v_a14b"
    elif "720" in resolution_label:
        key = "wan2.1_i2v_14b_720p"
    else:
        key = "wan2.1_i2v_14b_480p"
    if key not in I2V_HANDLES:
        I2V_HANDLES[key] = I2VHandle.for_key(key)
    return I2V_HANDLES[key]


def generate_i2v(
    image,                # PIL.Image or filepath from gr.Image
    prompt: str,
    generation: str,
    preset_label: str,
    resolution_label: str,
    duration_s: float,
    negative_prompt: str,
    seed: int,
    randomize: bool,
    steps_override: int,
    cfg_override: float,
    cfg_2_override: float,
    progress=gr.Progress(track_tqdm=True),
):
    import random
    from diffusers.utils import export_to_video

    if image is None:
        raise gr.Error("Please upload a source image first.")

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    if randomize:
        seed = random.randint(0, 2**31 - 1)

    preset = "fast" if preset_label.startswith("Fast") else "quality"
    handle = _get_i2v_handle(generation, resolution_label)
    progress(0.05, desc="Configuring preset...")
    preset_kwargs = handle.configure_preset(preset)

    inference_kwargs = {
        "num_inference_steps": steps_override if steps_override > 0 else preset_kwargs.num_inference_steps,
        "guidance_scale": cfg_override if cfg_override > 0 else preset_kwargs.guidance_scale,
    }
    if preset_kwargs.guidance_scale_2 is not None:
        inference_kwargs["guidance_scale_2"] = cfg_2_override if cfg_2_override > 0 else preset_kwargs.guidance_scale_2

    max_area = 720 * 1280 if "720" in resolution_label else 480 * 832
    num_frames = max(17, int(duration_s * 16) // 4 * 4 + 1)

    progress(0.2, desc="Generating frames...")
    frames = handle.generate(
        image=image, prompt=prompt, negative_prompt=negative_prompt,
        max_area=max_area, num_frames=num_frames,
        seed=seed, preset_kwargs=inference_kwargs,
    )

    progress(0.9, desc="Encoding video...")
    import tempfile
    fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="wan_i2v_")
    os.close(fd)
    export_to_video(frames, out_path, fps=16)

    info = f"**Generated** — {len(frames)} frames · preset={preset_kwargs.effective_preset} · seed={seed}"
    if preset_kwargs.fallback_message:
        info = f"⚠️ {preset_kwargs.fallback_message}\n\n{info}"
    return out_path, info


i2v_inputs = tabs["i2v"]["inputs"]
i2v_outputs = tabs["i2v"]["outputs"]

i2v_inputs["generate"].click(
    fn=generate_i2v,
    inputs=[
        i2v_inputs["image"],
        i2v_inputs["prompt"],
        header["generation"],
        header["preset"],
        i2v_inputs["resolution"],
        i2v_inputs["duration"],
        i2v_inputs["negative_prompt"],
        i2v_inputs["seed"],
        i2v_inputs["randomize"],
        i2v_inputs["steps"],
        i2v_inputs["cfg"],
        i2v_inputs["cfg_2"],
    ],
    outputs=[i2v_outputs["video"], i2v_outputs["eta"]],
)
```

- [ ] **Step 2: Smoke-test locally**

```bash
.venv/bin/python app.py
```

Open `http://localhost:7860`, click I2V in sidebar, upload any small image, type a motion prompt, hit Generate. With Wan 2.1 I2V-14B-480P selected (default per generation=Wan 2.1 + non-720p resolution), this will try to load 14B on MPS — likely OOM. For dev, only test the UI flow; full I2V verification happens on ZeroGPU.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Wire I2V Generate handler — checkpoint routing by gen + resolution"
```

---

## Task 14: Make the Wan 2.2 dual-CFG slider toggle visible per generation

**Files:**
- Modify: `app.py` (add generation `.change()` handler)

- [ ] **Step 1: Add `.change()` handlers to hide cfg_2 when not on Wan 2.2**

After the existing `header["generation"].change(...)` handler in `app.py`, add:

```python
def _toggle_cfg_2(generation: str):
    """Wan 2.2 MoE has a dual CFG; hide the second slider on Wan 2.1."""
    is_moe = (generation == "wan2.2")
    return [gr.update(visible=is_moe) for _ in range(2)]  # t2v cfg_2 + i2v cfg_2

header["generation"].change(
    fn=_toggle_cfg_2,
    inputs=[header["generation"]],
    outputs=[
        tabs["t2v"]["inputs"]["cfg_2"],
        tabs["i2v"]["inputs"]["cfg_2"],
    ],
)
```

- [ ] **Step 2: Re-launch + verify**

```bash
WAN_STUDIO_T2V_LOCAL_KEY=wan2.1_t2v_1.3b .venv/bin/python app.py
```

Switch Generation between Wan 2.1 ↔ Wan 2.2; observe the `CFG (low-noise)` slider showing only on Wan 2.2.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Hide low-noise CFG slider on Wan 2.1 (single-transformer)"
```

---

## Task 15: Write `scripts/duplicate_upstream.py`

**Files:**
- Create: `scripts/duplicate_upstream.py`

- [ ] **Step 1: Write the duplication script**

```python
"""One-shot script — duplicate upstream Wan-AI / lightx2v repos into our account
for Phase 1 resilience. Run BEFORE create_space.py.

Idempotent — skips destinations that already exist.

Phase 1 scope:
  - 5 base model repos (T2V / I2V on Wan 2.1 14B + Wan 2.2 A14B)
  - 3 Lightning LoRA upstream repos consolidated into a single mirror via curated
    file uploads (avoids mirroring all of Kijai/WanVideo_comfy's terabytes)

Usage:
  python scripts/duplicate_upstream.py --dry-run    # print plan
  python scripts/duplicate_upstream.py              # execute
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


# Base model duplicates — full repo copies.
PHASE_1_BASE_DUPLICATES: list[tuple[str, str]] = [
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers",      "techfreakworm/wan2.1-t2v-14b"),
    ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "techfreakworm/wan2.1-i2v-14b-480p"),
    ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", "techfreakworm/wan2.1-i2v-14b-720p"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",     "techfreakworm/wan2.2-t2v-a14b"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers",     "techfreakworm/wan2.2-i2v-a14b"),
]

# Lightning LoRA files — pulled from various upstream and uploaded into ONE mirror.
# (upstream_repo, upstream_filename, mirror_path_inside_techfreakworm/wan-lightning-loras)
LIGHTNING_FILES: list[tuple[str, str, str]] = [
    # Wan 2.1 T2V-14B (single LoRA — community-recommended rank-128 v2)
    (
        "Kijai/WanVideo_comfy",
        "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
        "wan2.1-t2v-14b/lightning.safetensors",
    ),
    # Wan 2.1 I2V-14B 480P
    (
        "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
        "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        "wan2.1-i2v-14b-480p/lightning.safetensors",
    ),
    # Wan 2.1 I2V-14B 720P (same LoRA file actually works per RESEARCH §5.1; we
    # mirror the 720P-tagged copy for clarity)
    (
        "lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v",
        "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        "wan2.1-i2v-14b-720p/lightning.safetensors",
    ),
    # Wan 2.2 T2V-A14B — paired HIGH + LOW (V2.0 / 250928 — latest stable)
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
        "wan2.2-t2v-a14b/lightning_high.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
        "wan2.2-t2v-a14b/lightning_low.safetensors",
    ),
    # Wan 2.2 I2V-A14B — V1 Seko (no V2 as of May 2026 per RESEARCH §5.1.3)
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
        "wan2.2-i2v-a14b/lightning_high.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
        "wan2.2-i2v-a14b/lightning_low.safetensors",
    ),
]

LIGHTNING_MIRROR = "techfreakworm/wan-lightning-loras"


def duplicate_base(api: HfApi, dry_run: bool) -> None:
    for upstream, dest in PHASE_1_BASE_DUPLICATES:
        if dry_run:
            print(f"  [dry] duplicate_repo({upstream!r} → {dest!r})")
            continue
        try:
            api.model_info(dest)
            print(f"  ✓ already exists: {dest}")
            continue
        except Exception:
            pass
        print(f"  ↻ duplicating {upstream} → {dest}", flush=True)
        api.duplicate_repo(from_id=upstream, to_id=dest, repo_type="model")
        print(f"  ✓ done: https://huggingface.co/{dest}")


def build_lightning_mirror(api: HfApi, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry] create_repo({LIGHTNING_MIRROR!r})")
        for src_repo, src_file, dst_path in LIGHTNING_FILES:
            print(f"    [dry] {src_repo}/{src_file} → {LIGHTNING_MIRROR}/{dst_path}")
        return

    try:
        api.model_info(LIGHTNING_MIRROR)
        print(f"  ✓ mirror repo exists: {LIGHTNING_MIRROR}")
    except Exception:
        api.create_repo(repo_id=LIGHTNING_MIRROR, repo_type="model", private=False)
        print(f"  ✓ created mirror repo: {LIGHTNING_MIRROR}")

    with tempfile.TemporaryDirectory() as tmpdir:
        for src_repo, src_file, dst_path in LIGHTNING_FILES:
            print(f"  ↻ {src_repo}/{src_file}  →  {LIGHTNING_MIRROR}/{dst_path}", flush=True)
            local = hf_hub_download(
                repo_id=src_repo, filename=src_file, cache_dir=tmpdir,
            )
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=dst_path,
                repo_id=LIGHTNING_MIRROR,
                repo_type="model",
                commit_message=f"Mirror {src_file}",
            )
            print(f"  ✓ uploaded {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")

    print("=== Phase 1 base model duplicates ===")
    duplicate_base(api, args.dry_run)

    print("\n=== Phase 1 Lightning LoRA mirror ===")
    build_lightning_mirror(api, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Update `pipelines/registry.py` Lightning fields to match mirror layout**

The duplicate script names the mirrored Lightning files predictably. Update each MoE / Lightning-supporting registry entry to use:

- `lightning_high_lora = "wan2.X-<mode>-<size>/lightning_high.safetensors"` (MoE) or `"<...>/lightning.safetensors"` (single-transformer)
- `lightning_low_lora = "<...>/lightning_low.safetensors"` (MoE only)

Open `pipelines/registry.py` and update the 4 Lightning-supporting Phase-1 entries:

```python
# Wan 2.1 T2V-14B
"lightning_lora_repo": None,  # resolved at runtime via _lora_repo_for(card)
"lightning_high_lora": "wan2.1-t2v-14b/lightning.safetensors",

# Wan 2.1 I2V-14B-480P
"lightning_high_lora": "wan2.1-i2v-14b-480p/lightning.safetensors",

# Wan 2.1 I2V-14B-720P
"lightning_high_lora": "wan2.1-i2v-14b-720p/lightning.safetensors",

# Wan 2.2 T2V-A14B (MoE)
"lightning_high_lora": "wan2.2-t2v-a14b/lightning_high.safetensors",
"lightning_low_lora":  "wan2.2-t2v-a14b/lightning_low.safetensors",

# Wan 2.2 I2V-A14B (MoE)
"lightning_high_lora": "wan2.2-i2v-a14b/lightning_high.safetensors",
"lightning_low_lora":  "wan2.2-i2v-a14b/lightning_low.safetensors",
```

The `lightning_lora_repo` field on each card becomes the **upstream fallback** for local dev (when `_lora_repo_for` doesn't find the mount). Set it per card to a sensible upstream (e.g. `"Kijai/WanVideo_comfy"` for the Kijai-hosted ones, `"lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v"` for the lightx2v ones). The `weight_name` in `load_lora_weights` always uses the path relative to the mounted mirror — locally, `huggingface_hub` resolves the upstream repo + that path via the cache.

Run the registry tests to make sure consistency holds:

```bash
.venv/bin/pytest tests/test_registry.py -v
```

- [ ] **Step 3: Dry-run the duplicate script**

```bash
.venv/bin/python scripts/duplicate_upstream.py --dry-run
```

Expected: 5 `[dry] duplicate_repo(...)` lines for bases + 1 mirror-create line + 7 `[dry] upload` lines for Lightning files.

- [ ] **Step 4: Execute (takes ~15-25 min depending on HF Xet transfer speed)**

```bash
.venv/bin/python scripts/duplicate_upstream.py
```

Base duplications: 1-5 min each. Lightning file uploads: ~30s each (files are 0.5-1.2 GB).

- [ ] **Step 5: Verify on HF web UI**

Open `https://huggingface.co/techfreakworm` in browser. Confirm:
- 5 new base model repos (wan2.1-t2v-14b, wan2.1-i2v-14b-480p, wan2.1-i2v-14b-720p, wan2.2-t2v-a14b, wan2.2-i2v-a14b)
- 1 new `wan-lightning-loras` repo containing 7 .safetensors files organized into per-card subdirectories

- [ ] **Step 6: Commit**

```bash
git add scripts/duplicate_upstream.py pipelines/registry.py
git commit -m "Add duplicate_upstream.py + retarget registry LoRA paths to mirror layout"
```

---

## Task 16: Write `scripts/create_space.py` (with volume mounts)

**Files:**
- Create: `scripts/create_space.py`

- [ ] **Step 1: Write the script**

```python
"""Programmatic Space configuration — sets `space_volumes` for our duplicated mirrors.

Run AFTER scripts/duplicate_upstream.py. The Space repo `techfreakworm/wan-studio` is
ALREADY CREATED (we did this manually for the probe). This script just updates volumes
+ hardware.

Volume mount paths inside the Space container:
  /models/wan2.1-t2v-14b
  /models/wan2.1-i2v-14b-480p
  /models/wan2.1-i2v-14b-720p
  /models/wan2.2-t2v-a14b
  /models/wan2.2-i2v-a14b
  /models/wan2.2-lightning      (Lightning LoRA bundle)

Usage:
  python scripts/create_space.py --dry-run
  python scripts/create_space.py
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.hf_api import Volume


SPACE_ID = "techfreakworm/wan-studio"


PHASE_1_VOLUMES = [
    # Base models (one mount per checkpoint — slug convention from pipelines/handle.py:_slug_for)
    Volume(type="model", source="techfreakworm/wan2.1-t2v-14b",
           mount_path="/models/wan2.1-t2v-14b", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.1-i2v-14b-480p",
           mount_path="/models/wan2.1-i2v-14b-480p", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.1-i2v-14b-720p",
           mount_path="/models/wan2.1-i2v-14b-720p", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.2-t2v-a14b",
           mount_path="/models/wan2.2-t2v-a14b", read_only=True),
    Volume(type="model", source="techfreakworm/wan2.2-i2v-a14b",
           mount_path="/models/wan2.2-i2v-a14b", read_only=True),
    # Consolidated Lightning LoRA mirror — single mount, multiple subdirs inside
    Volume(type="model", source="techfreakworm/wan-lightning-loras",
           mount_path="/models/wan-lightning-loras", read_only=True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    print(f"Configuring {SPACE_ID}")
    print(f"  {len(PHASE_1_VOLUMES)} volumes:")
    for v in PHASE_1_VOLUMES:
        print(f"    {v.source}  →  {v.mount_path}")

    if args.dry_run:
        return

    api.set_space_volumes(repo_id=SPACE_ID, volumes=PHASE_1_VOLUMES)
    print("✓ Volumes set")

    api.request_space_hardware(repo_id=SPACE_ID, hardware=SpaceHardware.ZERO_A10G)
    print(f"✓ Hardware requested: {SpaceHardware.ZERO_A10G}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry-run**

```bash
.venv/bin/python scripts/create_space.py --dry-run
```

Verify the 6 volumes listed.

- [ ] **Step 3: Execute (after Task 15 finishes duplicating)**

```bash
.venv/bin/python scripts/create_space.py
```

Setting volumes triggers a Space rebuild. Watch with:

```bash
hf spaces info techfreakworm/wan-studio 2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['runtime']['stage'])"
```

Wait until stage is `RUNNING` again (may take 2-5 min — the build re-mounts volumes).

- [ ] **Step 4: Verify mounts inside the Space**

Add a temporary debug entry to the live probe app — but easier: just trust the `set_space_volumes` API. The next deploy (Task 18) will confirm by actually loading from `/models/...`.

- [ ] **Step 5: Commit**

```bash
git add scripts/create_space.py
git commit -m "Add create_space.py — volume mounts for Phase 1 + ZeroGPU hardware request"
```

---

## Task 17: Write `NOTICE.md` + update `README.md`

**Files:**
- Create: `NOTICE.md`
- Modify: `README.md`

- [ ] **Step 1: Write `NOTICE.md`**

```markdown
# NOTICE

Wan Studio incorporates and redistributes the following third-party assets, all
licensed under Apache License 2.0 (the same license as this project's source
code). Attribution is required by Apache 2.0 §4.

## Wan-AI (Alibaba) — Wan 2.1 + Wan 2.2 model weights

Wan Studio mirrors the following Wan-AI repositories into the user's HF account
(`techfreakworm/wan2.*-*`) and mounts them read-only into the Space:

- `Wan-AI/Wan2.1-T2V-14B-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers` — Apache 2.0

Upstream: https://huggingface.co/Wan-AI · Paper: https://arxiv.org/abs/2503.20314

## lightx2v / ModelTC — Lightning step-distillation LoRAs

- `lightx2v/Wan2.2-Lightning` — Apache 2.0 (HIGH / LOW LoRA pairs for Wan 2.2 MoE)

Upstream: https://huggingface.co/lightx2v

## diffusers (Hugging Face)

Wan pipeline classes (`WanPipeline`, `WanImageToVideoPipeline`, etc.) — Apache 2.0
Upstream: https://github.com/huggingface/diffusers
```

- [ ] **Step 2: Update `README.md` with the Space YAML frontmatter + content**

Replace the existing `README.md` with:

```markdown
---
title: Wan Studio
emoji: 🎬
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: "5.49.0"
app_file: app.py
pinned: false
short_description: "Every Wan mode, one clean UI."
python_version: "3.12.12"
startup_duration_timeout: "30m"
preload_from_hub:
  - techfreakworm/wan2.2-lightning
# ZeroGPU hardware is set programmatically by scripts/create_space.py
# (using SpaceHardware.ZERO_A10G — the value still binds to the current
# Blackwell ZeroGPU V2 pool empirically verified May 2026).
---

# Wan Studio

Multi-mode Gradio Studio for the Alibaba Wan video diffusion family. Phase 1
ships **T2V** and **I2V** on **Wan 2.1** (T2V-14B, I2V-14B 480P + 720P) and
**Wan 2.2** (T2V-A14B MoE, I2V-A14B MoE) with two presets:

- **Fast (Lightning)** — 4 steps, CFG=1.0, official Lightning LoRA loaded
- **Quality** — 40-50 steps, full sampler, no LoRA

Pick generation and preset from the header. Modes are in the left sidebar.

Backed by HF ZeroGPU. Models are mounted read-only from duplicated mirrors
for resilience.

## Roadmap

| Phase | Modes | Status |
|---|---|---|
| 1 | T2V, I2V | **in progress** |
| 2 | FLF2V, V2V, TI2V-5B | planned |
| 3 | VACE | planned |
| 4 | Animate | planned |
| 5 | S2V | planned |
| 6 | Cross-mode chaining + Gallery + Settings | planned |

## Attribution

See [NOTICE.md](NOTICE.md) for Apache 2.0 attribution to Wan-AI, lightx2v, and the diffusers project.
```

- [ ] **Step 3: Commit**

```bash
git add NOTICE.md README.md
git commit -m "Write NOTICE.md attribution; Phase-1 README with preload_from_hub YAML"
```

---

## Task 18: Deploy Phase 1 code to ZeroGPU Space

**Files:**
- No new files. Push the entire wan-studio repo to the Space.

- [ ] **Step 1: Push code to the Space repo**

```bash
cd /Users/techfreakworm/Projects/llm/wan-studio
hf upload techfreakworm/wan-studio . --repo-type=space \
  --commit-message "Phase 1 deploy: T2V + I2V on Wan 2.1 + Wan 2.2 MoE with Fast/Quality preset" \
  --exclude ".venv/*" \
  --exclude ".superpowers/*" \
  --exclude "raw/*" \
  --exclude "wireframes/_codex*" \
  --exclude "tests/outputs/*" \
  --exclude "docs/*"
```

(Exclude flags strip local-only artifacts. `docs/` is fine to exclude from the Space — the spec/plans don't need to ship publicly.)

- [ ] **Step 2: Watch the build**

```bash
until python3 -c "
from huggingface_hub import HfApi
rt = HfApi().get_space_runtime('techfreakworm/wan-studio')
print(rt.stage)
" | grep -E '^(RUNNING|BUILD_ERROR|RUNTIME_ERROR)$' >/dev/null; do sleep 15; done
hf spaces info techfreakworm/wan-studio 2>&1 | head -30
```

If stage is `RUNTIME_ERROR`, fetch logs:

```bash
python3 -c "
from huggingface_hub import HfApi
for line in HfApi().fetch_space_logs('techfreakworm/wan-studio'):
    print(line, end='')
"
```

Common Phase-1 issues to expect:
- **`ImportError: WanPipeline`** — diffusers too old. Pin `diffusers>=0.38.0` is in `requirements.txt`, verify it stuck.
- **Mount paths missing** — Volume mounts not set yet. Re-run `scripts/create_space.py`.
- **OOM at module load** — too much eager loading. The handle is lazy by design; if there's eager loading creeping in (e.g. someone calls `shared.text_encoder()` at import), trace and remove.

- [ ] **Step 3: Visit the live Space**

Open `https://huggingface.co/spaces/techfreakworm/wan-studio` in browser. Verify UI renders: sidebar with all modes, T2V tab as default, header with Generation dropdown + Preset radio.

- [ ] **Step 4: Commit (the deploy itself doesn't need a separate local commit)**

Nothing to commit — push was direct to the Space repo. The local repo and the Space repo are separate (the Space pulls from its own commits via `hf upload`).

---

## Task 19: Live verification — Wan 2.1 T2V Fast preset

**Files:**
- No code changes. Manual + scripted verification.

- [ ] **Step 1: Run T2V Fast on Wan 2.1 14B via gradio API**

```bash
HFTOKEN=$(cat ~/.cache/huggingface/token)
BASE="https://techfreakworm-wan-studio.hf.space/gradio_api/call"
EID=$(curl -sS -X POST "$BASE/generate_t2v" \
  -H "Authorization: Bearer $HFTOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data":[
      "A cinematic shot of a fox running through autumn leaves at golden hour",
      "wan2.1",
      "Fast (Lightning)",
      "832x480 (16:9)",
      2.0,
      "static, blurred, low quality",
      42, false, 0, 0, 0
    ]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['event_id'])")
echo "event_id = $EID"
curl -sS --max-time 180 "$BASE/generate_t2v/$EID" -H "Authorization: Bearer $HFTOKEN" > /tmp/t2v-21-fast.txt
sed -n '1,40p' /tmp/t2v-21-fast.txt
```

Expected: SSE stream with `event: complete` and a `data: ["<filepath>", "<info_md>"]` line. The `<filepath>` should be a `.mp4` URL on the Space.

- [ ] **Step 2: Verify the output**

Extract the URL from the response and download:

```bash
URL=$(python3 -c "
import json, sys, re
text = open('/tmp/t2v-21-fast.txt').read()
m = re.search(r'data: (\\[.*\\])', text)
data = json.loads(m.group(1))
print(data[0]['url'] if isinstance(data[0], dict) else data[0])
")
echo "Output URL: $URL"
curl -sS "$URL" -o /tmp/t2v-21-fast.mp4
ls -la /tmp/t2v-21-fast.mp4
open /tmp/t2v-21-fast.mp4
```

Expected: `.mp4` is >100 KB, plays as a 2-second video at 832×480.

- [ ] **Step 3: If failure, debug from logs and patch**

Common issues:
- Lightning LoRA load fails → check `pipelines/handle.py:_load_lightning_lora`, verify mount path
- `vae_scale_factor_spatial` AttributeError → diffusers version mismatch
- frame count violation (`4k+1`) → check the `num_frames` formula in `app.py`

Iterate via `hf upload` + watch build + retry.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "Fix <whatever broke during live T2V Wan 2.1 verification>"
hf upload techfreakworm/wan-studio . --repo-type=space --commit-message "Fix T2V Wan 2.1 Fast preset"
```

---

## Task 20: Live verification — Wan 2.2 T2V MoE Fast preset

**Files:**
- No code changes (in the happy path).

- [ ] **Step 1: Run T2V Fast on Wan 2.2 A14B (MoE)**

Same `curl` as Task 19 but with `"wan2.2"` and `"1280x720 (16:9)"`:

```bash
curl -sS -X POST "$BASE/generate_t2v" \
  -H "Authorization: Bearer $HFTOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data":[
      "A close-up of dew on a leaf in morning sunlight, ultra-detailed",
      "wan2.2",
      "Fast (Lightning)",
      "1280x720 (16:9)",
      3.0,
      "static, blurred, watermarks",
      42, false, 0, 0, 0
    ]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['event_id'])"
```

Then GET the result as in Task 19.

- [ ] **Step 2: Verify dual-LoRA loaded both transformers**

If output is blurry or motion is sluggish, the LOW LoRA likely didn't load into `transformer_2`. Verify by adding a print in `pipelines/handle.py:_load_lightning_lora`:

```python
if self.card.is_moe:
    print(f"[LoRA] Loading LOW into transformer_2: {self.card.lightning_low_lora}", flush=True)
```

Re-deploy, watch Space logs:

```bash
python3 -c "
from huggingface_hub import HfApi
for line in HfApi().fetch_space_logs('techfreakworm/wan-studio', follow=False):
    print(line, end='')
" | grep LoRA
```

Should see two lines: one for HIGH onto transformer, one for LOW onto transformer_2.

- [ ] **Step 3: If both LoRAs loaded correctly but output is still soft, enable the hybrid trick**

Add a Settings checkbox + advanced setting that swaps the Wan 2.2 I2V Lightning V1 for the Wan 2.1 I2V Lightning LoRA. **Defer this to Task 22 — for now just confirm V2.0 T2V Lightning works.**

- [ ] **Step 4: Commit any fixes + re-deploy**

```bash
git add -u && git commit -m "Verify Wan 2.2 T2V-A14B dual-LoRA load"
hf upload techfreakworm/wan-studio . --repo-type=space --commit-message "..."
```

---

## Task 21: Live verification — I2V both generations

**Files:**
- No code changes.

- [ ] **Step 1: Upload a small test image to the Space and call I2V via gradio API**

```bash
# Use any small JPG as test input
TEST_IMG="/tmp/test-i2v-input.jpg"
curl -sS https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2v_input.jpg -o "$TEST_IMG"
ls -la "$TEST_IMG"

# Upload to the gradio file endpoint
UPLOAD=$(curl -sS -X POST "https://techfreakworm-wan-studio.hf.space/upload" \
  -F "files=@$TEST_IMG" \
  -H "Authorization: Bearer $HFTOKEN" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)[0])")
echo "Uploaded path: $UPLOAD"
```

Then POST to `/generate_i2v` with the upload path as the image arg. (Exact JSON shape depends on Gradio's File handling — confirm by inspecting the Space's `/info` endpoint or trying both `{"path": "..."}` and string forms.)

- [ ] **Step 2: Verify Wan 2.1 I2V Fast**

Use `"wan2.1"` + `"832x480 (16:9)"` → should auto-route to `wan2.1_i2v_14b_480p`.

- [ ] **Step 3: Verify Wan 2.2 I2V Fast**

Use `"wan2.2"` → routes to `wan2.2_i2v_a14b`. Expect motion to be softer at V1 Lightning per RESEARCH §5.

- [ ] **Step 4: Verify Quality preset on both**

Same calls with `"Quality"` preset_label. Quality runs are slow (40 steps × 81 frames × 720p), expect 90-150s per call.

- [ ] **Step 5: Commit any fixes + re-deploy**

```bash
git add -u && git commit -m "Live verification: I2V Wan 2.1 + Wan 2.2 + Quality preset"
hf upload techfreakworm/wan-studio . --repo-type=space --commit-message "..."
```

---

## Task 22: Add examples to T2V and I2V tabs

**Files:**
- Modify: `ui/tabs.py`
- Create: `assets/examples/` (small thumbnails — optional for Phase 1)

- [ ] **Step 1: Add `gr.Examples` to `build_t2v_tab`**

In `ui/tabs.py`, inside `build_t2v_tab` after the two-col block:

```python
gr.Examples(
    examples=[
        ["A cinematic shot of a fox running through autumn leaves at golden hour", "1280x720 (16:9)", 3.0],
        ["Aerial drone over a mountain range at sunrise, slow forward motion", "1280x720 (16:9)", 4.0],
        ["A close-up of dew drops on a spider web, shallow depth of field", "832x480 (16:9)", 2.0],
    ],
    inputs=[
        components["inputs"]["prompt"],
        components["inputs"]["resolution"],
        components["inputs"]["duration"],
    ],
    cache_examples=False,  # MANDATORY on ZeroGPU per RESEARCH §9.7
)
```

- [ ] **Step 2: Add `gr.Examples` to `build_i2v_tab`**

For I2V, we need example images. Use Hugging Face's diffusers demo images for now:

```python
gr.Examples(
    examples=[
        ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2v_input.jpg",
         "Slow zoom in, leaves rustling in the wind", "832x480 (16:9)", 2.5],
    ],
    inputs=[
        components["inputs"]["image"],
        components["inputs"]["prompt"],
        components["inputs"]["resolution"],
        components["inputs"]["duration"],
    ],
    cache_examples=False,
)
```

- [ ] **Step 3: Verify locally**

```bash
WAN_STUDIO_T2V_LOCAL_KEY=wan2.1_t2v_1.3b .venv/bin/python app.py
```

Open browser, click an example — fields should pre-fill.

- [ ] **Step 4: Deploy + verify on live Space**

```bash
hf upload techfreakworm/wan-studio . --repo-type=space --commit-message "Add Phase-1 examples for T2V and I2V"
```

- [ ] **Step 5: Commit**

```bash
git add ui/tabs.py
git commit -m "Add 3 T2V + 1 I2V examples (cache_examples=False for ZeroGPU)"
```

---

## Task 23: Add basic error UX (toast on Lightning fallback)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Use `gr.Info` for the fallback message**

In both `generate_t2v` and `generate_i2v` in `app.py`, replace:

```python
info = f"**Generated** ..."
if preset_kwargs.fallback_message:
    info = f"⚠️ {preset_kwargs.fallback_message}\n\n{info}"
return out_path, info
```

with:

```python
if preset_kwargs.fallback_message:
    gr.Info(preset_kwargs.fallback_message, duration=8)
info = f"**Generated** — {len(frames)} frames · preset={preset_kwargs.effective_preset} · seed={seed}"
return out_path, info
```

The fallback message now shows as a transient toast (which doesn't apply on Phase-1 modes since both T2V and I2V have Lightning, but the plumbing is correct for Phase 2+).

- [ ] **Step 2: Wrap the whole generate in try/except for graceful failures**

Inside both `generate_*` functions, wrap the body:

```python
try:
    # ... existing body ...
except torch.cuda.OutOfMemoryError as e:
    raise gr.Error(f"GPU out of memory. Try a smaller resolution or shorter duration. ({e})")
except FileNotFoundError as e:
    raise gr.Error(f"Model files not found on Space — volume mount may be missing. Contact admin. ({e})")
except Exception as e:
    raise gr.Error(f"Generation failed: {type(e).__name__}: {e}")
```

- [ ] **Step 3: Re-deploy + verify**

Force a failure (e.g. resolution `4096x4096`) and confirm a clean `gr.Error` modal appears, not a stack trace.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add gr.Info toast for Lightning fallback + try/except wrapping with gr.Error"
hf upload techfreakworm/wan-studio . --repo-type=space --commit-message "Phase-1 error UX polish"
```

---

## Task 24: Final Phase-1 sanity sweep

**Files:** none — manual verification + final commit.

- [ ] **Step 1: Run the full test suite locally one more time**

```bash
.venv/bin/pytest tests/ -v --ignore=tests/test_smoke_t2v_local.py
```

Expected: all non-slow tests PASS in <5 seconds.

Optional (slow, ~5 min):

```bash
.venv/bin/pytest tests/test_smoke_t2v_local.py -v -s
```

Expected: 1 PASS, MP4 produced.

- [ ] **Step 2: Test the live Space end-to-end via the browser**

Open `https://huggingface.co/spaces/techfreakworm/wan-studio` and exercise:

1. T2V tab + Wan 2.1 + Fast → generates in <30s
2. T2V tab + Wan 2.2 + Fast → generates in <60s (xlarge tier)
3. T2V tab + Wan 2.2 + Quality → generates in <120s (xlarge, 40 steps)
4. I2V tab + Wan 2.1 + Fast (upload an image) → generates
5. I2V tab + Wan 2.2 + Fast → generates
6. Switch generation from 2.2 → 2.1: cfg_2 slider disappears
7. Pick "Fast" on a Quality-only tab (no Phase-1 modes hit this — skip until Phase 2 adds VACE)
8. Mobile viewport (resize browser to 390×844): sidebar collapses, columns stack

If anything fails, debug + redeploy as in earlier tasks.

- [ ] **Step 3: Tag the commit**

```bash
git tag -a v0.1.0-phase1 -m "Phase 1: T2V + I2V on Wan 2.1 + Wan 2.2 MoE live on ZeroGPU"
```

- [ ] **Step 4: Optionally push to a remote git mirror**

If you want a backup repo on GitHub:

```bash
gh repo create techfreakworm/wan-studio --private --source=. --remote=origin --push
git push origin --tags
```

(Skip if you're keeping the Studio source only on HF Spaces.)

---

## Phase 1 done. Acceptance criteria

- [ ] All unit tests pass: `pytest tests/ -v --ignore=tests/test_smoke_t2v_local.py` is green
- [ ] Live Space at `https://huggingface.co/spaces/techfreakworm/wan-studio` renders without errors
- [ ] T2V Wan 2.1 Fast preset generates a 2-3s video in <30s
- [ ] T2V Wan 2.2 Fast preset generates a 3s 720p video in <60s
- [ ] I2V both generations produce video from a source image
- [ ] Quality preset works on all four checkpoints (slower but visually correct)
- [ ] Generation dropdown hides Wan 2.2 dual-CFG slider on Wan 2.1
- [ ] No raw stack traces visible in the UI on edge cases (resolution too big, missing image)
- [ ] Git tag `v0.1.0-phase1` exists on the local repo

**Phase 2 plan target:** FLF2V + V2V + TI2V-5B. Will be written after Phase 1 ships and we learn the actual ZeroGPU latency / quota profile on the real Blackwell.
