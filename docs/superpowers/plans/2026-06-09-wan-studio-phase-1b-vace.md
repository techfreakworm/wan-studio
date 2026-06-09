# Wan Studio Phase #1b — VACE (core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Wan 2.1 **VACE** (Versatile Animation Control & Editing) — `WanVACEPipeline`, single-transformer, Quality-only — onto the #0 foundation, with all 9 sub-modes functional via **user-provided** control video / masks / references. Auto-extraction annotators (DWPose/MiDaS/RAFT) are deferred to a follow-on (**#1b-preproc**), gated on the `wan-preproc` mount.

**Architecture:** `VACEHandle(WanModelHandle)` builds `WanVACEPipeline` on the mounted VACE checkpoint, injecting the shared VAE + text encoder (VACE has **no** image encoder). The 9 sub-modes are *not* separate models — they're determined by what gets passed for `video` / `mask` / `reference_images`. Pure-logic builders (`vace_inputs.py`) construct those per sub-mode; deferred sub-modes require a pre-extracted control video and otherwise raise a clear error. App wiring is append-only via the existing `_MODE_RUNNERS`/`_inputs_for`/`_ui_dispatch`. Quality-only via the registry cards' `lightning_available=False`.

**Tech Stack:** Python 3.12 · diffusers 0.38 (`WanVACEPipeline`) · gradio 6.14 · pytest 9 · MPS local (1.3B) / ZeroGPU `large` bf16.

**References:** [`program architecture & risks`](../specs/2026-06-08-wan-studio-program-architecture-and-risks.md) (VACE analysis, R16/R17/R24) · [`#1a plan`](./2026-06-09-wan-studio-phase-1a-flf2v-v2v.md) (`HANDLER_REGISTRY` wiring + `pipelines/video_io.py` `decode_video` to reuse). **Verified API:** `WanVACEPipeline.__call__(prompt, negative_prompt, video: list[PIL]|None, mask: list[PIL]|None, reference_images: list[PIL]|None, conditioning_scale=1.0, height=480, width=832, num_frames=81, num_inference_steps=50, guidance_scale=5.0, ...)` → `out.frames[0]`. Single transformer (no `transformer_2`). VAE loaded fp32.

**Out of scope:** auto-extraction annotators DWPose/MiDaS/RAFT (#1b-preproc); SAM2/GroundingDINO/InsightFace sub-modes (permanently user-provided per NG8); Animate (#2); vendored S2V/TI2V (#3); Send-to/Gallery (#4).

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `pipelines/vace.py` | **create** | `VACEHandle` (`WanVACEPipeline`, vae+text_encoder injection) + generate + self-register |
| `pipelines/vace_inputs.py` | **create** | pure builders: `gallery_to_references`, `prepare_video_and_mask` (gray-fill), `outpaint_video_and_mask`, `resolve_submode` (sub-mode → which inputs + deferred gating) |
| `pipelines/__init__.py` | modify | import `vace` so it self-registers; export `VACEHandle` |
| `app.py` | modify | `_run_vace` + `_MODE_RUNNERS`/`_inputs_for`/`_ui_dispatch` branches |
| `tests/test_vace.py` | **create** | VACEHandle construction + registration |
| `tests/test_vace_inputs.py` | **create** | builders + sub-mode resolution + deferred gating |
| `tests/test_app_wiring.py` | modify | assert vace wired (not toast) |

**Reuse:** `pipelines/video_io.py::decode_video` (#1a) for `source_video`→frames. `pipelines/i2v.py::aspect_ratio_resize` for sizing. **Read first:** `pipelines/v2v.py` (the closest handle template — non-MoE, vae+text_encoder only), `app.py` `_run_v2v` (runner template), `ui/tabs.py::build_vace_tab` (the real component keys), `pipelines/handle.py` `_configure_scheduler` (flow_shift).

**VACE size selection:** the VACE tab has no size selector. `key_for` returns `wan2.1_vace_14b` by default; `WAN_STUDIO_VACE_LOCAL_KEY=wan2.1_vace_1.3b` forces the 1.3B locally (same local-override pattern as t2v/i2v). VACE is Wan 2.1 only → `key_for` ignores `generation`.

---

## Task 1: `vace_inputs.py` — pure builders + sub-mode resolution

**Files:**
- Create: `pipelines/vace_inputs.py`
- Test: `tests/test_vace_inputs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vace_inputs.py`:

```python
"""Tests for pipelines.vace_inputs — pure VACE input builders + sub-mode routing."""
import pytest
from PIL import Image

from pipelines.vace_inputs import (
    DEFERRED_SUBMODES,
    gallery_to_references,
    outpaint_video_and_mask,
    prepare_video_and_mask,
    resolve_submode,
)


def _frames(n, w=64, h=32):
    return [Image.new("RGB", (w, h), (i, 0, 0)) for i in range(n)]


def test_gallery_to_references_coerces_to_rgb_pil():
    raw = [(Image.new("L", (10, 10), 128), "cap"), Image.new("RGB", (8, 8), "red")]
    refs = gallery_to_references(raw)
    assert all(isinstance(r, Image.Image) and r.mode == "RGB" for r in refs)
    assert len(refs) == 2


def test_gallery_to_references_empty_is_none():
    assert gallery_to_references([]) is None
    assert gallery_to_references(None) is None


def test_prepare_video_and_mask_lengths_and_fill():
    frames = _frames(5)
    video, mask = prepare_video_and_mask(frames, generate_indices={1, 3}, height=32, width=64)
    assert len(video) == len(mask) == 5
    assert all(m.mode == "L" for m in mask)
    # generate frames are gray + white mask; keep frames are real + black mask
    assert mask[0].getpixel((0, 0)) == 0 and mask[1].getpixel((0, 0)) == 255


def test_outpaint_video_and_mask_expands_and_masks_border():
    frames = _frames(3, w=32, h=32)
    video, mask = outpaint_video_and_mask(frames, pad=8)
    assert video[0].size == (48, 48)          # 32 + 2*8
    assert mask[0].getpixel((0, 0)) == 255    # border is to-generate
    assert mask[0].getpixel((24, 24)) == 0    # centre is kept


def test_resolve_submode_inpaint_needs_video():
    with pytest.raises(ValueError, match="control video"):
        resolve_submode("Inpaint", source_frames=None, references=None, height=32, width=64, num_frames=5)


def test_resolve_submode_reference_uses_references_only():
    refs = [Image.new("RGB", (8, 8))]
    plan = resolve_submode("Reference", source_frames=None, references=refs, height=32, width=64, num_frames=5)
    assert plan.reference_images == refs
    assert plan.video is None and plan.mask is None


def test_resolve_submode_deferred_requires_preextracted():
    assert "Animate-Anything" in DEFERRED_SUBMODES
    with pytest.raises(ValueError, match="pre-extracted"):
        resolve_submode("Animate-Anything", source_frames=None, references=None, height=32, width=64, num_frames=5)


def test_resolve_submode_control_passes_source_as_video():
    """Depth/Pose/Sketch/Flow in v1 treat the uploaded source as the control video."""
    frames = _frames(5)
    plan = resolve_submode("Depth", source_frames=frames, references=None, height=32, width=64, num_frames=5)
    assert plan.video == frames
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_vace_inputs.py -v`
Expected: FAIL — no `pipelines.vace_inputs`.

- [ ] **Step 3: Implement `pipelines/vace_inputs.py`**

```python
"""Pure builders for VACE inputs (video / mask / reference_images) per sub-mode.

VACE has one pipeline; the sub-mode is decided by what you pass. v1 ships the
control-via-user-upload path: control-extraction sub-modes (Depth/Pose/Sketch/Flow)
treat the uploaded source video AS the control signal (auto-extraction is #1b-preproc);
Inpaint/Outpaint/Extension build a gray-fill video+mask; Reference uses reference_images.
SAM2/GroundingDINO/InsightFace-dependent sub-modes are DEFERRED — they require a
pre-extracted control video.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

# Sub-modes whose annotators (SAM2/GroundingDINO/InsightFace) are not shipped → the
# user must supply a pre-extracted control video.
DEFERRED_SUBMODES = {"Animate-Anything"}

# Control-extraction sub-modes: in v1 the uploaded source IS the control video.
CONTROL_SUBMODES = {"Depth", "Pose", "Sketch", "Flow"}


@dataclass
class VacePlan:
    video: list | None = None
    mask: list | None = None
    reference_images: list | None = None


def gallery_to_references(gallery) -> list | None:
    """Coerce a gr.Gallery value (list of PIL or (path/PIL, caption) tuples) → list[PIL RGB]."""
    if not gallery:
        return None
    out = []
    for item in gallery:
        img = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(img, str):
            img = Image.open(img)
        out.append(img.convert("RGB"))
    return out or None


def prepare_video_and_mask(frames: list, generate_indices: set[int], height: int, width: int):
    """Inpaint/Extension: keep frames not in generate_indices; gray-fill the rest.

    Returns (video, mask): video is list[PIL RGB] (gray where generated), mask is
    list[PIL 'L'] (white=255 where the model should generate, black=0 where kept).
    """
    gray = Image.new("RGB", (width, height), (128, 128, 128))
    white = Image.new("L", (width, height), 255)
    black = Image.new("L", (width, height), 0)
    video, mask = [], []
    for i, f in enumerate(frames):
        if i in generate_indices:
            video.append(gray.copy())
            mask.append(white.copy())
        else:
            video.append(f.convert("RGB").resize((width, height)))
            mask.append(black.copy())
    return video, mask


def outpaint_video_and_mask(frames: list, pad: int):
    """Outpaint: paste each frame centred on a (w+2*pad, h+2*pad) gray canvas; mask the border."""
    video, mask = [], []
    for f in frames:
        f = f.convert("RGB")
        w, h = f.size
        nw, nh = w + 2 * pad, h + 2 * pad
        canvas = Image.new("RGB", (nw, nh), (128, 128, 128))
        canvas.paste(f, (pad, pad))
        m = Image.new("L", (nw, nh), 255)
        m.paste(Image.new("L", (w, h), 0), (pad, pad))
        video.append(canvas)
        mask.append(m)
    return video, mask


def resolve_submode(submode: str, *, source_frames, references, height: int, width: int, num_frames: int) -> VacePlan:
    """Map a sub-mode + the available inputs → a VacePlan(video, mask, reference_images).

    Raises ValueError with an actionable message when a required input is missing or
    the sub-mode is deferred.
    """
    if submode in DEFERRED_SUBMODES:
        if not source_frames:
            raise ValueError(
                f"{submode} needs a pre-extracted control video (its annotator is not shipped in v1) — "
                "upload one as the source video."
            )
        return VacePlan(video=source_frames)

    if submode == "Reference":
        if not references:
            raise ValueError("Reference mode needs at least one reference image.")
        return VacePlan(reference_images=references)

    if submode in CONTROL_SUBMODES:
        if not source_frames:
            raise ValueError(f"{submode} needs a control video — upload one as the source video.")
        return VacePlan(video=source_frames, reference_images=references)

    if submode == "Outpaint":
        if not source_frames:
            raise ValueError("Outpaint needs a source video.")
        video, mask = outpaint_video_and_mask(source_frames, pad=height // 4)
        return VacePlan(video=video, mask=mask, reference_images=references)

    if submode in ("Inpaint", "Extension"):
        if not source_frames:
            raise ValueError(f"{submode} needs a control video.")
        # Inpaint v1: regenerate ALL frames guided by the control (whole-clip restyle);
        # Extension v1: keep the first half, generate the second half.
        n = len(source_frames)
        gen_idx = set(range(n)) if submode == "Inpaint" else set(range(n // 2, n))
        video, mask = prepare_video_and_mask(source_frames, gen_idx, height, width)
        return VacePlan(video=video, mask=mask, reference_images=references)

    raise ValueError(f"Unknown VACE sub-mode: {submode!r}")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_vace_inputs.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add pipelines/vace_inputs.py tests/test_vace_inputs.py
git commit -m "Add VACE input builders: gray-fill/outpaint mask + sub-mode resolution"
```

---

## Task 2: `VACEHandle` + self-registration

**Files:**
- Create: `pipelines/vace.py`
- Modify: `pipelines/__init__.py`
- Test: `tests/test_vace.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vace.py`:

```python
"""Tests for VACEHandle (no model load)."""
import os

import pipelines  # noqa: F401
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.vace import VACEHandle


def test_vace_registered_large_quality():
    assert "vace" in HANDLER_REGISTRY
    spec = HANDLER_REGISTRY["vace"]
    assert spec.handle_cls is VACEHandle
    assert spec.tier == "large"


def test_vace_key_default_14b():
    spec = HANDLER_REGISTRY["vace"]
    assert spec.key_for("wan2.1") == "wan2.1_vace_14b"
    assert spec.key_for("wan2.2") == "wan2.1_vace_14b"  # Wan 2.1 only


def test_vace_key_local_override(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    monkeypatch.setenv("WAN_STUDIO_VACE_LOCAL_KEY", "wan2.1_vace_1.3b")
    spec = HANDLER_REGISTRY["vace"]
    assert spec.key_for("wan2.1") == "wan2.1_vace_1.3b"


def test_vace_handle_card_quality_only():
    h = VACEHandle.for_key("wan2.1_vace_14b")
    assert h.card.mode == "vace"
    assert h.card.lightning_available is False
    assert h.card.requires_image_encoder is False
    assert h.pipe is None
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_vace.py -v`
Expected: FAIL — no `pipelines.vace`.

- [ ] **Step 3: Implement `pipelines/vace.py`**

> Mirror `pipelines/v2v.py` (non-MoE, vae+text_encoder injection, no image_encoder). Read it first.

```python
"""VACE — Wan 2.1 control/edit (WanVACEPipeline). Single transformer, Quality-only.

VACE has no CLIP image branch → inject only vae + text_encoder. The sub-mode is
decided by which of video/mask/reference_images is passed (see vace_inputs).
"""
from __future__ import annotations

import os
from typing import Any

import torch
from diffusers import WanVACEPipeline

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from pipelines.handlers import HandlerSpec, register
from utils.backend import detect


class VACEHandle(WanModelHandle):
    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)
        return WanVACEPipeline.from_pretrained(
            path,
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            torch_dtype=backend.dtype,
        )

    def generate(
        self,
        *,
        prompt: str,
        video: list | None = None,
        mask: list | None = None,
        reference_images: list | None = None,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        conditioning_scale: float = 1.0,
        seed: int = 42,
        preset_kwargs: dict,
    ) -> list:
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            video=video,
            mask=mask,
            reference_images=reference_images,
            conditioning_scale=conditioning_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]


def _vace_key_for(generation: str, **_ui) -> str:
    # Wan 2.1 only; default to 14B, allow a local 1.3B override (dev).
    if os.getenv("SPACES_ZERO_GPU") is None:
        override = os.getenv("WAN_STUDIO_VACE_LOCAL_KEY")
        if override:
            return override
    return "wan2.1_vace_14b"


register(HandlerSpec(mode="vace", handle_cls=VACEHandle, key_for=_vace_key_for, tier="large"))
```

> **Verify-at-execution:** confirm `WanVACEPipeline.from_pretrained` accepts the injected `vae=`/`text_encoder=` and does NOT require an `image_encoder` (VACE has no image branch). Confirm `__call__` accepts `video`/`mask`/`reference_images`/`conditioning_scale` (it does per the diffusers source).

- [ ] **Step 4: Register in `pipelines/__init__.py`**

Add `from pipelines.vace import VACEHandle  # noqa: F401`; append `"VACEHandle"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_vace.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add pipelines/vace.py pipelines/__init__.py tests/test_vace.py
git commit -m "Add VACEHandle (WanVACEPipeline, vae+text_encoder only) + self-register"
```

---

## Task 3: Wire VACE Generate into `app.py`

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_wiring.py`

> Read `app.py` `_run_v2v` (closest runner template), `_MODE_RUNNERS`, `_inputs_for`, `_ui_dispatch`, and `ui/tabs.py::build_vace_tab` for the real input keys (submode, source_video, mask_source, mask_input, references, prompt, +advanced).

- [ ] **Step 1: Write the failing test**

```python
def test_vace_is_wired_not_toast():
    import app
    from pipelines.handlers import HANDLER_REGISTRY
    assert "vace" in HANDLER_REGISTRY
    assert "vace" in app._MODE_RUNNERS
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_app_wiring.py::test_vace_is_wired_not_toast -v`
Expected: FAIL — `_MODE_RUNNERS` lacks vace.

- [ ] **Step 3: Add `_run_vace` + dispatch**

Add `_run_vace(spec, ui_args, progress)` mirroring `_run_v2v`. Use this UI arg order (define `_inputs_for("vace", ...)` to produce exactly this, threading `hdr["generation"]`/`hdr["preset_state"]` into slots 1–2):

`ui_args = (submode, generation, preset_state, source_video, references, prompt, negative_prompt, seed, randomize, steps, cfg, cfg_2)`

```python
def _run_vace(spec, ui_args, progress):
    import random
    from pipelines.video_io import decode_video
    from pipelines.vace_inputs import gallery_to_references, resolve_submode
    (submode, generation, preset_label, source_video, references, prompt,
     negative_prompt, seed, randomize, steps, cfg, cfg_2) = ui_args
    if not str(prompt or "").strip():
        raise gr.Error("Prompt is required.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    key = _key_for("vace", generation)
    handle = REGISTRY.acquire(key)
    pk = handle.configure_preset(_coerce_preset(preset_label))

    refs = gallery_to_references(references)
    src_frames, h, w = (None, 480, 832)
    if source_video:
        src_frames, h, w = decode_video(source_video, handle.pipe, 480 * 832)
    try:
        plan = resolve_submode(submode, source_frames=src_frames, references=refs,
                               height=h, width=w, num_frames=(len(src_frames) if src_frames else 81))
    except ValueError as e:
        raise gr.Error(str(e))

    num_frames = len(plan.video) if plan.video else 81
    inf = _build_inference_kwargs(pk, steps, cfg, cfg_2)
    out = handle.generate(prompt=prompt, video=plan.video, mask=plan.mask,
                          reference_images=plan.reference_images, negative_prompt=negative_prompt or "",
                          height=h, width=w, num_frames=num_frames, seed=int(seed), preset_kwargs=inf)
    return _export(out, "vace", pk.fallback_message)
```

Add to the dispatch + extend `_inputs_for`/`_ui_dispatch`:

```python
_MODE_RUNNERS = {"t2v": _run_t2v, "i2v": _run_i2v, "flf2v": _run_flf2v, "v2v": _run_v2v, "vace": _run_vace}
```

In `_inputs_for`, add the `vace` branch returning `[submode, generation, preset_state, source_video, references, prompt, negative_prompt, seed, randomize, steps, cfg, cfg_2]` (read the real keys from the vace tab dict; note `mask_input`/`mask_source` are NOT passed to the runner in v1 — they drive the deferred-mode UX only). In `_ui_dispatch`, add `vace` → `(generation=ui_args[1], "", duration_s=4.0)`.

- [ ] **Step 4: Run wiring test + smoke build + full suite**

Run: `.venv/bin/pytest tests/test_app_wiring.py -v && .venv/bin/python -c "from app import build; build()" && .venv/bin/pytest tests/ -q`
Expected: wiring test PASS; build clean; full suite green.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_wiring.py
git commit -m "Wire VACE Generate handler through HANDLER_REGISTRY"
```

---

## Task 4: Sub-mode UX — info banner + flow_shift by resolution

**Files:**
- Modify: `app.py` (or `ui/tabs.py` if a `.change()` handler is cleaner) + `pipelines/vace.py`
- Test: `tests/test_vace.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_vace_flow_shift_picks_by_resolution():
    """VACE-14B card flow_shift is 5.0 (720p) but must drop to 3.0 at 480p (risk R24)."""
    from pipelines.vace import flow_shift_for
    assert flow_shift_for("wan2.1_vace_14b", height=480) == 3.0
    assert flow_shift_for("wan2.1_vace_14b", height=720) == 5.0
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_vace.py::test_vace_flow_shift_picks_by_resolution -v`
Expected: FAIL — no `flow_shift_for`.

- [ ] **Step 3: Implement `flow_shift_for` + use it**

In `pipelines/vace.py`:

```python
def flow_shift_for(card_key: str, *, height: int) -> float:
    """VACE flow_shift: 3.0 at 480p, 5.0 at 720p (registry stores the 720p value)."""
    return 3.0 if height <= 480 else 5.0
```

Override `_configure_scheduler` in `VACEHandle` to use a resolution-aware flow_shift when known (default to the card value; the runner passes the chosen height). Simplest: store the last requested height on the handle in `generate()` before `ensure_loaded()` re-configures, OR pass `flow_shift` through `preset_kwargs` if `WanVACEPipeline.__call__` accepts it. **Confirm** the pipeline accepts a `flow_shift` kwarg; if not, set `self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=flow_shift_for(self.card.key, height=height))` inside `generate()` before the call.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_vace.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipelines/vace.py tests/test_vace.py
git commit -m "VACE: resolution-aware flow_shift (3.0 at 480p, 5.0 at 720p)"
```

---

## Task 5 (SLOW, deferred): Local MPS smoke for VACE-1.3B

**Files:**
- Create: `tests/test_smoke_vace_local.py`

> VACE-1.3B is the small dev target (~3.5 GB). Needs the bf16 mirror (post-#0a ops) or upstream fallback. `slow`/`mps`/`skip-by-default`.

- [ ] **Step 1: Write the skipped smoke**

```python
"""Slow local MPS smoke for VACE-1.3B Inpaint (needs the vace-1.3b mirror or upstream)."""
import os
from pathlib import Path
import pytest, torch
from PIL import Image

pytestmark = [
    pytest.mark.slow, pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available() or os.getenv("WAN_RUN_SLOW") != "1",
        reason="set WAN_RUN_SLOW=1 on Apple Silicon to run",
    ),
]


def test_vace_1_3b_inpaint_smoke(tmp_path):
    from diffusers.utils import export_to_video
    from pipelines.handle import ModelRegistry
    from pipelines.vace import VACEHandle
    from pipelines.vace_inputs import resolve_submode

    reg = ModelRegistry(factory=lambda k: VACEHandle.for_key(k))
    handle = reg.acquire("wan2.1_vace_1.3b")
    pk = handle.configure_preset("quality")
    frames = [Image.new("RGB", (832, 480), (i * 10 % 255, 40, 80)) for i in range(17)]
    plan = resolve_submode("Inpaint", source_frames=frames, references=None,
                           height=480, width=832, num_frames=17)
    out = handle.generate(prompt="a calm ocean, painterly", video=plan.video, mask=plan.mask,
                          height=480, width=832, num_frames=17, seed=1,
                          preset_kwargs={"num_inference_steps": 8, "guidance_scale": 5.0})
    dst = tmp_path / "vace.mp4"
    export_to_video(out, str(dst), fps=16)
    assert dst.stat().st_size > 10_000
```

- [ ] **Step 2: Confirm it SKIPS by default**

Run: `.venv/bin/pytest tests/test_smoke_vace_local.py -v`
Expected: `skipped`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_vace_local.py
git commit -m "Add (skipped-by-default) MPS smoke for VACE-1.3B Inpaint"
```

---

## Self-review checklist (completed)

- **Spec coverage:** `WanVACEPipeline` single-transformer, vae+text_encoder-only injection (T2 ✓); 9 sub-modes via video/mask/reference_images (T1 ✓); gray-fill `prepare_video_and_mask` + outpaint (T1 ✓ R16); deferred sub-mode gating with actionable errors (T1 ✓ R17); Quality-only via card (existing fallback ✓); gr.Video→frames (reuse decode_video ✓) + gr.Gallery→references (T1 ✓); append-only wiring (T3 ✓); resolution-aware flow_shift (T4 ✓ R24); 1.3B local override (T2 ✓). **Deferred (#1b-preproc):** DWPose/MiDaS/RAFT auto-extraction (control sub-modes accept user-provided control video in v1).
- **Placeholder scan:** none. Two verify-at-execution flags (WanVACEPipeline no-image_encoder load; whether `flow_shift` is a `__call__` kwarg vs scheduler reconfigure) are called out, not hidden.
- **Type consistency:** `resolve_submode(...) -> VacePlan(video, mask, reference_images)` (T1) consumed in `_run_vace` (T3); `gallery_to_references` / `prepare_video_and_mask` / `outpaint_video_and_mask` (T1) used in T1/T3/T5; `VACEHandle.generate(*, prompt, video, mask, reference_images, height, width, num_frames, seed, preset_kwargs)` (T2) called in T3/T5; `flow_shift_for(card_key, height)` (T4); `decode_video`/`_inputs_for`/`_ui_dispatch`/`_MODE_RUNNERS`/`REGISTRY` are existing symbols.

---

## Execution handoff

Tasks 1–4 are local code+TDD (subagent-driven). Task 5 is a skipped-by-default slow smoke for post-#0a. After #1b: **#1b-preproc** (DWPose/MiDaS/RAFT auto-extraction, gated on the `wan-preproc` mount) completes VACE; then **#2 (Animate)**.
