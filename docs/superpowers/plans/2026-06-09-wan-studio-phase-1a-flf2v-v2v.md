# Wan Studio Phase #1a — FLF2V + V2V Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire two diffusers-native Wan 2.1 modes — **FLF2V** (first-last-frame, `WanImageToVideoPipeline` + `last_image=`) and **V2V** (video restyle, `WanVideoToVideoPipeline`) — onto the #0 foundation, each as an append-only `HANDLER_REGISTRY` registration plus a runner.

**Architecture:** Both reuse the foundation's `WanModelHandle`/`ModelRegistry`/`_mount_path`/shared-encoder injection. FLF2V subclasses the existing `I2VHandle` (same pipeline class + `last_image=` kwarg). V2V is a new handle on the **shared** `/models/wan2.1-t2v-14b` mount. App wiring extends the existing `_MODE_RUNNERS`/`_inputs_for`/`_ui_dispatch` dispatch — no per-mode `.click()` blocks. Both are Quality-preset-only diffusers cards already in the registry.

**Tech Stack:** Python 3.12 · diffusers 0.38 (`WanImageToVideoPipeline`, `WanVideoToVideoPipeline`, `WanPipeline`) · gradio 6.14 · pytest 9 · MPS local / ZeroGPU `large` bf16.

**References:** [`program architecture & risks`](../specs/2026-06-08-wan-studio-program-architecture-and-risks.md) (FLF2V/V2V analysis) · [`#0b runtime plan`](./2026-06-09-wan-studio-foundation-0b-runtime.md) (the `HANDLER_REGISTRY` + `_MODE_RUNNERS` extension points). **Verified API:** `WanImageToVideoPipeline.__call__` accepts `last_image`; `WanVideoToVideoPipeline` exists with `video: list[Image]` + `strength: float = 0.8`.

**Out of scope:** VACE (#1b), Animate (#2), vendored S2V/TI2V (#3), Send-to/Gallery (#4).

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `pipelines/video_io.py` | **create** | `decode_video()` (gr.Video filepath → list[PIL] on the VAE grid) + `center_crop_resize()` (FLF2V end frame) |
| `pipelines/v2v.py` | **create** | `V2VHandle` (`WanVideoToVideoPipeline`) + self-register |
| `pipelines/flf2v.py` | **create** | `FLF2VHandle(I2VHandle)` + `last_image=` + self-register |
| `pipelines/__init__.py` | modify | import `v2v`/`flf2v` so they self-register; export handles |
| `app.py` | modify | add `_run_v2v`/`_run_flf2v` to `_MODE_RUNNERS`; extend `_inputs_for`/`_ui_dispatch`; wire the FLF2V `generate_end` T2I button |
| `tests/test_video_io.py` | **create** | helper unit tests |
| `tests/test_v2v.py` | **create** | V2VHandle construction + registration |
| `tests/test_flf2v.py` | **create** | FLF2VHandle construction + registration + last_image path |
| `tests/test_app_wiring.py` | modify | assert flf2v/v2v are wired (not toast) |

**Established pattern to mirror (read these first):** `pipelines/i2v.py` (`I2VHandle`, `aspect_ratio_resize`, `generate`), `pipelines/t2v.py` (non-MoE `_build_pipeline`), `pipelines/handlers.py` (`HandlerSpec`/`register`), `app.py` `_run_i2v` (the runner template) + `_inputs_for` + `_ui_dispatch` + `_MODE_RUNNERS`.

---

## Task 1: Video I/O helpers (`decode_video`, `center_crop_resize`)

**Files:**
- Create: `pipelines/video_io.py`
- Test: `tests/test_video_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_video_io.py`:

```python
"""Tests for pipelines.video_io helpers."""
from PIL import Image

from pipelines.video_io import center_crop_resize


def test_center_crop_resize_exact_dims():
    img = Image.new("RGB", (200, 100), "red")     # 2:1
    out = center_crop_resize(img, 64, 128)          # target (h=64, w=128) = 2:1
    assert out.size == (128, 64)                    # PIL .size is (w, h)


def test_center_crop_resize_taller_source():
    img = Image.new("RGB", (100, 400), "blue")    # tall
    out = center_crop_resize(img, 100, 100)
    assert out.size == (100, 100)


def test_center_crop_resize_returns_rgb():
    img = Image.new("L", (50, 50), 128)            # grayscale
    out = center_crop_resize(img, 32, 48)
    assert out.mode == "RGB"
    assert out.size == (48, 32)
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_video_io.py -v`
Expected: FAIL — no `pipelines.video_io`.

- [ ] **Step 3: Implement `pipelines/video_io.py`**

```python
"""Video/image helpers for video-input modes (V2V) and end-frame modes (FLF2V).

PIL-only (no torchvision — it is intentionally not installed; see #0b).
"""
from __future__ import annotations

from PIL import Image


def center_crop_resize(image: Image.Image, h: int, w: int) -> Image.Image:
    """Resize `image` to COVER (w, h) preserving aspect ratio, then center-crop to (w, h).

    Used for the FLF2V end frame so it matches the first frame's (h, w).
    """
    image = image.convert("RGB")
    target_ar = w / h
    src_ar = image.width / image.height
    if src_ar > target_ar:               # source wider → match height, crop width
        new_h, new_w = h, max(w, int(round(h * src_ar)))
    else:                                 # source taller → match width, crop height
        new_w, new_h = w, max(h, int(round(w / src_ar)))
    resized = image.resize((new_w, new_h))
    left, top = (new_w - w) // 2, (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def decode_video(path: str, pipe, max_area: int) -> tuple[list[Image.Image], int, int]:
    """Decode a video filepath (from gr.Video) into a list of PIL frames resized to the
    VAE/patch grid. Returns (frames, height, width). H/W derive from the FIRST frame's
    aspect ratio, snapped to `vae_scale_factor_spatial * patch_size[1]`, like I2V.
    """
    import numpy as np
    from diffusers.utils import load_video

    raw = load_video(path)               # list[PIL.Image]
    if not raw:
        raise ValueError("could not decode any frames from the input video")
    ar = raw[0].height / raw[0].width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = max(mod, int(round(np.sqrt(max_area * ar))) // mod * mod)
    w = max(mod, int(round(np.sqrt(max_area / ar))) // mod * mod)
    frames = [f.convert("RGB").resize((w, h)) for f in raw]
    return frames, h, w
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_video_io.py -v`
Expected: PASS (3 tests). (`decode_video` needs a real pipe + video; it is exercised by the slow smoke in Task 6, not here.)

- [ ] **Step 5: Commit**

```bash
git add pipelines/video_io.py tests/test_video_io.py
git commit -m "Add video_io helpers: center_crop_resize + decode_video"
```

---

## Task 2: `V2VHandle` + self-registration

**Files:**
- Create: `pipelines/v2v.py`
- Modify: `pipelines/__init__.py`
- Test: `tests/test_v2v.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v2v.py`:

```python
"""Tests for V2VHandle (no model load)."""
import pipelines  # noqa: F401 — triggers registration
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.v2v import V2VHandle


def test_v2v_registered_large_tier():
    assert "v2v" in HANDLER_REGISTRY
    spec = HANDLER_REGISTRY["v2v"]
    assert spec.handle_cls is V2VHandle
    assert spec.tier == "large"


def test_v2v_key_is_fixed_wan21():
    # V2V is Wan 2.1 only; key_for ignores generation
    spec = HANDLER_REGISTRY["v2v"]
    assert spec.key_for("wan2.2") == "wan2.1_v2v_14b"
    assert spec.key_for("wan2.1") == "wan2.1_v2v_14b"


def test_v2v_handle_card_and_lazy():
    h = V2VHandle.for_key("wan2.1_v2v_14b")
    assert h.card.mode == "v2v"
    assert h.card.mirror_repo == "techfreakworm/wan2.1-t2v-14b-bf16"  # shared backbone
    assert h.pipe is None
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_v2v.py -v`
Expected: FAIL — no `pipelines.v2v`.

- [ ] **Step 3: Implement `pipelines/v2v.py`**

> Mirror the **non-MoE branch** of `T2VHandle._build_pipeline` exactly (read `pipelines/t2v.py`), swapping the pipeline class to `WanVideoToVideoPipeline`. V2V never needs an image encoder.

```python
"""V2V — Wan 2.1 video restyle on the T2V-14B backbone (WanVideoToVideoPipeline)."""
from __future__ import annotations

from typing import Any

import torch
from diffusers import WanVideoToVideoPipeline

from pipelines import shared
from pipelines.handle import WanModelHandle, _mount_path
from pipelines.handlers import HandlerSpec, register
from utils.backend import detect


class V2VHandle(WanModelHandle):
    """WanVideoToVideoPipeline on the shared wan2.1-t2v-14b mount. Quality-only."""

    def _build_pipeline(self) -> Any:
        backend = detect()
        path = _mount_path(self.card)
        return WanVideoToVideoPipeline.from_pretrained(
            path,
            vae=shared.vae(),
            text_encoder=shared.text_encoder(),
            torch_dtype=backend.dtype,
        )

    def generate(
        self,
        video: list,
        prompt: str,
        *,
        negative_prompt: str = "",
        strength: float = 0.7,
        seed: int = 42,
        preset_kwargs: dict,
    ) -> list:
        self.ensure_loaded()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            video=video,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            strength=strength,
            generator=gen,
            **preset_kwargs,
        )
        return out.frames[0]


def _v2v_key_for(generation: str, **_ui) -> str:
    return "wan2.1_v2v_14b"  # Wan 2.1 only


register(HandlerSpec(mode="v2v", handle_cls=V2VHandle, key_for=_v2v_key_for, tier="large"))
```

> **Verify-at-execution:** confirm `WanVideoToVideoPipeline.__call__` accepts `video`+`strength` and returns `.frames[0]` (it does per the diffusers source); if it also requires `height`/`width`, derive them from the decoded frames in the runner (Task 4) and pass through.

- [ ] **Step 4: Register in `pipelines/__init__.py`**

Add `from pipelines.v2v import V2VHandle  # noqa: F401` (so importing the package self-registers v2v) and append `"V2VHandle"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_v2v.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add pipelines/v2v.py pipelines/__init__.py tests/test_v2v.py
git commit -m "Add V2VHandle (WanVideoToVideoPipeline on t2v-14b mount) + self-register"
```

---

## Task 3: `FLF2VHandle` + self-registration

**Files:**
- Create: `pipelines/flf2v.py`
- Modify: `pipelines/__init__.py`
- Test: `tests/test_flf2v.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_flf2v.py`:

```python
"""Tests for FLF2VHandle (no model load)."""
import pipelines  # noqa: F401
from pipelines.handlers import HANDLER_REGISTRY
from pipelines.flf2v import FLF2VHandle
from pipelines.i2v import I2VHandle


def test_flf2v_registered():
    assert "flf2v" in HANDLER_REGISTRY
    spec = HANDLER_REGISTRY["flf2v"]
    assert spec.handle_cls is FLF2VHandle
    assert spec.tier == "large"


def test_flf2v_subclasses_i2v():
    assert issubclass(FLF2VHandle, I2VHandle)


def test_flf2v_key_fixed_720p():
    spec = HANDLER_REGISTRY["flf2v"]
    assert spec.key_for("wan2.1") == "wan2.1_flf2v_14b_720p"
    assert spec.key_for("wan2.2") == "wan2.1_flf2v_14b_720p"  # Wan 2.1 only


def test_flf2v_handle_card():
    h = FLF2VHandle.for_key("wan2.1_flf2v_14b_720p")
    assert h.card.mode == "flf2v"
    assert h.card.requires_image_encoder is True
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_flf2v.py -v`
Expected: FAIL — no `pipelines.flf2v`.

- [ ] **Step 3: Implement `pipelines/flf2v.py`**

```python
"""FLF2V — Wan 2.1 first-last-frame. WanImageToVideoPipeline + last_image= kwarg.

Subclasses I2VHandle (identical pipeline class + shared-encoder injection); only
generate() differs: it resizes the first frame (aspect_ratio_resize) and
center-crop-resizes the last frame to match, then passes last_image=.
Lightning is BETA (reuses the I2V LoRA, not FLF2V-trained) — UI labels it Beta.
"""
from __future__ import annotations

import torch

from pipelines.flf2v_handlers import register_flf2v  # see Step 4 ordering note
from pipelines.i2v import I2VHandle, aspect_ratio_resize
from pipelines.video_io import center_crop_resize


class FLF2VHandle(I2VHandle):
    """720p-locked; max_area fixed at 720*1280."""

    def generate(
        self,
        image,
        last_image,
        prompt: str,
        *,
        negative_prompt: str = "",
        max_area: int = 720 * 1280,
        num_frames: int = 81,
        seed: int = 42,
        preset_kwargs: dict,
    ) -> list:
        self.ensure_loaded()
        first, h, w = aspect_ratio_resize(image, self.pipe, max_area)
        last = center_crop_resize(last_image, h, w)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            image=first,
            last_image=last,
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

Then self-register at the bottom (keep the import after the class def, like t2v/i2v):

```python
from pipelines.handlers import HandlerSpec, register  # noqa: E402


def _flf2v_key_for(generation: str, **_ui) -> str:
    return "wan2.1_flf2v_14b_720p"  # Wan 2.1 only, 720p


register(HandlerSpec(mode="flf2v", handle_cls=FLF2VHandle, key_for=_flf2v_key_for, tier="large"))  # noqa: E402
```

> Delete the stray `from pipelines.flf2v_handlers import register_flf2v` line shown above — it was a placeholder; FLF2V self-registers via the `register(...)` call at the bottom, exactly like t2v.py/i2v.py. (Do NOT create a `flf2v_handlers` module.)

- [ ] **Step 4: Register in `pipelines/__init__.py`**

Add `from pipelines.flf2v import FLF2VHandle  # noqa: F401` and append `"FLF2VHandle"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_flf2v.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add pipelines/flf2v.py pipelines/__init__.py tests/test_flf2v.py
git commit -m "Add FLF2VHandle (I2VHandle + last_image) + self-register"
```

---

## Task 4: Wire FLF2V + V2V Generate into `app.py`

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_wiring.py`

> Read `app.py` `_run_i2v` (the runner template, ~line 354), `_MODE_RUNNERS` (~416), `_inputs_for` (~1812), and `_ui_dispatch` (~257) first. The `.click()` wiring loop already iterates `HANDLER_REGISTRY`, so once `_MODE_RUNNERS` + `_inputs_for` know the new modes, the buttons wire automatically.

- [ ] **Step 1: Write the failing test (extend `tests/test_app_wiring.py`)**

```python
def test_flf2v_and_v2v_are_wired_not_toast():
    """After registration, flf2v/v2v Generate buttons route to a runner, not _generate_toast."""
    import app
    from pipelines.handlers import HANDLER_REGISTRY
    assert "flf2v" in HANDLER_REGISTRY
    assert "v2v" in HANDLER_REGISTRY
    assert "flf2v" in app._MODE_RUNNERS
    assert "v2v" in app._MODE_RUNNERS
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_app_wiring.py::test_flf2v_and_v2v_are_wired_not_toast -v`
Expected: FAIL — `_MODE_RUNNERS` lacks flf2v/v2v.

- [ ] **Step 3: Add the runners + dispatch entries**

In `app.py`, define `_run_v2v` and `_run_flf2v` matching the `_run_i2v(spec, ui_args, progress)` signature. Use these UI arg orders (define `_inputs_for` to produce exactly these, mirroring how `_run_i2v` unpacks):

**V2V** `ui_args = (video, generation, preset_state, prompt, strength, negative_prompt, seed, randomize, steps, cfg, cfg_2)`:

```python
def _run_v2v(spec, ui_args, progress):
    import random
    from pipelines.video_io import decode_video
    (video, generation, preset_label, prompt, strength,
     negative_prompt, seed, randomize, steps, cfg, cfg_2) = ui_args
    if not video:
        raise gr.Error("Please upload a video.")
    if not prompt:
        raise gr.Error("Restyle prompt is required.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    key = _key_for("v2v", generation)
    handle = REGISTRY.acquire(key)
    pk = handle.configure_preset(_coerce_preset(preset_label))
    frames_in, _, _ = decode_video(video, handle.pipe, 480 * 832)
    inf = _build_inference_kwargs(pk, steps, cfg, cfg_2)
    out = handle.generate(frames_in, prompt, negative_prompt=negative_prompt or "",
                          strength=float(strength), seed=int(seed), preset_kwargs=inf)
    return _export(out, "v2v", pk.fallback_message)
```

**FLF2V** `ui_args = (start_frame, generation, preset_state, end_frame_uploaded, end_frame_generated, prompt, negative_prompt, seed, randomize, steps, cfg, cfg_2)`:

```python
def _run_flf2v(spec, ui_args, progress):
    import random
    (start_frame, generation, preset_label, end_uploaded, end_generated, prompt,
     negative_prompt, seed, randomize, steps, cfg, cfg_2) = ui_args
    if start_frame is None:
        raise gr.Error("Please provide a start frame.")
    last = end_uploaded if end_uploaded is not None else end_generated
    if last is None:
        raise gr.Error("Please upload or generate an end frame.")
    if not prompt:
        raise gr.Error("Transition prompt is required.")
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    key = _key_for("flf2v", generation)
    handle = REGISTRY.acquire(key)
    pk = handle.configure_preset(_coerce_preset(preset_label))
    # FLF2V Quality default CFG 5.5 when the user didn't override
    cfg_eff = cfg if (cfg and cfg > 0) else 5.5
    inf = _build_inference_kwargs(pk, steps, cfg_eff, cfg_2)
    out = handle.generate(start_frame, last, prompt, negative_prompt=negative_prompt or "",
                          seed=int(seed), preset_kwargs=inf)
    return _export(out, "flf2v", pk.fallback_message)
```

> If `app.py` has no shared `_export(frames, mode, fallback_message)` helper (the #0b refactor may inline export in `_run_t2v`/`_run_i2v`), factor one out (tempfile `wan_<mode>_*.mp4` + `export_to_video(frames, path, fps=16)` + the `gr.Info(fallback_message)` toast) and use it in all runners. Match the existing inline logic exactly.

Add both to the dispatch table:

```python
_MODE_RUNNERS = {"t2v": _run_t2v, "i2v": _run_i2v, "flf2v": _run_flf2v, "v2v": _run_v2v}
```

- [ ] **Step 4: Extend `_inputs_for` and `_ui_dispatch`**

In `_inputs_for(mode, tab, hdr)`, add the `v2v` and `flf2v` branches returning the component lists in EXACTLY the arg order the runners unpack (above), threading `hdr["generation"]`/`hdr["preset_state"]` into slots 1–2. In `_ui_dispatch(mode, ui_args)`, add `v2v`/`flf2v` so it returns `(generation, "", duration_s)` — both have no resolution dropdown; use a fixed `duration_s` (e.g. 3.0 for V2V; FLF2V is 81-frame fixed) so `_get_duration` reserves sanely. Follow the existing t2v/i2v branch structure exactly.

- [ ] **Step 5: Run the wiring test + smoke build + full suite**

Run: `.venv/bin/pytest tests/test_app_wiring.py -v && .venv/bin/python -c "from app import build; build()" && .venv/bin/pytest tests/ -q`
Expected: wiring test PASS; build prints no exception; full suite green.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_wiring.py
git commit -m "Wire FLF2V + V2V Generate handlers through HANDLER_REGISTRY"
```

---

## Task 5: FLF2V end-frame "Generate" (T2I) sub-handler

**Files:**
- Modify: `app.py` (wire the `generate_end` button)
- Test: `tests/test_app_wiring.py`

> The FLF2V tab has a secondary `generate_end` button + `end_frame_prompt` that synthesizes an end frame via a Wan **T2I** call (`WanPipeline`, `num_frames=1`). To avoid two 14B transformers warm at once (LRU thrash, risk R21), run it as its own short load→generate→unload using the registry.

- [ ] **Step 1: Write the failing test**

```python
def test_flf2v_generate_end_handler_exists():
    import app
    assert callable(getattr(app, "generate_end_frame", None))
```

- [ ] **Step 2: Run, observe failure**

Run: `.venv/bin/pytest tests/test_app_wiring.py::test_flf2v_generate_end_handler_exists -v`
Expected: FAIL.

- [ ] **Step 3: Implement the T2I sub-handler + wire the button**

```python
@spaces_gpu_or_noop()(duration=lambda *a, **k: 30, size="large")
def generate_end_frame(end_frame_prompt, generation, progress=gr.Progress(track_tqdm=False)):
    """Synthesize an FLF2V end frame via Wan T2I (num_frames=1). Returns one PIL image."""
    if not end_frame_prompt:
        raise gr.Error("Enter a prompt to generate an end frame.")
    key = _key_for("t2v", generation)             # reuse the T2V backbone
    handle = REGISTRY.acquire(key)                 # LRU: evicts the warm transformer if needed
    pk = handle.configure_preset("quality")
    frames = handle.generate(
        prompt=end_frame_prompt, negative_prompt="", height=720, width=1280,
        num_frames=1, seed=0,
        preset_kwargs=_build_inference_kwargs(pk, 0, 0, 0),
    )
    return frames[0]                                # single PIL/np frame → gr.Image
```

In `build()`, wire the button (outside the `HANDLER_REGISTRY` loop, since it's a secondary button):

```python
flf2v_tab = tabs["flf2v"]["inputs"]
flf2v_tab["generate_end"].click(
    fn=generate_end_frame,
    inputs=[flf2v_tab["end_frame_prompt"], header["generation"]],
    outputs=[flf2v_tab["end_frame_generated"]],
)
```

> Note: the `_generate_toast` loop in `build()` currently also toasts `generate_end` for flf2v (the #0b loop wired all secondary buttons to toast). Remove the `generate_end` toast wiring before adding the real `.click()` so it doesn't double-fire (risk R18). Confirm against the actual loop.

- [ ] **Step 4: Run tests + smoke build**

Run: `.venv/bin/pytest tests/test_app_wiring.py -v && .venv/bin/python -c "from app import build; build()"`
Expected: PASS; build clean.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_wiring.py
git commit -m "FLF2V: wire end-frame Generate (Wan T2I num_frames=1) sub-handler"
```

---

## Task 6 (SLOW, deferred): Local MPS smoke for V2V

**Files:**
- Create: `tests/test_smoke_flf2v_v2v_local.py`

> Both are 14B → ~28 GB bf16 + ~11 GB UMT5 download/load on MPS, and need the bf16 mirrors (from #0a ops) or upstream fp32 fallback. Marked `slow`/`mps`/`skip-by-default`; run manually once mirrors exist.

- [ ] **Step 1: Write the smoke test (skipped by default)**

```python
"""Slow local MPS smoke for V2V (needs the t2v-14b mirror or upstream fallback)."""
import os
from pathlib import Path
import pytest, torch

pytestmark = [
    pytest.mark.slow, pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available() or os.getenv("WAN_RUN_SLOW") != "1",
        reason="set WAN_RUN_SLOW=1 on Apple Silicon to run",
    ),
]


def test_v2v_smoke(tmp_path):
    from diffusers.utils import export_to_video, load_video
    from pipelines.handle import ModelRegistry
    from pipelines.v2v import V2VHandle
    from pipelines.video_io import decode_video

    reg = ModelRegistry(factory=lambda k: V2VHandle.for_key(k))
    handle = reg.acquire("wan2.1_v2v_14b")
    pk = handle.configure_preset("quality")
    # a tiny synthetic 17-frame clip
    src = tmp_path / "in.mp4"
    from PIL import Image
    export_to_video([Image.new("RGB", (832, 480), (i * 8 % 255, 0, 0)) for i in range(17)],
                    str(src), fps=16)
    frames_in, _, _ = decode_video(str(src), handle.pipe, 480 * 832)
    out = handle.generate(frames_in[:17], "make it watercolor", strength=0.6, seed=1,
                          preset_kwargs={"num_inference_steps": 8, "guidance_scale": 5.0})
    dst = tmp_path / "out.mp4"
    export_to_video(out, str(dst), fps=16)
    assert dst.stat().st_size > 10_000
```

- [ ] **Step 2: Confirm it SKIPS by default**

Run: `.venv/bin/pytest tests/test_smoke_flf2v_v2v_local.py -v`
Expected: `skipped` (no `WAN_RUN_SLOW`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_flf2v_v2v_local.py
git commit -m "Add (skipped-by-default) MPS smoke for V2V"
```

---

## Self-review checklist (completed)

- **Spec coverage:** FLF2V via `WanImageToVideoPipeline`+`last_image` (T3 ✓), end-frame Upload-vs-Generate (T5 ✓), Beta-Lightning note (T3 docstring ✓), V2V via `WanVideoToVideoPipeline` on shared mount (T2 ✓, no workaround needed), strength slider (T4 ✓), Quality-only via the registry cards' `lightning_available=False` (existing fallback ✓), gr.Video→frames decode (T1 ✓), append-only `HANDLER_REGISTRY` wiring (T4 ✓), LRU-thrash mitigation for end-frame T2I (T5 ✓ R21).
- **Placeholder scan:** none. The one trap (a placeholder import line in the T3 snippet) is explicitly called out and instructed to be deleted. Two verify-at-execution flags (the `WanVideoToVideoPipeline` height/width need; the `_export` helper name) are noted, not hidden.
- **Type consistency:** `center_crop_resize(image, h, w)` / `decode_video(path, pipe, max_area) -> (frames, h, w)` (T1) used in T3/T4/T6; `V2VHandle.generate(video, prompt, *, strength, seed, preset_kwargs)` (T2) called in T4/T6; `FLF2VHandle.generate(image, last_image, prompt, ...)` (T3) called in T4; `_MODE_RUNNERS`/`_inputs_for`/`_ui_dispatch`/`_key_for`/`_build_inference_kwargs`/`REGISTRY.acquire` are the existing #0b symbols (T4 extends them).

---

## Execution handoff

Tasks 1–5 are local code+TDD (subagent-driven). Task 6 is a skipped-by-default slow smoke for when the bf16 mirrors exist (post-#0a ops). After #1a: **#1b (VACE)** is the next plan.
