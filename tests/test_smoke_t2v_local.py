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
