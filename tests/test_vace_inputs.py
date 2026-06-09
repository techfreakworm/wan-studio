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
