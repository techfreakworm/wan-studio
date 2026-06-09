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
