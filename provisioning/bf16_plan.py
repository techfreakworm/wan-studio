"""Per-card bf16 conversion plan.

Transformer-only mirrors: convert transformer(s) to bf16, keep the small
config/scheduler/tokenizer + model_index.json, DROP text_encoder/ and vae/
(they live once in wan-shared-encoders and are injected at load). Image-
conditioned cards (requires_image_encoder) additionally keep image_processor/
(a tiny config from_pretrained may require); Animate also keeps the large
image_encoder/ CLIP — for the other image cards it's injected at load
(amendment 2 + R9 insurance). Vendored S2V/TI2V (diffusers_class=None) return
None — handled in #3.
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
    # Keep image_processor for ANY image-conditioned card (I2V/FLF2V/Animate):
    # it's a tiny config that WanImageToVideoPipeline.from_pretrained may require
    # if model_index.json lists it, and the handler doesn't inject it. A
    # snapshot_download of a non-existent subfolder is a harmless no-op.
    if card.requires_image_encoder:
        keep |= {"image_processor"}
    # Keep image_encoder (the large CLIP) ONLY for Animate; for the others it is
    # injected at load (R9 spike will confirm).
    if card.mode == "animate":
        keep |= {"image_encoder"}
    return ConversionPlan(card.key, convert, sorted(keep))
