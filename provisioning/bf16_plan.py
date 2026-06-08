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
