"""Fast/Quality preset resolver.

Translates a (model_card, preset) tuple into the actual kwargs to pass to
`pipe(...)` at inference time. Implements the graceful fallback rule from
RESEARCH.md §5.3 — when the user picks Fast for a mode without a Lightning LoRA
(VACE, S2V, Animate, TI2V-5B, T2V-1.3B), silently route to Quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pipelines.registry import ModelCard

Preset = Literal["fast", "quality"]


@dataclass(frozen=True)
class PresetKwargs:
    num_inference_steps: int
    guidance_scale: float
    guidance_scale_2: float | None    # Wan 2.2 MoE low-noise stage
    flow_shift: float
    lora_active: bool                 # True iff Lightning LoRA should be enabled
    effective_preset: Preset          # may differ from requested due to fallback
    fallback_message: str | None      # toast text shown to user on fallback


def resolve(card: ModelCard, requested: Preset) -> PresetKwargs:
    """Map (model card, requested preset) → actual pipeline kwargs."""
    if requested == "fast" and not card.lightning_available:
        return PresetKwargs(
            num_inference_steps=card.quality_steps,
            guidance_scale=card.quality_guidance,
            guidance_scale_2=card.quality_guidance_2 if card.is_moe else None,
            flow_shift=card.flow_shift,
            lora_active=False,
            effective_preset="quality",
            fallback_message=(
                f"Lightning unavailable for {card.mode.upper()} on {card.generation} "
                f"{card.size} — using Quality preset ({card.quality_steps} steps)."
            ),
        )

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
