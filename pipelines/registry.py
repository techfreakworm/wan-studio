"""Wan model + mode registry.

Source-of-truth catalog of every supported (generation, mode, checkpoint) tuple,
their Diffusers pipeline class, Lightning LoRA availability, ZeroGPU duration
budgets. Pulled from RESEARCH.md §2-§5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Generation = Literal["wan2.1", "wan2.2"]
Mode = Literal["t2v", "i2v", "flf2v", "v2v", "vace", "s2v", "animate", "ti2v"]


@dataclass(frozen=True)
class ModelCard:
    key: str                              # stable id used in UI dispatch
    generation: Generation
    mode: Mode
    repo: str                             # upstream HF repo path (fp32)
    size: str                             # "1.3B" / "14B" / "5B" / "A14B" / etc.
    native_resolutions: tuple[str, ...]   # e.g. ("480p",) or ("480p", "720p")
    native_fps: int
    frames_default: int
    diffusers_class: str | None           # None => not in diffusers, vendor upstream wan
    is_moe: bool                          # Wan 2.2 A14B family
    requires_image_encoder: bool          # I2V, FLF2V, Animate
    lightning_available: bool             # Fast preset works
    lightning_lora_repo: str | None
    lightning_high_lora: str | None       # MoE high-noise expert LoRA file
    lightning_low_lora: str | None        # MoE low-noise expert LoRA file
    lightning_steps: int                  # 4 typical
    lightning_guidance: float             # 1.0 for CFG-distilled
    quality_steps: int
    quality_guidance: float
    flow_shift: float                     # QUALITY-preset scheduler shift (euler)
    zerogpu_duration: int                 # seconds for one run at Fast preset
    mirror_repo: str = ""                 # bf16 mirror (techfreakworm/<slug>-bf16); defaulted post-init
    quality_guidance_2: float | None = None  # MoE low-noise stage; None for non-MoE
    # FAST/Lightning scheduler shift. Lightning is a *distilled* sampler calibrated to a
    # specific noise schedule — it must run at the shift it was distilled with, NOT the
    # quality shift, or output goes soft/washed. lightx2v/ComfyUI Fast use shift≈5.0.
    lightning_flow_shift: float = 5.0
    notes: str = ""

    def __post_init__(self):
        if not self.mirror_repo:
            slug = self.key.replace("_", "-")
            object.__setattr__(self, "mirror_repo", f"techfreakworm/{slug}-bf16")


# --------------------------------------------------------------------------------------
# Wan 2.1 (Feb–May 2025) — single dense DiT, shared Wan-VAE
# --------------------------------------------------------------------------------------

WAN_2_1: list[ModelCard] = [
    ModelCard(
        key="wan2.1_t2v_1.3b",
        generation="wan2.1", mode="t2v",
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        size="1.3B",
        native_resolutions=("480p",), native_fps=16, frames_default=81,
        diffusers_class="WanPipeline",
        is_moe=False, requires_image_encoder=False,
        lightning_available=False,  # no LoRA — FastWan full ckpt exists separately
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=50, quality_guidance=5.0, flow_shift=3.0,
        zerogpu_duration=60,
        notes="No Lightning LoRA. Use FastWan2.1-T2V-1.3B full ckpt for Fast preset.",
    ),
    ModelCard(
        key="wan2.1_t2v_14b",
        generation="wan2.1", mode="t2v",
        repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        size="14B",
        native_resolutions=("480p", "720p"), native_fps=16, frames_default=81,
        diffusers_class="WanPipeline",
        is_moe=False, requires_image_encoder=False,
        lightning_available=True,
        lightning_lora_repo=None,  # resolved at runtime via _lora_repo_for(card)
        lightning_high_lora="wan2.1-t2v-14b/lightning.safetensors",
        lightning_low_lora=None,  # single transformer, no LOW expert
        lightning_steps=4, lightning_guidance=1.0,
        quality_steps=50, quality_guidance=5.0, flow_shift=5.0,  # 720p; use 3.0 for 480p
        zerogpu_duration=90,
    ),
    ModelCard(
        key="wan2.1_i2v_14b_480p",
        generation="wan2.1", mode="i2v",
        repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        size="14B",
        native_resolutions=("480p",), native_fps=16, frames_default=81,
        diffusers_class="WanImageToVideoPipeline",
        is_moe=False, requires_image_encoder=True,
        lightning_available=True,
        lightning_lora_repo="lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
        lightning_high_lora="wan2.1-i2v-14b-480p/lightning.safetensors",
        lightning_low_lora=None,
        lightning_steps=4, lightning_guidance=1.0,
        quality_steps=40, quality_guidance=5.0, flow_shift=3.0,
        zerogpu_duration=90,
    ),
    ModelCard(
        key="wan2.1_i2v_14b_720p",
        generation="wan2.1", mode="i2v",
        repo="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        size="14B",
        native_resolutions=("720p",), native_fps=16, frames_default=81,
        diffusers_class="WanImageToVideoPipeline",
        is_moe=False, requires_image_encoder=True,
        lightning_available=True,
        lightning_lora_repo="lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v",
        lightning_high_lora="wan2.1-i2v-14b-720p/lightning.safetensors",
        lightning_low_lora=None,
        lightning_steps=4, lightning_guidance=1.0,
        quality_steps=40, quality_guidance=5.0, flow_shift=5.0,
        zerogpu_duration=120,
    ),
    ModelCard(
        key="wan2.1_flf2v_14b_720p",
        generation="wan2.1", mode="flf2v",
        repo="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        size="14B",
        native_resolutions=("720p",), native_fps=16, frames_default=81,
        diffusers_class="WanImageToVideoPipeline",  # via last_image= kwarg
        is_moe=False, requires_image_encoder=True,
        # Lightning DISABLED: there is NO usable bf16 720p-i2v Lightning LoRA (lightx2v
        # ships fp8/int8 only — Metal-hostile; Kijai has 480p only). The "empirical
        # reuse" path 404s. flf2v is Quality-only on MPS (key-chunked flash makes 720p
        # Quality correct — see mps_patches). 2026-06-22.
        lightning_available=False,
        lightning_lora_repo=None,
        lightning_high_lora=None,
        lightning_low_lora=None,
        lightning_steps=4, lightning_guidance=1.0,
        quality_steps=40, quality_guidance=5.5, flow_shift=5.0,
        zerogpu_duration=150,
        notes="Quality-only on MPS (no bf16 720p Lightning LoRA exists). Chinese prompts recommended.",
    ),
    ModelCard(
        key="wan2.1_vace_1.3b",
        generation="wan2.1", mode="vace",
        repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        size="1.3B",
        native_resolutions=("480p",), native_fps=16, frames_default=81,
        diffusers_class="WanVACEPipeline",
        is_moe=False, requires_image_encoder=False,
        lightning_available=False,
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=30, quality_guidance=5.0, flow_shift=3.0,
        zerogpu_duration=150,
        notes="Quality preset only. VACE not Lightning-trained.",
    ),
    ModelCard(
        key="wan2.1_vace_14b",
        generation="wan2.1", mode="vace",
        repo="Wan-AI/Wan2.1-VACE-14B-diffusers",
        size="14B",
        native_resolutions=("480p", "720p"), native_fps=16, frames_default=81,
        diffusers_class="WanVACEPipeline",
        is_moe=False, requires_image_encoder=False,
        lightning_available=False,
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=30, quality_guidance=5.0, flow_shift=5.0,
        zerogpu_duration=180,
        notes="Quality preset only. VACE not Lightning-trained.",
    ),
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
]

# --------------------------------------------------------------------------------------
# Wan 2.2 (Jul–Nov 2025) — dense + MoE mix
# --------------------------------------------------------------------------------------

WAN_2_2: list[ModelCard] = [
    ModelCard(
        key="wan2.2_ti2v_5b",
        generation="wan2.2", mode="ti2v",
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        size="5B",
        native_resolutions=("720p",), native_fps=24, frames_default=121,
        # B2 fix (2026-06-21): TI2V-5B IS in diffusers as WanPipeline (Tier-3 spike).
        # The old diffusers_class=None made conversion_plan() skip it. Dense 5B
        # (is_moe=False → single transformer). It ships its OWN 16×16×4 VAE (NOT the
        # shared Wan 8×8×4 VAE) → the ti2v.py handler loads that VAE from this repo,
        # not shared.vae(). Image conditioning is VAE-encoded (no CLIP), so
        # requires_image_encoder stays False. Confirm pipeline _class_name + whether
        # a transformer_2 exists from model_index.json at build time.
        diffusers_class="WanPipeline",
        is_moe=False, requires_image_encoder=False,
        lightning_available=False,
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=50, quality_guidance=5.0, flow_shift=5.0,
        zerogpu_duration=60,
        notes="Own 16×16×4 VAE (not shared). Only 1280×704 / 704×1280 supported.",
    ),
    ModelCard(
        key="wan2.2_t2v_a14b",
        generation="wan2.2", mode="t2v",
        repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        size="A14B",
        native_resolutions=("480p", "720p"), native_fps=24, frames_default=81,
        diffusers_class="WanPipeline",
        is_moe=True, requires_image_encoder=False,
        lightning_available=True,
        # bf16 MoE Lightning from the cached lightx2v/Wan2.2-Lightning repo (Seko-V1
        # rank64). NOT the consolidated-mirror subpath (which lacks it) — these
        # filenames don't start with the slug, so _lora_repo_for routes here.
        lightning_lora_repo="lightx2v/Wan2.2-Lightning",
        lightning_high_lora="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        lightning_low_lora="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
        lightning_steps=4, lightning_guidance=1.0,
        quality_steps=40, quality_guidance=3.0, flow_shift=12.0,
        zerogpu_duration=120,
        quality_guidance_2=4.0,
        notes="MoE: HIGH→transformer, LOW→transformer_2 (load_into_transformer_2=True). "
              "Quality CFG per Wan repo wan_t2v_A14B.py sample_guide_scale=(3.0, 4.0).",
    ),
    ModelCard(
        key="wan2.2_i2v_a14b",
        generation="wan2.2", mode="i2v",
        repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        size="A14B",
        native_resolutions=("480p", "720p"), native_fps=24, frames_default=81,
        diffusers_class="WanImageToVideoPipeline",
        is_moe=True, requires_image_encoder=True,
        lightning_available=True,
        # bf16 MoE Lightning from the cached lightx2v/Wan2.2-Lightning repo (Seko-V1
        # rank64). Routes via _lora_repo_for to the upstream repo (not consolidated).
        lightning_lora_repo="lightx2v/Wan2.2-Lightning",
        lightning_high_lora="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        lightning_low_lora="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
        lightning_steps=4, lightning_guidance=1.0,
        quality_steps=40, quality_guidance=3.5, flow_shift=8.0,
        zerogpu_duration=150,
        quality_guidance_2=3.5,
        notes="Only V1 (Seko) I2V LoRA. Hybrid trick: reuse Wan 2.1 lightx2v I2V LoRA for sharper output.",
    ),
    ModelCard(
        key="wan2.2_s2v_14b",
        generation="wan2.2", mode="s2v",
        repo="Wan-AI/Wan2.2-S2V-14B",
        size="14B",
        native_resolutions=("480p", "720p"), native_fps=24, frames_default=0,  # variable, audio-driven
        diffusers_class=None,  # NOT in diffusers — vendor upstream wan
        is_moe=False, requires_image_encoder=True,
        lightning_available=False,
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=40, quality_guidance=4.5, flow_shift=3.0,
        zerogpu_duration=240,
        notes="Vendor upstream `wan` package. wav2vec2-large-xlsr-53 bundled in repo. Variable length.",
    ),
    ModelCard(
        key="wan2.2_animate_14b",
        generation="wan2.2", mode="animate",
        repo="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        size="14B",
        native_resolutions=("720p",), native_fps=30, frames_default=77,  # per segment
        diffusers_class="WanAnimatePipeline",
        is_moe=False, requires_image_encoder=True,
        lightning_available=False,
        lightning_lora_repo=None, lightning_high_lora=None, lightning_low_lora=None,
        lightning_steps=0, lightning_guidance=0.0,
        quality_steps=20, quality_guidance=1.0, flow_shift=5.0,  # CFG-disabled by default
        zerogpu_duration=300,
        notes="ViTPose-H + YOLOv10 + SAM2 preproc required (~2 GB). Multi-segment stitching native.",
    ),
]

ALL_MODELS: list[ModelCard] = WAN_2_1 + WAN_2_2

# Quick lookups
BY_KEY: dict[str, ModelCard] = {m.key: m for m in ALL_MODELS}


def for_generation(gen: Generation) -> list[ModelCard]:
    return [m for m in ALL_MODELS if m.generation == gen]


def for_mode(gen: Generation, mode: Mode) -> list[ModelCard]:
    return [m for m in ALL_MODELS if m.generation == gen and m.mode == mode]


def modes_in(gen: Generation) -> list[Mode]:
    """List of modes available in a given generation, deduplicated, in canonical order."""
    canonical_order: list[Mode] = ["t2v", "i2v", "ti2v", "flf2v", "v2v", "vace", "s2v", "animate"]
    available = {m.mode for m in for_generation(gen)}
    return [m for m in canonical_order if m in available]
