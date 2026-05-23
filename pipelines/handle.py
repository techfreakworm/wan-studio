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

    HF Volume mounts serve truncated copies of small JSON files (e.g.
    transformer/config.json is 290B mounted vs 495B on the mirror), so we
    bypass the mount for base models and always load via the mirror repo ID.
    `preload_from_hub` in README YAML caches the heavy safetensors into the
    container image at build time, so the first generate isn't a cold pull.
    """
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
