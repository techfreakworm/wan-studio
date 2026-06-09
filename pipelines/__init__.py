"""Pipeline layer — model registry, shared component loaders, preset resolver,
and (Phase 1+) per-mode pipeline wrappers.

See RESEARCH.md §7-§8 for the architecture rationale.
"""
from pipelines.registry import (
    ALL_MODELS,
    BY_KEY,
    Generation,
    Mode,
    ModelCard,
    WAN_2_1,
    WAN_2_2,
    for_generation,
    for_mode,
    modes_in,
)
from pipelines.preset import Preset, PresetKwargs, resolve
from pipelines.handle import WanModelHandle
from pipelines.t2v import T2VHandle
from pipelines.i2v import I2VHandle, aspect_ratio_resize
from pipelines.v2v import V2VHandle  # noqa: F401
from pipelines.handlers import HANDLER_REGISTRY, HandlerSpec, register  # noqa: F401

__all__ = [
    "ALL_MODELS",
    "BY_KEY",
    "Generation",
    "Mode",
    "ModelCard",
    "WAN_2_1",
    "WAN_2_2",
    "for_generation",
    "for_mode",
    "modes_in",
    "Preset",
    "PresetKwargs",
    "resolve",
    "WanModelHandle",
    "T2VHandle",
    "I2VHandle",
    "V2VHandle",
    "aspect_ratio_resize",
    "HANDLER_REGISTRY",
    "HandlerSpec",
    "register",
]
