"""HANDLER_REGISTRY — per-mode plugin registration.

Each mode module (t2v.py, i2v.py, vace.py, ...) calls register(...) at import
time. app.py and __init__.py iterate this registry instead of hard-coding
per-mode wiring, so later phases are append-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HandlerSpec:
    mode: str
    handle_cls: type            # WanModelHandle subclass
    key_for: Callable[..., str] # (generation, **ui_kwargs) -> registry key
    tier: str = "large"         # "large" | "xlarge" — @spaces.GPU size literal


HANDLER_REGISTRY: dict[str, HandlerSpec] = {}


def register(spec: HandlerSpec) -> None:
    HANDLER_REGISTRY[spec.mode] = spec
