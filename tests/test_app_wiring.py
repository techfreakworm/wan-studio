"""Tests for app.py routing through ModelRegistry + per-tier @spaces.GPU entrypoints.

Task 8: app.py no longer keeps per-mode handle dicts (T2V_HANDLES / I2V_HANDLES)
nor per-mode closures (_build_t2v_handler / _build_i2v_handler). Instead it owns
one `REGISTRY = ModelRegistry(...)`, two tier entrypoints (generate_large /
generate_xlarge) that dispatch by `mode` through HANDLER_REGISTRY, and a Generate
wiring loop in `build()` driven by HANDLER_REGISTRY.

The plan's headline test is the smoke build (`from app import build; build()`),
which these tests exercise alongside the structural assertions.
"""
import app
from pipelines.handle import ModelRegistry
from pipelines.handlers import HANDLER_REGISTRY


def test_registry_is_model_registry():
    """app owns a single ModelRegistry, not per-mode handle dicts."""
    assert isinstance(app.REGISTRY, ModelRegistry)


def test_per_mode_handle_dicts_removed():
    """The old per-mode caches are gone (replaced by the LRU ModelRegistry)."""
    assert not hasattr(app, "T2V_HANDLES")
    assert not hasattr(app, "I2V_HANDLES")


def test_old_per_mode_builders_removed():
    """The two per-mode closures are replaced by the generic tier entrypoints."""
    assert not hasattr(app, "_build_t2v_handler")
    assert not hasattr(app, "_build_i2v_handler")


def test_per_tier_entrypoints_exist():
    """One decorated entrypoint per ZeroGPU size tier."""
    assert callable(app.generate_large)
    assert callable(app.generate_xlarge)


def test_build_handle_resolves_through_registry():
    """_build_handle returns a handle whose card matches the requested key,
    using HANDLER_REGISTRY specs (not hard-coded per-mode lookups)."""
    handle = app._build_handle("wan2.2_t2v_a14b")
    assert handle.card.key == "wan2.2_t2v_a14b"
    handle = app._build_handle("wan2.1_i2v_14b_480p")
    assert handle.card.key == "wan2.1_i2v_14b_480p"


def test_t2v_and_i2v_modes_are_wired_from_registry():
    """Both wired modes live in HANDLER_REGISTRY so build() iterates them."""
    assert "t2v" in HANDLER_REGISTRY
    assert "i2v" in HANDLER_REGISTRY


def test_flf2v_and_v2v_are_wired_not_toast():
    """After registration, flf2v/v2v Generate buttons route to a runner, not _generate_toast."""
    import app
    from pipelines.handlers import HANDLER_REGISTRY
    assert "flf2v" in HANDLER_REGISTRY
    assert "v2v" in HANDLER_REGISTRY
    assert "flf2v" in app._MODE_RUNNERS
    assert "v2v" in app._MODE_RUNNERS


def test_smoke_build():
    """The plan's headline test: `from app import build; build()` builds a
    Blocks with no model load and no exception."""
    demo = app.build()
    assert type(demo).__name__ == "Blocks"
