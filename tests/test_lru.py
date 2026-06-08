"""Tests for handle.ModelRegistry — one-warm-transformer LRU."""
from pipelines.handle import ModelRegistry, WanModelHandle
from pipelines.registry import BY_KEY


class _FakeHandle(WanModelHandle):
    def __init__(self, card):
        super().__init__(card)
        self.unloaded = False

    def ensure_loaded(self):
        self.pipe = object()  # pretend built

    def unload_to_cpu(self):
        self.unloaded = True


def test_acquire_builds_and_caches():
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    h1 = reg.acquire("wan2.1_t2v_14b")
    assert reg.acquire("wan2.1_t2v_14b") is h1  # same key → same warm handle


def test_acquire_evicts_previous_on_switch():
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    h1 = reg.acquire("wan2.1_t2v_14b")
    h2 = reg.acquire("wan2.1_i2v_14b_480p")
    assert h1.unloaded is True          # previous transformer evicted
    assert h2.unloaded is False
    assert reg.warm_key == "wan2.1_i2v_14b_480p"


def test_acquire_unknown_raises():
    import pytest
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    with pytest.raises(KeyError):
        reg.acquire("nope")
