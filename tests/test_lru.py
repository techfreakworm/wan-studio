"""Tests for handle.ModelRegistry — bounded CPU-warm cache, one GPU resident.

The registry decouples two residencies:
  • GPU-attached — exactly one key (`warm_key`); switching moves the prior one
    off the GPU via `unload_to_cpu()` but leaves it built.
  • CPU-warm — up to `max_warm` handles stay resident so a swap-back is a cache
    hit (no cold disk→CPU reload on the GPU clock). Only an LRU victim beyond
    `max_warm` is fully freed (pipe=None + dropped from `_handles`).
"""
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
        self.cuda_attached = False


def test_acquire_builds_and_caches():
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    h1 = reg.acquire("wan2.1_t2v_14b")
    assert reg.acquire("wan2.1_t2v_14b") is h1  # same key → same warm handle


def test_switch_moves_prev_off_gpu_but_keeps_it_warm():
    """Within the warm budget a switch only moves GPU residency: the prior
    handle is unloaded from the GPU yet stays warm in CPU for an instant
    swap-back."""
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]), max_warm=2)
    h1 = reg.acquire("wan2.1_t2v_14b")
    h2 = reg.acquire("wan2.1_i2v_14b_480p")
    assert h1.unloaded is True               # prior moved off the GPU
    assert h2.unloaded is False
    assert reg.warm_key == "wan2.1_i2v_14b_480p"
    assert h1.pipe is not None               # still warm in CPU RAM
    assert "wan2.1_t2v_14b" in reg._handles  # still tracked


def test_swap_back_is_a_cache_hit():
    """Swapping back to a still-warm key returns the same handle (no rebuild),
    which is what keeps a T2V↔I2V swap off the GPU duration budget."""
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]), max_warm=2)
    h1 = reg.acquire("wan2.1_t2v_14b")
    reg.acquire("wan2.1_i2v_14b_480p")
    assert reg.acquire("wan2.1_t2v_14b") is h1   # same object, not re-built
    assert reg.warm_key == "wan2.1_t2v_14b"


def test_cpu_overflow_frees_lru_victim():
    """Beyond `max_warm`, the least-recently-used warm handle is fully freed
    (CPU RAM reclaimed) — never the currently GPU-attached key."""
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]), max_warm=1)
    k1, k2 = "wan2.1_t2v_14b", "wan2.1_i2v_14b_480p"
    h1 = reg.acquire(k1)
    reg.acquire(k2)
    assert h1.pipe is None          # CPU RAM reference dropped
    assert k1 not in reg._handles   # no longer tracked → eligible for GC
    assert reg.warm_key == k2


def test_overflow_never_evicts_the_gpu_resident_key():
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]), max_warm=2)
    a, b, c = "wan2.1_t2v_14b", "wan2.1_i2v_14b_480p", "wan2.1_i2v_14b_720p"
    reg.acquire(a)
    reg.acquire(b)
    reg.acquire(c)                  # warm set would be {a,b,c} > 2 → evict LRU (a)
    assert reg.warm_key == c        # current GPU resident survives
    assert c in reg._handles
    assert a not in reg._handles    # least-recently-used freed
    assert len(reg._handles) == 2


def test_acquire_unknown_raises():
    import pytest
    reg = ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key]))
    with pytest.raises(KeyError):
        reg.acquire("nope")


def test_default_max_warm_from_env(monkeypatch):
    monkeypatch.delenv("WAN_STUDIO_MAX_WARM", raising=False)
    assert ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key])).max_warm == 2
    monkeypatch.setenv("WAN_STUDIO_MAX_WARM", "3")
    assert ModelRegistry(factory=lambda key: _FakeHandle(BY_KEY[key])).max_warm == 3
