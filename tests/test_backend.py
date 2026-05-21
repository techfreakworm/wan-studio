"""Tests for utils.backend — device detection + dtype selection."""
import torch
from utils.backend import detect, Backend, spaces_gpu_or_noop


def test_detect_returns_backend_instance():
    backend = detect()
    assert isinstance(backend, Backend)


def test_detect_device_is_one_of_known():
    backend = detect()
    assert backend.device in ("cuda", "mps", "cpu")


def test_detect_vae_dtype_is_always_float32():
    """VAE must stay fp32 on every backend per RESEARCH §7.2."""
    backend = detect()
    assert backend.vae_dtype == torch.float32


def test_detect_mps_uses_float16_transformer():
    backend = detect()
    if backend.device == "mps":
        assert backend.dtype == torch.float16, (
            "MPS bf16 is patchy as of mid-2026; transformer must be fp16"
        )


def test_detect_cuda_uses_bfloat16():
    backend = detect()
    if backend.device == "cuda":
        assert backend.dtype == torch.bfloat16


def test_zerogpu_flag_false_outside_space(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    backend = detect()
    assert backend.is_zerogpu is False


def test_spaces_gpu_decorator_is_noop_offline():
    """Outside ZeroGPU, the decorator must not modify the function."""
    deco = spaces_gpu_or_noop()

    @deco(duration=60)
    def my_fn(x):
        return x * 2

    assert my_fn(21) == 42


def test_backend_label_is_human_readable():
    backend = detect()
    assert isinstance(backend.label, str)
    assert len(backend.label) > 0
