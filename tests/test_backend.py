"""Tests for utils.backend — device detection + dtype selection."""
import torch
from utils.backend import detect, Backend, spaces_gpu_or_noop


def test_detect_returns_backend_instance():
    backend = detect()
    assert isinstance(backend, Backend)


def test_detect_device_is_one_of_known():
    backend = detect()
    assert backend.device in ("cuda", "mps", "cpu")


def test_detect_mps_vae_defaults_to_bfloat16(monkeypatch):
    """MPS VAE defaults to bf16: measured to cut the 14B @17f peak 131.9→92.4GB
    (off the 137GB ceiling) with pixel-identical output (no NaN/banding). bf16
    keeps fp32's exponent range so no overflow. Verified empirically 2026-06."""
    monkeypatch.delenv("WAN_STUDIO_VAE_DTYPE", raising=False)
    backend = detect()
    if backend.device == "mps":
        assert backend.vae_dtype == torch.bfloat16


def test_detect_mps_vae_dtype_env_override(monkeypatch):
    """WAN_STUDIO_VAE_DTYPE escape hatch back to fp32 if any content bands."""
    monkeypatch.setenv("WAN_STUDIO_VAE_DTYPE", "float32")
    backend = detect()
    if backend.device == "mps":
        assert backend.vae_dtype == torch.float32


def test_detect_cuda_vae_stays_float32():
    """CUDA/ZeroGPU keeps fp32 VAE (ample VRAM; numerical safety)."""
    backend = detect()
    if backend.device == "cuda":
        assert backend.vae_dtype == torch.float32


def test_detect_mps_defaults_to_bfloat16(monkeypatch):
    """MPS defaults to bf16 — Wan transformers + lightx2v LoRAs are bf16-native
    and torch 2.11 MPS bf16 is validated clean (no NaN/black frames). fp16's
    narrow range risks overflow on these models. Verified empirically 2026-06."""
    monkeypatch.delenv("WAN_STUDIO_MPS_DTYPE", raising=False)
    backend = detect()
    if backend.device == "mps":
        assert backend.dtype == torch.bfloat16


def test_detect_mps_dtype_env_override(monkeypatch):
    """The WAN_STUDIO_MPS_DTYPE escape hatch lets us A/B fp16 per model."""
    monkeypatch.setenv("WAN_STUDIO_MPS_DTYPE", "float16")
    backend = detect()
    if backend.device == "mps":
        assert backend.dtype == torch.float16


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
