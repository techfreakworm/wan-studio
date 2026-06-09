"""Tests for pipelines.video_io helpers."""
from PIL import Image

from pipelines.video_io import center_crop_resize


def test_center_crop_resize_exact_dims():
    img = Image.new("RGB", (200, 100), "red")     # 2:1
    out = center_crop_resize(img, 64, 128)          # target (h=64, w=128) = 2:1
    assert out.size == (128, 64)                    # PIL .size is (w, h)


def test_center_crop_resize_taller_source():
    img = Image.new("RGB", (100, 400), "blue")    # tall
    out = center_crop_resize(img, 100, 100)
    assert out.size == (100, 100)


def test_center_crop_resize_returns_rgb():
    img = Image.new("L", (50, 50), 128)            # grayscale
    out = center_crop_resize(img, 32, 48)
    assert out.mode == "RGB"
    assert out.size == (48, 32)
