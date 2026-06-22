#!/usr/bin/env python3
"""Self-test for the local_verify quality gate.

Before we trust any PASS, prove the gate FAILS on known-bad video — otherwise a
harness that rubber-stamps black frames turns every later "pass" into a false
pass. Synthesizes four pathological clips and asserts each is rejected, plus one
synthetic-good clip that must pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from scripts.local_verify import quality_report  # noqa: E402

H, W, N = 96, 128, 16
rng = np.random.default_rng(0)


def all_black():
    return [np.zeros((H, W, 3), np.uint8) for _ in range(N)]


def frozen_duplicate():
    # One textured frame repeated → no motion.
    base = (rng.integers(0, 256, (H, W, 3))).astype(np.uint8)
    return [base.copy() for _ in range(N)]


def nan_render():
    frames = [np.full((H, W, 3), np.nan, np.float32) for _ in range(N)]
    return frames


def neon_noise():
    # Under-denoised look: extreme per-pixel saturation, high-frequency, moving.
    out = []
    for _ in range(N):
        hue = rng.integers(0, 256, (H, W, 1)).astype(np.uint8)
        # Force near-max saturation: one channel near 0, one near 255.
        f = np.concatenate([hue, np.full((H, W, 1), 5, np.uint8),
                            np.full((H, W, 1), 250, np.uint8)], axis=-1)
        out.append(f)
    return out


def good_clip():
    # Moderate saturation, real detail, gentle motion (panning gradient + texture).
    out = []
    base = np.zeros((H, W, 3), np.uint8)
    for x in range(W):
        base[:, x, :] = int(40 + 120 * x / W)
    tex = rng.integers(-20, 20, (H, W, 3))
    for i in range(N):
        shifted = np.roll(base.astype(np.int32) + tex, i * 2, axis=1)
        out.append(np.clip(shifted, 0, 255).astype(np.uint8))
    return out


def vivid_correct():
    # The exact failure mode that fooled the old metric: HIGH saturation but a
    # COHERENT structured subject (orange disc on green) with gentle motion. Must
    # PASS — this proves the gate won't reject a vivid-but-correct frame like the
    # real red panda (sat ~0.86). Structured (smooth shape, not noise) so it isn't
    # pure-noise, and saturated enough to land in the panda's range.
    out = []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(N):
        cx = W * 0.4 + i * 1.5
        cy = H * 0.5
        d = np.sqrt(((xx - cx) / (W * 0.22)) ** 2 + ((yy - cy) / (H * 0.3)) ** 2)
        # Ambient light floor (real scenes are never pure-channel) + slight texture
        # so saturation lands in the real-panda range (~0.83), not pure-color 0.98.
        amb = rng.integers(0, 18, (H, W, 3)).astype(np.float32)
        f = np.zeros((H, W, 3), np.float32)
        f[..., 0] = 55; f[..., 1] = 150; f[..., 2] = 60   # green background w/ floor
        disc = d < 1.0
        f[..., 0][disc] = 235; f[..., 1][disc] = 120; f[..., 2][disc] = 70  # orange w/ floor
        f = f + amb
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return out


CASES = [
    ("all_black", all_black, False),
    ("frozen_duplicate", frozen_duplicate, False),
    ("nan_render", nan_render, False),
    ("neon_noise", neon_noise, False),
    ("good_clip", good_clip, True),
    ("vivid_correct", vivid_correct, True),  # high-sat but coherent → MUST pass
]


def main() -> int:
    ok = True
    for name, fn, want_pass in CASES:
        rep, _ = quality_report(fn())
        got = rep["PASS"]
        status = "OK" if got == want_pass else "WRONG"
        if got != want_pass:
            ok = False
        failed = [k for k, v in rep["checks"].items() if not v]
        print(f"[{status:5s}] {name:18s} PASS={got!s:5s} (want {want_pass!s:5s}) "
              f"sat={rep['saturation_mean']} motion={rep['motion_mean']} "
              f"bright={rep['brightness_mean']} failed_gates={failed}")
    print("\nSELFTEST", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
