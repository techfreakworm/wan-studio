#!/usr/bin/env python3
"""Memory preflight + serial guard for wan-studio local MPS tests.

WHY: a prior session ran THREE heavy jobs in parallel (CPU-gen + 14B-Lightning +
a 30-block activation probe) → exhausted the 128 GB unified pool → wired-memory
starvation → KERNEL PANIC / OS restart. Two defenses, both enforced here:

  1. SERIAL GUARD — a lockfile (/tmp/wan_mem.lock). Only one heavy job at a time.
     Refuses to start if another guarded job holds the lock and is alive.
  2. PEAK-MEMORY PREFLIGHT — estimate the peak resident bytes for a planned
     (model, frames, resolution, dtype, decode-plan) and refuse if it would leave
     the OS less than SAFETY_HEADROOM_GB free.

The model is deliberately CONSERVATIVE (over-estimates) — a false "unsafe" wastes
time; a false "safe" crashes the machine.

CLI:
  python scripts/memcheck.py --model wan2.1_t2v_14b --frames 33 --res 480p
    → prints an estimate + SAFE/UNSAFE and exits 0 (safe) / 1 (unsafe).

Library:
  from memcheck import estimate_peak_gb, free_gb, assert_safe, SerialLock
  with SerialLock():            # raises if another job holds it
      assert_safe(model, frames, res)   # raises RuntimeError if unsafe
      ... run the generation ...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TOTAL_RAM_GB = 137.4          # M5 Max (128 GiB ≈ 137.4 GB)
SAFETY_HEADROOM_GB = 45.0     # leave this much for OS+apps; below it = panic risk
HARD_CAP_GB = TOTAL_RAM_GB - SAFETY_HEADROOM_GB   # ≈ 92 GB usable for a job
LOCK = Path("/tmp/wan_mem.lock")

# ── CALIBRATION INVARIANTS (when tightening the static model toward measured reality) ──
# The static estimate above is deliberately worst-case (materialized-N² attention +
# non-tiled decode) and over-refuses; real runs use SDPA + spatial-tiled decode and peak
# far lower. We may tighten the MODEL using measured peaks — but NEVER at the cost of
# safety. Measured peak from sampling is a LOWER BOUND (a transient spike can hide between
# samples), so:
#   1. HARD CEILING is inviolable. Calibration tightens the model; it never raises
#      HARD_CAP_GB. Authorize a config only if measured_peak × ~1.3 margin < HARD_CAP_GB.
#   2. NO EXTRAPOLATION. Authorize at most ONE length step beyond the last MEASURED point
#      (17→25→33→49→65→81), each measured before the next. Never curve-fit and jump.
#   3. PER (mode × model × preset), not per length. i2v/vace/flf2v add conditioning
#      memory; Lightning-4-step and quality-50-step have different activation peaks; a t2v
#      calibration must NOT authorize another mode/preset at the same length. Re-measure each.
#   4. Use driver_allocated_memory (sticky high-water; MPS reserves & doesn't release),
#      not current_allocated. Treat it as estimate+margin. SerialLock stays absolute.

# Per-model weight footprint at bf16 (transformer) + shared encoders + vae.
# UMT5-XXL text encoder ≈ 11 GB (bf16), VAE fp32 ≈ 0.5 GB, CLIP (image modes) ≈ 2.5 GB.
_TX_BF16_GB = {
    "wan2.1_t2v_1.3b": 2.6, "wan2.1_t2v_14b": 28.0, "wan2.1_v2v_14b": 28.0,
    "wan2.1_i2v_14b_480p": 28.0, "wan2.1_i2v_14b_720p": 28.0,
    "wan2.1_flf2v_14b_720p": 28.0, "wan2.1_vace_1.3b": 2.6, "wan2.1_vace_14b": 28.0,
    "wan2.2_t2v_a14b": 56.0, "wan2.2_i2v_a14b": 56.0,   # MoE: BOTH experts resident
    "wan2.2_ti2v_5b": 10.0, "wan2.2_s2v_14b": 28.0, "wan2.2_animate_14b": 28.0,
}
_HEADS = {"1.3b": 12, "14b": 40, "5b": 24, "a14b": 40}
_HDIM = 128
_TEXT_ENCODER_GB = 11.0
_VAE_GB = 0.5
_CLIP_GB = 2.5

# fp32 multiplies transformer + activations by 2 (vs bf16). Set via dtype arg.
def _latent_frames(px_frames: int) -> int:
    return (px_frames - 1) // 4 + 1

def _tokens(px_frames: int, res: str) -> int:
    # 480p = 832x480 → patch2 → 52 x 30 ; 720p = 1280x720 → 80 x 45
    pph, ppw = (45, 80) if "720" in res else (30, 52)
    return _latent_frames(px_frames) * pph * ppw

def _heads_for(model: str) -> int:
    for k, v in _HEADS.items():
        if k in model:
            return v
    return 40

def estimate_peak_gb(model: str, frames: int, res: str = "480p",
                     dtype: str = "bf16", decode_chunk_latent: int | None = None) -> dict:
    """Conservative peak-resident estimate (GB), broken down."""
    tx = _TX_BF16_GB.get(model, 28.0)
    dt_mul = 2.0 if dtype == "float32" else 1.0
    tx_resident = tx * dt_mul
    weights = tx_resident + _TEXT_ENCODER_GB + _VAE_GB
    if any(s in model for s in ("i2v", "flf2v", "animate", "s2v")):
        weights += _CLIP_GB

    N = _tokens(frames, res)
    heads = _heads_for(model)
    abytes = 4.0  # MPS attention often computes scores in fp32 → 4 bytes (conservative)
    # Peak transformer activation ≈ ONE layer's full N×N attention scores (transient,
    # but the allocator may hold it alongside Q/K/V + hidden). Conservative: materialized.
    attn_scores_gb = heads * (N ** 2) * abytes / 1e9
    qkv_hidden_gb = N * heads * _HDIM * 2 * dt_mul * 3 / 1e9  # q,k,v
    tx_activation = attn_scores_gb + qkv_hidden_gb

    # VAE decode peak — the historical OS-crasher AND process-OOM driver. 3D convs
    # over the FULL temporal extent + the torch.cat-accumulated output grow ~quadratically
    # in latent frames. CALIBRATED to REAL data: 6 latent (T=21) decoded FINE; 9 latent
    # (T=33) at 14B OOM-killed the process (weights 40GB + decode > 94GB > 134GB avail);
    # 13 latent (T=49) crashed the OS. Coefficient 1.2 (was 0.5 — under-estimated; the
    # 14B@33 OOM proved decode(9 latent)≈95GB, not 40GB). Fit: 1.2 * lf^2 * (spatial/1560).
    # → correctly refuses 14B@33 now (40+97=137>92 cap). If decode_chunk_latent is set
    # (temporal chunking / free-transformer-first), only that many decode at once.
    lf = decode_chunk_latent if decode_chunk_latent else _latent_frames(frames)
    spatial = (45 * 80) if "720" in res else (30 * 52)
    vae_decode_gb = 1.2 * (lf ** 2) * (spatial / 1560.0)

    gen_peak = weights + tx_activation
    decode_peak = weights + vae_decode_gb   # weights still resident during decode
    peak = max(gen_peak, decode_peak)
    return {
        "model": model, "frames": frames, "latent_frames": _latent_frames(frames),
        "res": res, "dtype": dtype, "tokens": N,
        "weights_gb": round(weights, 1),
        "tx_activation_gb": round(tx_activation, 1),
        "vae_decode_gb": round(vae_decode_gb, 1),
        "gen_peak_gb": round(gen_peak, 1),
        "decode_peak_gb": round(decode_peak, 1),
        "peak_gb": round(peak, 1),
        "hard_cap_gb": round(HARD_CAP_GB, 1),
        "safe": peak <= HARD_CAP_GB,
    }

def free_gb() -> float:
    """TRUE available memory (GB): total physical − genuinely-unavailable
    (wired + active + compressed). free/inactive/speculative/purgeable are all
    reclaimable. NOTE: summing only free+inactive under-reports massively on a
    fresh boot (untouched pages aren't categorized), which wrongly refuses safe
    jobs — so we subtract the unavailable from total instead."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        pg = 4096
        def pages(label):
            for line in out.splitlines():
                if label in line:
                    return int(line.split(":")[1].strip().rstrip(".")) * pg
            return 0
        unavailable = (pages("Pages wired down") + pages("Pages active")
                       + pages("Pages occupied by compressor")) / 1e9
        avail = TOTAL_RAM_GB - unavailable
        # Clamp to a sane range; never report more than physical.
        return max(0.0, min(avail, TOTAL_RAM_GB))
    except Exception:
        return 0.0

def assert_safe(model: str, frames: int, res: str = "480p", dtype: str = "bf16",
                decode_chunk_latent: int | None = None) -> dict:
    est = estimate_peak_gb(model, frames, res, dtype, decode_chunk_latent)
    if not est["safe"]:
        raise RuntimeError(
            f"MEMCHECK REFUSED: {model} @{frames}f {res} {dtype} → peak ~{est['peak_gb']}GB "
            f"> cap {est['hard_cap_gb']}GB (gen {est['gen_peak_gb']} / decode {est['decode_peak_gb']}). "
            f"Reduce frames, or chunk VAE decode (decode_chunk_latent<=6), or run on CPU.")
    fg = free_gb()
    if est["peak_gb"] > fg + 30:  # peak must fit reclaimable free + some compressible slack
        raise RuntimeError(
            f"MEMCHECK REFUSED: peak ~{est['peak_gb']}GB but only ~{fg:.0f}GB reclaimable free now. "
            f"Close other jobs first.")
    return est

class SerialLock:
    """Refuse to run if another guarded heavy job is alive. Prevents the parallel
    stacking that caused the OS panic."""
    def __enter__(self):
        if LOCK.exists():
            try:
                pid = int(LOCK.read_text().split()[0])
                os.kill(pid, 0)  # alive?
                raise RuntimeError(
                    f"SERIAL GUARD: another heavy job (pid {pid}) holds {LOCK}. "
                    f"Wait for it or kill it — NEVER run heavy MPS jobs in parallel (OS-panic risk).")
            except (ProcessLookupError, ValueError):
                pass  # stale lock
        LOCK.write_text(f"{os.getpid()} {int(time.time())}\n")
        return self
    def __exit__(self, *a):
        try:
            if LOCK.exists() and int(LOCK.read_text().split()[0]) == os.getpid():
                LOCK.unlink()
        except Exception:
            pass

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--res", default="480p")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--decode-chunk-latent", type=int, default=None)
    args = ap.parse_args()
    est = estimate_peak_gb(args.model, args.frames, args.res, args.dtype, args.decode_chunk_latent)
    import json
    print(json.dumps(est, indent=2))
    print(f"free-now ~{free_gb():.0f}GB | {'SAFE ✅' if est['safe'] else 'UNSAFE ❌'}")
    return 0 if est["safe"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
