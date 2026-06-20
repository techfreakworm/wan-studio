#!/usr/bin/env python3
"""Local MPS end-to-end verifier for Wan Studio modes.

For a given mode key: build the handle, configure the preset, generate a short
clip on MPS, export an mp4, extract start/mid/end frames via ffmpeg, and compute
quality metrics (brightness, contrast, sharpness, inter-frame motion) so we
PROVE the output is real, moving video — not a black/static placeholder or just
a file that happens to exist.

A run only reports PASS when every gate holds:
  not_black, has_contrast, has_detail, has_motion, no_nan.

Usage:
  python scripts/local_verify.py --mode wan2.1_t2v_1.3b --preset quality --steps 8 --frames 17
  python scripts/local_verify.py --mode wan2.1_i2v_14b_480p --image in.png --preset fast
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for memcheck
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

OUT_DIR = REPO / "tests" / "outputs"

# Canonical Wan negative prompt (suppresses the "static / greyish / low-quality"
# failure modes the quality gates check for). Same string the Wan repo ships.
WAN_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

DEFAULT_PROMPT = (
    "A red panda eating bamboo on a mossy rock in a sunlit forest, "
    "cinematic, photorealistic, shallow depth of field, gentle camera push-in"
)


# ──────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────
def to_uint8_frames(frames) -> list[np.ndarray]:
    """Normalize pipeline frames (PIL list, or np array/list, float or uint8)
    into a list of HxWx3 uint8 arrays."""
    out = []
    for f in frames:
        if hasattr(f, "convert"):  # PIL.Image
            a = np.asarray(f.convert("RGB"))
        else:
            a = np.asarray(f)
            if a.dtype != np.uint8:
                scale = 255.0 if float(np.nanmax(a)) <= 1.0 + 1e-3 else 1.0
                a = np.clip(np.nan_to_num(a) * scale, 0, 255).astype(np.uint8)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.shape[-1] == 4:
                a = a[..., :3]
        out.append(a)
    return out


def _laplacian_var(gray: np.ndarray) -> float:
    from scipy.signal import convolve2d
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return float(convolve2d(gray.astype(np.float32), k, mode="valid").var())


def _saturation(a: np.ndarray) -> float:
    """Mean HSV saturation of an HxWx3 uint8 frame. Under-denoised latents decode
    to neon-saturated noise (S≈0.7-0.95); real photoreal scenes sit ~0.2-0.5."""
    af = a.astype(np.float32)
    mx = af.max(-1)
    mn = af.min(-1)
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(s.mean())


def quality_report(frames_raw) -> dict:
    arrs = to_uint8_frames(frames_raw)
    grays = [a.mean(-1) for a in arrs]
    bright = [float(g.mean()) for g in grays]
    contrast = [float(g.std()) for g in grays]
    sharp = [_laplacian_var(g) for g in grays]
    sat = [_saturation(a) for a in arrs]
    motion = [
        float(np.abs(grays[i].astype(np.float32) - grays[i - 1].astype(np.float32)).mean())
        for i in range(1, len(grays))
    ]
    rep = {
        "n_frames": len(arrs),
        "resolution": f"{arrs[0].shape[1]}x{arrs[0].shape[0]}",
        "brightness_mean": round(float(np.mean(bright)), 2),
        "brightness_range": [round(min(bright), 2), round(max(bright), 2)],
        "contrast_mean": round(float(np.mean(contrast)), 2),
        "sharpness_mean": round(float(np.mean(sharp)), 2),
        "saturation_mean": round(float(np.mean(sat)), 3),
        "motion_mean": round(float(np.mean(motion)), 3) if motion else 0.0,
        "motion_max": round(float(np.max(motion)), 3) if motion else 0.0,
    }
    checks = {
        "not_black": rep["brightness_mean"] > 5 and rep["brightness_range"][1] > 15,
        "has_contrast": rep["contrast_mean"] > 8,
        "has_detail": rep["sharpness_mean"] > 1.0,
        "has_motion": rep["motion_mean"] > 0.4,  # real video, not a frozen still
        "not_static": rep["motion_max"] < 80,    # not strobing/garbage either
        # NOTE: saturation alone does NOT separate a vivid-but-correct scene (an
        # orange red-panda on green grass legitimately hits ~0.86) from psychedelic
        # under-denoised noise — a clean vivid frame can out-saturate a broken one.
        # So this gate only catches PURE neon noise (≈0.98); coherence is judged by
        # VISUAL inspection of the start/mid/end PNGs, which is the real arbiter.
        "not_pure_noise": rep["saturation_mean"] < 0.93,
        "no_nan": all(np.isfinite(a).all() for a in arrs),
    }
    rep["checks"] = checks
    rep["PASS"] = all(checks.values())
    return rep, arrs


def ffmpeg_extract(mp4: Path, name: str, n: int) -> dict:
    """Extract start/mid/end frames as PNG via ffmpeg (proves the mp4 decodes)."""
    fdir = OUT_DIR / f"{name}_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    picks = {"start": 0, "mid": max(0, n // 2), "end": max(0, n - 1)}
    saved = {}
    for label, idx in picks.items():
        out = fdir / f"{label}.png"
        if out.exists():
            out.unlink()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
             "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", str(out)],
            check=True,
        )
        saved[label] = str(out) if out.exists() else None
    return saved


# ──────────────────────────────────────────────────────────────────────────
# Generation dispatch
# ──────────────────────────────────────────────────────────────────────────
def _inference_kwargs(pk, steps_override):
    d = {
        "num_inference_steps": int(steps_override) if steps_override else pk.num_inference_steps,
        "guidance_scale": pk.guidance_scale,
    }
    if pk.guidance_scale_2 is not None:
        d["guidance_scale_2"] = pk.guidance_scale_2
    return d


def _progress(done, total):
    print(f"    step {done}/{total}", flush=True)


def _load_video_frames(handle, path, max_area, n_max):
    """Decode an mp4 → list[PIL] resized to the pipe's VAE/patch grid."""
    from pipelines.video_io import decode_video
    frames, h, w = decode_video(path, handle.pipe, max_area)
    if n_max and len(frames) > n_max:
        frames = frames[:n_max]
    return frames, h, w


def generate_for_mode(card, handle, args, ik):
    """Call the right generate() for this mode and return a list of frames."""
    from PIL import Image
    mode = card.mode
    if mode == "t2v":
        return handle.generate(
            prompt=args.prompt, negative_prompt=args.neg,
            height=args.height, width=args.width, num_frames=args.frames,
            seed=args.seed, preset_kwargs=ik, step_callback=_progress,
        )
    if mode == "i2v":
        assert args.image, "i2v requires --image"
        img = Image.open(args.image).convert("RGB")
        max_area = args.height * args.width
        return handle.generate(
            img, args.prompt, negative_prompt=args.neg, max_area=max_area,
            num_frames=args.frames, seed=args.seed, preset_kwargs=ik,
            step_callback=_progress,
        )
    if mode == "flf2v":
        assert args.image and args.last_image, "flf2v requires --image and --last-image"
        first = Image.open(args.image).convert("RGB")
        last = Image.open(args.last_image).convert("RGB")
        max_area = args.height * args.width
        return handle.generate(
            first, last, args.prompt, negative_prompt=args.neg, max_area=max_area,
            num_frames=args.frames, seed=args.seed, preset_kwargs=ik,
        )
    if mode == "v2v":
        assert args.video, "v2v requires --video"
        handle.ensure_loaded()
        frames, h, w = _load_video_frames(handle, args.video, args.height * args.width, args.frames)
        return handle.generate(
            frames, args.prompt, negative_prompt=args.neg, strength=args.strength,
            seed=args.seed, preset_kwargs=ik,
        )
    if mode == "vace":
        # Reference sub-mode: condition on a reference image (no control video).
        # Control sub-mode: pass the source video as the control signal.
        ref = [Image.open(args.image).convert("RGB")] if args.image else None
        ctrl = None
        if args.video:
            handle.ensure_loaded()
            ctrl, h, w = _load_video_frames(handle, args.video, args.height * args.width, args.frames)
        return handle.generate(
            prompt=args.prompt, video=ctrl, reference_images=ref,
            negative_prompt=args.neg, height=args.height, width=args.width,
            num_frames=args.frames, seed=args.seed, preset_kwargs=ik,
        )
    raise NotImplementedError(f"generate dispatch for mode {mode!r} not wired yet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--preset", default="quality", choices=["fast", "quality"])
    ap.add_argument("--steps", type=int, default=0, help="override num_inference_steps (0=preset)")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--neg", default=WAN_NEG, help="negative prompt ('' to disable)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--last-image", dest="last_image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for output filename")
    ap.add_argument("--shift", type=float, default=None,
                    help="override FlowMatchEuler scheduler shift (per-mode shift tuning)")
    ap.add_argument("--force-unsafe", action="store_true",
                    help="bypass the memory preflight (DANGEROUS — can panic the OS)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── MEMORY PREFLIGHT + SERIAL GUARD (a prior run OOM-panicked the OS) ──────
    # Refuse to even import torch if the planned run would exceed the safe memory
    # budget, and hold a lockfile so two heavy jobs can never run concurrently.
    from memcheck import estimate_peak_gb, free_gb, SerialLock, HARD_CAP_GB
    _res = "720p" if (args.width >= 1100 or args.height >= 700) else "480p"
    _dtype = os.getenv("WAN_STUDIO_MPS_DTYPE", "bf16")
    _est = estimate_peak_gb(args.mode, args.frames, _res,
                            "float32" if _dtype in ("float32", "fp32") else "bf16")
    print(f"=== MEMCHECK: {args.mode} @{args.frames}f {_res} → peak ~{_est['peak_gb']}GB "
          f"(gen {_est['gen_peak_gb']} / decode {_est['decode_peak_gb']}) | cap {HARD_CAP_GB:.0f}GB "
          f"| free-now ~{free_gb():.0f}GB | {'SAFE' if _est['safe'] else 'UNSAFE'} ===", flush=True)
    if not _est["safe"] and not args.force_unsafe:
        print(f"!! REFUSED (static cap): peak ~{_est['peak_gb']}GB exceeds safe cap {HARD_CAP_GB:.0f}GB. "
              f"This config can PANIC THE OS. Lower --frames (decode is the limit: ≤9 latent ≈ ≤33f "
              f"at 480p), or implement chunked VAE decode. --force-unsafe only if you've verified headroom.",
              flush=True)
        return 4
    # Live-memory gate: even a statically-safe peak must fit the CURRENTLY reclaimable
    # memory plus a compressible margin, or the box thrashes/panics. macOS compresses
    # aggressively, so allow free + 50GB; refuses right after a reboot until RAM recovers.
    _fg = free_gb()
    if _est["peak_gb"] > _fg + 50 and not args.force_unsafe:
        print(f"!! REFUSED (live memory): peak ~{_est['peak_gb']}GB but only ~{_fg:.0f}GB reclaimable now. "
              f"Close other apps / wait for RAM to recover, then retry.", flush=True)
        return 5

    _lock = SerialLock()
    _lock.__enter__()
    try:
        return _run(args)
    finally:
        _lock.__exit__()


def _run(args) -> int:
    import torch  # noqa
    import pipelines  # populate HANDLER_REGISTRY + handle classes  # noqa
    from pipelines.registry import BY_KEY
    from pipelines.handlers import HANDLER_REGISTRY
    from utils.backend import detect

    backend = detect()
    card = BY_KEY[args.mode]
    spec = HANDLER_REGISTRY.get(card.mode)
    if spec is None:
        print(f"!! no handler registered for mode {card.mode!r} (key {args.mode}) — needs a new handler")
        return 3

    name = f"verify_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    print(f"=== {args.mode} | backend={backend.label} dtype={backend.dtype} "
          f"vae={backend.vae_dtype} | preset={args.preset} ===", flush=True)

    handle = spec.handle_cls.for_key(args.mode)

    t0 = time.time()
    pk = handle.configure_preset(args.preset)
    t_load = time.time() - t0
    print(f"  loaded+attached in {t_load:.0f}s | effective_preset={pk.effective_preset} "
          f"lora_active={pk.lora_active}", flush=True)
    if pk.fallback_message:
        print(f"  (fallback) {pk.fallback_message}", flush=True)

    # Optional per-mode shift override (re-key the scheduler AFTER configure_preset set it).
    if args.shift is not None:
        handle._configure_scheduler(shift=args.shift)
        print(f"  [shift] overrode scheduler shift → {args.shift}", flush=True)

    # TRIAGE PROBE: log the ACTUAL pipe state (not the intended preset) so a soft/broken
    # result is diagnosable immediately — real scheduler+shift, and whether the Lightning
    # LoRA is genuinely active (the enable_lora-before-set_adapters trap fails silently).
    try:
        _sched = type(handle.pipe.scheduler).__name__
        _shift = getattr(handle.pipe.scheduler.config, "shift", None)
        _act = (handle.pipe.get_active_adapters()
                if hasattr(handle.pipe, "get_active_adapters") else "n/a")
        print(f"  [probe] scheduler={_sched} shift={_shift} | active_adapters={_act}", flush=True)
    except Exception as _e:
        print(f"  [probe] state introspect failed: {_e}", flush=True)

    ik = _inference_kwargs(pk, args.steps)
    print(f"  inference kwargs: {ik} | frames={args.frames} {args.width}x{args.height}", flush=True)

    # MEASURED peak MPS memory — calibrates memcheck against the SDPA(non-materialized
    # N²)+tiled-decode REALITY (the static estimate is deliberately worst-case and
    # over-refuses; these numbers let us tighten it safely instead of guessing).
    def _mps_gb():
        try:
            return torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            return float("nan")
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    _mem_before = _mps_gb()

    t1 = time.time()
    frames = generate_for_mode(card, handle, args, ik)
    t_gen = time.time() - t1
    _mem_after = _mps_gb()  # driver high-water proxy (cache not freed) — covers gen+decode
    print(f"  generated {len(frames)} frames in {t_gen:.0f}s "
          f"({t_gen / max(1, ik['num_inference_steps']):.1f}s/step)", flush=True)
    print(f"  [MEM] MPS driver-allocated: before {_mem_before:.1f}GB → after {_mem_after:.1f}GB "
          f"(peak proxy for gen+decode; static memcheck estimated higher)", flush=True)

    # Quality metrics on the raw frames (most reliable signal).
    rep, arrs = quality_report(frames)

    # DIAGNOSTIC: dump RAW frames directly as PNG (bypass mp4/ffmpeg) to isolate
    # generation-vs-export corruption.
    from PIL import Image as _Img
    _rawdir = OUT_DIR / f"{name}_RAW"; _rawdir.mkdir(parents=True, exist_ok=True)
    for _l, _i in (("start", 0), ("mid", len(arrs) // 2), ("end", len(arrs) - 1)):
        _Img.fromarray(arrs[_i]).save(_rawdir / f"{_l}.png")
    print(f"  [diag] raw frames -> {_rawdir}", flush=True)

    # Export mp4 (imageio; diffusers export_to_video corrupts frames on MPS) +
    # ffmpeg-extract start/mid/end frames.
    from pipelines.video_io import save_video
    out_mp4 = OUT_DIR / f"{name}.mp4"
    if out_mp4.exists():
        out_mp4.unlink()
    save_video([a for a in arrs], str(out_mp4), fps=args.fps)
    rep["mp4"] = str(out_mp4)
    rep["mp4_bytes"] = out_mp4.stat().st_size if out_mp4.exists() else 0
    rep["frames_png"] = ffmpeg_extract(out_mp4, name, len(arrs))
    rep["load_s"] = round(t_load, 1)
    rep["gen_s"] = round(t_gen, 1)
    rep["mps_peak_gb"] = round(_mem_after, 1)  # measured; for memcheck calibration

    print("\n=== QUALITY REPORT ===")
    print(json.dumps(rep, indent=2))
    verdict = "PASS ✅" if rep["PASS"] else "FAIL ❌"
    print(f"\n{verdict}  {args.mode}  →  {out_mp4}")

    # Persist the report alongside the mp4.
    (OUT_DIR / f"{name}.json").write_text(json.dumps(rep, indent=2))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
