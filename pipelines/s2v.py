"""S2V handler — Wan 2.2 Speech-to-Video.

S2V is NOT in diffusers. It runs the upstream `wan.WanS2V` port (scripts/s2v_smoke.py)
as a SCOPED SUBPROCESS so that its invasive, process-global MPS shims — torch.device
cuda→mps replacement, float64→fp32 downcast, autocast remapping, flash_attention
replacement — stay isolated from the 12 diffusers modes living in the app process.

The handle is a registration stub (so the mode is detected as WIRED); the app never
.acquire()s it. `_run_s2v` calls `run_s2v_subprocess()` directly.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

from pipelines.handle import WanModelHandle
from pipelines.handlers import HandlerSpec, register

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SMOKE = os.path.join(REPO, "scripts", "s2v_smoke.py")
_CKPT_HINT = os.path.join(REPO, "tests", "outputs", "s2v_ckpt_path.txt")


def _resolve_ckpt() -> str:
    """Prefer the recorded snapshot path; else glob the HF cache for the S2V weights."""
    if os.path.exists(_CKPT_HINT):
        p = open(_CKPT_HINT).read().strip()
        if p and os.path.isdir(p):
            return p
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-S2V-14B/snapshots/*/")))
    if not hits:
        raise FileNotFoundError(
            "Wan2.2-S2V-14B snapshot not found. Download with: "
            "hf download Wan-AI/Wan2.2-S2V-14B")
    return hits[-1]


def run_s2v_subprocess(image_path: str, audio_path: str, prompt: str, out_path: str,
                       frames: int = 17, steps: int = 16, size: str = "832*480",
                       timeout: int = 2400, progress_cb=None) -> str:
    """Spawn the scoped S2V subprocess (scripts/s2v_smoke.py). Returns the output mp4
    path on success; raises RuntimeError with the captured tail on failure.

    The subprocess applies PYTORCH_ENABLE_MPS_FALLBACK=1 (RoPE complex op-gap) and runs
    SerialLock-guarded inside the smoke; the caller should free the app's warm models
    first (REGISTRY.free_all) since the subprocess needs ~110GB of unified memory."""
    ckpt = _resolve_ckpt()
    env = dict(os.environ)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["WAN_STUDIO_MAX_WARM"] = "1"
    cmd = [
        sys.executable, _SMOKE,
        "--ckpt", ckpt,
        "--image", image_path,
        "--audio", audio_path,
        "--prompt", prompt or "a person talking, natural motion, cinematic",
        "--frames", str(frames),
        "--steps", str(steps),
        "--size", size,
        "--out", out_path,
    ]
    proc = subprocess.Popen(
        cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            if progress_cb and "it/s" in line:
                progress_cb(line.strip().split("\r")[-1])
        proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"S2V subprocess failed (rc={proc.returncode}):\n{''.join(tail[-15:])}")
    return out_path


class S2VHandle(WanModelHandle):
    """Registration stub — S2V runs as a scoped subprocess; never built in-process."""

    def _build_pipeline(self):
        raise NotImplementedError(
            "S2V runs as a scoped subprocess (scripts/s2v_smoke.py via run_s2v_subprocess).")


def _s2v_key_for(generation: str, **_kw) -> str:
    return "wan2.2_s2v_14b"


register(HandlerSpec(mode="s2v", handle_cls=S2VHandle, key_for=_s2v_key_for, tier="xlarge"))
