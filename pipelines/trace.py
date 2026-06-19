"""Worker-side phase tracing that survives a ZeroGPU SIGKILL.

The @spaces.GPU function runs in a forked worker whose stdout is NOT captured
in the Space run logs, and ZeroGPU SIGKILLs it when it overruns its duration —
so any buffered stdout is lost. We instead append timestamped phase markers to
a file in /tmp (shared with the main process), then read them back through a
non-GPU `/_debug_trace` Gradio endpoint. This is how we locate which phase of a
generation (disk→CPU load / host→GPU transfer / inference) eats the budget.

Best-effort and dependency-free: a failed write must never break a generation.
"""
from __future__ import annotations

import os
import time

TRACE_PATH = os.getenv("WAN_STUDIO_TRACE_PATH", "/tmp/wan_worker_trace.log")

_t0 = time.time()


def trace(phase: str) -> None:
    """Append a timestamped phase marker. Never raises."""
    try:
        now = time.time()
        line = f"{time.strftime('%H:%M:%S', time.localtime(now))} +{now - _t0:7.2f}s  pid={os.getpid()}  {phase}\n"
        with open(TRACE_PATH, "a") as fh:
            fh.write(line)
            fh.flush()
    except Exception:
        pass


def mark(phase: str) -> float:
    """Like trace() but also returns a monotonic stamp for delta math."""
    trace(phase)
    return time.time()


def read_trace(tail: int = 200) -> str:
    """Return the last `tail` lines of the trace file (for the debug endpoint)."""
    try:
        with open(TRACE_PATH) as fh:
            lines = fh.readlines()
        return "".join(lines[-tail:]) or "(trace empty)"
    except FileNotFoundError:
        return "(no trace file yet)"
    except Exception as e:  # pragma: no cover
        return f"(trace read error: {e})"


def reset_trace() -> None:
    try:
        if os.path.exists(TRACE_PATH):
            os.remove(TRACE_PATH)
    except Exception:
        pass
