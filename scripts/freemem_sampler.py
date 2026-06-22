#!/usr/bin/env python3
"""Standalone, OUT-OF-PROCESS free-memory sampler — the ground-truth peak monitor.

WHY A SEPARATE PROCESS: an in-process Python thread sampling free memory gets
GIL-/Metal-starved during long synchronous MPS calls (a single vae.decode() can
block the interpreter for seconds). It then MISSES the transient peak and reports
an optimistic min-free ≈ end-free. A separate OS process is scheduled independently
of the worker's Metal/GIL stalls, so it actually catches the dip.

Contract:
  - Polls free memory every INTERVAL_MS using the SAME definition as
    memcheck.free_gb() (total − wired − active − compressor) so the calibration
    metric and the runtime gate metric are identical (no apples-vs-oranges).
  - Writes a JSONL trace (one sample per line) AND, on exit/SIGTERM, a final
    summary line with min_free_gb (the true peak consumption proxy).

Usage:
  python scripts/freemem_sampler.py --out /tmp/trace.jsonl [--interval-ms 150]
  # parent runs the e2e job, then sends SIGTERM; this flushes the summary.

Validation (wan-brain's check): run against the known-good 2s T2V. If the reported
min_free is ~equal to the free measured AFTER the job ends, the sampler is broken
(it slept through the peak). A correct sampler shows min_free meaningfully BELOW
end-free.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time

TOTAL_RAM_GB = 137.4  # M5 Max (128 GiB ≈ 137.4 GB) — keep in sync with memcheck.py


def free_gb() -> float:
    """TRUE available memory (GB): total − (wired + active + compressor).
    IDENTICAL definition to memcheck.free_gb() so gate and calibration agree."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        pg = 4096

        def pages(label: str) -> int:
            for line in out.splitlines():
                if label in line:
                    return int(line.split(":")[1].strip().rstrip(".")) * pg
            return 0

        unavailable = (pages("Pages wired down") + pages("Pages active")
                       + pages("Pages occupied by compressor")) / 1e9
        return max(0.0, min(TOTAL_RAM_GB - unavailable, TOTAL_RAM_GB))
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSONL trace output path")
    ap.add_argument("--interval-ms", type=int, default=150)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    state = {"min_free": 1e9, "min_t": 0.0, "n": 0, "start_free": None, "last_free": None}
    t0 = time.time()
    stop = {"flag": False}

    def handle(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    def write_summary(f):
        end_free = free_gb()
        summary = {
            "summary": True, "tag": args.tag,
            "samples": state["n"],
            "start_free_gb": round(state["start_free"], 2) if state["start_free"] is not None else None,
            "end_free_gb": round(end_free, 2),
            "min_free_gb": round(state["min_free"], 2),
            "min_at_s": round(state["min_t"], 2),
            # peak real consumption = how far free dropped below where it started
            "peak_consumed_gb": round((state["start_free"] - state["min_free"]), 2)
                if state["start_free"] is not None else None,
            "dip_below_end_gb": round((end_free - state["min_free"]), 2),
            "duration_s": round(time.time() - t0, 2),
        }
        f.write(json.dumps(summary) + "\n")
        f.flush()
        print(f"[freemem_sampler] {json.dumps(summary)}", file=sys.stderr, flush=True)

    interval = args.interval_ms / 1000.0
    with open(args.out, "w") as f:
        while not stop["flag"]:
            fg = free_gb()
            t = time.time() - t0
            if state["start_free"] is None and fg >= 0:
                state["start_free"] = fg
            if 0 <= fg < state["min_free"]:
                state["min_free"], state["min_t"] = fg, t
            state["last_free"] = fg
            state["n"] += 1
            f.write(json.dumps({"t": round(t, 3), "free_gb": round(fg, 2)}) + "\n")
            f.flush()
            time.sleep(interval)
        write_summary(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
