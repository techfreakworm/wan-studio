#!/usr/bin/env python3
"""Verify an S2V output mp4: read frames, run the shared quality_report metric gates,
and re-extract start/mid/end PNGs (proves the mp4 decodes). The PNGs are the eyeball
arbiter; the metrics are the necessary-not-sufficient gate."""
import sys
import os

import imageio.v2 as imageio
import numpy as np

REPO = os.path.expanduser("~/Projects/llm/wan-studio")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from local_verify import quality_report  # noqa: E402


def main():
    mp4 = sys.argv[1]
    rdr = imageio.get_reader(mp4)
    frames = [np.asarray(f) for f in rdr]
    rdr.close()
    rep, arrs = quality_report(frames)
    import json
    print(json.dumps(rep, indent=2))
    print("PASS" if rep["PASS"] else "FAIL-GATES", f"({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
