#!/usr/bin/env python3
"""Minimal Wan-Animate preprocess (animation mode): driving video → pose_video + face_video.

Reuses the vendored ViTPose+YOLO ONNX pose2d + skeleton-draw + face-crop, on CPU
(onnxruntime CPUExecutionProvider), WITHOUT the heavy ProcessPipeline deps
(sam2/flux/decord/moviepy). Produces src_pose.mp4 (DWPose skeleton frames) +
src_face.mp4 (512×512 face crops) — the pose_video/face_video the diffusers
WanAnimatePipeline needs. The reference CHARACTER image is animated separately.

Usage:
  python scripts/animate_preprocess.py --video <driving.mp4> --out tests/inputs/animate --frames 13
"""
import argparse
import os
import sys

import cv2
import numpy as np

REPO = os.path.expanduser("~/Projects/llm/wan-studio")
PRE = os.path.join(REPO, "vendor/Wan2.2/wan/modules/animate/preprocess")
# PRE FIRST (and ONLY) on the path: the vendored preprocess has a `utils.py` that
# collides with wan-studio's `utils` package — give PRE priority and avoid importing
# any wan-studio module here (save via imageio directly).
sys.path.insert(0, PRE)


def read_frames(path, n):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="driving video (provides pose+face motion)")
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/wan-bf16/animate-preproc-ckpt/process_checkpoint"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--area", type=int, default=1280 * 720)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from pose2d import Pose2d
    from pose2d_utils import AAPoseMeta
    from human_visualization import draw_aapose_by_meta_new
    from utils import get_face_bboxes, resize_by_area

    print("[preproc] loading ViTPose+YOLO (CPU onnxruntime)…", flush=True)
    pose2d = Pose2d(
        checkpoint=os.path.join(a.ckpt, "pose2d/vitpose_h_wholebody.onnx"),
        detector_checkpoint=os.path.join(a.ckpt, "det/yolov10m.onnx"),
        device="cpu",
    )
    frames = read_frames(a.video, a.frames)
    frames = [resize_by_area(f, a.area, divisor=16) for f in frames]
    H, W = frames[0].shape[:2]
    print(f"[preproc] {len(frames)} frames @ {W}x{H}; running pose2d…", flush=True)
    metas = pose2d(frames)

    pose_frames, face_frames = [], []
    for idx, meta in enumerate(metas):
        fb = get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(H, W))
        x1, x2, y1, y2 = fb
        face = cv2.resize(frames[idx][y1:y2, x1:x2], (512, 512))
        face_frames.append(face.astype(np.uint8))
        aameta = AAPoseMeta.from_humanapi_meta(meta)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        pose_frames.append(draw_aapose_by_meta_new(canvas, aameta).astype(np.uint8))

    import imageio
    pp = os.path.join(a.out, "src_pose.mp4")
    fp = os.path.join(a.out, "src_face.mp4")
    for path, seq in ((pp, pose_frames), (fp, face_frames)):
        imageio.mimsave(path, seq, fps=16, codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"], macro_block_size=8)
    print(f"PREPROC_DONE pose={pp} face={fp} ({len(pose_frames)} frames @ {W}x{H})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
