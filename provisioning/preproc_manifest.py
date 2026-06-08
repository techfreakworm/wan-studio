"""Contents of the wan-shared-encoders and wan-preproc mirrors.

These are DATA manifests consumed by duplicate_upstream.py. Source paths are
the upstream repos; dest paths are inside the maintainer's mirror repos.
Amendment 1: Animate's ViTPose/YOLO/SAM2 live ONLY in the non-Diffusers
Wan-AI/Wan2.2-Animate-14B repo (the Diffusers mirror has no process_checkpoint/),
so they are provisioned here, not assumed bundled.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedComponent:
    source_repo: str
    source_subfolder: str
    dest_subfolder: str
    dtype: str  # "bfloat16" | "float32"


@dataclass(frozen=True)
class PreprocAsset:
    name: str
    source_repo: str
    source_path: str   # file or glob in the source repo
    dest_path: str     # path inside techfreakworm/wan-preproc


SHARED_ENCODERS = [
    SharedComponent("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "text_encoder", "text_encoder", "bfloat16"),
    SharedComponent("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "vae", "vae", "float32"),
    SharedComponent("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "image_encoder", "image_encoder", "float32"),
    SharedComponent("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "image_processor", "image_processor", "float32"),
]

PREPROC_ASSETS = [
    # VACE lightweight subset (~1 GB)
    PreprocAsset("dwpose", "ali-vilab/VACE", "models/dwpose/*", "vace/dwpose/"),
    PreprocAsset("midas_dpt_hybrid", "Intel/dpt-hybrid-midas", "*", "vace/midas/"),
    PreprocAsset("raft", "ali-vilab/VACE", "models/raft/*", "vace/raft/"),
    # Animate (~2 GB) — amendment 1
    PreprocAsset("vitpose_h_wholebody", "Wan-AI/Wan2.2-Animate-14B",
                 "process_checkpoint/pose2d/vitpose_h_wholebody.onnx", "animate/pose2d/vitpose_h_wholebody.onnx"),
    PreprocAsset("yolov10m", "Wan-AI/Wan2.2-Animate-14B",
                 "process_checkpoint/det/yolov10m.onnx", "animate/det/yolov10m.onnx"),
    PreprocAsset("sam2_hiera_large", "Wan-AI/Wan2.2-Animate-14B",
                 "process_checkpoint/sam2/sam2_hiera_large.pt", "animate/sam2/sam2_hiera_large.pt"),
]
