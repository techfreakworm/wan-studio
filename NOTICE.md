# NOTICE

Wan Studio incorporates and redistributes the following third-party assets, all
licensed under Apache License 2.0 (the same license as this project's source
code). Attribution is required by Apache 2.0 §4.

## Wan-AI (Alibaba) — Wan 2.1 + Wan 2.2 model weights

Wan Studio mirrors the following Wan-AI repositories into the user's HF account
(`techfreakworm/wan2.*-*`) and mounts them read-only into the Space at runtime
via the `huggingface_hub` Volume API:

- `Wan-AI/Wan2.1-T2V-14B-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` — Apache 2.0
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers` — Apache 2.0

Upstream org: https://huggingface.co/Wan-AI · Paper: https://arxiv.org/abs/2503.20314

## lightx2v / ModelTC — Lightning step-distillation LoRAs

Lightning LoRA weights (HIGH+LOW pairs for Wan 2.2 MoE; single LoRAs for Wan 2.1)
are mirrored into `techfreakworm/wan-lightning-loras` and mounted read-only.
Upstream sources are Apache 2.0:

- `lightx2v/Wan2.2-Lightning` (Wan 2.2 T2V-A14B + I2V-A14B paired LoRAs)
- `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`
- `lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v`
- Kijai/WanVideo_comfy (community mirror of T2V-14B and Wan 2.2 LoRA files)

Upstream org: https://huggingface.co/lightx2v

## diffusers (Hugging Face)

Wan pipeline classes (`WanPipeline`, `WanImageToVideoPipeline`, `WanVACEPipeline`,
`WanAnimatePipeline`, `WanVideoToVideoPipeline`) — Apache 2.0
Upstream: https://github.com/huggingface/diffusers

## Gradio (Hugging Face)

Gradio framework — Apache 2.0
Upstream: https://github.com/gradio-app/gradio
