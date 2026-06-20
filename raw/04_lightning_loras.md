# Lightning / Accelerated LoRA Strategy for Wan Studio

Research date: **2026-05-21**. Scope: Lightning + distillation LoRAs only — model inventory, MoE dual-LoRA pattern, diffusers integration, and the "Fast"/"Quality" preset architecture. Other agents own base-model inventory, modes, ZeroGPU, UX.

---

## TL;DR

| Family | Wan generation covered | T2V | I2V | VACE | Animate | TI2V-5B | Steps | CFG-distilled |
|---|---|:-:|:-:|:-:|:-:|:-:|---|:-:|
| **lightx2v Wan2.1 StepDistill-CfgDistill (LoRA)** | 2.1 (14B only) | yes (rank4-256) | yes (480P/720P, same LoRA) | no | n/a | n/a | 4 | yes |
| **lightx2v Wan2.1 CausVid (LoRA)** | 2.1 (T2V-14B) | yes | no | no | n/a | n/a | 4-8 | partial |
| **lightx2v Wan2.2-Lightning (MoE LoRA pair)** | 2.2 (T2V-A14B, I2V-A14B) | yes | yes | no | no | no | 4 | yes |
| **lightx2v Wan2.2-Distill-Loras (MoE LoRA pair)** | 2.2 (I2V-A14B only) | no | yes (4-step, rank 64) | no | no | no | 4 | yes |
| **lightx2v Wan2.2-Distill-Models (full distilled MoE)** | 2.2 (T2V-A14B, I2V-A14B) | full ckpt | full ckpt | no | no | no | 4 | yes |
| **FastWan (Hao AI Lab, full ckpt)** | 2.1 (T2V-1.3B), 2.2 (TI2V-5B) | yes | n/a | n/a | n/a | yes | 3 | yes (DMD) |
| **Self-Forcing (Wan 2.1 T2V-1.3B, full ckpt)** | 2.1 (T2V-1.3B) | yes | no | no | n/a | n/a | autoregressive | yes |
| **AccVideo** | 2.1 (T2V-14B) | yes | no | no | n/a | n/a | ~5 | partial |

**Note on naming:** the community uses "Lightning" and "lightx2v" almost interchangeably. Both refer to ModelTC's family of DMD/Self-Forcing distillation LoRAs hosted at the `lightx2v` HF org. "Wan2.2-Lightning" is the brand for the Wan 2.2 paired-LoRA release.

**Wan 2.5 / 2.6:** no open weights as of May 2026. Wan 2.5-Preview is API-only via Alibaba Cloud (shipped September 2025). No accelerated LoRAs exist because the base model is closed. Studio should hard-disable the "Fast" preset for any future 2.5/2.6 modes until weights are public. Sources: [Spheron deploy guide](https://www.spheron.network/blog/deploy-wan-2-5-gpu-cloud/).

---

## 1. Lightning LoRA family inventory

### 1.1 lightx2v Wan2.1 StepDistill-CfgDistill (the canonical "Wan 2.1 Lightning")

- **HF org**: <https://huggingface.co/lightx2v>
- **Full distilled checkpoints**: `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`, `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`, `lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v`. The 720P variant is *delta-applied* from the 480P distill — the LoRA file is identical between 480P and 720P. ([lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v))
- **LoRA file** (T2V): `loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors` (and rank4/8/16/32/128/256 variants on Kijai mirror).
- **LoRA file** (I2V): `loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`. ([HF link](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/blob/main/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors))
- **Kijai mirror with rank ladder** (rank 4 → 256, all bf16): <https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v>. Files like `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank{4,8,16,32,64,128,256}_bf16.safetensors` and `lightx2v_I2V_14B_480p_cfg_step_distill_rank{4-256}_bf16.safetensors`. The community currently considers **`lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16`** the best-quality T2V Lightning LoRA (often used *in place of* Wan2.2-Lightning, see §1.3). ([HF discussion](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/56))
- **Steps**: **4** (the distillation was trained for exactly NFE=4 with `denoising_step_list=[1000, 750, 500, 250]`). ([Lightx2v docs](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html))
- **CFG / guidance_scale**: **1.0**. The model name `CfgDistill` means CFG was distilled into the single-stream pass. The docs explicitly warn: "enabling CFG may result in completely blurred video." ([Lightx2v docs](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html))
- **Scheduler**: `UniPCMultistepScheduler` with `flow_shift=5.0` (diffusers default per the Wan pipeline doc) *or* LCMScheduler with `sample_shift=5` (lightx2v native). UniPC is what diffusers integration uses; Euler also works empirically with the same shift. ([diffusers Wan docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md))
- **License**: Apache-2.0. ([model card](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill))
- **Quality**: "near-identical" to base at 4 steps per maintainer; community feedback says still the gold standard, **even for Wan 2.2** (see §1.3).

### 1.2 lightx2v CausVid (Wan 2.1, T2V-only)

- **HF**: <https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid>
- **Base**: Wan2.1-T2V-14B. **T2V only.** No I2V/VACE.
- **Steps**: 4–9 (originally trained 9-step bidirectional → causal student; community uses 4–8).
- **CFG**: not strictly distilled; community uses guidance_scale 1–2.
- **License**: **cc-by-nc-4.0** (non-commercial) — **flag for the Studio**: if the deployment is monetized, do not ship CausVid. ([model card](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid))
- **Quality**: V2 known for boosted colors/saturation but adds bias; superseded by StepDistill-CfgDistill for general use.

### 1.3 lightx2v Wan2.2-Lightning (MoE dual LoRA — primary target for Wan 2.2)

- **HF**: <https://huggingface.co/lightx2v/Wan2.2-Lightning>
- **Folder layout** (each version is a folder containing `high_noise_model.safetensors` + `low_noise_model.safetensors`):
  - `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1`, `…-V1.1`, `…-V2.0` (released **2025-11-08**, latest stable)
  - `Wan2.2-T2V-A14B-4steps-lora-250928` (preview, 2025-09-28 — referenced as "0928" in changelogs)
  - `Wan2.2-T2V-A14B-4steps-250928-dyno` (dynamic-rank experimental)
  - `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1` (released 2025-08-07; **no V2.0 I2V LoRA yet** as of May 2026)
- **Kijai's hand-renamed mirror** (better for diffusers — single file per expert, no V-folder structure): <https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Wan22-Lightning>
  - `Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors` (1.23 GB)
  - `Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors` (614 MB) — note rank asymmetry: HIGH is rank 128, LOW is rank 64
  - `old/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors`
  - For I2V: `LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors` and `_LOW_` counterpart
- **Steps**: **4**. **CFG**: **1.0** for both stages. **Scheduler**: Euler (lightx2v ref) or UniPC (diffusers) with `flow_shift=5.0` (T2V) / `flow_shift=8.0` (I2V — see §2). **Strength**: officially `1.0/1.0`; **community-tuned** strength: `HIGH=1.5, LOW=1.0` at 6 steps gives better motion. ([discussion #14](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/14), [discussion #41](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/41))
- **License**: Apache-2.0.
- **Quality caveats** — this matters: community consensus is that **Wan2.2-Lightning V1/V1.1 produced softer, less dynamic video than the 2.1 lightx2v LoRA**, and **the workaround was to load the Wan2.1 T2V-14B cfg-step-distill-v2 rank128 LoRA onto Wan 2.2** instead. V2.0 (2025-11-08) is reportedly close to par but still flagged for "kills complex motions" in high-noise. ([discussion #14](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/14), [discussion #41](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/41))
- **VACE compatibility**: officially **untested**. Maintainer comment: "The LoRA is trained with Wan2.2-T2V-A14B and has not been tested on VACE yet." Treat as a gap. ([discussion #41](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/41))

### 1.4 lightx2v Wan2.2-Distill-Loras (parallel I2V-only LoRA release)

- **HF**: <https://huggingface.co/lightx2v/Wan2.2-Distill-Loras>
- **Files**: `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_<version>.safetensors` and `_low_noise_` counterpart. Only **I2V-A14B**, only **rank 64**. Apache-2.0.
- **Difference vs Wan2.2-Lightning**: This repo is the "LoRA extraction from the full Wan2.2-Distill-Models distilled checkpoints," whereas Wan2.2-Lightning was trained as LoRA from the start (Seko recipe). **Wan2.2-Lightning is the better-known and more-supported set; Wan2.2-Distill-Loras is the alternative.** ([model card](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras))
- **Strength**: 1.0/1.0 (fixed per README).
- **Denoising step list**: `[1000, 750, 500, 250]`.

### 1.5 lightx2v Wan2.2-Distill-Models (full distilled MoE checkpoints — not LoRAs)

- **HF**: <https://huggingface.co/lightx2v/Wan2.2-Distill-Models>
- **Files**: `wan2.2_{t2v|i2v}_A14b_{high|low}_noise_{bf16|scaled_fp8_e4m3|int8}_lightx2v_4step.safetensors` (with `_comfyui` variants).
- **April 2026 release**: `Wan2.2-I2V-A14B-4step-720p-high` and `-low` — 720p-trained variants with optimized low-noise algorithm. ([issue #1030](https://github.com/ModelTC/lightx2v/issues/1030)) **Still under discussion whether a LoRA-only version will ship** — as of April 21, 2026 only full distilled checkpoints exist for the 720p-trained variant.
- **Use case for Studio**: skip these for the LoRA architecture (they're full distilled MoE; loading them = swapping the base model, not toggling a LoRA). Mention in §6 as the "Fast Plus" path if the user has VRAM to spare and wants 720p I2V quality.

### 1.6 FastWan / FastVideo (Hao AI Lab, sparse distillation)

- **HF**: <https://huggingface.co/FastVideo>
- **Released models** ([blog](https://haoailab.com/blogs/fastvideo_post_training/)):
  - `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` — T2V, 1.3B, **3-step**, **full checkpoint** (not LoRA). Pipeline class: `WanDMDPipeline`. Apache-2.0.
  - `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` — TI2V 5B, **3-step**, full checkpoint, `WanDMDPipeline`. Apache-2.0.
  - `FastWan2.1-T2V-14B-Preview` — listed as "coming soon" on the blog.
- **Key gap**: **no LoRA versions** — the FastWan release is full-weight checkpoints only. Loading FastWan in the Studio = switching the base transformer, not toggling a LoRA. Not a fit for the "Fast" LoRA preset; relevant only if you support multiple base models in the model picker.
- **Innovation**: jointly trained step distillation + Video Sparse Attention (VSA). At long-sequence resolutions the sparse-attention path is a 5×+ win on top of step distillation; at TI2V-5B's 20K-token sequences VSA does not help much, so FastWan2.2-TI2V-5B is the "FullAttn" variant.

### 1.7 Self-Forcing (Wan 2.1, T2V-1.3B)

- **HF**: <https://huggingface.co/lightx2v/Self-Forcing-FP8>, `lightx2v/Self-Forcing-NVFP4`.
- Full-model checkpoint, autoregressive causal generation, MIT-licensed. Used as the training framework for `Wan2.1-T2V-14B-StepDistill-CfgDistill` (Self-Forcing-Plus). Not a runtime LoRA you'd toggle in the Studio.

### 1.8 AccVideo / CausVid (T2V-14B Wan 2.1 distillation LoRAs)

- AccVideo: Wan2.1-T2V-14B variant referenced by Kijai's repo notes; ~5-step. License/quality inconsistent — community has largely consolidated onto lightx2v StepDistill-CfgDistill.
- CausVid: covered in §1.2; cc-by-nc-4.0 limits commercial use.

---

## 2. Wan 2.2 MoE dual-LoRA pattern

Wan 2.2 14B is a *two-expert MoE*: a **high-noise transformer** denoises the early (noisy) timesteps, and a **low-noise transformer** denoises the late (clean) timesteps. The handoff happens at a `boundary_ratio * num_train_timesteps` threshold. In diffusers, the two experts are exposed as `transformer` (high-noise) and `transformer_2` (low-noise) on `WanPipeline` / `WanImageToVideoPipeline` / `WanVACEPipeline`. The `boundary_ratio` lives on the pipeline. ([diffusers Wan docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md))

**Key rules** (verified from PRs #12040 and #12074, both merged into diffusers):

1. **You must load BOTH LoRAs.** There is no single combined Lightning LoRA for Wan 2.2 MoE. Loading only the HIGH LoRA leaves the LOW transformer un-accelerated, which causes blur/artifacts at 4 steps. (Files for `Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16` and `_LOW_fp16` are paired.)
2. The HIGH LoRA goes on `pipe.transformer` (default destination).
3. The LOW LoRA goes on `pipe.transformer_2` via `load_into_transformer_2=True` on `pipe.load_lora_weights(...)`.
4. **Adapter names must be distinct** (e.g. `"lightning_high"` and `"lightning_low"`) — they target different transformers, and `set_adapters([...])` can then set per-adapter weights.
5. The pipeline's `boundary_ratio` is **already set correctly** by the official Wan-AI/Wan2.2-*-Diffusers configs — do not override unless you know what you're doing.
6. The same diffusers call accepts both `guidance_scale` (high-noise stage) and `guidance_scale_2` (low-noise stage). For Lightning, pass `guidance_scale=1.0, guidance_scale_2=1.0`.

### Canonical Wan 2.2 T2V Lightning code (diffusers ≥ 0.38, merged PR #12074)

```python
import torch
from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler

dtype = torch.bfloat16
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=dtype
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=5.0
)
pipe.to("cuda")

# HIGH-noise expert LoRA (goes onto pipe.transformer)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
    adapter_name="lightning_high",
)

# LOW-noise expert LoRA (goes onto pipe.transformer_2)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
    adapter_name="lightning_low",
    load_into_transformer_2=True,
)

# Activate both at strength 1.0 (or community-tuned 1.5/1.0 for stronger motion)
pipe.set_adapters(["lightning_high", "lightning_low"], adapter_weights=[1.0, 1.0])

out = pipe(
    prompt="...",
    num_frames=81,
    num_inference_steps=4,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
).frames[0]
```

### Wan 2.2 I2V Lightning code (with fuse_lora + per-transformer scale — preferred for production)

The fused path is faster, avoids the runtime LoRA hook, and lets you assign different scales to high vs low. Drawback: `unfuse_lora()` only works for a single adapter — if you want to swap back to base, reload the pipeline. From the PR #12074 comment thread:

```python
pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers", torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=8.0   # 8.0 is the empirical I2V value
)
pipe.to("cuda")

# Load the SAME lightx2v I2V LoRA twice — once per transformer
LIGHTX2V_I2V = "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy", weight_name=LIGHTX2V_I2V, adapter_name="lightning_high"
)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy", weight_name=LIGHTX2V_I2V,
    adapter_name="lightning_low", load_into_transformer_2=True,
)

# Fuse with asymmetric scale (HIGH at 3.0 was found empirically by the PR author;
# the standard "safe" config is 1.0/1.0)
pipe.fuse_lora(adapter_names=["lightning_high"], lora_scale=3.0, components=["transformer"])
pipe.fuse_lora(adapter_names=["lightning_low"],  lora_scale=1.0, components=["transformer_2"])
pipe.unload_lora_weights()

out = pipe(image=img, prompt="...", num_inference_steps=4, guidance_scale=1.0).frames[0]
```

**Important note**: the example above re-uses the **Wan 2.1** lightx2v I2V LoRA on the Wan 2.2 base. This is the community's "hybrid" trick — it works because the lightx2v I2V LoRA targets attention shapes that are largely shared between Wan 2.1 and Wan 2.2's I2V branches, and it often gives sharper output than the official Wan2.2-Lightning I2V LoRA. You will see harmless "unmatched key" warnings on load. For the Studio's "Fast" preset, prefer the **official Wan2.2-Lightning HIGH/LOW pair** for forward compatibility, with this hybrid as a configurable fallback.

### Wan 2.2 VACE Lightning — **GAP**

- `WanVACEPipeline` also exposes `transformer` + `transformer_2` + `boundary_ratio`, so the API supports the pattern.
- **However**, neither lightx2v/Wan2.2-Lightning nor lightx2v/Wan2.2-Distill-Loras were trained against the VACE control branch. The maintainer has explicitly not tested it. Practical result: applying the T2V Lightning LoRA onto a `WanVACEPipeline` works but the conditioning fidelity (depth/pose/mask) degrades because VACE introduces extra control layers that the LoRA does not target.
- **Recommendation for the Studio**: **disable "Fast" preset for VACE mode**, with a tooltip "Lightning LoRA not validated for VACE — use Quality preset (40 steps)." Sources: [VACE issue #63](https://github.com/ali-vilab/VACE/issues/63), [Wan2.2-Lightning discussion #41](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/41).

---

## 3. Diffusers loading recipes

### 3.1 Wan 2.1 T2V Lightning (single transformer, single LoRA)

```python
import torch
from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler

vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=5.0   # 5.0 for 720p, 3.0 for 480p
)
pipe.to("cuda")

pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
    adapter_name="lightning",
)
pipe.set_adapters(["lightning"], adapter_weights=[1.0])

out = pipe(
    prompt="...",
    num_frames=81,
    num_inference_steps=4,
    guidance_scale=1.0,
).frames[0]
```

### 3.2 Wan 2.1 I2V Lightning (single transformer, 480P or 720P — same LoRA file)

```python
from diffusers import WanImageToVideoPipeline
pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

pipe.load_lora_weights(
    "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
    weight_name="loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    adapter_name="lightning",
)
pipe.set_adapters(["lightning"], adapter_weights=[1.0])
```

### 3.3 Wan 2.2 T2V / I2V Lightning (paired MoE, see §2 for full code)

The pattern is the dual-`load_lora_weights` call with `load_into_transformer_2=True` on the second call, distinct `adapter_name`s, then `set_adapters([h, l], [w_h, w_l])`.

### 3.4 fuse vs hot-swap vs unload — decision matrix

| Strategy | Pros | Cons | When to use in Studio |
|---|---|---|---|
| **Load + set_adapters** (no fuse) | LoRA can be enabled/disabled per-call with `pipe.set_adapters([], [])` or `pipe.disable_lora()`; preset toggle costs zero seconds | Runtime LoRA hooks add ~3–8% inference overhead | **Recommended default for the Studio.** Lets you toggle Fast ↔ Quality in the same session without reloading. |
| **fuse_lora + unload_lora_weights** | Fastest inference (no hooks). Required if you `torch.compile` the transformer. Required for fp8 quantization on top of LoRA. | `unfuse_lora()` only works with a single fused adapter; for Wan 2.2 you'd have to reload the pipeline to revert to Quality | Use when you ship a **single dedicated Fast endpoint** that never serves Quality. Also use when targeting ZeroGPU compile + fp8. |
| **Hot-swap (`hotswap=True`)** | Avoids accumulating memory across multiple LoRAs; preserves `torch.compile` graph | Limited to text-encoder-free LoRAs (Wan Lightning qualifies). Requires `enable_lora_hotswap(target_rank=max_rank)` before first load when compiled. Doesn't natively help with HIGH/LOW dual-transformer pattern. | Use only if you support **multiple style LoRAs** stacked on top of Lightning. For Lightning toggling alone, plain `set_adapters` is simpler. |
| **disable_lora / set_adapters([], [])** | Zero memory cost to "remove" — LoRA stays loaded but inactive | Inference still has hook overhead even when disabled | Use this for the Fast → Quality runtime toggle. ~50MB LoRA stays resident. |
| **delete_adapters + unload_lora_weights** | Frees LoRA memory entirely | Adds reload time when user switches back | Only on memory pressure (ZeroGPU is generous, so not needed). |

**Memory cost of inactive LoRA**: rank-128 Wan 2.2 Lightning HIGH ≈ 1.23 GB on disk, ~620 MB on GPU (bf16). For an A10G / L40S that's negligible; on ZeroGPU the spinning-up host has 80 GB so it does not matter. Plan to leave LoRAs loaded and toggle via `set_adapters`.

### 3.5 set_adapters semantics — scope

From the diffusers PEFT docs ([link](https://huggingface.co/docs/diffusers/en/tutorials/using_peft_for_inference)): `set_adapters()` only scales attention LoRA weights. If a LoRA targets ResNets or up/downsamplers, those keep scale 1.0. Wan Lightning LoRAs target only attention modules, so `adapter_weights` works as expected on them.

---

## 4. Sampler and step-schedule details

### 4.1 4-step Lightning timestep schedule

Wan Lightning was trained against a fixed 4-step schedule. The lightx2v native engine uses `denoising_step_list = [1000, 750, 500, 250]` with `sample_shift = 5` and the LCM scheduler. In diffusers, the equivalent is `UniPCMultistepScheduler` (flow-match variant) with `flow_shift=5.0` and `num_inference_steps=4`. Both produce statistically equivalent timestep allocations because Wan uses flow matching natively and UniPC's flow-match path bakes the same `[1000, 750, 500, 250]` partition.

For T2V at 480p: `flow_shift=3.0` is the doc-recommended value (Wan 2.1 official example). For 720p T2V: `flow_shift=5.0`. For I2V (any resolution on Wan 2.2): empirical sweet-spot is `flow_shift=8.0` (from PR #12040 author).

### 4.2 8-step variants

There is **no** dedicated "8-step Lightning" LoRA in the lightx2v org. Community 8-step usage = the same 4-step LoRA with `num_inference_steps=8` and reduced strength (~0.7–0.8) for smoother motion. This trades 2× speed for measurably better motion. CausVid was trained against a longer schedule and is the one family where 8 steps is canonical. For the Studio's "Fast" preset, **stick to 4 steps + strength 1.0**; expose 6/8/14 only as a "Fast Plus" advanced slider if you want.

### 4.3 Flow matching vs DDIM/DPM

Wan uses **flow matching** in training. The diffusers default `UniPCMultistepScheduler` configured from the Wan pipeline checkpoint has `prediction_type="flow_prediction"` and uses the FM-compatible solver path. You do *not* need to swap to DPM++ / DDIM / Karras — those are SD-style schedulers that don't speak flow matching. Lightning LoRAs are flow-aware by construction. **Action**: keep `UniPCMultistepScheduler` everywhere; only vary `flow_shift`.

### 4.4 guidance_scale

`guidance_scale=1.0` is **required** for all CFG-distilled Lightning LoRAs (CfgDistill, Wan2.2-Lightning, Wan2.2-Distill-Loras). On Wan 2.2 also pass `guidance_scale_2=1.0` to apply the same to the low-noise stage. Using guidance_scale > 1 with a CFG-distilled LoRA produces washed-out video. CausVid is an exception — it tolerates and slightly benefits from guidance_scale ≈ 2 because CFG was not distilled into it.

---

## 5. "Fast" vs "Quality" preset architecture for the Studio

### 5.1 Preset definition

| Field | Fast preset | Quality preset |
|---|---|---|
| `num_inference_steps` | 4 (with optional 6/8 advanced slider) | 30 (default) / 40 (high) / 50 (max) |
| `guidance_scale` | 1.0 | 5.0 (T2V), 5.5 (I2V FLF2V), 5.0 (VACE) |
| `guidance_scale_2` (Wan 2.2 only) | 1.0 | 5.0 (or pipeline default) |
| `flow_shift` | 5.0 (T2V-14B 720p), 3.0 (T2V-1.3B / I2V-480p), 8.0 (Wan 2.2 I2V) | same as Fast — `flow_shift` is independent of the LoRA |
| Scheduler | `UniPCMultistepScheduler` (flow-match) | `UniPCMultistepScheduler` (flow-match) |
| Lightning LoRA loaded | yes, active (set_adapters weight 1.0) | yes, loaded but **disabled via `set_adapters([], [])` or `disable_lora()`** |
| `negative_prompt` | ignored (CFG=1) | used |
| Expected speedup vs Quality | 6–10× | 1× (reference) |
| VAE decoder | identical | identical |

### 5.2 UI: single radio + advanced collapsible

Recommended Gradio layout (the simplest mental model is *one* toggle):

```python
preset = gr.Radio(
    choices=["Fast (Lightning, 4 steps)", "Quality (base, 30 steps)"],
    value="Fast (Lightning, 4 steps)",
    label="Quality preset",
    info="Fast = Lightning LoRA, 6–10x faster, slight motion softness. Quality = full sampler.",
)

with gr.Accordion("Advanced", open=False):
    steps_override = gr.Slider(2, 50, value=0, step=1, label="Override steps (0 = preset default)")
    guidance_override = gr.Slider(0.5, 10.0, value=0.0, step=0.1, label="Override guidance (0 = preset default)")
    flow_shift = gr.Slider(2.0, 12.0, value=5.0, step=0.5, label="Flow shift")
```

### 5.3 Per-mode preset support — the coverage matrix

| Mode \ Generation | Wan 2.1 T2V-1.3B | Wan 2.1 T2V-14B | Wan 2.1 I2V-14B | Wan 2.1 FLF2V-14B | Wan 2.1 VACE-1.3B/14B | Wan 2.2 T2V-A14B | Wan 2.2 I2V-A14B | Wan 2.2 TI2V-5B | Wan 2.2 Animate-14B |
|---|---|---|---|---|---|---|---|---|---|
| Fast (Lightning) | **gap** (no LoRA — use FastWan2.1-1.3B full ckpt instead) | yes (`lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128`) | yes (`lightx2v_I2V_14B_480p_cfg_step_distill_rank128`) | likely works via I2V LoRA (untested by maintainer) | **gap** — Lightning not trained on VACE | yes (`Wan22_A14B_T2V_HIGH+LOW_Lightning_4steps_250928_rank128/64`) | yes (`Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1` HIGH+LOW) | **gap as LoRA** — use FastWan2.2-TI2V-5B full ckpt (3-step) instead | **gap** — Animate not covered |
| Quality (base) | yes (50 steps) | yes (50 steps) | yes (50 steps) | yes (50 steps) | yes (30–50 steps) | yes (40 steps default) | yes (40 steps default) | yes (40 steps default) | yes (50 steps) |

**Gap handling** — graceful fallback rule:

```python
LIGHTNING_AVAILABLE = {
    ("wan2.1", "t2v",   "1.3b"): False,
    ("wan2.1", "t2v",   "14b"):  True,
    ("wan2.1", "i2v",   "14b"):  True,
    ("wan2.1", "flf2v", "14b"):  True,    # community-confirmed, use I2V LoRA
    ("wan2.1", "vace",  "1.3b"): False,
    ("wan2.1", "vace",  "14b"):  False,
    ("wan2.2", "t2v",   "a14b"): True,
    ("wan2.2", "i2v",   "a14b"): True,
    ("wan2.2", "ti2v",  "5b"):   False,    # only as full FastWan checkpoint, not LoRA
    ("wan2.2", "animate","14b"): False,
}

def resolve_preset(mode, gen, size, requested):
    if requested == "fast" and not LIGHTNING_AVAILABLE.get((gen, mode, size), False):
        # show toast: "Lightning unavailable for this mode — falling back to Quality (30 steps)"
        return "quality"
    return requested
```

### 5.4 Pipeline lifecycle — model-loading state machine

For the Studio (single-base-model session, preset toggled at request time):

```python
class WanModelHandle:
    """One per (mode, generation) tuple. Lazy-loads on first use."""
    def __init__(self, mode, gen):
        self.pipe = None
        self.lora_loaded = False
        self.mode, self.gen = mode, gen

    def ensure_loaded(self):
        if self.pipe is None:
            self.pipe = self._build_pipeline()      # WanPipeline / WanI2VPipeline / WanVACEPipeline
            self._configure_scheduler()
            if LIGHTNING_AVAILABLE.get((self.gen, self.mode, self.size), False):
                self._load_lightning_lora()        # leaves it loaded; set_adapters controls activity
                self.lora_loaded = True

    def configure_preset(self, preset: Literal["fast", "quality"]):
        if not self.lora_loaded:
            # gen/mode has no Lightning; pin to Quality regardless
            return self._quality_kwargs()
        if preset == "fast":
            self.pipe.set_adapters(["lightning_high", "lightning_low"], [1.0, 1.0]) \
                if self.gen == "wan2.2" else \
                self.pipe.set_adapters(["lightning"], [1.0])
            return dict(num_inference_steps=4, guidance_scale=1.0, guidance_scale_2=1.0)
        else:  # quality
            self.pipe.disable_lora()                # keeps LoRA in memory, deactivates hooks
            return dict(num_inference_steps=30, guidance_scale=5.0, guidance_scale_2=5.0)
```

**Why this design**

- Load once, toggle infinitely. The LoRA is loaded eagerly on first model use; preset switching is `disable_lora()` ↔ `set_adapters(...)`, which is microsecond-fast.
- One handle per `(mode, gen)`, so the Studio's mode picker can lazy-load on demand and keep only the active mode hot (ZeroGPU spins the host down between requests anyway).
- The kwargs dict returned from `configure_preset()` is splatted into `pipe(**kwargs, **user_kwargs)`. The `negative_prompt` user input is forwarded but ignored at `guidance_scale=1.0` (a no-op).
- For ZeroGPU specifically: ZeroGPU re-imports modules per request. Persist `WanModelHandle` at module-level so it's recreated only if process restarts; `@spaces.GPU` decorates the inference call, which is fine because LoRA loading must happen on-GPU.

### 5.5 fp8 + Lightning (optional fast-path)

For ZeroGPU H200, fp8 + Lightning gives a further ~30% over bf16 + Lightning at near-zero quality cost. Pattern:

1. Load base + Lightning LoRA in bf16.
2. `pipe.fuse_lora(...)` then `pipe.unload_lora_weights()`.
3. Quantize the fused transformer with `optimum-quanto` or diffusers' `PipelineQuantizationConfig` to `qfloat8` or `int8`.
4. Cache the quantized fused pipeline to disk.

Trade-off: **the Quality preset is no longer available** on the same process (unfuse + dequantize is not supported). If you take this path, run **two separate ZeroGPU endpoints** — one Fast-fp8, one Quality-bf16.

---

## 6. Open issues / known footguns

1. **Wan2.2-Lightning quality regression** — V1.0 and V1.1 are softer than the Wan 2.1 lightx2v LoRA. V2.0 (2025-11-08, T2V only) is the recommended T2V. For I2V, **the hybrid trick of loading the Wan 2.1 lightx2v I2V LoRA on a Wan 2.2 I2V pipeline** still gives the sharpest results per community testing through Oct 2025. Ship V2.0 as default, expose the hybrid in advanced settings. ([discussion #41](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/41), [discussion #56](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/56))
2. **PR #12147 reported a LoRA loading bug** for some Wan 2.2 Lightning files via `transformer.load_lora_adapter` direct API. Use the `pipe.load_lora_weights(..., load_into_transformer_2=True)` path (PR #12074) instead — that's the supported path.
3. **VACE Lightning is unsupported.** Hard-disable in UI.
4. **Wan-Animate Lightning is unsupported.** Wan-Animate already runs with `guidance_scale=1.0` natively but at 50 steps; no 4-step LoRA exists. Hard-disable Fast for Animate or set Fast = base-model 30 steps (still 40% faster than Quality's 50).
5. **CausVid license**: cc-by-nc-4.0. Do not enable for any commercial/monetized Spaces.
6. **diffusers version**: PR #12074 (the `load_into_transformer_2` flag) landed in `v0.38.0`. Pin `diffusers >= 0.38.0`.
7. **Wan 2.5 / 2.6**: closed weights. If a future research drop publishes weights, the LoRA story will start over.

---

## Citations index

- [diffusers Wan pipeline docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md)
- [diffusers PEFT/LoRA tutorial](https://huggingface.co/docs/diffusers/en/tutorials/using_peft_for_inference)
- [diffusers PR #12040 — LightX2V LoRA support code](https://github.com/huggingface/diffusers/pull/12040#issuecomment-3144185272)
- [diffusers PR #12074 — load_into_transformer_2 for Wan 2.2 dual LoRA](https://github.com/huggingface/diffusers/pull/12074)
- [diffusers issue #12146 — Wan acceleration feature request](https://github.com/huggingface/diffusers/issues/12146)
- [diffusers issue #12147 — Wan2.2 Lightning loading failures](https://github.com/huggingface/diffusers/issues/12147)
- [lightx2v org index](https://huggingface.co/lightx2v)
- [lightx2v/Wan2.2-Lightning model card](https://huggingface.co/lightx2v/Wan2.2-Lightning)
- [lightx2v/Wan2.2-Distill-Loras model card](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- [lightx2v/Wan2.2-Distill-Models model card](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- [lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill model card](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)
- [lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)
- [lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v)
- [lightx2v/Wan2.1-T2V-14B-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid) (cc-by-nc-4.0)
- [Kijai/WanVideo_comfy — full mirror of Lightning LoRAs](https://huggingface.co/Kijai/WanVideo_comfy)
- [ModelTC/Wan2.2-Lightning GitHub](https://github.com/ModelTC/Wan2.2-Lightning)
- [LightX2V framework docs — step distillation](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html)
- [FastVideo / FastWan blog post](https://haoailab.com/blogs/fastvideo_post_training/)
- [FastVideo/FastWan2.1-T2V-1.3B-Diffusers](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)
- [FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers)
- [VACE LoRA support issue (ali-vilab/VACE #63)](https://github.com/ali-vilab/VACE/issues/63)
- [lightx2v/Wan2.2-Lightning discussion #14 — vs Wan2.1-Lightx2v](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/14)
- [lightx2v/Wan2.2-Lightning discussion #41 — V2.0 release notes + VACE compat note](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/41)
- [lightx2v/Wan2.2-Lightning discussion #56 — community current-best LoRA recommendation](https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/56)
- [ModelTC/lightx2v issue #1030 — 720p April 2026 LoRA request](https://github.com/ModelTC/lightx2v/issues/1030)
- [Spheron Wan 2.5 deploy guide](https://www.spheron.network/blog/deploy-wan-2-5-gpu-cloud/) (confirms 2.5 API-only)
