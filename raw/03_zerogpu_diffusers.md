# Wan Studio — ZeroGPU + Diffusers Integration Layer

Research date: **2026-05-21**. Scope: ZeroGPU runtime, diffusers Wan pipeline loading, multi-mode swapping, MPS/ZeroGPU backend agnosticism, progress + cancellation.

This document is intentionally code-heavy. Every snippet is copy-pasteable. Every non-obvious claim links to its source.

---

## 1. ZeroGPU current state (May 2026)

### 1.1 Hardware — hard reset on 2026-05-13

ZeroGPU moved off NVIDIA H200 on **May 13, 2026**. As of this writing it allocates **NVIDIA RTX Pro 6000 Blackwell** GPUs.

| size              | backing hardware                   | VRAM | quota cost |
|-------------------|------------------------------------|------|-----------:|
| `large` (default) | half NVIDIA RTX Pro 6000 Blackwell | 48 GB | 1× |
| `xlarge`          | full NVIDIA RTX Pro 6000 Blackwell | 96 GB | 2× |

Sources:
- <https://huggingface.co/docs/hub/spaces-zerogpu> (current ZeroGPU docs — "Half NVIDIA RTX Pro 6000 Blackwell" / 48GB / "Full NVIDIA RTX Pro 6000 Blackwell" / 96GB)
- <https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960> (the transition thread, dated 2026-05-13; HF staff confirmed it's a hardware refresh, not a temporary swap)

**Compute capability change.** RTX Pro 6000 Blackwell is **sm_120**, not sm_90 (H200). PyTorch < 2.8 will print
`sm_120 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.`
Some existing Spaces broke on the switch for exactly this reason; multimodalart's Wan 2.2 FLF2V Space `pip install`s a PyTorch nightly at startup to work around it (see §2.5).

### 1.2 Per-call time budget

- **Default `@spaces.GPU` duration: 60 seconds.**
- Override via `@spaces.GPU(duration=N)` where `N` is seconds.
- Dynamic durations: pass a callable that receives the same args as the decorated function and returns an int.

```python
@spaces.GPU(duration=120)
def generate(prompt):
    return pipe(prompt).images

def get_duration(prompt, steps):
    return min(300, int(steps * 3.75))

@spaces.GPU(duration=get_duration)
def generate(prompt, steps):
    return pipe(prompt, num_inference_steps=steps).images
```

Source: <https://huggingface.co/docs/hub/spaces-zerogpu> ("Duration Management" + "Dynamic duration" sections).

**Practical ceiling.** Daily quota is consumed at the *effective* runtime. PRO users get 40 min/day on `large`, 60 min on Enterprise; pre-paid credits extend past quota at $1 per 10 min of GPU time. There is no hard per-call cap documented, but durations >300s are uncommon in practice; community Spaces use up to 500 (see alexnasa Animate Space, §3.2).

### 1.3 The `spaces` Python package

- Current PyPI: **0.50.2** released 2026-05-14.
- Import: `import spaces`
- Source: <https://pypi.org/project/spaces/>

Public API used in practice:
```python
spaces.GPU(duration=int | callable, size='large' | 'xlarge')   # decorator
spaces.aoti_capture(module)                                    # context manager (AOT)
spaces.aoti_compile(exported_program)                          # → CompiledModel
spaces.aoti_apply(compiled, module)                            # patches forward
spaces.aoti_blocks_load(module, repo_id, variant='fp8da')      # load precompiled graphs from Hub
```

Source for AOT API: <https://huggingface.co/blog/zerogpu-aoti>.

### 1.4 Cold-start, warm-state, and process lifecycle

- The Space's **main Python process stays warm** while the Space is "running" — module-level code runs once at boot.
- **Each `@spaces.GPU` call spawns a fresh sub-process** with a real GPU attached. CUDA must not be initialized in the parent. The PyTorch CUDA emulation mode that ZeroGPU enables lets you call `.to('cuda')` at module-level safely; under the hood that's a no-op marker until a child process inherits and re-binds.
- **Lazy `.to('cuda')` inside the decorated function is officially discouraged** — CUDA transfers happen during the cold-start of the GPU subprocess and are optimized to happen there. Docs say verbatim: "Lazy-loading or moving models to CUDA inside `@spaces.GPU` is discouraged, as it is significantly less efficient."

Source: <https://huggingface.co/docs/hub/spaces-zerogpu> ("Model loading" section).

### 1.5 RAM / VRAM / disk

| resource                | value | source |
|-------------------------|-------|--------|
| GPU VRAM (large)        | 48 GB | docs |
| GPU VRAM (xlarge)       | 96 GB | docs |
| CPU RAM                 | not documented per-tier; community reports ~70 GB on ZeroGPU class instances, but treat as best-effort | forum threads |
| Ephemeral disk          | Small; not numerically documented. Lost on restart. Persistent storage purchase is **no longer offered for new Spaces** (as of forum thread 173028). Attach Storage Buckets if you need persistence. | <https://huggingface.co/docs/hub/spaces-storage>, <https://discuss.huggingface.co/t/zerogpu-disk-space-limits/173028> |
| HF Hub cache (`HF_HOME`)| Defaults to a path on the ephemeral disk. Re-downloads on every restart unless you stage to a Storage Bucket. | spaces-storage docs |

**Cannot confirm authoritatively as of May 2026:** exact GB number for ephemeral disk on ZeroGPU. The HF docs intentionally say "a small amount" without naming a figure. Treat as <= ~50 GB and design with snapshot_download → cache reuse during the same boot.

### 1.6 bf16 — confirmed-yes

RTX Pro 6000 Blackwell supports native bf16 (it's Blackwell tensor cores). Every Wan reference Space loads `torch_dtype=torch.bfloat16`. Confirmed by inspection of multimodalart Wan 2.2 FLF2V app.py (loads transformer in bf16, transformer_2 in bf16) and Wan-AI's Wan2.2-5B Space (uses `convert_model_dtype=True` which auto-targets bf16).

### 1.7 Can `@spaces.GPU` be async / generator?

**Generator: yes, supported and used in production.**
- Pattern used in community Spaces: yield intermediate values from a generator-decorated function and bind via `.click(...)` to a Gradio output. Internal Gradio iteration consumes the generator and streams updates.
- Documented usage with `yield from sd_gen.load_new_model(*args)` style is confirmed working in ZeroGPU.
- See discussion at <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/119> for stream-related caveats ("GPU task aborted" if generator is dropped client-side).

**Async (`async def`): no official documentation either way.**
- The official ZeroGPU docs only show synchronous `def` functions.
- The HF debugging knowledge base does not document `async def` with `@spaces.GPU`.
- **Recommendation: do not use `async def` inside `@spaces.GPU`.** Use `def` (Gradio runs it in a threadpool) or a generator. The ZeroGPU process model spawns a sub-process per call; the parent event loop is not what runs the GPU code.

Sources:
- <https://huggingface.co/docs/hub/spaces-zerogpu> (only shows sync examples)
- <https://www.gradio.app/guides/streaming-outputs> (yield-based streaming)
- <https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/hf_spaces_debug_zerogpu_20251203.md> (RAG-indexed ZeroGPU debug notes — explicitly state async patterns are not documented)

### 1.8 "Warm up" / preload pattern

Yes — **always load weights at module top-level**, not inside the decorated function.

```python
import spaces, torch
from diffusers import WanPipeline

# Module level: loads to "cuda" via PyTorch's ZeroGPU emulation. Cheap.
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

@spaces.GPU(duration=120)
def generate(prompt: str):
    # Real GPU is bound HERE. Weights are inherited efficiently.
    return pipe(prompt, num_frames=81).frames[0]
```

If you also want AOTI compilation, do it at module load too — it persists across calls via shared compiled binaries from `zerogpu-aoti/*` repos on the Hub.

### 1.9 Compatibility table summary

| component | requirement |
|-----------|-------------|
| Gradio    | ≥ 4.x (and 5.49.1+ recommended for current docs) |
| Python    | 3.10.13 or 3.12.12 (ZeroGPU-supported only) |
| PyTorch   | 2.8.0–2.11.0 (must be ≥ 2.8 for sm_120) |
| Diffusers | latest (Wan 2.2 Animate landed; S2V still community contribution) |
| `spaces`  | 0.50.2 |

Source: <https://huggingface.co/docs/hub/spaces-zerogpu> + multimodalart's pip-shim line at the top of `wan-2-2-first-last-frame/app.py`.

---

## 2. Pipeline loading recipes — one snippet per Wan class

Diffusers `main` (as of May 2026) exposes exactly **five** Wan pipeline classes via `diffusers.pipelines.wan.__init__`:

```
WanPipeline
WanImageToVideoPipeline
WanVideoToVideoPipeline
WanVACEPipeline
WanAnimatePipeline
```

Source: <https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan>

**FLF2V (first-last-frame) does NOT have a dedicated class.** It is implemented via `WanImageToVideoPipeline` with `last_image=...` and the `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` checkpoint. **Wan 2.2 TI2V-5B** ships with its own non-diffusers Python package (`wan` from the original repo) and is used via `wan.WanTI2V(...)` not via diffusers. **Wan 2.2 S2V** is not yet in diffusers — open RFC at <https://github.com/huggingface/diffusers/issues/12257>; use the original `wan` package or fffiloni-style API wrapping.

### 2.1 Wan 2.1 T2V (1.3B / 14B) — `WanPipeline`

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"   # or Wan2.1-T2V-1.3B-Diffusers
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)  # 5.0 for 720p, 3.0 for 480p
pipe.to("cuda")

frames = pipe(
    prompt="A cat and dog baking a cake together.",
    negative_prompt="...",
    height=720, width=1280, num_frames=81, guidance_scale=5.0,
).frames[0]
```

Notes:
- **VAE must stay in float32** for quality. The docs explicitly say "Set the `AutoencoderKLWan` dtype to `torch.float32` for better decoding quality."
- `num_frames` should obey `4*k+1` (so 81, 85, 89, ...).
- `flow_shift` 2–5 for low-res, 7–12 for high-res.

Source: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan>

### 2.2 Wan 2.1 I2V — `WanImageToVideoPipeline`

```python
import numpy as np, torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import load_image
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"   # or -720P-Diffusers
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = load_image("...")
# resize to a multiple of pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
max_area = 480 * 832
ar = image.height / image.width
mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
H = round(np.sqrt(max_area * ar)) // mod * mod
W = round(np.sqrt(max_area / ar)) // mod * mod
image = image.resize((W, H))

frames = pipe(image=image, prompt="...", negative_prompt="...",
              height=H, width=W, num_frames=81, guidance_scale=5.0).frames[0]
```

### 2.3 Wan 2.1 FLF2V (first + last frame) — `WanImageToVideoPipeline` with `last_image`

```python
# Same loading as I2V but with the FLF2V checkpoint and CLIPVisionModel
model_id = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# call signature differs:
frames = pipe(
    image=first_frame, last_image=last_frame, prompt="...",
    height=H, width=W, guidance_scale=5.5
).frames[0]
```

### 2.4 Wan 2.1 VACE (control / inpaint / outpaint / subject / reference) — `WanVACEPipeline`

```python
import torch, PIL.Image
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"   # or Wan2.1-VACE-14B-diffusers
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

# VACE expects (video, mask, reference_images). Black mask = keep, white = generate.
frames = pipe(
    video=video_frames_list, mask=mask_frames_list, reference_images=ref_imgs,
    prompt="...", negative_prompt="...", height=H, width=W, num_frames=81,
    num_inference_steps=30, guidance_scale=5.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]
```

VACE is the workhorse for: depth-to-video, pose-to-video, sketch-to-video, scribble-to-video, inpaint, outpaint, subject-to-video, composition-to-video. All controlled by what you feed into `video` (RGB conditioning), `mask` (B/W), and `reference_images`. Preprocessing recipes: <https://github.com/ali-vilab/VACE/blob/main/UserGuide.md>.

### 2.5 Wan 2.2 T2V / I2V / TI2V — `WanPipeline` / `WanImageToVideoPipeline` with **MoE dual-expert**

Wan 2.2 14B is a **two-stage MoE**: a *high-noise* transformer (early timesteps, broad composition) and a *low-noise* transformer (late timesteps, detail). Diffusers wires this transparently via the `transformer_2` arg on `WanPipeline` / `WanImageToVideoPipeline`. Set `boundary_ratio` to control the switching timestep. Independent CFG for each stage via `guidance_scale` and `guidance_scale_2`.

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16, device_map="cuda",
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16, device_map="cuda",
    ),
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=8.0)
pipe.to("cuda")
```

(Verbatim from <https://huggingface.co/spaces/multimodalart/wan-2-2-first-last-frame/blob/main/app.py> — currently running on ZeroGPU.)

Call-site: pass both `guidance_scale` and `guidance_scale_2`. If `guidance_scale_2=None` it inherits from `guidance_scale`.

**LoRA quirk: by default LoRAs only load into `transformer`. Pass `load_into_transformer_2=True` to also load into the second denoiser.** Reference: <https://github.com/huggingface/diffusers/pull/12074#issuecomment-3155896144>.

**Wan 2.2 TI2V-5B is NOT a diffusers pipeline.** Wan-AI's own Space uses the original `wan.WanTI2V` class from the upstream `wan` package — not via diffusers. If you want it in the Studio, vendor the upstream `wan` package and call `wan.WanTI2V(config=WAN_CONFIGS['ti2v-5B'], checkpoint_dir=ckpt_dir, ...)` then `.generate(input_prompt, img, size, ...)`. Source: <https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B/blob/main/app.py>. A `Wan2.2-TI2V-5B-Diffusers` repo exists but the integration class isn't in `diffusers.pipelines.wan/__init__.py` yet.

### 2.6 Wan 2.2 Animate (character animation + replacement) — `WanAnimatePipeline`

```python
import numpy as np, torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video

model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanAnimatePipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image("character.jpg")
pose_video = load_video("pose.mp4")    # PREPROCESSED skeletal keypoints (not raw video)
face_video = load_video("face.mp4")    # PREPROCESSED facial features

# Animation mode (default)
frames = pipe(
    image=image, pose_video=pose_video, face_video=face_video,
    prompt="...", negative_prompt="...", height=720, width=1280,
    segment_frame_length=77, guidance_scale=1.0,        # CFG OFF by default
    mode="animate",
).frames[0]

# Replacement mode: also pass background_video + mask_video
frames = pipe(
    image=image, pose_video=pose_video, face_video=face_video,
    background_video=bg_video, mask_video=mask_video,
    prompt="...", mode="replace", segment_frame_length=77, guidance_scale=1.0,
).frames[0]
```

**Critical gotcha:** `pose_video` and `face_video` must be **preprocessed** (skeleton + face landmarks), not raw videos. The preprocessing scripts live in the original Wan repo; diffusers does not yet bundle them. See <https://github.com/Wan-Video/Wan2.2?tab=readme-ov-file#1-preprocessing>.

Source: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan> (Wan-Animate section).

### 2.7 Wan 2.2 S2V (audio-driven) — NOT in diffusers

Status as of 2026-05-21:
- Diffusers issue <https://github.com/huggingface/diffusers/issues/12257> is **open**. No PR linked. No `WanSpeechToVideoPipeline` class exists.
- Wan-AI's own Space (`Wan-AI/Wan2.2-S2V`) wraps DashScope's hosted inference API. Not local.

**Integration options for the Studio:**
1. Vendor the upstream `wan` package and call its S2V entry point directly. Treat it as a sibling to TI2V-5B (non-diffusers).
2. Wrap DashScope API as a fallback when running on Spaces without local weights.
3. Wait for the diffusers community contribution to land and switch.

### 2.8 Memory-saving knobs (all 14B pipelines on `large` tier)

In order of impact on a 48GB tier:

```python
# 1. bf16 (mandatory)
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. VAE tiling — slices the VAE decode pass spatially, massive savings
pipe.vae.enable_tiling()
# optional finer control:
pipe.vae.enable_slicing()

# 3. Quantize transformers (Wan 2.2 14B in fp8 fits comfortably)
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer,  Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())   # if MoE

# 4. Group-offloading (block-level keeps text encoder + transformer in mixed RAM/VRAM)
from diffusers.hooks.group_offloading import apply_group_offloading
apply_group_offloading(pipe.text_encoder,
    onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
    offload_type="block_level", num_blocks_per_group=4)
pipe.transformer.enable_group_offload(
    onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
    offload_type="leaf_level", use_stream=True)

# 5. Model CPU offload (slowest but most aggressive — local-MPS fallback only)
pipe.enable_model_cpu_offload()        # whole modules
pipe.enable_sequential_cpu_offload()   # per-layer; even more aggressive
```

Sources: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan> (memory section + group-offload code), <https://huggingface.co/spaces/multimodalart/wan-2-2-first-last-frame/blob/main/app.py> (quantize_ + aoti_blocks_load).

### 2.9 AOTI: precompiled graphs from the Hub

ZeroGPU's `aoti_blocks_load` shortcut downloads a precompiled inductor graph + applies it to your pipeline in one call. Saves 6 minutes of cold start.

```python
import spaces

# After quantize_(...)
spaces.aoti_blocks_load(pipe.transformer,   'zerogpu-aoti/Wan2', variant='fp8da')
spaces.aoti_blocks_load(pipe.transformer_2, 'zerogpu-aoti/Wan2', variant='fp8da')
```

Sources:
- <https://huggingface.co/blog/zerogpu-aoti>
- <https://huggingface.co/spaces/multimodalart/wan-2-2-first-last-frame/blob/main/app.py>

Build your own once with:
```python
@spaces.GPU(duration=1500)   # max duration for the one-off compile
def compile_transformer():
    with spaces.aoti_capture(pipe.transformer) as call:
        pipe("warmup prompt")
    exported = torch.export.export(pipe.transformer, args=call.args, kwargs=call.kwargs)
    return spaces.aoti_compile(exported)

compiled = compile_transformer()
spaces.aoti_apply(compiled, pipe.transformer)
```

Note: `torch.compile` itself does NOT work on ZeroGPU (each subprocess would recompile). AOT is the only viable path.

---

## 3. Multi-mode swapping strategy on a single Space

The Studio needs ~6 modes × 3 generations (Wan 2.1, 2.2, 2.5/2.6) = up to ~12 distinct pipeline configurations. ZeroGPU's hosting limits (10 Spaces per PRO account, 50 per Org) and the 48GB VRAM ceiling on the default tier make "one Space per mode" cleaner than "one Space for everything." But if you want them in one Space, here is the working pattern.

### 3.1 The cheap shared-component strategy

Every Wan pipeline shares the same UMT5-XXL text encoder, the same `AutoencoderKLWan` VAE, and (for I2V/FLF2V/Animate) the same `CLIPVisionModel` image encoder. Load these *once* at module level and inject into every pipeline via `from_pretrained(text_encoder=..., vae=..., image_encoder=...)`. The only thing you swap per mode is the transformer.

```python
# Module top — runs once
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline, WanVACEPipeline, WanAnimatePipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from transformers import UMT5EncoderModel, CLIPVisionModel

dtype = torch.bfloat16

# Shared, single copy in RAM
text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=dtype)
vae_fp32 = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
image_encoder = CLIPVisionModel.from_pretrained(
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder="image_encoder", torch_dtype=torch.float32)

# Per-mode transformers (lazy-load on first request, cache thereafter)
TRANSFORMER_CACHE = {}

def get_transformer(repo, subfolder="transformer"):
    key = (repo, subfolder)
    if key not in TRANSFORMER_CACHE:
        TRANSFORMER_CACHE[key] = WanTransformer3DModel.from_pretrained(
            repo, subfolder=subfolder, torch_dtype=dtype, device_map="cuda",
        )
    return TRANSFORMER_CACHE[key]

# Build (or rebuild) a pipeline object cheaply — components are references, not copies
def make_pipe_t2v(repo="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
    pipe = WanPipeline.from_pretrained(
        repo, vae=vae_fp32, text_encoder=text_encoder,
        transformer=get_transformer(repo), torch_dtype=dtype,
    )
    pipe.to("cuda")
    return pipe

def make_pipe_i2v(repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"):
    pipe = WanImageToVideoPipeline.from_pretrained(
        repo, vae=vae_fp32, text_encoder=text_encoder, image_encoder=image_encoder,
        transformer=get_transformer(repo), torch_dtype=dtype,
    )
    pipe.to("cuda")
    return pipe

# ...same pattern for VACE, Animate, MoE 2.2 (pass transformer_2 too)
```

### 3.2 Dispatch + dynamic duration

```python
import spaces

PIPELINE_BUILDERS = {
    "t2v_2_1_14b": (lambda: make_pipe_t2v("Wan-AI/Wan2.1-T2V-14B-Diffusers"), 90),
    "i2v_2_1_14b_480p": (lambda: make_pipe_i2v("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"), 120),
    "flf2v_2_1_14b_720p": (lambda: make_pipe_i2v("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"), 180),
    "vace_2_1_14b": (lambda: make_pipe_vace("Wan-AI/Wan2.1-VACE-14B-diffusers"), 180),
    "i2v_2_2_a14b": (lambda: make_pipe_moe_i2v("Wan-AI/Wan2.2-I2V-A14B-Diffusers"), 200),
    "animate_2_2_14b": (lambda: make_pipe_animate("Wan-AI/Wan2.2-Animate-14B-Diffusers"), 300),
}

ACTIVE_PIPE = None     # only ever ONE pipe object is on GPU at a time
ACTIVE_KEY = None

def select(mode_key):
    global ACTIVE_PIPE, ACTIVE_KEY
    if ACTIVE_KEY == mode_key:
        return ACTIVE_PIPE
    # release old transformer's VRAM
    if ACTIVE_PIPE is not None:
        ACTIVE_PIPE.transformer.to("cpu")
        if hasattr(ACTIVE_PIPE, "transformer_2") and ACTIVE_PIPE.transformer_2 is not None:
            ACTIVE_PIPE.transformer_2.to("cpu")
        del ACTIVE_PIPE
        torch.cuda.empty_cache()
    builder, _ = PIPELINE_BUILDERS[mode_key]
    ACTIVE_PIPE = builder()
    ACTIVE_KEY = mode_key
    return ACTIVE_PIPE

def get_duration(mode_key, *args, **kwargs):
    _, default_dur = PIPELINE_BUILDERS[mode_key]
    return default_dur

@spaces.GPU(duration=get_duration)
def generate(mode_key, **kwargs):
    pipe = select(mode_key)
    return pipe(**kwargs).frames[0]
```

### 3.3 Hub cache + persistence

- HF Hub cache lives on the Space's ephemeral disk. `snapshot_download(repo_id, local_dir="./models/...")` works but the dir disappears on restart.
- For ~12 pipelines × tens-of-GB this won't fit. Two recipes:
  - **Lazy: only download the first time a mode is invoked.** First user of each mode pays the snapshot cost (15–60s for 14B in shards).
  - **Eager: at boot, `snapshot_download` only the few you expect to be hot.** Use the cache for the cold ones.
- If you have a Storage Bucket: mount it at `/data` (or wherever `HF_HOME` points) and you get persistent cache across restarts. Buckets cost $5–$100/month depending on size. <https://huggingface.co/docs/hub/spaces-storage>

### 3.4 Prior art

- **multimodalart's Wan 2.2 FLF2V Space** uses a single MoE I2V pipeline loaded at boot with full optimization stack (bf16 → torchao FP8 → `spaces.aoti_blocks_load`). Single mode. Confirms the load-once-optimize-once pattern works. <https://huggingface.co/spaces/multimodalart/wan-2-2-first-last-frame/blob/main/app.py>
- **multimodalart's Wan 2.1 fast** loads I2V + a fused CausVid LoRA at boot, uses dynamic `get_duration`. <https://huggingface.co/spaces/multimodalart/wan2-1-fast/blob/main/app.py>
- **alexnasa's Wan2.2-Animate-ZEROGPU** is the best multi-tier reference: defines two GPU functions, one with `size='large'` and one with `size='xlarge'`, switches per resolution choice. Dynamic durations ramping 75–500s. Uses `snapshot_download` once at boot, then loads weights with a custom `load_model()`. <https://huggingface.co/spaces/alexnasa/Wan2.2-Animate-ZEROGPU/blob/main/app.py>
- **fffiloni-style API wrapping** (Wan-AI's S2V Space) wraps DashScope rather than running locally — useful template for S2V until diffusers integration lands. <https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V/blob/main/app.py>

### 3.5 `safe_serialization`, `low_cpu_mem_usage`, `variant="bf16"`

- `safe_serialization=True` is the diffusers default since v0.20; you don't need to set it.
- `low_cpu_mem_usage=True` is the default in modern `from_pretrained` paths; explicitly set it only if loading via the older API.
- `variant="bf16"` is supported on the official Wan checkpoints — many published with `bf16` variant. multimodalart uses `cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers` which is the bf16 variant repackaged at the repo level; for repos with a `variant` flag, pass `variant="bf16"` to skip downloading fp32 shards.

---

## 4. MPS-vs-ZeroGPU backend table

The Studio must run locally on M5 Max (MPS) for the user's dev loop. Wan's MPS story is **painful but workable for the smaller variants**. The 14B and the 2.2 MoE variants will not fit comfortably; assume MPS-local development uses **Wan 2.1 T2V 1.3B** and **Wan 2.1 VACE 1.3B**, with ZeroGPU for the full 14B/MoE/Animate stack.

### 4.1 Device selection

```python
import torch

def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = best_device()

def best_dtype(device):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # MPS bf16 support exists in PyTorch 2.6+ but is partial.
        # Many SDPA / inductor paths still error on bf16. Default to fp16 on MPS.
        return torch.float16
    return torch.float32

DTYPE = best_dtype(DEVICE)
```

Sources: <https://github.com/pytorch/pytorch/issues/141864> (MPS bf16 still patchy as of late 2025), <https://github.com/comfyanonymous/ComfyUI/issues/9255> (Wan 2.2 specifically fails on MPS with fp8; fp16 path is the workaround).

### 4.2 Wan-on-MPS specific issues

| issue | source | workaround |
|-------|--------|-----------|
| `Trying to convert Float8_e4m3fn to the MPS backend` when running Wan 2.2 14B template | ComfyUI #9255 | Skip fp8 quantization on MPS; use fp16 weights only. multimodalart's quantize_ block must be guarded by `if DEVICE == "cuda"`. |
| `torch.compile` AOTI on MPS | not supported | Skip AOTI on MPS. ZeroGPU-only. |
| bf16 SDPA path | partial, varies by PyTorch nightly | Force `torch_dtype=torch.float16` on MPS for transformer + image_encoder. **Keep VAE in float32** (decode quality already requires this; float32 on MPS is fine). |
| `convert_model_dtype` in upstream `wan` package | unknown on MPS | Wan 2.2 5B's official Space hard-requires CUDA. Treat the TI2V-5B and Animate paths as ZeroGPU-only for now. |

There is no published "Wan-on-MPS fork" or merged monkey-patch. The pragmatic path: **load fp16, skip fp8 quant, skip AOTI, keep VAE fp32, use Wan 2.1 1.3B locally**.

### 4.3 `@spaces.GPU` as a no-op locally

`spaces` is safe to import in any environment: the decorator is effect-free when not running inside a ZeroGPU Space. The docs explicitly state:

> The `@spaces.GPU` decorator is designed to be effect-free in non-ZeroGPU environments, ensuring compatibility across different setups.

(<https://huggingface.co/docs/hub/spaces-zerogpu>)

So no manual no-op shim is needed. But if you want to be extra-safe and avoid the `import spaces` dependency on a clean local install:

```python
try:
    import spaces
except ImportError:
    class _NoSpaces:
        @staticmethod
        def GPU(duration=None, size=None):
            def deco(fn):
                return fn
            return deco
        @staticmethod
        def aoti_blocks_load(*a, **kw): return None
        @staticmethod
        def aoti_capture(m):
            from contextlib import contextmanager
            @contextmanager
            def _c():
                yield type("Call", (), {"args": (), "kwargs": {}})()
            return _c()
    spaces = _NoSpaces()
```

### 4.4 VAE on MPS

- **Keep VAE in float32** on both MPS and CUDA — Wan docs are explicit that quality drops with fp16/bf16 VAE.
- VAE tiling/slicing is supported on MPS and crucial for 720p decode RAM:
  ```python
  pipe.vae.enable_tiling()
  pipe.vae.enable_slicing()
  ```
- Decoded frames sometimes hit Metal kernel bugs at very high resolutions (>1280) on macOS 14.x. Pin macOS 15+ if possible; M5 Max ships with 15+ so this should be moot.

### 4.5 Attention backend

| backend | ZeroGPU | MPS |
|---------|---------|-----|
| `sdpa` (default torch SDPA) | works | works (default) |
| FlashAttention-3 via `kernels` lib | **recommended** for ZeroGPU H100/H200/Blackwell | not available |
| `xformers` | optional | not built for Metal |

ZeroGPU FA3 setup:
```python
from kernels import get_kernel
vllm_flash_attn3 = get_kernel("kernels-community/vllm-flash-attn3")
# Then route via attention_kwargs={"processor": vllm_flash_attn3} or AOT-compile with FA3.
```

Source: <https://huggingface.co/blog/zerogpu-aoti> ("Flash Attention 3 Integration"). On MPS just leave SDPA alone — it picks the fastest Metal kernel automatically.

### 4.6 Side-by-side backend table

| concern | ZeroGPU (large/xlarge) | MPS (M5 Max 128 GB) |
|---------|------------------------|---------------------|
| device   | `"cuda"`              | `"mps"`             |
| dtype    | `torch.bfloat16`      | `torch.float16` (transformer) + `torch.float32` (VAE) |
| compile  | AOTI via `spaces.aoti_*` | none — `torch.compile` and inductor MPS path are flaky for video DiTs |
| quant    | torchao FP8 + Int8 → `spaces.aoti_blocks_load` | none (skip fp8) |
| attention | FA3 via `kernels` | torch SDPA |
| VAE      | float32 + tiling | float32 + tiling **mandatory** |
| offload  | not needed at xlarge; group-offload optional at large | `enable_sequential_cpu_offload()` for 14B; otherwise 1.3B only |
| `@spaces.GPU` | applies | no-op (effect-free) |
| max practical model | Wan 2.2 I2V-A14B with fp8 + AOTI | Wan 2.1 T2V-1.3B / VACE-1.3B |
| Wan-2.2 Animate | yes (preprocess sub-step out of GPU window) | not recommended |
| Wan-2.2 S2V | no diffusers class; vendor `wan` pkg or wrap DashScope | same |

---

## 5. Progress, queue config, cancellation

### 5.1 Progress: prefer `gr.Progress(track_tqdm=True)`

The canonical pattern in every reference Space (multimodalart, alexnasa, Wan-AI 5B) is:

```python
import gradio as gr, spaces

@spaces.GPU(duration=120)
def generate(prompt, steps, progress=gr.Progress(track_tqdm=True)):
    progress(0.1, desc="Preprocessing...")
    # diffusers calls `tqdm` internally for the denoise loop —
    # track_tqdm=True surfaces those updates to the UI for free.
    progress(0.2, desc=f"Denoising {steps} steps...")
    out = pipe(prompt, num_inference_steps=steps).frames[0]
    progress(0.9, desc="Encoding video...")
    video_path = export_to_video(out, fps=16)
    progress(1.0, desc="Done!")
    return video_path
```

Why this is robust on ZeroGPU:
- `gr.Progress` is a special object; Gradio threads its updates over the WebSocket independently of the GPU subprocess.
- `track_tqdm=True` snoops the tqdm bar diffusers already creates → no callback-on-step-end gymnastics needed.

### 5.2 Custom callback-on-step-end (for fancier UIs)

Wan pipelines expose `callback_on_step_end` and `callback_on_step_end_tensor_inputs`. Use this if you want to (a) emit a preview frame mid-denoise, or (b) write a per-step progress percentage.

```python
def cb(pipe, step_index, timestep, callback_kwargs):
    progress((step_index + 1) / num_inference_steps, desc=f"step {step_index+1}/{num_inference_steps}")
    return callback_kwargs

out = pipe(..., callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).frames[0]
```

Source: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan> (every Wan pipeline accepts these).

### 5.3 Generator-based streaming (intermediate previews)

```python
@spaces.GPU(duration=120)
def generate_streaming(prompt, num_inference_steps=20, progress=gr.Progress(track_tqdm=True)):
    state = {"latents": None}
    def cb(pipe, i, t, kw):
        state["latents"] = kw["latents"]
        return kw
    # ... but you can't yield from inside the pipe() call directly.
    # Instead, run pipe() in a thread + drain progress, OR break the denoise into smaller pipe() calls.
    yield None  # final video at end
```

Practical reality: **most Wan Spaces do not stream intermediate frames** because the VAE decode is too expensive to run mid-denoise. Stick with `gr.Progress` + `track_tqdm=True` for users; use `callback_on_step_end` only if you need server-side telemetry.

### 5.4 Queue config

Gradio queue defaults are reasonable on ZeroGPU. The community pattern is:

```python
demo.queue().launch()                                 # default
# or
demo.queue(max_size=20, default_concurrency_limit=1).launch()
```

ZeroGPU itself enforces per-user fair-share via the `X-IP-Token` header; you do NOT need to limit `concurrency_limit` to 1 to "prevent GPU contention." The platform handles that.

**One thing to set:** when this Space is called from *another* Space via `gradio_client.Client`, propagate the IP token, or your callers will be rate-limited fast:

```python
def proxy(prompt, request: gr.Request):
    x_ip_token = request.headers["x-ip-token"]
    client = Client("you/your-wan-studio", headers={"x-ip-token": x_ip_token})
    return client.predict(prompt, api_name="/generate")
```

Source: <https://www.gradio.app/main/docs/python-client/using-zero-gpu-spaces>

### 5.5 Cancellation when the user closes the tab

- Gradio fires a `disconnect` event when the WebSocket dies.
- The default ZeroGPU behaviour: the GPU subprocess **keeps running until your function returns**, then is released. Closing the tab does NOT preempt the GPU call (this is by design — billing already happened).
- If you want soft-cancel: poll `progress.is_canceled` (Gradio 4+) inside `callback_on_step_end` and `return` early.
- Per the docs and community thread <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/119>, you may see `gradio.exceptions.Error: 'GPU task aborted'` if the parent process tears down mid-flight. Catch it and clean up.

### 5.6 Logging to the UI

- `print()` from inside `@spaces.GPU` → goes to the Space logs (visible to the maintainer in the HF UI), NOT to the client browser.
- For per-call user-visible status, use `progress(..., desc="...")` calls and `gr.Info(...)` / `gr.Warning(...)`.
- Do not stream large log buffers via Gradio — they will consume the per-call WebSocket budget.

---

## 6. Gotchas & known issues appendix

1. **The H200 → RTX Pro 6000 Blackwell hardware switch broke any Space pinned to PyTorch < 2.8.** If you see `sm_120 is not compatible`, bump PyTorch. multimodalart's FLF2V Space does this with a one-line `pip install` shim at the top of `app.py` until the official base image catches up. <https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960>

2. **CUDA cannot be initialized in the parent process.** `nvidia-smi` calls at import, custom CUDA extensions that probe at import, etc. will all crash the Space. `import torch` is fine; `torch.cuda.is_available()` is fine; `torch.cuda.set_device(0)` from main process is not. <https://huggingface.co/docs/hub/spaces-zerogpu>

3. **Lazy `.to('cuda')` inside the decorator is officially discouraged.** Load at module-level so the GPU subprocess inherits ready tensors. Confirmed in docs. <https://huggingface.co/docs/hub/spaces-zerogpu>

4. **VAE dtype must be float32.** Both for ZeroGPU and MPS. Diffusers docs say so explicitly; visible decode quality regression with bf16 VAE in Wan.

5. **Wan 2.2 MoE LoRA loads into transformer_1 only by default.** Pass `load_into_transformer_2=True` to also patch the low-noise denoiser. <https://github.com/huggingface/diffusers/pull/12074>

6. **Wan 2.2 Animate inputs are pre-processed.** The `pose_video` and `face_video` you pass aren't raw videos — they're outputs of the upstream `wan` preprocessing pipeline (skeleton + face landmarks). Diffusers doesn't bundle the preprocessor. Plan for ffmpeg/CPU work before the `@spaces.GPU` call. <https://github.com/Wan-Video/Wan2.2?tab=readme-ov-file#1-preprocessing>

7. **Wan 2.2 S2V is not in diffusers.** Issue is open with no PR linked. Use the original `wan` package or wrap DashScope.

8. **Wan 2.2 TI2V-5B is in the diffusers wan.md catalog but is NOT exposed via diffusers pipeline classes.** Wan-AI's own 5B Space imports the upstream `wan.WanTI2V` class. If you want it, vendor that package.

9. **fp8 + Apple Silicon = crash.** ComfyUI issue #9255 documents this for Wan 2.2; same applies to any torchao FP8 path. Guard quant-block with `if device == "cuda"`.

10. **Ephemeral disk has no documented limit.** Forum reports of failed packing on 22B / 80GB models suggest the ceiling is well under 100GB. Stage carefully: don't try to hold all 12 modes' weights pre-downloaded. Lazy-load per-mode.

11. **`torch.compile` does NOT work on ZeroGPU.** It would recompile in every subprocess. Use AOTI (`spaces.aoti_*`) instead.

12. **`asyncio` + `@spaces.GPU` is undocumented.** Stick with `def` / generators.

13. **Per-call cancellation does not preempt the GPU subprocess.** Closing the tab still runs the job to completion. Implement soft-cancel via `callback_on_step_end` if needed.

14. **Wan 2.5 and 2.6 are NOT confirmed to exist in HF.** The only "Wan 2.5" reference on the Hub is `wangkanai/wan25-fp16-i2v`, a community FP16 repack, not an official Wan-AI release. **Cannot find authoritative info as of 2026-05-21 for a Wan 2.5/2.6 official release on Hugging Face.** Treat anything labeled 2.5/2.6 as third-party until Wan-AI publishes officially. Plan the Studio to scope-down to 2.1 and 2.2 confirmed modes for the initial deploy.

15. **`hosting limit`**: PRO accounts cap at 10 ZeroGPU Spaces. If you split mode-by-mode, count that. Orgs (Team/Enterprise) get 50.

---

## Concise summary

**ZeroGPU as of 2026-05-21:**
- Hardware: half/full **NVIDIA RTX Pro 6000 Blackwell** (48 GB / 96 GB). Switched from H200 on **May 13, 2026**. Requires PyTorch ≥ 2.8 for sm_120.
- Decorator: `@spaces.GPU(duration=60..N, size='large'|'xlarge')`. `duration` accepts a callable for dynamic budgets. Default 60s, no documented hard ceiling — community Spaces go up to ~500s.
- `spaces==0.50.2` on PyPI (2026-05-14). Provides `GPU`, `aoti_capture`, `aoti_compile`, `aoti_apply`, `aoti_blocks_load`.
- Lifecycle: Space main process stays warm; each GPU call spawns a fresh subprocess. **Load weights at module top-level**, never inside the decorated function. Decorator is a no-op outside ZeroGPU.
- bf16 supported. `torch.compile` not supported — use AOT (`spaces.aoti_*`) instead. Generators work; `async def` is undocumented (avoid).

**Recommended swap strategy for the Studio:**
1. **Share text_encoder, VAE, image_encoder across all modes** — load once at module top, inject via `from_pretrained(..., text_encoder=, vae=, image_encoder=)`. Saves ~15 GB of duplicated weights.
2. **Cache transformer modules in a dict keyed by (repo, subfolder)**. Build pipeline objects on demand — they're cheap wrappers around shared components.
3. **One active pipeline on GPU at a time.** On mode switch, `.to("cpu")` the outgoing transformer (and transformer_2 if MoE), `del` the pipe, `empty_cache()`, then build the new one.
4. **Lazy-load weights on first call to each mode.** First user pays the snapshot cost; subsequent calls hit the in-boot cache.
5. **Per-mode `get_duration` callable** sized to the realistic worst case (T2V 1.3B ≈ 60s, I2V 14B ≈ 120s, MoE 14B ≈ 200s, Animate ≈ 300s, fixed at boot).
6. **Apply torchao FP8 + `spaces.aoti_blocks_load('zerogpu-aoti/Wan2', variant='fp8da')` on `cuda` only.** Skip on MPS. multimodalart's FLF2V Space is the canonical reference.

**Top-3 gotchas to design around:**
1. **The hardware switch hasn't fully stabilized.** Pin PyTorch ≥ 2.8 explicitly; expect intermittent OOM at `large`/48GB on un-quantized 14B variants; budget for fp8-via-torchao + AOTI as a *standard* part of the loading recipe, not an optimization.
2. **MoE (Wan 2.2) and Animate have non-obvious quirks.** Dual-transformer LoRA loading (`load_into_transformer_2=True`), pre-processed pose+face video inputs for Animate, separate guidance scales for the two stages. Bake these into the Studio's per-mode adapter, not the user UX.
3. **Multi-mode on one Space is a memory + ergonomics tradeoff.** Better to scope down to (a) shared text/vae/clip components, (b) lazy transformer cache, (c) per-mode `get_duration`, and (d) only ever ONE active pipeline on GPU. Plan that S2V and TI2V-5B paths will NOT use diffusers — wrap the upstream `wan` package or DashScope.
