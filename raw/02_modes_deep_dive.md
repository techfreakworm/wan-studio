# Wan Video Diffusion — Modes / Capabilities Deep Dive

> Scope: every officially supported MODE in the Alibaba Wan video diffusion family that is relevant for the HF ZeroGPU Studio app.
> Date of survey: 2026-05-21. Anchors: Wan2.1 (open weights), Wan2.2 (open weights — current production family), Wan2.5 (mixed: API-first; only the VAE component has open weights on HF as of this date), Wan2.6 (released Dec 2025, partially open, builds on Wan2.2 arch).
> Companion docs in this folder: `01_checkpoints.md` (model registry), `03_zerogpu_integration.md`, `04_lightning_loras.md`, `05_ux_patterns.md`.

Sources (top-of-file index, cited inline below):
- Wan2.1 GitHub: <https://github.com/Wan-Video/Wan2.1>
- Wan2.2 GitHub: <https://github.com/Wan-Video/Wan2.2>
- Diffusers Wan docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/wan>
- Diffusers pipeline source dir: <https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan>
- VACE official repo + UserGuide: <https://github.com/ali-vilab/VACE> and <https://github.com/ali-vilab/VACE/blob/main/UserGuide.md>
- Wan-Animate paper: <https://huggingface.co/papers/2509.14055>
- Wan-Animate project page: <https://humanaigc.github.io/wan-animate>
- HF org listing: <https://huggingface.co/Wan-AI>

---

## 0. Capability matrix

| Mode | text | image (single) | image (first+last) | video | audio | pose video | depth video | edge/scribble | mask | reference images | -> output video | -> output frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **T2V** | required | — | — | — | — | — | — | — | — | — | yes (np frames -> mp4) | yes |
| **I2V** | required | required | — | — | — | — | — | — | — | — | yes | yes |
| **TI2V** (Wan2.2-5B) | required | optional | — | — | — | — | — | — | — | — | yes | yes |
| **FLF2V** | required | — | required (2 imgs) | — | — | — | — | — | — | — | yes | yes |
| **V2V** (low-level) | required | — | — | required | — | — | — | — | — | — | yes | yes |
| **VACE — Depth** | required | — | — | required (depth maps) | — | — | required | — | optional | optional | yes | yes |
| **VACE — Pose** | required | — | — | required (pose maps) | — | required | — | — | optional | optional | yes | yes |
| **VACE — Flow / Gray / Scribble / Layout-BBox / Layout-Track** | required | — | — | required (control maps) | — | — | — | optional | optional | optional | yes | yes |
| **VACE — Inpainting (Mask / BBox / MaskTrack / BBoxTrack / Label / Caption)** | required | — | — | required | — | — | — | — | required (or implicit) | optional | yes | yes |
| **VACE — Outpainting** | required | — | — | required | — | — | — | — | required (implicit ratio) | optional | yes | yes |
| **VACE — Reference-Anything / Animate-Anything / Swap-Anything / Expand-Anything / Move-Anything** | required | — | — | optional | — | — | — | — | optional | required (1-3 refs) | yes | yes |
| **VACE — Frame/Clip Extension** | required | — | — | optional | — | — | — | — | optional | optional (first/last/clip) | yes | yes |
| **S2V (Speech-to-Video)** | required | required (ref char image) | — | — | required (wav/mp3) | optional (pose.mp4) | — | — | — | — | yes | yes |
| **Wan-Animate — Animation mode** | optional | required (char image) | — | — | — | required (preproc pose.mp4) | — | — | — | — (face video required) | yes | yes |
| **Wan-Animate — Replacement mode** | optional | required (char image) | — | required (background.mp4) | — | required (preproc pose.mp4) | — | — | required (mask.mp4) | — (face video required) | yes | yes |
| **T2I** (Wan2.1 unified) | required | — | — | — | — | — | — | — | — | — | — | single frame |

(Reference: input/output columns derived from the call signatures of each pipeline; see per-mode sections for full input semantics.)

---

## Reference pipeline class table

Every mode mapped to a diffusers class (if any), the source file on GitHub, and the fall-back inference script in the official Wan-Video repo.

| Mode | Diffusers class (main, May 2026) | Diffusers source file | Official Wan repo script (`generate.py --task`) |
|---|---|---|---|
| **T2V (Wan2.1 / Wan2.2)** | `WanPipeline` | `pipeline_wan.py` <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py> | `t2v-14B`, `t2v-1.3B`, `t2v-A14B` |
| **I2V (Wan2.1 / Wan2.2)** | `WanImageToVideoPipeline` | `pipeline_wan_i2v.py` <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py> | `i2v-14B`, `i2v-A14B` |
| **TI2V (Wan2.2-5B)** | `WanImageToVideoPipeline` (image is optional) | same as above | `ti2v-5B` |
| **FLF2V** | `WanImageToVideoPipeline` (with `last_image=` kwarg) | same as above | `flf2v-14B` |
| **V2V (basic, latent-space strength remix)** | `WanVideoToVideoPipeline` | `pipeline_wan_video2video.py` <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_video2video.py> | (no dedicated task — exposed via VACE in the repo) |
| **VACE (all 25+ sub-modes — Wan2.1 only)** | `WanVACEPipeline` (single class, sub-mode is controlled by how you set `video`/`mask`/`reference_images`) | `pipeline_wan_vace.py` <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py> | `vace-1.3B`, `vace-14B` (in repo: <https://github.com/ali-vilab/VACE>) |
| **S2V (Speech-to-Video)** | **no diffusers support yet** (PR open, listed in `Wan2.2/TODO`) | n/a | `s2v-14B` |
| **Wan-Animate (animation + replacement)** | `WanAnimatePipeline` | `pipeline_wan_animate.py` <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_animate.py> | `animate-14B` |
| **T2I (Wan2.1 unified)** | use `WanPipeline` with `num_frames=1` | same as T2V | `t2i-14B` |

> The diffusers `wan/` directory contains exactly 5 pipeline files in main as of 2026-05-21: `pipeline_wan.py`, `pipeline_wan_i2v.py`, `pipeline_wan_vace.py`, `pipeline_wan_video2video.py`, `pipeline_wan_animate.py`. Reference: <https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan>
>
> **S2V is the only major Wan2.2 mode that is NOT yet in diffusers main** — the Studio will need to wrap the official `generate.py` from `Wan-Video/Wan2.2`, or re-implement it on top of `WanTransformer3DModel` + `AutoencoderKLWan` plus the bundled `wav2vec2-large-xlsr-53-english` encoder.

---

## 1. T2V — Text-to-Video

### a) Inputs
- `prompt: str | list[str]` — required. UMT5-XXL is the text encoder; `max_sequence_length=512` tokens.
- `negative_prompt: str | list[str]` — optional but strongly recommended; the official negative prompt is in Chinese in `wan/configs/shared_config.py` (long boilerplate "static, blurred details, subtitles, low quality, JPEG compression…").
- `height: int = 480`, `width: int = 832` — diffusers defaults; full-rez recipe is `1280×720`.
- `num_frames: int = 81` — must satisfy `4k+1` (VAE temporal stride is 4 — see "Notes" in <https://huggingface.co/docs/diffusers/en/api/pipelines/wan>).
- `guidance_scale: float = 5.0` (single-stage) / `(3.0, 4.0)` for Wan2.2 A14B two-stage (high-noise / low-noise) — see `wan/configs/wan_t2v_A14B.py`.
- `guidance_scale_2: float | None` — for the low-noise expert in Wan2.2 MoE; ignored if `transformer_2` not loaded.
- `num_inference_steps: int = 50` for Wan2.1; `40` for Wan2.2 A14B.
- `generator: torch.Generator` — for seeded reproducibility.
- `latents`, `prompt_embeds`, `negative_prompt_embeds`, `attention_kwargs`, `callback_on_step_end` — advanced.

Source: `WanPipeline.__call__`: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py>

### b) Output
`WanPipelineOutput.frames` — a list (one entry per `num_videos_per_prompt`). Each entry is a numpy array of shape `(num_frames, H, W, 3)` in uint8 / float depending on `output_type` ("np" default; "pt" tensor). Convention is to call `diffusers.utils.export_to_video(output, "out.mp4", fps=16)` to write an MP4.

### c) Native resolutions / frame counts
| Checkpoint | 480p | 720p | Frames (default) | FPS |
|---|---|---|---|---|
| Wan2.1-T2V-1.3B | 832×480 | — | 81 | 16 |
| Wan2.1-T2V-14B | 832×480 | 1280×720 | 81 | 16 |
| Wan2.2-T2V-A14B | 832×480 | 1280×720 | 81 | 16 (default in shared cfg) |

`sample_fps=16` from `wan/configs/shared_config.py` — but examples in the diffusers docs commonly export at `fps=16`. Wan2.2-Animate and Wan2.2-TI2V-5B override to **24 fps** (see those sections).

### d) Default sampler / steps / CFG (official Wan repo)
- Scheduler: `UniPCMultistepScheduler` (Wan2.1) — `flow_shift=5.0` for 720p, `flow_shift=3.0` for 480p. Cited: docstring of `WanPipeline.__call__`.
- Wan2.2-T2V-A14B: `sample_steps=40`, `sample_guide_scale=(3.0, 4.0)`, `sample_shift=12.0`, MoE `boundary=0.875`. Source: <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py>
- Wan2.2 uses MoE — `transformer` handles t ≥ `boundary * num_train_timesteps`, `transformer_2` handles t < that boundary.

### e) Supporting checkpoints
- `Wan-AI/Wan2.1-T2V-1.3B[-Diffusers]`
- `Wan-AI/Wan2.1-T2V-14B[-Diffusers]`
- `Wan-AI/Wan2.2-T2V-A14B[-Diffusers]`

### f) Reference pipeline class
`diffusers.WanPipeline` — supports both Wan2.1 single-transformer and Wan2.2 dual-transformer (set `transformer_2=…` and `boundary_ratio=0.875`).

### g) Quirks
- The VAE is `AutoencoderKLWan` and **must be loaded in `torch.float32`** for clean decoding — bfloat16 VAE produces artifacts (docs note: "Set the AutoencoderKLWan dtype to torch.float32 for better decoding quality." <https://huggingface.co/docs/diffusers/en/api/pipelines/wan>).
- Text encoder is `UMT5EncoderModel` (`google/umt5-xxl`) — large (~11 GB bf16), benefits from `enable_group_offload` or `enable_model_cpu_offload`.
- `num_frames` constraint: `4 * k + 1` (because Wan2.1 VAE has 4× temporal stride; Wan2.2-5B VAE has temporal stride 4 too — TI2V-5B default 121 = 4×30+1).

### h) Aspect ratios / orientation
Any 16:9 or 9:16 sized so that both H and W are multiples of `vae_scale_factor_spatial * patch_size[1]` (typically 8 for Wan2.1; 16 for Wan2.2-5B since the 5B VAE has spatial stride 16). The diffusers example uses `aspect_ratio_resize(image, pipe, max_area=720*1280)` to pick a legal H/W pair.

### i) Long-video extension
- **No official "extend" for pure T2V**. Users get 81 frames at 16 fps = ~5 s.
- Wan2.2-Animate has built-in multi-segment stitching via `segment_frame_length` + `prev_segment_conditioning_frames`, but that mechanism is **only inside `WanAnimatePipeline`**.
- For T2V, the community pattern is to feed the last frame of a T2V output as the first frame of an I2V continuation — but this is not officially a single pipeline.

---

## 2. I2V — Image-to-Video

### a) Inputs
- `image: PIL.Image | np.ndarray | torch.Tensor | list[…]` — required, single reference frame.
- `prompt: str` — required, describes motion.
- `negative_prompt`, `height`, `width`, `num_frames=81`, `num_inference_steps=50`, `guidance_scale=5.0` (Wan2.1) / `(3.5, 3.5)` Wan2.2 (low-noise, high-noise tuple in repo).
- `image_embeds: torch.Tensor` — precomputed CLIP embeddings if you want to bypass the image encoder.
- `last_image: torch.Tensor` — when set, this same class implements **FLF2V** (mode 3).
- `guidance_scale_2`, `generator`, `latents`, etc.

Source: `WanImageToVideoPipeline.__call__`: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py>

### b) Output
Same `WanPipelineOutput`-style frames object (`.frames[0]`); MP4 via `export_to_video`.

### c) Native resolutions / frame counts
| Checkpoint | Native rez | Frames | FPS |
|---|---|---|---|
| Wan2.1-I2V-14B-480P | 832×480 | 81 | 16 |
| Wan2.1-I2V-14B-720P | 1280×720 | 81 | 16 |
| Wan2.2-I2V-A14B | 832×480 + 1280×720 (same checkpoint) | 81 | 16 |
| Wan2.2-TI2V-5B | 1280×704 / 704×1280 | 121 | 24 |

The diffusers I2V docstring uses `max_area = 480 * 832` and resizes the input image accordingly: see the `aspect_ratio_resize` helper in the model card.

### d) Default sampler / steps / CFG
- Wan2.1-I2V-14B: `sample_steps=40` (per official repo), `sample_guide_scale=5.0`.
- Wan2.2-I2V-A14B: `sample_steps=40`, `sample_guide_scale=(3.5, 3.5)`, `sample_shift=5.0`, `boundary=0.900`. Source: <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_i2v_A14B.py>
- Scheduler `UniPCMultistepScheduler`.

### e) Checkpoints
- `Wan-AI/Wan2.1-I2V-14B-480P[-Diffusers]`
- `Wan-AI/Wan2.1-I2V-14B-720P[-Diffusers]`
- `Wan-AI/Wan2.2-I2V-A14B[-Diffusers]`

### f) Reference pipeline class
`diffusers.WanImageToVideoPipeline`. Requires an `image_encoder` (`CLIPVisionModel`, specifically `clip-vit-huge-patch14` / `xlm-roberta-large` per the docstring).

### g) Quirks
- For Wan2.1, two **separate** checkpoints exist for 480P vs 720P (different LoRA / different fine-tuning). For Wan2.2 there is one unified `A14B`.
- `image_encoder` must be loaded at `torch.float32` (per `pipe.image_encoder.from_pretrained(..., torch_dtype=torch.float32)` example).
- Frame count `4k+1` constraint same as T2V.

### h) Aspect ratios
Driven by the input image's aspect ratio — the helper rounds H,W to multiples of `vae_scale_factor_spatial * patch_size[1]`. Both portrait and landscape supported.

### i) Long-video extension
Same constraint as T2V — ~5s native. Studio will need to stitch externally.

---

## 3. TI2V — Text-Image-to-Video (Wan2.2 5B only)

### a) Inputs
Same call signature as `WanImageToVideoPipeline`. The novelty is the **5B model** + ultra-high-compression VAE (`Wan2.2-VAE`), which yields 121 frames at 24 fps from a 5B-param transformer on consumer-grade GPUs.

### b) Output
`WanPipelineOutput.frames[0]`, ~5 s at 24 fps = 121 frames.

### c) Native rez / frames / FPS
- 1280×704 or 704×1280 — these are the ONLY trained sizes per the repo (`wan/configs/wan_ti2v_5B.py`).
- 121 frames.
- 24 FPS (the 5B model overrides the shared default of 16).

### d) Default sampler / steps / CFG
- `sample_steps=50`, `sample_guide_scale=5.0`, `sample_shift=5.0`, `sample_fps=24`, `frame_num=121`, `vae_stride=(4, 16, 16)`. Source: <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py>

### e) Checkpoints
- `Wan-AI/Wan2.2-TI2V-5B[-Diffusers]`

### f) Reference pipeline class
Uses `WanImageToVideoPipeline` from diffusers; in the Wan repo it's task `ti2v-5B`.

### g) Quirks
- The **5B VAE has a 4×16×16 compression ratio** (vs 4×8×8 for Wan2.1 / Wan2.2 14B family). That is the key reason this can run on consumer GPUs — but it also means H and W must be multiples of 16×patch=32, and the 5B model can ONLY synthesise at the 1280×704 grid (or its 90°-rotated 704×1280 portrait counterpart). Other resolutions are not supported.
- Image input is optional — when omitted, this degrades gracefully to T2V.

### h) Aspect ratios
Landscape `1280×704` (≈16:9) and portrait `704×1280` only.

### i) Extension
No native extension. Same ~5 s cap.

---

## 4. FLF2V — First-Last-Frame-to-Video

### a) Inputs
- `image: PIL.Image` — first frame.
- `last_image: PIL.Image` — last frame.
- `prompt: str` — motion description. The Wan team recommends Chinese prompts for FLF2V for best quality (Wan2.1 readme note).
- Standard `negative_prompt`, `height`, `width`, `num_frames`, `num_inference_steps`, `guidance_scale=5.5` per the diffusers example.

### b) Output
Same `frames[0]` mp4 export pattern.

### c) Native rez / frames / FPS
- 720p only: `1280×720` (or portrait equivalent). The Wan2.1 repo only ships `Wan2.1-FLF2V-14B-720P`.
- 81 frames at 16 fps.

### d) Default sampler / steps / CFG
- `UniPCMultistepScheduler`, `flow_shift=5.0`, `guidance_scale=5.5` (per the diffusers example).
- `sample_steps=40` per the Wan repo conventions for I2V family.

### e) Checkpoints
- `Wan-AI/Wan2.1-FLF2V-14B-720P[-diffusers]`
- Wan2.2 family does **not** yet ship a dedicated FLF2V checkpoint — for Wan2.2 you'd reach for VACE's "First-Last-Frame" sub-mode instead.

### f) Reference pipeline class
`diffusers.WanImageToVideoPipeline` — pass the second image as `last_image=…`. The model card walks through `aspect_ratio_resize(first_frame, …)` then `center_crop_resize(last_frame, height, width)` so both end up identical-shaped. Source: <https://huggingface.co/docs/diffusers/en/api/pipelines/wan>

### g) Quirks
- Aspect ratio for the OUTPUT is taken from the **first** frame. The last frame is center-cropped to match — be aware that if your two frames have very different aspect ratios, content will be cropped.
- Image-encoder load (CLIPVisionModel) is mandatory — there's no text-only fallback.

### h) Aspect ratios
Whatever the first-frame's ratio resolves to after the rounding helper, up to a max area of 720×1280.

### i) Extension
None native — the two anchors fully constrain the video.

---

## 5. V2V — basic latent video-to-video (low-level / "image-to-image" for video)

### a) Inputs
- `video: list[PIL.Image]` — required.
- `prompt`, `negative_prompt`, `height=480`, `width=832`, `guidance_scale=5.0`, `num_inference_steps=50`, `strength: float = 0.8` (how much to deviate from the input video).
- No image-encoder, no last_image.

Source: `WanVideoToVideoPipeline.__call__`: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_video2video.py>

### b) Output
`frames[0]`.

### c) Native rez / frames / FPS
Inherits from underlying T2V checkpoint (Wan2.1-T2V-{1.3B,14B}); same 81 / 16fps defaults.

### d) Default sampler / steps / CFG
`UniPCMultistepScheduler`, `flow_shift=3.0` (480P) / 5.0 (720P), `guidance_scale=5.0`, `strength=0.7-0.8` (example uses 0.7).

### e) Checkpoints
Reuses the T2V checkpoints — `Wan-AI/Wan2.1-T2V-{1.3B,14B}-Diffusers`. There is no separate V2V checkpoint.

### f) Reference pipeline class
`diffusers.WanVideoToVideoPipeline`. Not advertised in the Wan repo's `generate.py` — exposed only via diffusers and via VACE-style sub-modes.

### g) Quirks
This is the diffusers-style "noise + restyle" V2V (a la SD's `Img2Img`), NOT the VACE controlled V2V. It encodes the input video into latents, adds noise proportional to `strength`, then re-denoises with a new prompt. Use it for restyling, not for structural control.

### h) Aspect ratios
Driven by the input video frames.

### i) Extension
No native extension. Same per-clip cap.

---

## 6. VACE — Versatile Animation Control & Editing (Wan2.1 only)

VACE is the single biggest mode in the family. It uses one pipeline class with one set of inputs (`video`, `mask`, `reference_images`) but supports **25+ sub-modes** by varying what you put in those slots. **Wan2.2 has NOT released a VACE-equivalent checkpoint** — VACE remains Wan2.1-only as of 2026-05-21. Source: <https://huggingface.co/Wan-AI> (no Wan2.2-VACE present).

### General

#### a) Inputs (common to all sub-modes)
- `prompt: str` (required) — UMT5-XXL.
- `negative_prompt: str`.
- `video: list[PIL.Image] | None` — the control video / source video / placeholder frames.
- `mask: list[PIL.Image] | None` — per-frame mask. Black = condition (keep), White = generate. Must match `len(video)`.
- `reference_images: list[PIL.Image] | None` — 1-3 reference images (faces, objects, subjects) for ID-preserving generation.
- `conditioning_scale: float | list[float] | torch.Tensor = 1.0` — applied to each VACE control layer; can be per-layer (length = `len(transformer.config.vace_layers)`).
- Standard `height=480`, `width=832`, `num_frames=81`, `num_inference_steps=50`, `guidance_scale=5.0`.

Source: `WanVACEPipeline.__call__`: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py>

#### b) Output
`frames[0]` — np frames -> mp4.

#### c) Native rez / frames / FPS
- 1.3B model: 480P only (~81×480×832).
- 14B model: 480P and 720P (~81×720×1280).
- 16 fps.

#### d) Default sampler / steps / CFG
`UniPCMultistepScheduler`, `flow_shift=3.0` (480P) / 5.0 (720P), `guidance_scale=5.0`, `num_inference_steps=30-50` (the diffusers example uses 30).

#### e) Checkpoints
- `Wan-AI/Wan2.1-VACE-1.3B[-diffusers]`
- `Wan-AI/Wan2.1-VACE-14B[-diffusers]`

#### f) Reference pipeline class
`diffusers.WanVACEPipeline`. Single class, sub-modes are configuration of `video`/`mask`/`reference_images` slots. Diffusers integration PR: <https://github.com/huggingface/diffusers/pull/11582>

#### g) Quirks (the big one)
The pre-processing step that turns your raw input into `video` + `mask` is **not in diffusers**. The official tool is `vace_preprocess.py` in <https://github.com/ali-vilab/VACE>; the annotators live under `vace/annotators/` and depend on:

| Annotator | Underlying model | Library / install |
|---|---|---|
| `DepthVideoAnnotator` | **MiDaS** (variant `dpt_hybrid`) | bundled in VACE repo as `vace/annotators/midas` |
| `PoseBodyFaceVideoAnnotator` | **DWPose** (Wholebody: body + face + hands) | bundled `vace/annotators/dwpose`. NB: this is the same DWPose used everywhere (RTMPose backbone + body-pose checkpoint). |
| `FlowVisAnnotator` | **RAFT** | `git+https://github.com/martin-chobanyan-sdc/RAFT.git` per `requirements/annotator.txt` |
| `GrayVideoAnnotator` | plain numpy greyscale | (no model) |
| `ScribbleVideoAnnotator` | HED / PiDi (community variant) | bundled in VACE annotators |
| `LayoutBboxAnnotator` / `LayoutTrackAnnotator` | linear interp + tracking (SAM2 / GroundingDINO for tracking variants) | `sam2`, `segment-anything`, `GroundingDINO` from `requirements/annotator.txt` |
| `InpaintingAnnotator` (all variants) | uses SAM2 + GroundingDINO + recognize-anything for mask-tracking / label / caption flows | same as above |
| `OutpaintingVideoAnnotator` | pure compositor (no neural model) | (no model) |
| `SubjectAnnotator` (Face / Object reference) | **InsightFace** for face crops | `insightface` |
| `FrameRefExpandAnnotator` | replication / blanking | (no model) |
| `ReferenceAnythingAnnotator`, `AnimateAnythingAnnotator`, `SwapAnythingAnnotator`, `ExpandAnythingAnnotator`, `MoveAnythingAnnotator` | composition of the above (SAM2 + GroundingDINO + InsightFace) | composite of the above |

Source: <https://github.com/ali-vilab/VACE/blob/main/requirements/annotator.txt>, `vace/annotators/pose.py`, `vace/annotators/depth.py`.

> **Studio implementation note**: shipping all of VACE's annotator deps in a ZeroGPU Space adds ~5 GB of model weights and several CUDA-only deps (`sam2`, `GroundingDINO`, `insightface`'s onnx models). For ZeroGPU, prefer to (a) accept user-pre-extracted control videos (depth/pose maps) directly in the UI, OR (b) only ship the lightweight subset: DWPose + MiDaS + RAFT.

#### h) Aspect ratios
Same as T2V/I2V — `vae_scale_factor_spatial * patch_size[1]` multiples. Driven by the supplied `video` shape.

#### i) Extension
VACE has explicit **Frame/Clip Extension** sub-modes (see below). Plus the diffusers PR notes that VACE can do start-clip → middle → end-clip stitching.

### 6.1 — VACE Sub-mode: **Depth-driven V2V** (control)
- Inputs: pre-extracted depth-map video as `video=`, all-white masks, prompt.
- Preprocessor: `DepthVideoAnnotator` → MiDaS dpt_hybrid.
- Use case: control structure but freely re-style appearance.

### 6.2 — VACE Sub-mode: **Pose-driven V2V** (control)
- Inputs: pre-extracted pose-skeleton video as `video=`, all-white masks, prompt + optional `reference_images` for character ID.
- Preprocessor: `PoseBodyFaceVideoAnnotator` → DWPose Wholebody (body + face + hands).
- Use case: animate a character (defined by reference) to perform the motion in a driving video. **Note**: this overlaps with Wan-Animate but VACE is on Wan2.1; Wan-Animate is on Wan2.2 — see §8 for the comparison.

### 6.3 — VACE Sub-mode: **Optical-flow V2V** (control)
- Preprocessor: `FlowVisAnnotator` → RAFT.
- Use case: motion-pattern transfer.

### 6.4 — VACE Sub-mode: **Grayscale V2V** (control)
- Preprocessor: `GrayVideoAnnotator` (no neural model).
- Use case: structure-preserving colorisation.

### 6.5 — VACE Sub-mode: **Scribble V2V** (control)
- Preprocessor: `ScribbleVideoAnnotator` (HED-like edge maps).
- Use case: sketch-driven generation.

### 6.6 — VACE Sub-mode: **Layout-BBox** (control)
- Inputs: two static bboxes (start + end), interpolated linearly.
- Preprocessor: `LayoutBboxAnnotator` (pure geometry, no model).
- Use case: "move object from A to B" without supplying a driving video.

### 6.7 — VACE Sub-mode: **Layout-Track** (control)
- Inputs: video + tracking spec (mask-track, bbox-track, label, or caption).
- Preprocessor: `LayoutTrackAnnotator` → SAM2 + GroundingDINO (label/caption variants).

### 6.8 — VACE Sub-mode: **Inpainting — Mask**
- Inputs: source video + per-frame static masks (you provide the masks).
- Preprocessor: `InpaintingAnnotator` (passes through; you supply the mask).

### 6.9 — VACE Sub-mode: **Inpainting — BBox**
- Inputs: source video + a single bbox; the annotator builds masks.
- Preprocessor: `InpaintingAnnotator`.

### 6.10 — VACE Sub-mode: **Inpainting — MaskTrack**
- Inputs: source video + initial mask (auto-propagated through time).
- Preprocessor: `InpaintingAnnotator` + SAM2 (for tracking).

### 6.11 — VACE Sub-mode: **Inpainting — BBoxTrack**
- Like MaskTrack but starts from a bbox.

### 6.12 — VACE Sub-mode: **Inpainting — Label**
- Inputs: source video + a class label string ("dog", "car"). Annotator runs GroundingDINO to detect, SAM2 to segment+track.

### 6.13 — VACE Sub-mode: **Inpainting — Caption**
- Like Label but with a free-form caption.

### 6.14 — VACE Sub-mode: **Outpainting**
- Inputs: source video + direction + expansion ratio (e.g. "left=0.3, right=0.3, up=0, down=0").
- Preprocessor: `OutpaintingVideoAnnotator` (compositor).

### 6.15 — VACE Sub-mode: **Reference — Face**
- Inputs: 1 face image as `reference_images=[face]`, plus prompt.
- Preprocessor: `SubjectAnnotator` → InsightFace for crop/embed.

### 6.16 — VACE Sub-mode: **Reference — Object**
- 1 object image as reference.

### 6.17 — VACE Sub-mode: **Extension — First-Frame**, **Last-Frame**, **First-Last-Frame**, **First-Clip**, **Last-Clip**, **First-Last-Clip**
- Inputs: 1 or 2 images/clips to anchor the start/end/middle of generation.
- Preprocessor: `FrameRefExpandAnnotator` — replicates anchor frames, blanks (grey 128) the rest, and builds black/white masks accordingly.
- See the diffusers VACE docstring `prepare_video_and_mask(first_img, last_img, height, width, num_frames)` helper for an explicit implementation of First-Last-Frame.

### 6.18 — VACE Sub-mode: **Reference-Anything**
- 1-3 reference images (faces + objects), free-form prompt.
- Preprocessor: composite of SubjectAnnotator + FrameRefExpand.

### 6.19 — VACE Sub-mode: **Animate-Anything**
- Reference image + driving video. Animate the subject in the reference to perform actions from the driving video.
- Preprocessor: `AnimateAnythingAnnotator` (SAM2 mask of subject + DWPose of driver).

### 6.20 — VACE Sub-mode: **Swap-Anything**
- Reference image + source video. Swap the subject identity while preserving the scene.
- Preprocessor: `SwapAnythingAnnotator` (SAM2 + InsightFace).

### 6.21 — VACE Sub-mode: **Expand-Anything**
- Reference image + reference-image-list. Used for canvas expansion with content references.

### 6.22 — VACE Sub-mode: **Move-Anything**
- Reference image + two bboxes.

Source for the full sub-mode list: <https://github.com/ali-vilab/VACE/blob/main/UserGuide.md>

---

## 7. S2V — Speech-to-Video (Wan2.2 only, NOT in diffusers yet)

### a) Inputs
- `--audio`: WAV or MP3 file. Sample rate **16 kHz** (Wav2Vec2 standard; resampled with `librosa` in `audio_encoder.py`).
- `--image`: reference character image (JPG/PNG). Determines output aspect ratio.
- `--prompt`: text prompt — required, describes scene/style around the speaker.
- `--pose_video` (optional): driving pose video to combine with audio.
- `--num_clip` (optional): how many ~5s clips to chain. If omitted, **the generated video length auto-adjusts to the input audio length** — this is the only Wan mode with native variable-length output.
- `--size`: `1024*704` is the example; supported pair area is 480P or 720P.

Source: <https://huggingface.co/Wan-AI/Wan2.2-S2V-14B>

### b) Output
MP4 file at 24 fps, lip-synced to the audio. The official script writes to disk; there's no diffusers in-memory return path (yet).

### c) Native rez / frames / FPS
- 480P and 720P. Example: `1024×704`. Aspect ratio matches input image.
- **24 fps** (the audio encoder interpolates to 30 fps internally — see `video_rate = 30` in `audio_encoder.py` — but the sampled output is 24 fps per `wan_s2v_14B` config).
- Frame count: variable; auto-chunked by audio length.

### d) Default sampler / steps / CFG
- `sample_steps=40`, `sample_guide_scale=4.5`, `sample_shift=3`. Source: <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_s2v_14B.py>
- Scheduler: same UniPC family as the rest of Wan2.2.

### e) Checkpoints
- `Wan-AI/Wan2.2-S2V-14B` (raw, official).
- **No `-Diffusers` variant exists** as of 2026-05-21.

### f) Reference pipeline class
**Not in diffusers.** Use the official inference script: `python generate.py --task s2v-14B --ckpt_dir ./Wan2.2-S2V-14B/ --image ... --audio ... --prompt ...` from <https://github.com/Wan-Video/Wan2.2>. To integrate into a Gradio Studio you have two options:
1. Shell-out to `generate.py` (simplest, but harder to ZeroGPU-ify since ZeroGPU expects a `@spaces.GPU`-decorated function).
2. Import `wan.s2v` and call `WanS2V(...)` from the repo directly — the repo's `Wan2.2/wan/__init__.py` exposes the model class. Cleaner for ZeroGPU.

### g) Special inference quirks — audio backbone (CRITICAL for packaging)
The S2V model **bundles its own audio encoder inside the HF repo**:
- Path: `Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/` — sub-folder with `model.safetensors` (~1.26 GB), `config.json`, `preprocessor_config.json`, etc.
- `config.json` shows `_name_or_path: "facebook/wav2vec2-large-xlsr-53"` — the bundle is the official Facebook XLSR-53 large model (1024 hidden, 24 layers, 16 attn heads), packaged together with the diffusion weights.
- The code default in `wan/modules/s2v/audio_encoder.py` points at `facebook/wav2vec2-base-960h` — but at runtime the script loads the bundled local sub-folder, **NOT** the base-960h. **For packaging:** if you download `Wan-AI/Wan2.2-S2V-14B` you get the audio encoder for free; no separate download.
- Sample rate hard-coded to 16 kHz (`librosa.load(..., sr=16000)`).
- Hidden-state features (not CTC logits) are extracted, then linearly interpolated to 30 fps to align with the video latents.
- Optional `requirements_s2v.txt` adds CosyVoice for built-in TTS (so you can feed text → CosyVoice → S2V end-to-end). Studio can offer this as a "Text-to-Talking-Head" convenience mode.

Sources: <https://huggingface.co/Wan-AI/Wan2.2-S2V-14B/tree/main>, <https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/s2v/audio_encoder.py>, <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_s2v_14B.py>

### h) Aspect ratios
Driven by the input reference image. The example `1024×704` is ~3:2 landscape; portrait is supported.

### i) Long-video extension
**Yes — natively.** S2V auto-chains via `num_clip` (or auto-derives from audio length). This makes it the only Wan mode with first-class long-form video generation. Each clip is conditioned on the prior clip's tail frames (overlap-and-stitch).

---

## 8. Wan-Animate — Unified Character Animation & Replacement (Wan2.2 only)

### a) Inputs
Mode is selected via `mode: str = "animate" | "replace"`.

**Animation mode** (default):
- `image: PIL.Image` — character reference (single image).
- `pose_video: list[PIL.Image]` — **pre-processed** pose video (skeletons drawn frame-by-frame). Not raw RGB.
- `face_video: list[PIL.Image]` — **pre-processed** face video (cropped face regions, used for facial-expression replication via implicit features).
- `prompt: str` (optional in `animate` mode).
- `negative_prompt`.

**Replacement mode** (in addition):
- `background_video: list[PIL.Image]` — original RGB scene.
- `mask_video: list[PIL.Image]` — black = preserve original, white = generate new character.

**Both modes:**
- `height=720`, `width=1280` (defaults — see `WanAnimatePipeline.__call__`).
- `segment_frame_length: int = 77` — frames per stitched segment.
- `prev_segment_conditioning_frames: int = 1` — overlap frames for temporal continuity (1 or 5 recommended; 5 = smoother but more VRAM).
- `num_inference_steps: int = 20` (per `wan_animate_14B.py`).
- `guidance_scale: float = 1.0` (CFG **disabled** by default — Wan-Animate is trained without classifier-free guidance; you can re-enable with `guidance_scale > 1` to use the negative prompt).
- `motion_encode_batch_size: int | None` — for VRAM-constrained machines.

Source: `WanAnimatePipeline.__call__`: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_animate.py>

### b) Output
`frames[0]` — np frames at 30 fps. Note: the diffusers example exports at `fps=30`; the repo config `wan_animate_14B.py` sets `sample_fps=30`. **This is the only Wan2.2 mode at 30 fps.**

### c) Native rez / frames / FPS
- 480p and 720p (default `1280×720`); also accepts other sizes that respect VAE × patch grid.
- **Multi-segment**: arbitrary length via stitching. Effective segment = `segment_frame_length − prev_segment_conditioning_frames` = 76 frames. Total generated frames = `ceil(len(pose_video) / 76) × 76 + 1`.
- 30 FPS.

### d) Default sampler / steps / CFG
- `sample_steps=20`, `sample_guide_scale=1.0`, `sample_shift=5.0`, `sample_fps=30`, `frame_num=77`. Source: <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_animate_14B.py>
- Scheduler: same UniPC family.

### e) Checkpoints
- `Wan-AI/Wan2.2-Animate-14B` (raw).
- `Wan-AI/Wan2.2-Animate-14B-Diffusers` (diffusers).
- Plus the bundled `Wan2.2-Animate-14B/process_checkpoint/` directory which contains:
  - `pose2d/vitpose_h_wholebody.onnx` — **ViTPose Huge wholebody** (NOT DWPose; this is the only Wan mode that uses ViTPose).
  - `det/yolov10m.onnx` — YOLOv10-Medium for person detection.
  - `sam2/sam2_hiera_large.pt` — SAM2 Hiera Large for mask extraction in replacement mode.
  - Optional Flux model for off-axis pose retargeting (`--use_flux`).

Source: <https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/preprocess_data.py>

### f) Reference pipeline class
`diffusers.WanAnimatePipeline`. **Caveat from the diffusers docstring**: "Raw videos should not be used for inputs such as `pose_video`, which the pipeline expects to be preprocessed to extract the proper information. Preprocessing scripts to prepare these inputs are available in the [original Wan-Animate repository](https://github.com/Wan-Video/Wan2.2). Integration of these preprocessing steps into Diffusers is planned for a future release."

So the Studio MUST either:
1. Bundle the Wan2.2 repo's `preprocess_data.py` script and the three ONNX/PyTorch model files above (~2 GB of ViTPose + YOLOv10 + SAM2 weights), OR
2. Use `diffusers`'s upcoming preprocessor when it lands, OR
3. Accept user-uploaded pre-processed pose+face videos.

### g) Quirks
- **Animation vs Replacement preprocessing differ**:
  - Animation: needs pose + face videos only. Has `--retarget_flag` and optional `--use_flux` (for non-front-facing poses, Flux performs an image edit to make retargeting cleaner).
  - Replacement: additionally needs background + mask videos. Uses iterative mask-tightening (`--iterations 3 --k 7 --w_len 1 --h_len 1 --replace_flag`).
- **Relighting LoRA**: There's an auxiliary LoRA for replacement mode (`--use_relighting_lora`) that adapts the new character's lighting to the original scene. **Do not** use it in animation mode (the model card warns the LoRA can cause unexpected behavior in animate).
- **Multi-segment stitching is built-in**: Wan-Animate is the one Wan2.2 mode where the pipeline itself handles long-form output. Default segment is 77 frames; with `prev_segment_conditioning_frames=5` you get smoother transitions at the cost of more VRAM.
- `motion_encode_batch_size` — control face-feature extraction batch size when VRAM is tight.

### h) Aspect ratios
Default 1280×720 landscape; the `aspect_ratio_resize` helper in the model card rounds H,W to multiples of `vae_scale_factor_spatial * patch_size[1]` (typically 16 for Wan2.2). Portrait fully supported.

### i) Long-video extension
**Yes — natively** via the segment loop. There is no documented hard cap; practical limit is your VRAM × time budget.

### Difference vs VACE pose-driven (§6.2)
| Aspect | VACE Pose | Wan-Animate Animation |
|---|---|---|
| Base model | Wan2.1 (1.3B or 14B) | Wan2.2 (14B) |
| Pose extractor | DWPose (Wholebody) | ViTPose-H Wholebody |
| Face | included in DWPose | **separate face video** with implicit feature extraction |
| Lip sync to audio | no | no (uses motion only — for audio sync, use S2V) |
| Multi-segment stitching | no (one 81-frame clip) | yes (built into the pipeline) |
| Replacement mode | via SAM2 + Swap-Anything sub-mode | first-class (mode="replace") + Relighting LoRA |
| Diffusers support | yes (`WanVACEPipeline`) | yes (`WanAnimatePipeline`) |
| Output FPS | 16 | 30 |

---

## 9. Edit / Inpaint / Outpaint

Not exposed as standalone Wan modes — see VACE §6.8 – 6.14 (Inpainting variants) and §6.14 (Outpainting). All routed through `WanVACEPipeline` on Wan2.1 checkpoints.

Wan2.2 does NOT currently have a dedicated inpaint/outpaint mode. If the Studio wants Wan2.2-quality inpainting, the only option is to extend `WanVideoToVideoPipeline` with a mask kwarg (community PRs exist), or fall back to Wan2.1-VACE.

---

## 10. T2I — Text-to-Image (Wan2.1 unified, niche)

- Task: `t2i-14B` in the Wan2.1 repo.
- Uses the SAME T2V-14B checkpoint with `--frame_num 1` (single frame, no temporal denoise).
- Recommended size `1024*1024`.
- In diffusers, just call `WanPipeline(prompt=..., num_frames=1).frames[0][0]` and save as PNG.

Source: <https://github.com/Wan-Video/Wan2.1>

This is unlikely to be a primary Studio mode (you have purpose-built image models), but the Studio could expose it as a "preview frame" feature.

---

## 11. Future / partial — Wan 2.5, Wan 2.6, Wan 2.7

As of 2026-05-21:

| Generation | Status | Open weights? | New modes? |
|---|---|---|---|
| **Wan 2.5** | API-first (Alibaba cloud); only the VAE component on HF | `wangkanai/wan25-vae` is community-mirrored; no transformer | adds native audio generation (synchronized voice + ambient + music) and 1080p; 4K in preview; supports T2V, I2V, A2V (audio-to-video), V2V, T2I, image editing, **video extend** |
| **Wan 2.6** | Released Dec 2025; partially open | Diffusion weights not yet on HF Wan-AI org | adds multi-shot storytelling, 5-15 s extended duration, **R2V (Reference-to-Video) extracting movement + voice + appearance from a reference video**, 16:9/9:16/1:1/4:3/3:4 |
| **Wan 2.7** | Speculative / pre-release | Unknown | Speculation only — flaq.ai and wavespeed posts suggest further extension of Wan 2.6 |

Sources: <https://www.mindstudio.ai/blog/what-is-wan-2-5-video-open-source>, <https://www.mindstudio.ai/blog/what-is-wan-2-6-video-open-source>, <https://huggingface.co/wangkanai/wan25-vae>

**Implication for the Studio**: build for Wan 2.1 + 2.2 today. Wire in a "model family" enum so Wan 2.5/2.6 transformer weights can be slotted in when they drop on the Wan-AI HF org. The new modes to plan for are **A2V** (similar to S2V but with full lip-sync + ambient audio output, not just input-driven), **video extend** (Wan 2.5 native), and **R2V** (Wan 2.6).

---

## Studio implementation cheat-sheet

For each mode the Studio will expose, here's the bare minimum it must do.

| Mode | Pipeline | Custom pre-process | Custom post-process | ZeroGPU cost (~) |
|---|---|---|---|---|
| T2V | `WanPipeline.from_pretrained(...)` then `.to(cuda)` then call | none | `export_to_video` | ~13 GB VRAM, ~3-5 min on H100 |
| I2V | `WanImageToVideoPipeline` + CLIPVisionModel image encoder | `aspect_ratio_resize` helper | `export_to_video` | similar to T2V |
| TI2V (5B) | `WanImageToVideoPipeline` on `Wan2.2-TI2V-5B-Diffusers` | none | `export_to_video` | ~9 GB VRAM, fast |
| FLF2V | `WanImageToVideoPipeline` with `last_image=...` | aspect_ratio_resize on first frame, center_crop_resize on last | `export_to_video` | ~13 GB VRAM |
| V2V (restyle) | `WanVideoToVideoPipeline` | none | `export_to_video` | similar to T2V |
| VACE — control sub-modes | `WanVACEPipeline` + ship DWPose + MiDaS + RAFT annotators | run annotator over user-uploaded video to produce `video`/`mask` | export | ~14-22 GB VRAM, plus ~3 GB for annotators |
| VACE — inpaint/outpaint/extension | `WanVACEPipeline` | mask creation helpers (`prepare_video_and_mask` from docstring) | export | same |
| VACE — reference / animate-anything / swap-anything | `WanVACEPipeline` + SAM2 + GroundingDINO + InsightFace | composite annotators | export | adds ~5 GB extras |
| S2V | NO diffusers — wrap `Wan-Video/Wan2.2/wan` module directly. Use bundled `wav2vec2-large-xlsr-53-english`. | resample audio to 16 kHz with `librosa` | merge audio track back into mp4 with `ffmpeg` | ~22-30 GB VRAM (S2V is heaviest), supports very long clips |
| Wan-Animate | `WanAnimatePipeline` + bundle `vitpose_h_wholebody.onnx`, `yolov10m.onnx`, `sam2_hiera_large.pt` (~2 GB) | run `preprocess_data.py` to get pose_video + face_video (+ mask_video for replace) | export at fps=30 | ~18-25 GB; multi-segment loop dominates wall time |

---

## Summary — 3 most surprising findings

1. **S2V is NOT in diffusers and bundles its own wav2vec2 inside the HF model repo.** The `Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/` sub-folder is the audio encoder — Facebook's XLSR-53 large model (multilingual, fine-tuned on English data per `eval.py` in the bundle). Sample rate 16 kHz, hidden states (not CTC logits) interpolated to 30 fps. This means: download the S2V repo and you have everything; no separate `transformers`-style `from_pretrained("facebook/...")` call.
2. **Wan-Animate uses ViTPose-H wholebody, NOT DWPose** (which the rest of the ecosystem, including VACE, uses). It also bundles YOLOv10-Medium + SAM2 Hiera Large for preprocessing — a ~2 GB ONNX/PyTorch checkpoint trio that is **not in diffusers** yet. The diffusers docstring explicitly warns: "Preprocessing scripts to prepare these inputs are available in the original Wan-Animate repository. Integration of these preprocessing steps into Diffusers is planned for a future release."
3. **VACE has 25+ sub-modes but only Wan2.1 ships VACE weights** — Wan2.2 has no VACE checkpoint as of 2026-05-21. For Wan2.2 you get equivalent pose-driven animation via Wan-Animate, but you lose depth/flow/sketch/inpaint/outpaint/reference-anything/swap-anything etc. If the Studio needs structural control (depth, sketch, flow, etc.) it has to drop down to Wan2.1-VACE. The Wan2.2-5B's tiny 4×16×16 VAE compresses the latent so aggressively that adding a VACE control branch isn't a straightforward port.
