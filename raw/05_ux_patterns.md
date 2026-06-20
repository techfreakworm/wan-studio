# Wan Studio — Gradio UX Patterns & Reference-Space Study

Research date: 2026-05-21
Scope: UX architecture only — model selection, mode coverage, ZeroGPU, and Lightning LoRA work is owned by other agents.
Target stack: Gradio 5.x on Hugging Face ZeroGPU (Wan 2.1, 2.2, 2.5/2.6 generations).

---

## 0. Methodology

- Visited each Space with Playwright, captured a full-page screenshot, and pulled the source `app.py` (when reachable) to extract the real Gradio component graph.
- Saved screenshots to `/Users/techfreakworm/Projects/llm/wan-studio/raw/space_screenshots/`.
- Verified Gradio component APIs against the May 2026 docs (Gradio 5.x: `gr.Sidebar`, `Blocks.route()`, native navbar, streaming `gr.Video`).
- Tested mobile responsiveness by resizing the Playwright viewport to a 390x844 iPhone-class viewport against the Wan-AI S2V Space.

---

## 1. Reference Space study

Each entry: URL, layout, mode exposure, model-picker placement, parameter-panel pattern, video output, progress UX, mobile behavior, screenshot.

### 1.1 Wan-AI/Wan2.1 (official multi-mode demo)
- URL: https://huggingface.co/spaces/Wan-AI/Wan2.1
- Screenshot: `raw/space_screenshots/wan-ai-2.1.png`
- Layout: **`gr.Tabs`** inside a single `gr.Row`. Left column = tabs (T2V + I2V), right column = output + status panel.
- Mode exposure: One `gr.TabItem` per mode (`"Text to Video"`, `"Image to Video"`). Tab `.select()` events toggle visibility of mode-specific Examples blocks. A `gr.State(value="t2v")` tracks the active mode.
- Model picker: **Implicit per tab** — no global model dropdown. Resolution is a `gr.Dropdown` inside the T2V tab (`["1280*720", "960*960", "720*1280", "1088*832", "832*1088"]`); I2V has no resolution control.
- Parameter panel: Sparse and per-tab; T2V has only a 19-line prompt textbox + resolution dropdown. I2V has image upload + 5-line prompt. **No accordion** for advanced params on this Space — it offloads everything to the backend.
- Video output: `gr.Video(label="Generated Video", interactive=False, height=500)`. No autoplay/loop set explicitly.
- Progress UX: **Polling-based** — `gr.Number(label="Cost Time(secs)")`, `gr.Number(label="Estimated Waiting Time(secs)")`, a `gr.Slider` used as a 0-100 progress meter, and a hidden "Refresh Generating Status" button. Task ID stored in `gr.State`.
- Mobile: Hugging Face's iframe shrinks but the two-column row stacks vertically; tabs remain usable.

### 1.2 Wan-AI/Wan2.2-S2V (audio-driven cinematic)
- URL: https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V
- Screenshot: `raw/space_screenshots/wan-ai-s2v.png` (desktop), `wan-ai-s2v-mobile.png` (mobile)
- Layout: Single-page, single column. Labels are bilingual (`"Input image(输入图像)"`).
- Mode exposure: Single-purpose Space — no mode switching.
- Model picker: None visible (banner notes a distilled model is used). A `Resolution(分辨率)` `listbox` is the only model-related control.
- Parameter panel: Three stacked inputs (image, audio, resolution) then a single `Generate Video(生成视频)` button. No accordion.
- Video output: `Output Video(输出视频)` slot — empty placeholder until generation.
- Progress UX: Examples table (9 audio clips) doubles as a "queue feeder" — one-click prefill. No explicit progress UI surfaced.
- Mobile: Mobile screenshot confirms the page reflows cleanly to a single column — input image + audio + button stack vertically; the Examples table becomes horizontally scrollable.

### 1.3 Wan-AI/Wan2.2-Animate (currently paused, source-extracted)
- URL: https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate
- Screenshot: `raw/space_screenshots/wan-ai-animate.png` (shows "Space paused" page)
- Layout (from source): `gr.Blocks` -> `gr.Row` -> two columns. Left = inputs, right = output.
- Mode exposure: **`gr.Dropdown` for mode** — `"wan2.2-animate-move"` (drive reference with template motion) vs `"wan2.2-animate-mix"` (replace template character with reference). Second `gr.Dropdown` for quality (`wan-pro` = 25fps 720p, `wan-std` = 15fps 720p).
- Model picker: The mode dropdown doubles as model picker.
- Parameter panel: `gr.Image` (reference), `gr.Video` (template), two dropdowns (mode + quality), generate button. No prompt because Animate is driven by the template video.
- Video output: `gr.Video` + `gr.Textbox` for status.
- Progress UX: Status textbox prints `"SUCCEEDED"` or error.
- Examples: 4 example sets.

### 1.4 alexnasa/Wan2.2-Animate-ZEROGPU (ZeroGPU community variant)
- URL: https://huggingface.co/spaces/alexnasa/Wan2.2-Animate-ZEROGPU
- Screenshot: `raw/space_screenshots/alexnasa-wan22-animate.png`
- Layout: 3-column workflow inside a single `gr.Column(elem_id="col-container")`. Cols = (1) input video + duration slider, (2) reference image + mode radio, (3) output + advanced trigger.
- Mode exposure: Custom `RadioAnimated` (animated pill) for **mode** (`"Character Swap"` / `"Pose Retarget"`) **and** for **resolution** (`"Low Res"` / `"Medium Res"`).
- Model picker: Effectively merged with mode radio.
- Parameter panel: Duration slider (1-20s). The big win is a **dynamic time estimate**: `gr.Text(value="⌚ Zero GPU Required: ~110.0s (1.8 mins)")` updates on every duration/mode/resolution change.
- Video output: Main `gr.Video` plus an `gr.Accordion("Processing Outputs 🎭")` exposing 4 intermediate videos (pose, background, face, mask) for debugging.
- Progress UX: `gr.Progress(track_tqdm=True)` for live tqdm bridging. The headline pattern: **estimated GPU seconds shown before submit**, computed via a `get_duration()` function that the `@spaces.GPU(duration=get_duration, size='large')` decorator also reads from — so display time and actual reservation stay in sync.
- Extra: A `gr.Column(visible=False, elem_id="fake-modal")` is toggled to act as a SAM2-based mask-painting "Pro" modal.

### 1.5 multimodalart/wan2-1-fast (Lightning-distilled I2V)
- URL: https://huggingface.co/spaces/multimodalart/wan2-1-fast
- Screenshot: `raw/space_screenshots/multimodalart-wan21-fast.png`
- Layout: `gr.Row` with two columns. Left = inputs, right = output.
- Mode exposure: Single-mode (I2V).
- Model picker: None — implicit.
- Parameter panel: Input image, prompt, duration slider (0.3-3.4 s), then a collapsed **`gr.Accordion("Advanced Settings")`** with negative prompt, seed + randomize, height/width sliders (128-896 step 32), inference steps (1-30 default 4), and a hidden CFG slider. This is the **cleanest "fast" preset UX** in the entire study: distilled defaults are tuned for `steps=4`.
- Video output: `gr.Video(label="Generated Video", autoplay=True, interactive=False)`.
- Progress UX: Implicit via `gr.Progress(track_tqdm=True)` in the function signature.
- Examples: 2 cached examples with `cache_mode="lazy"`.

### 1.6 multimodalart/wan-2-2-first-last-frame (FLF2V)
- URL: https://huggingface.co/spaces/multimodalart/wan-2-2-first-last-frame
- Screenshot: `raw/space_screenshots/multimodalart-wan22-flf.png`
- Layout: Two-column row.
- Mode exposure: Single-mode (FLF2V) but the end-frame slot is itself **tabbed** — `gr.Tabs` with `"Upload"` and `"Generate"` sub-tabs. `"Generate"` exposes a `Generate scene 5 seconds in the future` button that synthesizes the end frame.
- Model picker: None.
- Parameter panel: Start `gr.Image(type="pil", sources=["upload","clipboard"])`, end-frame tabs, prompt, then `gr.Accordion("Advanced Settings")` with duration (0.5-5.1 s), negative prompt, two CFG sliders (high noise / low noise — Wan 2.2 dual-noise MoE), inference steps default 8, seed + randomize.
- Video output: `gr.Video(label="Generated Video", autoplay=True)`.
- Progress UX: implicit.
- Examples: 3 cached examples, lazy cache.
- **Pattern worth stealing:** the nested `gr.Tabs` for "upload vs auto-generate" sub-flows inside one mode.

### 1.7 linoyts/wan2-1-VACE-fast (multi-input multimodal control)
- URL: https://huggingface.co/spaces/linoyts/wan2-1-VACE-fast
- Screenshot: `raw/space_screenshots/linoyts-vace-fast.png`
- Layout: Two-column row.
- Mode exposure: A single VACE pipeline but **sub-modes via `gr.Radio`** — `"Reference"`, `"First - Last Frame"`, `"Random Transitions"`. The radio updates the default prompt and toggles visibility of a `"Remove Background"` checkbox.
- Model picker: None — single VACE checkpoint.
- Parameter panel: `gr.Gallery` for multi-image input (3 cols x 2 rows), mode radio, conditional `Remove Background` checkbox, prompt, duration slider (0.5-5.06 s mapped to 8-81 frames at 16 fps), then accordion with neg-prompt, seed, H/W (128-896 step 32), steps (1-10 default 6), hidden CFG.
- Video output: `gr.Video(label="Generated Video", autoplay=True, interactive=False)`.
- Progress UX: implicit.
- **Pattern worth stealing:** mode radio with conditional visibility (the bg-removal checkbox only appears in `"Reference"` mode). The dimension auto-calc on gallery upload.

### 1.8 multimodalart/self-forcing (streaming I2V)
- URL: https://huggingface.co/spaces/multimodalart/self-forcing
- Screenshot: `raw/space_screenshots/multimodalart-self-forcing.png`
- Layout: Two-column row, **scale=2 left vs scale=3 right** so the video output dominates.
- Mode exposure: Single-mode.
- Model picker: None.
- Parameter panel: Prompt textbox with an `"✨ Enhance Prompt"` button (Qwen3-8B rewrite), then a `gr.Number(seed)` and `gr.Slider(fps, 1-30)`. Tiny settings surface.
- Video output: **`gr.Video(streaming=True, loop=True)`** — true frame-streamed playback as the generator yields `.ts` chunks.
- Progress UX: Custom HTML progress bar drawn in `gr.HTML`, updated every yielded frame: `<div>Block {idx+1}/{num_blocks} | Frame {n} | {percent}%</div>`. Generator function yields `(ts_path, status_html)` tuples; status HTML updates per-frame, the video chunk yields per-block.
- **Pattern worth stealing:** custom HTML progress card next to a streaming `gr.Video`. This is the gold-standard live UX for diffusion video.

### 1.9 fffiloni/Wan2.1 (legacy fffiloni Space)
- URL: https://huggingface.co/spaces/fffiloni/Wan2.1
- Screenshot: `raw/space_screenshots/fffiloni-wan21.png`
- Layout: Single-column `gr.Blocks` -> single `gr.Column`.
- Mode exposure: Single (`task = "t2v-1.3B"` hardcoded).
- Model picker: **GPU-availability banners** rendered via `gr.HTML` with `elem_id="warning-duplicate"` / `"warning-ready"` / `"warning-setgpu"`. Toggle UI based on detected hardware.
- Parameter panel: Just a prompt textbox and a Submit button.
- Video output: `gr.Video(label="Generated Video")`.
- Progress UX: **Multi-level tqdm streaming via subprocess stdout** — `select.select()` polls every 40 ms, parses INFO messages with regex `r"(\d+)%\|.*\| (\d+)/(\d+)"`, drives three independent `tqdm` bars at positions 0/1/2 (overall, sub-step, video gen).
- **Pattern worth stealing:** the GPU-availability detection banner — useful when running locally vs on ZeroGPU.

### 1.10 Lightricks/ltx-video-distilled ("LTX Video Fast")
- URL: https://huggingface.co/spaces/Lightricks/ltx-video-distilled
- Screenshot: `raw/space_screenshots/lightricks-ltx-fast.png`
- Layout: `gr.Row` two-column. Output dominates the right column.
- Mode exposure: **Three `gr.Tab`** items inside the left column — `"image-to-video"`, `"text-to-video"`, `"video-to-video"` (last is `visible=False` toggled by code). Each tab has its own input set and its own generate button, but they all route to a single `generate()` function with mode passed as an argument. Three click handlers expose three separate API names.
- Model picker: None — single distilled checkpoint.
- Parameter panel: Common duration slider (0.3-8.5 s) and Improve-Texture checkbox below the tabs. `gr.Accordion("Advanced settings", open=False)` holds mode dropdown (hidden), negative prompt, seed + randomize row, height/width sliders (256-1280 step 32), CFG (hidden).
- Video output: `gr.Video` on the right column.
- Progress UX: implicit.
- **Pattern worth stealing:** **three tabs but one shared parameter panel below them**. The mode-specific inputs live in tabs, the shared params live outside. Critical for our Studio.

### 1.11 Fabrice-TIERCELIN/HunyuanVideo (large-T2V demo)
- URL: https://huggingface.co/spaces/Fabrice-TIERCELIN/HunyuanVideo
- Screenshot: `raw/space_screenshots/hunyuan-video.png`
- Layout: Two-column.
- Mode exposure: Single (T2V).
- Model picker: Resolution `gr.Dropdown` with 10 preset aspect ratios at two resolution tiers (720p / 540p). Video-Length `gr.Dropdown` (2 s / 5 s).
- Parameter panel: Prompt, resolution dropdown, video-length dropdown, inference-steps slider (1-100 default 5). Collapsed `gr.Accordion("Advanced Options")` with seed (-1 = random), CFG (1-20 step 0.5), flow shift (0-10 step 0.1), embedded CFG (1-20 step 0.5).
- Video output: `gr.Video` with autoplay.
- Progress UX: Queue of size 10, implicit progress.
- Banner: Red GPU-availability warning when no GPU detected — same pattern as fffiloni Wan2.1.
- **Pattern worth stealing:** the resolution dropdown as preset aspect ratios (`"1280x720 (16:9)"`, `"720x1280 (9:16)"`...) rather than free-form H/W sliders. Much friendlier for non-power users.

### 1.12 THUDM/CogVideoX-5B-Space (multi-mode in one page)
- URL: https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space (alias zai-org)
- Screenshot: `raw/space_screenshots/cogvideox-5b.png`
- Layout: Two-column row.
- Mode exposure: **No tabs — two mutually-exclusive `gr.Accordion`s** for `"I2V: Image Input (cannot be used simultaneously with video input)"` and `"V2V: Video Input (cannot be used simultaneously with image input)"`. The empty-prompt path becomes T2V by default. Anti-pattern: relies on the user reading the label to avoid conflicts.
- Model picker: Two checkboxes for **post-processing**: super-resolution (RIFE 720x480 -> 2880x1920) and frame interpolation (8 -> 16 fps). Useful pattern: optional post-processing toggles next to the main inputs.
- Parameter panel: 5-line prompt, `"✨ Enhance Prompt"` button (GLM-4), seed `gr.Number`, two checkboxes for post-processing. V2V accordion adds a strength slider (0.1-1.0 step 0.01).
- Video output: `gr.Video(width=720, height=480)` plus hidden `gr.File` download buttons that become visible after success, plus a hidden seed-output `gr.Number`.
- Progress UX: `gr.Progress(track_tqdm=True)`, `@spaces.GPU(duration=300)`. Queue max 15.
- **Pattern to avoid:** mutually-exclusive accordions for modes. Tabs (LTX, Wan-AI) handle this much better.

---

## 2. Proposed Studio UX architecture

### 2.1 Top-level navigation

With 6+ modes (T2V, I2V, FLF2V, VACE, S2V, Animate, plus future Inpaint / Outpaint / Long-Video) and 3 generations (2.1, 2.2, 2.5/2.6), a single flat tab bar at the root would have 6-10 tabs and is unsuitable. A second axis (generation) needs to live somewhere too.

Two viable layouts:

**Recommendation: Option A — Left Sidebar (mode) + Top Header (global controls).** Gradio 5's native `gr.Sidebar(position="left", open=True, width=260)` is the right primitive. Modes live as buttons / radio in the sidebar; the main column shows the active mode's panel.

```
+--------------------------------------------------------------+
|  Wan Studio    [Gen: 2.6 v] [Preset: Fast | Quality]    [SAVE] [HISTORY] [SETTINGS] |
+---------+----------------------------------------------------+
|         |                                                    |
| T2V     |  [Active mode panel — see section 2.4]             |
| I2V     |                                                    |
| FLF2V   |                                                    |
| VACE    |                                                    |
|  > Ref  |    (input zone)    |    (output zone)              |
|  > FLF  |                    |                               |
|  > Anim |                    |                               |
| S2V     |                    |                               |
| Animate |                                                    |
|---------|                                                    |
| Gallery |                                                    |
| Chain   |                                                    |
+---------+----------------------------------------------------+
```

**Option B (fallback) — Root-level `gr.Tabs` with one tab per mode**, mirroring LTX Video Fast's pattern. Cleaner code but starts crowding once we add Inpaint / Long-Video / Frame-Interp. Use only if `gr.Sidebar` proves janky in production.

### 2.2 Generation selector (2.1 / 2.2 / 2.5 / 2.6)

Place a **`gr.Dropdown` in the top header** labeled "Wan Generation". When the user switches generation, the parameter panel re-renders to expose generation-specific knobs (Wan 2.2 has dual-noise CFG, Wan 2.5 has different scheduler defaults, etc.). Modes unavailable in a generation get greyed out / hidden in the sidebar — sidebar entries are computed from `MODE_AVAILABILITY[generation]`.

### 2.3 Fast / Quality preset toggle

A **`gr.Radio(["Fast (Lightning)", "Quality"], value="Fast")` in the top header**, global. The preset writes default values into the parameter panel (steps, CFG, scheduler) when toggled. Internally maps to Lightning-LoRA on/off plus a step-count preset (e.g. Fast = 4 steps, Quality = 30 steps). The preset is sticky across mode switches.

### 2.4 Per-mode panel layout (two-column inside the main area)

Per-mode panels share a stable layout — only the inputs in the left column change:

```
+--------------------------- Active Mode: T2V ----------------------+
|                                                                  |
| Inputs (col-left, scale=2)        Output (col-right, scale=3)    |
| ---------------------------       ----------------------------    |
| Prompt                            Video player (gr.Video,         |
| (mode-specific inputs)            autoplay, loop, dl button)      |
| Duration / Frames slider                                          |
| > Advanced (Accordion)            ETA: ~32s on ZeroGPU            |
|   Negative prompt                 [Progress card]                 |
|   Seed + randomize                                                |
|   CFG / Steps                     [Send to: I2V | VACE | Anim]    |
| [Generate]                                                        |
+--------------------------------------------------------------------+
| Examples (3-6 prefill cards)                                       |
+--------------------------------------------------------------------+
```

The left column is rebuilt per mode (different input components). The right column is shared across modes.

### 2.5 Output player + cross-mode "Send to"

The output `gr.Video(autoplay=True, loop=True, show_download_button=True)` is permanent. Below it sits a row of **`gr.Button` "Send to ..."** chips, each wired to copy the current video into the input slot of another mode and switch the active sidebar entry. This is the headline differentiator from any single-mode Space — users can chain T2V -> VACE -> Animate without leaving the app.

### 2.6 Queue / progress

ZeroGPU per-call budget is finite and visible. Steal alexnasa's pattern: a `gr.Text` field labeled **"ZeroGPU reservation"** that recomputes on every parameter change via `.change()` events, showing `"~110.0s (1.8 mins)"`. The same function feeds `@spaces.GPU(duration=get_duration, size=...)`.

During inference, render a **custom HTML progress card** (steal multimodalart/self-forcing pattern) inside a `gr.HTML` component — block counter, percent, eta. Render it next to (not on) the video player. For streaming-capable modes (T2V on Wan 2.5 distill), set `gr.Video(streaming=True)` and yield `.ts` chunks.

### 2.7 Gallery / history

A dedicated `"Gallery"` sidebar entry shows a `gr.Gallery(columns=4, height=600, allow_preview=True)` of the session's last N generations. Each item is a tuple `(video_path, caption=mode+prompt[:60])`. Clicking an item fires a `.select()` handler that repopulates the active mode's params from `session_state[gallery_index]`.

Session state lives in `gr.State` (per-browser-session, in-memory). For longer history, write a small JSON sidecar to `/tmp` per session id. Don't promise cross-session persistence on Spaces — the sandbox blows away storage.

### 2.8 Examples

Every mode panel includes a `gr.Examples(..., cache_mode="lazy", cache_examples=False)` block with 3-6 curated one-click prefills. Borrow Wan-AI/Wan2.2-S2V's pattern of grouping examples by emotion/genre.

### 2.9 Theme & mobile

`gr.themes.Default(primary_hue="indigo", neutral_hue="slate")` with the built-in dark variant. Dark mode is default — video looks better on dark backgrounds. On viewports <768 px the sidebar collapses (`gr.Sidebar(open=False)` toggle event), and the input/output two-col stacks vertically. Confirmed via mobile screenshot of Wan-AI/Wan2.2-S2V that Gradio reflows columns cleanly.

---

## 3. Component spec table

| UI piece | Gradio component | Constructor (May 2026 API) | Notes |
|---|---|---|---|
| Root nav (mode) | `gr.Sidebar(position="left", open=True, width=260)` + `gr.Button` list inside | gradio>=5.x | Each mode = a `gr.Button` whose `.click` swaps the visible main panel via `gr.update(visible=...)`. |
| Root nav (generation) | `gr.Dropdown(choices=["2.1","2.2","2.5","2.6"], value="2.6", label="Wan Generation")` | std | Lives in a header `gr.Row`. |
| Preset toggle | `gr.Radio(["Fast (Lightning)","Quality"], value="Fast", label="Preset")` | std | Drives default values on change. |
| Prompt | `gr.Textbox(lines=4, placeholder="Describe the scene...")` | std | Add an "Enhance Prompt" button next to it (CogVideoX / self-forcing pattern). |
| Negative prompt | `gr.Textbox(lines=2)` inside `gr.Accordion("Advanced", open=False)` | std | Pre-fill Wan's default Chinese negative prompt. |
| Image input (I2V, FLF2V start) | `gr.Image(type="pil", sources=["upload","clipboard"], image_mode="RGB")` | gradio 5.x | `clipboard` source is the killer feature — users paste from screenshot tools. |
| FLF2V end-frame | Nested `gr.Tabs` ("Upload" / "Generate") around a second `gr.Image` | std | Steals multimodalart/wan-2-2-first-last-frame pattern. |
| Audio (S2V) | `gr.Audio(sources=["upload","microphone"], type="filepath")` | std | `format="wav"`. |
| Driving video (Animate) | `gr.Video(sources=["upload"], include_audio=False)` | std | Disable audio to avoid duplicate audio streams. |
| Control gallery (VACE) | `gr.Gallery(columns=3, rows=2, allow_preview=True, sources=["upload"])` | std | Multi-image for VACE Reference/FLF/Random modes; auto-calc target H/W on `.change()`. |
| VACE sub-mode | `gr.Radio(["Reference","First-Last","Random"], value="Reference", label="Control mode")` | std | `.change` toggles visibility of bg-removal checkbox, updates default prompt. |
| Duration | `gr.Slider(0.5, 8.0, value=2.0, step=0.1, label="Duration (s)")` | std | Mirrors LTX Video Fast / VACE Fast. |
| Resolution preset | `gr.Dropdown(["1280x720 (16:9)","720x1280 (9:16)","960x960 (1:1)","832x480 (16:9)","480x832 (9:16)"], value="...")` | std | Borrow HunyuanVideo's labeled-aspect-ratio dropdown. |
| Steps | `gr.Slider(1, 50, value=4, step=1, label="Inference steps")` | std | Default 4 for Fast preset, 30 for Quality preset. |
| CFG (Wan 2.2 dual) | Two `gr.Slider(0, 10, step=0.1)` — high-noise / low-noise | std | Only render both when generation == 2.2. |
| Seed | `gr.Slider(0, 2**31-1, value=42, step=1)` + `gr.Checkbox("Randomize", value=True)` | std | Wan-AI pattern. |
| Generate button | `gr.Button("Generate", variant="primary", size="lg")` | std | Always full-width at bottom of input column. |
| ETA / GPU reservation | `gr.Markdown` showing `"⌚ ZeroGPU reservation: ~Ns"` updated on `.change()` of params | std | Borrow alexnasa's pattern verbatim. |
| Progress card | `gr.HTML` updated by generator function via `yield` | std | Steal multimodalart/self-forcing custom-HTML pattern. |
| Output video | `gr.Video(autoplay=True, loop=True, interactive=False, buttons=["download","share"])` | gradio 5.x adds `buttons` | `show_download_button` is the older API. |
| Output streaming (T2V Wan 2.5 distill) | `gr.Video(streaming=True, loop=True)` | std | Yield `.ts` chunks. |
| "Send to" chips | `gr.Row` of `gr.Button(size="sm")` — one per target mode | std | Wire each via `.click` to copy the video path into the target mode's input slot + switch sidebar. |
| Gallery | `gr.Gallery(columns=4, height=520, allow_preview=True, object_fit="contain")` | std | Store list of `(video_path, caption)` in `gr.State`. |
| Examples | `gr.Examples(..., cache_examples=False, cache_mode="lazy")` | std | Don't cache — ZeroGPU rebuilds nightly. |
| GPU banner | `gr.HTML` with `elem_id="warning-setgpu"` shown when `os.getenv("SPACES_ZERO_GPU") is None` | std | fffiloni / HunyuanVideo pattern for local-dev awareness. |
| Theme | `gr.themes.Default(primary_hue="indigo", neutral_hue="slate")` | std | Dark-mode default. |
| Multi-page (optional) | `demo.route("Settings", "/settings")` + auto-navbar | gradio 5.x | Settings page hosts model-manager + cache controls; main demo stays single-page. |

---

## 4. Wireframe brief (one paragraph per screen)

### S1 — Landing / global shell
The landing view IS the active mode panel — there is no separate landing screen. On first load, the active mode is `T2V` and the sidebar is open. Header strip: app logo + name on the left, then `[Wan Generation v]` dropdown, `[Preset: Fast | Quality]` radio (segmented control), then right-aligned `[History] [Settings]` icon buttons. Sidebar lists the modes as buttons with monochrome icons; gallery and pipeline-chain views sit at the bottom of the sidebar as separate entries. The empty state of the right output column shows a muted illustration plus "Generate to preview here". On mobile (<768 px) the sidebar collapses to a hamburger.

### S2 — T2V tab
Two-column main area. Left column (scale=2): prompt textbox (4 rows), `"✨ Enhance Prompt"` button row, duration slider, resolution preset dropdown, collapsed `Advanced Settings` accordion (negative prompt, seed+randomize, steps, CFG — dual-CFG only if Wan 2.2), and a large `Generate` button. Right column (scale=3): `gr.Video` output, ETA markdown above it, progress card replaces ETA during generation. Below the output, a "Send to" row with chips for I2V/VACE/Animate. Examples grid below the two-column area. State: empty (placeholder svg) -> generating (progress card + spinner over the video element) -> success (video autoplays, dl button visible, send-to chips enabled). Distinct from other modes because the input column is just text — no image/audio/video slots.

### S3 — I2V tab
Same shell as T2V but the input column has a `gr.Image` upload slot above the prompt; the prompt is shorter (3 lines). Resolution dropdown auto-suggests an aspect ratio matching the uploaded image (handler runs on `.upload`). The duration slider has a smaller range (0.3-3.4 s for Lightning Fast variant; 5 s for Quality). Distinct because: pasted image triggers automatic resolution selection; the placeholder text in the prompt nudges "describe the motion you want" rather than the scene. Examples are image-prompt pairs.

### S4 — FLF2V tab
Input column has two `gr.Image` slots side-by-side in a `gr.Row`: "Start frame" and "End frame". The end-frame slot is itself wrapped in nested tabs (Upload | Generate-from-LLM). Below the frames: a prompt describing the transition. Advanced accordion shows duration, dual-CFG, seed. Right column same as T2V. Distinct because it's the only mode with two image inputs and the optional generated-end-frame sub-flow. Examples include action transitions (penguin -> takeoff style).

### S5 — VACE tab
Input column is the most complex. Top: a `gr.Radio` for sub-mode (Reference / First-Last / Random). Below: a `gr.Gallery(columns=3, rows=2)` for the control images (mode-dependent semantic — reference, anchor frames, transition keyframes). Conditional `"Remove background"` checkbox appears only for `Reference`. Then prompt, duration, advanced accordion. Distinct because it's the only mode using a gallery input AND a radio for sub-mode, and the only mode where the prompt label changes based on sub-mode. Examples grouped by sub-mode.

### S6 — S2V tab
Input column has `gr.Image` (reference subject), then `gr.Audio(sources=["upload","microphone"])`, then a prompt, then resolution dropdown (S2V default 480p). Advanced accordion exposes duration auto-derived from audio length (the duration slider is read-only here). Distinct because audio drives the duration — the slider is informational rather than interactive, and the mic input encourages "test with your own voice" demos. Examples are emotion-tagged audio clips (sing, speech, cinematic).

### S7 — Animate tab
Input column has `gr.Image` (reference character) and `gr.Video` (driving / template video) in a vertical stack. Below: a `gr.Radio` for animation mode (Character Swap / Pose Retarget) and a `gr.Radio` for resolution (Low / Medium). Duration slider clamped to 1-20 s (Animate is expensive). Output column shows the main animated video plus a collapsed accordion "Processing outputs" (pose, mask, bg, face). Distinct because two media inputs (image + video) and the explicit "this is GPU-expensive — pick low res to test" framing in helper text. Examples are short dance / sports clips.

### S8 — Settings / Model manager
Separate page via `demo.route("Settings", "/settings")`. Layout is a single column. Sections: (1) Active models per mode — read-only list of which Wan checkpoints are loaded for the current generation, (2) Cache controls — buttons "Clear video cache", "Clear LoRA cache", "Re-download base model", (3) Per-mode default presets (override Fast/Quality numbers), (4) About / version info. Distinct because no video player — it's pure configuration with `gr.Button`s and `gr.Accordion`s.

### S9 — Gallery / history
Single-column main area. Top: `gr.Gallery(columns=4, height=600)` of the session's last 24 generations. Each cell is a thumbnail (first frame of the video) with the mode badge in a corner. Selecting a cell expands a side panel (`gr.Column(scale=1)`) showing the full video with autoplay+loop, the original params (read-only), a `"Reload into ..."` button per applicable mode, and a `"Delete"` button. Distinct because it's the only screen with no generation controls — purely retrospective. The empty state ("Nothing yet — generate something!") includes shortcut buttons that route to T2V/I2V.

### S10 — Pipeline chain (optional, MVP+1)
A graph view showing the chain of operations applied so far this session — `T2V -> VACE refine -> Animate` for example. Implemented as a `gr.HTML` SVG renderer plus a row of `gr.Button` "Add step" entries that append new modes. Each node opens its own param panel below the graph. Distinct because it visualizes cross-mode dependencies — the rest of the app treats each generation as isolated.

---

## 5. Sources

- https://huggingface.co/spaces/Wan-AI/Wan2.1
- https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V
- https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate
- https://huggingface.co/spaces/alexnasa/Wan2.2-Animate-ZEROGPU
- https://huggingface.co/spaces/multimodalart/wan2-1-fast
- https://huggingface.co/spaces/multimodalart/wan-2-2-first-last-frame
- https://huggingface.co/spaces/multimodalart/self-forcing
- https://huggingface.co/spaces/linoyts/wan2-1-VACE-fast
- https://huggingface.co/spaces/fffiloni/Wan2.1
- https://huggingface.co/spaces/Lightricks/ltx-video-distilled
- https://huggingface.co/spaces/Fabrice-TIERCELIN/HunyuanVideo
- https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space (alias zai-org)
- https://www.gradio.app/docs/gradio/video
- https://www.gradio.app/docs/gradio/image
- https://www.gradio.app/docs/gradio/audio
- https://www.gradio.app/docs/gradio/sidebar
- https://www.gradio.app/docs/gradio/progress
- https://www.gradio.app/docs/gradio/gallery
- https://www.gradio.app/guides/multipage-apps
- https://www.gradio.app/guides/theming-guide
- https://huggingface.co/blog/gradio-5
- https://huggingface.co/docs/hub/spaces-zerogpu
