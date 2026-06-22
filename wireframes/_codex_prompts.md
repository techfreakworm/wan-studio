# Codex CLI Wireframe Generation Prompts

Once `codex login` is working, run these 8 prompts in parallel (each writes one PNG to `wireframes/`).

Suggested invocation pattern (adjust to your codex CLI version):

```bash
codex exec --image -o wireframes/w1_shell_t2v.png "$(cat wireframes/_prompts/w1.txt)" &
codex exec --image -o wireframes/w2_i2v.png       "$(cat wireframes/_prompts/w2.txt)" &
# ...etc, all 8 in background
wait
```

Or if codex outputs to a fixed path, adjust the invocation accordingly.

---

## Common style brief (prepend to every prompt)

> Generate a high-detail, low-fidelity wireframe mockup of a Gradio web app. Style: monochrome with one accent color (indigo), thin black borders on every UI box, labeled rectangles indicating component types ("gr.Video", "gr.Image upload", "gr.Slider"), clean sans-serif font, light grey background, 16:10 desktop aspect ratio (1536×960 or similar). Annotate every distinct UI region with a small label. The viewer should be able to read the entire layout at a glance. NO photorealistic content inside the mockups — just placeholder rectangles labeled "video player", "image upload slot", etc.

---

## W1 — Global shell + T2V active

A full Gradio Studio dashboard. Top header: "Wan Studio" logo on the left, then a "Generation: 2.2" dropdown, then a segmented control "Preset: Fast | Quality", then right-aligned "History" and "Settings" icon buttons. Left sidebar 260px wide with mode buttons: T2V (highlighted active, with left accent border), I2V, FLF2V, V2V, VACE, S2V, Animate. Below a divider: Gallery and Settings entries. Main area split into two columns: left "Inputs" column (scale 2) contains a prompt textbox (4 rows) labeled "Prompt", a small "✨ Enhance Prompt" button below it, then "Resolution: 720x1280" dropdown, "Duration: 3.4s" slider, a closed "▶ Advanced" accordion, and a large indigo "[Generate]" button. Right "Output" column (scale 3) contains an empty `gr.Video` placeholder rectangle, "⌚ ZeroGPU: ~110s" text below it, then "Send to:" row with three small pill buttons "I2V", "VACE", "Animate". Below the two columns: a row of 4 "Examples" cards. Aspect 16:10 desktop.

## W2 — I2V mode

Same shell as W1, but I2V is highlighted in the sidebar. Inputs column: a large "Source image" upload slot (with "📁 📋 sources" hint), then a 3-row "Motion prompt" textbox, then "Resolution: 720x1280" dropdown, then a "Duration: 3.0s" slider, then a yellow info banner "⚠ Wan 2.2 I2V Lightning V2 not yet released — V1 in use; enable hybrid trick in Advanced", then closed "▶ Advanced" accordion, then "[Generate]". Output column: empty video player, "⌚ ZeroGPU: ~120s", "Send to: VACE Animate". Aspect 16:10 desktop.

## W3 — FLF2V mode

Same shell, FLF2V active. Inputs column has TWO image slots side-by-side in a row: left "Start frame" (simple upload), right "End frame" wrapped in a small nested tab bar with "Upload" / "Generate" tabs. The active "Upload" tab shows a placeholder upload slot. Below the frames: a "Transition prompt" textbox (4 rows) with placeholder text including a Chinese-character hint icon. Below: open "Advanced" accordion showing "Negative prompt", "CFG: 5.5" slider, "Seed" slider + "Randomize" checkbox. "[Generate]" at the bottom. Output column: empty video, "⌚ ZeroGPU: ~150s", "Send to: V2V VACE". Aspect 16:10 desktop.

## W4 — VACE mode (most complex)

Same shell, VACE active. Inputs column (scale 2) is dense:
- Top: "Sub-mode" radio group with 9 options in a 3×3 grid: Depth, Pose, Sketch, Flow, **Inpaint (highlighted active)**, Outpaint, Reference, Extension, Animate-Anything.
- Below: "Source video" upload slot.
- Below: "Mask source" sub-radio with 6 options: Provide mask, Bbox, **Track from mask (active)**, Track bbox, Label, Caption.
- Below: an "Initial mask / bbox / label" input field.
- Below: "Optional reference images (1-3)" — a row of 3 small "+" upload boxes.
- Below: "Prompt" textbox with a sub-mode-aware placeholder.
- Below: closed "▶ Advanced" accordion.
- "[Generate]" button.

Output column: empty video, "⌚ ZeroGPU: ~180s", a small "Quality preset — no Lightning for VACE" badge, "Send to: Animate I2V". Aspect 16:10 desktop. This wireframe should look noticeably more dense than the others to convey VACE's complexity.

## W5 — S2V mode

Same shell, S2V active. Inputs column:
- "Reference character" image upload slot.
- "Driving audio" upload slot with a horizontal waveform preview (placeholder squiggle) and "🎤 Record" mic icon, plus "12.4s" duration text.
- "Optional pose video" upload slot.
- "Scene / style prompt" textbox.
- "Resolution: 1024x704 (≈3:2)" dropdown.
- "Duration: 12.4s (read-only, driven by audio)" — greyed-out info field, NOT a slider.
- Closed "▶ Advanced (CFG 4.5, 40 steps)" accordion.
- "[Generate]" button.

Output column: empty video player, "⌚ ZeroGPU: ~240s (variable — driven by audio)", small "Quality only — vendor `wan` pkg" badge, "Send to: Animate" chip. Aspect 16:10 desktop.

## W6 — Animate mode

Same shell, Animate active. Inputs column:
- "Character reference" image upload slot.
- "Driving / template video" upload slot with a small video thumbnail and "▶ 0:05" play icon.
- "Mode" radio with 3 options: **Character Swap (active)**, Pose Retarget, Replacement (bg+mask).
- "Resolution" radio with 2 options: **Low 480p (active)**, Medium 720p.
- "Duration (1-20s): 6s" slider.
- "Optional prompt" textbox.
- Yellow info banner "⚠ Pose+face preproc runs on CPU before GPU (~30s extra)".
- Closed "▶ Advanced" accordion.
- "[Generate]" button.

Output column: main `gr.Video` placeholder, "⌚ ZeroGPU: ~300s (xlarge tier)" text, "Send to: VACE" chip, then an OPEN "▶ Processing outputs" accordion below the main video showing 4 small thumbnails labeled "🎭 pose", "🎭 face", "🎭 bg", "🎭 mask". Aspect 16:10 desktop.

## W7 — Settings / Model manager

Same shell, **Settings highlighted in sidebar**. Main area is a single full-width column (no two-col split). Sections from top to bottom:

1. "Active models per mode" — a 6-row read-only table: T2V / I2V / FLF2V / VACE / S2V / Animate, each with the HF repo path and a "loaded ✓" or "not loaded" status badge.
2. "Lightning LoRA status" — 2 rows + checkbox: T2V Wan 2.2 V2.0 (loaded), I2V Wan 2.2 V1 (loaded), and a "☐ Use Wan 2.1 lightx2v I2V LoRA hybrid trick on Wan 2.2 I2V" checkbox.
3. "Cache controls" — 3 buttons: "Clear video cache", "Clear LoRA cache", "Force re-download base model" + per-mode dropdown.
4. "Per-mode default presets" — 2 rows: Fast steps `[4]` CFG `[1.0]`, Quality steps `[30]` CFG `[5.0]`.
5. "About" — a small monospace block: "Wan Studio v0.1 · diffusers 0.38.2 · spaces 0.50.2 · Backend: ZeroGPU large (Blackwell 48 GB) · GPU: NVIDIA RTX Pro 6000 Blackwell (sm_120)".

NO video player anywhere — this is pure configuration. Aspect 16:10 desktop.

## W8 — Gallery / history

Same shell, Gallery highlighted in sidebar. Main area:
- Header: "Gallery — last 24 generations (session)".
- A 3×4 grid of video thumbnail tiles. Each tile is a labeled rectangle with a small "mode badge" in the top-right corner (T2V, I2V, FLF2V, VACE, S2V, Animate, etc.). One of the tiles (a "VACE — depth control" tile) is highlighted with an indigo border indicating selection.
- Below the grid, a horizontal divider.
- Below the divider, the selected item's detail view: on the left a `gr.Video` placeholder (autoplay+loop), on the right a "Params" read-only block listing prompt, sub-mode, seed, steps, CFG, resolution.
- Bottom row of action buttons: "Reload into VACE", "Reload into Animate", "Delete", "Export".

Aspect 16:10 desktop.

---

## Notes for the dispatcher

- Run all 8 in parallel via `&` + `wait` (each codex job takes ~30-60s typically).
- Codex may want a specific image-size arg — check `codex exec --help` for `--image-size` or `--aspect-ratio`.
- If codex doesn't write to `-o` reliably, capture stdout to a file or check codex's default output dir.
- After all 8 finish, embed them in RESEARCH.md §10 by appending image refs like `![W1](wireframes/w1_shell_t2v.png)` under each wireframe section.
