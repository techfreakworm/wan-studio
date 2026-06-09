"""Per-mode tab panels.

Phase 0: each panel is a stubbed `gr.Column` with mode-appropriate inputs but no
actual generation function wired. The output column is a placeholder `gr.Video`.

Round 1 fix — navigation is now driven 100% client-side. All 10 panels are
mounted with `visible=True` (Gradio never toggles visibility), and a JS event
handler in app.py adds/removes a `ws-mode-panel-active` class on whichever
panel the sidebar selected. Gradio sends ZERO backend updates per nav click,
so there's no cascade for Svelte to choke on.

Phase 1+ will plug `pipelines.{t2v,i2v,...}` into the Generate buttons.
"""
from __future__ import annotations

import gradio as gr


RESOLUTION_PRESETS = [
    "1280x720 (16:9)",
    "720x1280 (9:16)",
    "960x960 (1:1)",
    "832x480 (16:9)",
    "480x832 (9:16)",
]


def _two_col(input_builder, output_builder):
    """Standard 2-col layout: 40% inputs / 60% output."""
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            inputs = input_builder()
        with gr.Column(scale=3):
            outputs = output_builder()
    return inputs, outputs


def _output_column(default_eta: str = "~?s"):
    video = gr.Video(label="Output", autoplay=True, loop=True, interactive=False)
    eta = gr.Markdown(f"⌚ ZeroGPU reservation: **{default_eta}**")
    progress = gr.HTML(visible=False)
    with gr.Row():
        with gr.Column(scale=1, min_width=70):
            gr.Markdown("Send to:")
        sendto = {
            "i2v": gr.Button("I2V", size="sm", scale=0),
            "vace": gr.Button("VACE", size="sm", scale=0),
            "animate": gr.Button("Animate", size="sm", scale=0),
        }
    return {"video": video, "eta": eta, "progress": progress, "sendto": sendto}


def _advanced_accordion():
    with gr.Accordion("Advanced", open=False):
        negative_prompt = gr.Textbox(
            label="Negative prompt", lines=2,
            placeholder="Things you don't want to see (CFG must be > 1 to apply)",
        )
        with gr.Row():
            seed = gr.Slider(0, 2**31 - 1, value=42, step=1, label="Seed", scale=4)
            randomize = gr.Checkbox(value=True, label="Randomize", scale=1)
        with gr.Row():
            steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
            cfg = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG")
            cfg_2 = gr.Slider(
                0.0, 10.0, value=1.0, step=0.1,
                label="CFG (low-noise)", visible=False,  # Wan 2.2 MoE only
            )
    return {
        "negative_prompt": negative_prompt, "seed": seed, "randomize": randomize,
        "steps": steps, "cfg": cfg, "cfg_2": cfg_2,
    }


def _panel(elem_id: str, initial: bool = False):
    """Mount a mode panel.  Initial-active gets the `ws-mode-panel-active`
    class so it's visible on first paint; the others get only `ws-mode-panel`
    and CSS hides them until JS toggles them on."""
    classes = ["ws-mode-panel"] + (["ws-mode-panel-active"] if initial else [])
    # NOTE: visible=True everywhere — JS does the show/hide, not Gradio.
    return gr.Column(visible=True, elem_id=elem_id, elem_classes=classes)


def build_t2v_tab() -> dict:
    components = {}
    with _panel("tab-t2v", initial=True) as tab:
        gr.Markdown("## T2V — Text-to-Video", elem_classes=["mode-title"])
        def _inputs():
            prompt = gr.Textbox(
                label="Prompt", lines=4,
                placeholder="A cinematic shot of a fox running through autumn leaves...",
            )
            enhance = gr.Button("✨ Enhance Prompt", variant="secondary", size="sm")
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=RESOLUTION_PRESETS, value="1280x720 (16:9)", label="Resolution",
                )
                duration = gr.Slider(0.5, 5.1, value=3.4, step=0.1, label="Duration (s)")
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {"prompt": prompt, "enhance": enhance, "resolution": resolution,
                    "duration": duration, "generate": generate, **advanced}
        def _outputs():
            return _output_column(default_eta="~110s")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_i2v_tab() -> dict:
    components = {}
    with _panel("tab-i2v") as tab:
        gr.Markdown("## I2V — Image-to-Video", elem_classes=["mode-title"])
        def _inputs():
            image = gr.Image(
                type="pil", sources=["upload", "clipboard"],
                label="Source image", image_mode="RGB", height=300,
            )
            prompt = gr.Textbox(
                label="Motion prompt", lines=3,
                placeholder="Slow zoom in, leaves rustling in the wind...",
            )
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=RESOLUTION_PRESETS, value="1280x720 (16:9)", label="Resolution",
                )
                duration = gr.Slider(0.5, 5.1, value=3.0, step=0.1, label="Duration (s)")
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {"image": image, "prompt": prompt, "resolution": resolution,
                    "duration": duration, "generate": generate, **advanced}
        def _outputs():
            return _output_column(default_eta="~120s")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_flf2v_tab() -> dict:
    components = {}
    with _panel("tab-flf2v") as tab:
        gr.Markdown("## FLF2V — First-Last-Frame to Video (Wan 2.1 only)", elem_classes=["mode-title"])
        def _inputs():
            with gr.Row():
                start_frame = gr.Image(
                    type="pil", sources=["upload", "clipboard"],
                    label="Start frame", height=240,
                )
                with gr.Column():
                    with gr.Tabs():
                        with gr.Tab("Upload"):
                            end_frame_uploaded = gr.Image(
                                type="pil", sources=["upload", "clipboard"],
                                label="End frame", height=240,
                            )
                        with gr.Tab("Generate"):
                            gr.Markdown("Synthesize end frame from a prompt (T2I).")
                            end_frame_prompt = gr.Textbox(label="End-frame prompt", lines=2)
                            generate_end = gr.Button("Generate end frame", size="sm")
                            end_frame_generated = gr.Image(
                                type="pil", label="Generated end frame", height=200,
                            )
            prompt = gr.Textbox(
                label="Transition prompt", lines=3,
                placeholder="A penguin spreads its wings and takes flight (中文 hint)...",
            )
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {
                "start_frame": start_frame, "end_frame_uploaded": end_frame_uploaded,
                "end_frame_prompt": end_frame_prompt, "generate_end": generate_end,
                "end_frame_generated": end_frame_generated,
                "prompt": prompt, "generate": generate, **advanced,
            }
        def _outputs():
            return _output_column(default_eta="~150s")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_vace_tab() -> dict:
    components = {}
    with _panel("tab-vace") as tab:
        gr.Markdown("## VACE — Versatile Animation Control & Editing (Wan 2.1 only)", elem_classes=["mode-title"])
        def _inputs():
            submode = gr.Radio(
                choices=[
                    "Depth", "Pose", "Sketch", "Flow",
                    "Inpaint", "Outpaint", "Reference",
                    "Extension", "Animate-Anything",
                ],
                value="Inpaint", label="Sub-mode",
            )
            source_video = gr.Video(sources=["upload"], label="Source video")
            mask_source = gr.Radio(
                choices=[
                    "Provide mask", "Bbox", "Track from mask",
                    "Track bbox", "Label", "Caption",
                ],
                value="Track from mask", label="Mask source", visible=True,
            )
            mask_input = gr.Textbox(label="Initial mask / bbox / label")
            references = gr.Gallery(
                label="Optional reference images (1-3)",
                columns=3, rows=1, height=120, object_fit="contain", interactive=True,
            )
            prompt = gr.Textbox(label="Prompt", lines=3)
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {
                "submode": submode, "source_video": source_video,
                "mask_source": mask_source, "mask_input": mask_input,
                "references": references, "prompt": prompt,
                "generate": generate, **advanced,
            }
        def _outputs():
            return _output_column(default_eta="~180s")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_s2v_tab() -> dict:
    components = {}
    with _panel("tab-s2v") as tab:
        gr.Markdown("## S2V — Speech to Video (Wan 2.2, via upstream wan package)", elem_classes=["mode-title"])
        def _inputs():
            reference_image = gr.Image(
                type="pil", sources=["upload", "clipboard"],
                label="Reference character", height=240,
            )
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                             label="Driving audio")
            pose_video = gr.Video(sources=["upload"], label="Optional pose video",
                                  include_audio=False)
            prompt = gr.Textbox(
                label="Scene / style prompt", lines=3,
                placeholder="A cinematic close-up...",
            )
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=RESOLUTION_PRESETS + ["1024x704 (S2V default)"],
                    value="1024x704 (S2V default)", label="Resolution",
                )
                duration = gr.Markdown("Duration: **auto** (driven by audio length)")
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {
                "reference_image": reference_image, "audio": audio,
                "pose_video": pose_video, "prompt": prompt,
                "resolution": resolution, "duration": duration,
                "generate": generate, **advanced,
            }
        def _outputs():
            return _output_column(default_eta="~240s (audio-driven)")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_animate_tab() -> dict:
    components = {}
    with _panel("tab-animate") as tab:
        gr.Markdown("## Animate — Character Animation & Replacement (Wan 2.2)", elem_classes=["mode-title"])
        def _inputs():
            character = gr.Image(
                type="pil", sources=["upload", "clipboard"],
                label="Character reference", height=280,
            )
            driving = gr.Video(sources=["upload"], label="Driving / template video")
            mode = gr.Radio(
                choices=["Character Swap", "Pose Retarget", "Replacement (bg+mask)"],
                value="Character Swap", label="Mode",
            )
            res = gr.Radio(
                choices=["Low 480p", "Medium 720p"], value="Low 480p", label="Resolution",
            )
            duration = gr.Slider(1, 20, value=6, step=1, label="Duration (s)")
            prompt = gr.Textbox(label="Optional prompt", lines=2)
            gr.Markdown(
                "⚠ Pose+face preprocessing runs on CPU before GPU (~30s extra).",
                elem_classes=["warning-banner"],
            )
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {
                "character": character, "driving": driving, "mode": mode, "res": res,
                "duration": duration, "prompt": prompt,
                "generate": generate, **advanced,
            }
        def _outputs():
            output = _output_column(default_eta="~300s (xlarge tier)")
            with gr.Accordion("🎭 Processing outputs", open=True):
                output["pose_preview"] = gr.Video(label="pose", interactive=False)
                output["face_preview"] = gr.Video(label="face", interactive=False)
                output["bg_preview"] = gr.Video(label="bg", interactive=False)
                output["mask_preview"] = gr.Video(label="mask", interactive=False)
            return output
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_v2v_tab() -> dict:
    components = {}
    with _panel("tab-v2v") as tab:
        gr.Markdown("## V2V — Video-to-Video Restyle", elem_classes=["mode-title"])
        def _inputs():
            video = gr.Video(sources=["upload"], label="Source video")
            prompt = gr.Textbox(label="Restyle prompt", lines=3)
            strength = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Strength")
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {"video": video, "prompt": prompt, "strength": strength,
                    "generate": generate, **advanced}
        def _outputs():
            return _output_column(default_eta="~90s")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_ti2v_tab() -> dict:
    components = {}
    with _panel("tab-ti2v") as tab:
        gr.Markdown("## TI2V — Text+Image to Video (Wan 2.2-5B, via upstream wan package)", elem_classes=["mode-title"])
        def _inputs():
            image = gr.Image(
                type="pil", sources=["upload", "clipboard"],
                label="Optional image (omit for T2V-only)", height=240,
            )
            prompt = gr.Textbox(label="Prompt", lines=4)
            gr.Markdown(
                "TI2V-5B is locked to **1280×704** (landscape) or **704×1280** (portrait), 121 frames @ 24 fps.",
            )
            orientation = gr.Radio(
                choices=["Landscape (1280x704)", "Portrait (704x1280)"],
                value="Landscape (1280x704)", label="Orientation",
            )
            advanced = _advanced_accordion()
            generate = gr.Button("Generate", variant="primary", size="lg")
            return {"image": image, "prompt": prompt, "orientation": orientation,
                    "generate": generate, **advanced}
        def _outputs():
            return _output_column(default_eta="~60s")
        components["inputs"], components["outputs"] = _two_col(_inputs, _outputs)
    components["tab"] = tab
    return components


def build_gallery_tab() -> dict:
    components = {}
    with _panel("tab-gallery") as tab:
        gr.Markdown("## Gallery — session history", elem_classes=["mode-title"])
        # Empty-state hint shown until the first generation lands.
        empty_state = gr.Markdown(
            "_Generated videos will appear here._",
            elem_classes=["ws-gallery-empty"],
        )
        # Read-only display of past outputs. `interactive=False` removes the
        # "Drop Media Here / Click to Upload" affordance that made the gallery
        # look like an upload zone.
        gallery = gr.Gallery(
            label=None,
            show_label=False,
            columns=4,
            rows=3,
            height=560,
            allow_preview=True,
            object_fit="cover",
            interactive=False,
            value=[],
            elem_classes=["ws-gallery-readonly"],
        )
        with gr.Row():
            preview = gr.Video(label="Selected", autoplay=True, loop=True, interactive=False)
            params = gr.Markdown("Params will appear here.")
        with gr.Row():
            reload_t2v = gr.Button("Reload into T2V")
            reload_vace = gr.Button("Reload into VACE")
            reload_animate = gr.Button("Reload into Animate")
            delete_btn = gr.Button("Delete", variant="stop")
            export_btn = gr.Button("Export")
        components.update(dict(
            empty_state=empty_state,
            gallery=gallery, preview=preview, params=params,
            reload_t2v=reload_t2v, reload_vace=reload_vace, reload_animate=reload_animate,
            delete_btn=delete_btn, export_btn=export_btn,
        ))
    components["tab"] = tab
    return components


def build_settings_tab() -> dict:
    components = {}
    with _panel("tab-settings") as tab:
        gr.Markdown("## Settings — Model Manager", elem_classes=["mode-title"])
        with gr.Accordion("Active models per mode", open=True):
            components["model_status"] = gr.Markdown("Model load status appears here.")
        with gr.Accordion("Lightning LoRA status", open=False):
            components["lora_status"] = gr.Markdown("LoRA status appears here.")
            components["use_hybrid"] = gr.Checkbox(
                value=False,
                label="Use Wan 2.1 lightx2v I2V LoRA on Wan 2.2 I2V (hybrid trick)",
            )
        with gr.Accordion("Cache controls", open=False):
            components["clear_video_cache"] = gr.Button("Clear video cache")
            components["clear_lora_cache"] = gr.Button("Clear LoRA cache")
        with gr.Accordion("About", open=False):
            components["about"] = gr.Markdown(
                "Wan Studio v0.1 · loading...\n\n"
                "*Run `python -c 'from utils.backend import detect; print(detect())'` "
                "to see the active backend.*"
            )
    components["tab"] = tab
    return components


def build_all_tabs() -> dict[str, dict]:
    """Return all mode tabs keyed by mode name. Each value is the tab's component dict."""
    return {
        "t2v":      build_t2v_tab(),
        "i2v":      build_i2v_tab(),
        "ti2v":     build_ti2v_tab(),
        "flf2v":    build_flf2v_tab(),
        "v2v":      build_v2v_tab(),
        "vace":     build_vace_tab(),
        "s2v":      build_s2v_tab(),
        "animate":  build_animate_tab(),
        "gallery":  build_gallery_tab(),
        "settings": build_settings_tab(),
    }
