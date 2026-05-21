"""Top header — app name, generation selector, preset toggle, history+settings.

Returns the Gradio components so app.py can wire them into state + change handlers.
"""
from __future__ import annotations

import gradio as gr


def build_header() -> dict:
    with gr.Row(equal_height=True, elem_id="wan-studio-header"):
        with gr.Column(scale=2, min_width=200):
            gr.Markdown("# ◉ Wan Studio")
        with gr.Column(scale=2, min_width=160):
            generation = gr.Dropdown(
                choices=[("Wan 2.1", "wan2.1"), ("Wan 2.2", "wan2.2")],
                value="wan2.2",
                label="Generation",
                interactive=True,
            )
        with gr.Column(scale=2, min_width=200):
            preset = gr.Radio(
                choices=["Fast (Lightning)", "Quality"],
                value="Fast (Lightning)",
                label="Preset",
                interactive=True,
            )
        with gr.Column(scale=1, min_width=120):
            history_btn = gr.Button("📜 History", variant="secondary", size="sm")
            settings_btn = gr.Button("⚙ Settings", variant="secondary", size="sm")

    return {
        "generation": generation,
        "preset": preset,
        "history_btn": history_btn,
        "settings_btn": settings_btn,
    }
