"""Left sidebar — mode picker. Modes grey out when unavailable in the current
generation (driven by `pipelines.registry.modes_in(generation)`).
"""
from __future__ import annotations

import gradio as gr

# Mode → (label, emoji, default visible in Wan 2.2)
MODE_PILLS = [
    ("t2v",      "🎬 T2V"),
    ("i2v",      "🖼️ I2V"),
    ("ti2v",     "🌗 TI2V"),
    ("flf2v",    "⇄ FLF2V"),
    ("v2v",      "🎞️ V2V"),
    ("vace",     "🎛️ VACE"),
    ("s2v",      "🔊 S2V"),
    ("animate",  "💃 Animate"),
]


def build_sidebar() -> dict:
    components = {}
    with gr.Column(scale=0, min_width=240, elem_id="wan-studio-sidebar"):
        gr.Markdown("### Modes")
        for key, label in MODE_PILLS:
            btn = gr.Button(label, variant="secondary", size="sm", elem_id=f"mode-btn-{key}")
            components[f"mode_{key}"] = btn

        gr.Markdown("---")
        components["gallery_btn"] = gr.Button("🖼 Gallery", variant="secondary", size="sm")
        components["settings_btn"] = gr.Button("⚙ Settings", variant="secondary", size="sm")

    return components
