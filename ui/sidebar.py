"""Left sidebar — Linear-style mode list with section headings and muted labels.

Active mode is tracked via state in app.py; sidebar buttons receive updated
`elem_classes` to flip the active-pill style.
"""
from __future__ import annotations

import gradio as gr

# Mode → display label (no emoji; we use a leading dot glyph via CSS for the
# active state, matching Linear's restrained iconography).
MODE_PILLS = [
    ("t2v",      "Text → Video"),
    ("i2v",      "Image → Video"),
    ("ti2v",     "Text + Image → Video"),
    ("flf2v",    "First / Last Frame"),
    ("v2v",      "Video → Video"),
    ("vace",     "VACE · Edit & Control"),
    ("s2v",      "Speech → Video"),
    ("animate",  "Character Animate"),
]


def build_sidebar() -> dict:
    components: dict = {}
    with gr.Column(
        scale=0, min_width=248,
        elem_id="ws-sidebar", elem_classes=["ws-sidebar-col"],
    ):
        gr.HTML(
            '<div class="ws-side-heading">'
            '<span class="ws-side-heading-text">Generate</span>'
            '</div>'
        )
        for key, label in MODE_PILLS:
            classes = ["ws-side-btn"]
            if key == "t2v":
                classes.append("ws-side-btn-active")
            btn = gr.Button(
                label,
                elem_classes=classes,
                elem_id=f"ws-mode-{key}",
            )
            components[f"mode_{key}"] = btn

        gr.HTML(
            '<div class="ws-side-heading ws-side-heading-divider">'
            '<span class="ws-side-heading-text">Workspace</span>'
            '</div>'
        )
        components["gallery_btn"] = gr.Button(
            "Gallery",
            elem_classes=["ws-side-btn"],
            elem_id="ws-mode-gallery",
        )
        components["settings_btn"] = gr.Button(
            "Settings",
            elem_classes=["ws-side-btn"],
            elem_id="ws-mode-settings",
        )

        # Footer.
        gr.HTML(
            '<div class="ws-side-footer">'
            '<div class="ws-side-footer-row">'
            '<span class="ws-side-footer-key">v0.2 · design v2/2</span>'
            '<span class="ws-side-footer-status">●  ready</span>'
            '</div>'
            '<div class="ws-side-footer-hint">Linear-inspired build</div>'
            '</div>'
        )

    return components
