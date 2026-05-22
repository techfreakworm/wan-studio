"""Top header — brand mark + Generation + Preset + chrome nav.

Linear-inspired: hairline border, 13px nav labels, restrained type, light pill CTA.
NOTE: NO `gr.Radio` for chrome. Preset uses two `gr.Button`s with active class
flipped via state in app.py. Generation stays as `gr.Dropdown`.
"""
from __future__ import annotations

import gradio as gr


def build_header() -> dict:
    with gr.Row(elem_id="ws-header", elem_classes=["ws-header-row"]):
        # ── Brand mark ────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=240, elem_classes=["ws-chrome-col", "ws-brand-col"]):
            gr.HTML(
                """
                <div class="ws-brand">
                  <div class="ws-brand-mark"></div>
                  <div class="ws-brand-text">
                    <span class="ws-brand-name">Wan Studio</span>
                    <span class="ws-brand-sub">studio · video diffusion</span>
                  </div>
                </div>
                """,
                elem_id="ws-brand-html",
            )

        # ── Generation dropdown ──────────────────────────────────────────
        with gr.Column(scale=0, min_width=150, elem_classes=["ws-chrome-col"]):
            generation = gr.Dropdown(
                choices=[("Wan 2.2", "wan2.2"), ("Wan 2.1", "wan2.1")],
                value="wan2.2",
                label="Generation",
                show_label=False,
                interactive=True,
                container=False,
                elem_classes=["ws-dropdown"],
            )

        # ── Preset toggle (two pill buttons + state) ─────────────────────
        with gr.Column(scale=0, min_width=160, elem_classes=["ws-chrome-col"]):
            preset_state = gr.State("fast")
            with gr.Row(elem_classes=["ws-preset-group"]):
                preset_fast = gr.Button(
                    "Fast",
                    elem_classes=["ws-pill", "ws-pill-active"],
                    elem_id="ws-preset-fast",
                )
                preset_quality = gr.Button(
                    "Quality",
                    elem_classes=["ws-pill"],
                    elem_id="ws-preset-quality",
                )

        # ── Chrome nav buttons ───────────────────────────────────────────
        with gr.Column(scale=0, min_width=200, elem_classes=["ws-chrome-col", "ws-chrome-right"]):
            with gr.Row(elem_classes=["ws-chrome-actions"]):
                history_btn = gr.Button(
                    "History",
                    elem_classes=["ws-nav-btn"],
                    elem_id="ws-history-btn",
                )
                settings_btn = gr.Button(
                    "Settings",
                    elem_classes=["ws-nav-btn"],
                    elem_id="ws-settings-btn",
                )

    return {
        "generation": generation,
        # back-compat alias: app.py reads header["preset"] to seed about-block.
        "preset": preset_state,
        "preset_state": preset_state,
        "preset_fast": preset_fast,
        "preset_quality": preset_quality,
        "history_btn": history_btn,
        "settings_btn": settings_btn,
    }
