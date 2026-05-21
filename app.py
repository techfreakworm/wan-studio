"""Wan Studio — Gradio entry point.

Phase 0: UI shell only. Sidebar + header + per-mode tabs render, but no actual
generation runs yet. Run with `python app.py` to verify the shell builds.

Phase 1+ will wire `pipelines.{t2v,i2v,...}` into each tab's Generate button.
"""
from __future__ import annotations

import gradio as gr

from pipelines import BY_KEY, ModelCard, Preset, modes_in, resolve
from ui import build_all_tabs, build_header, build_sidebar, MODE_PILLS
from utils import detect


CSS = """
#wan-studio-header { padding: 8px 16px; border-bottom: 1px solid #2a2a2a; }
#wan-studio-sidebar { padding: 12px 8px; border-right: 1px solid #2a2a2a; min-height: 80vh; }
#wan-studio-sidebar button { width: 100%; text-align: left; margin-bottom: 4px; }
.warning-banner { background: #443811; border-left: 3px solid #d4a23b; padding: 8px 12px; border-radius: 4px; }
"""


def build() -> gr.Blocks:
    backend = detect()

    with gr.Blocks(
        title="Wan Studio",
        theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"),
        css=CSS,
    ) as demo:
        # ── Header ───────────────────────────────────────────────────────────
        header = build_header()

        # ── Main area: sidebar + tabs ────────────────────────────────────────
        with gr.Row(equal_height=False):
            sidebar = build_sidebar()
            with gr.Column(scale=10):
                tabs = build_all_tabs()

        # Backend banner (only visible locally / off-ZeroGPU)
        if not backend.is_zerogpu:
            gr.HTML(
                f"<div style='padding:6px 12px;background:#222;border-radius:4px;font-size:0.9em;'>"
                f"💻 <b>Local backend:</b> {backend.label} · "
                f"dtype={backend.dtype} · vae={backend.vae_dtype}"
                f"</div>",
                elem_id="local-backend-banner",
            )

        # ── Tab switching ────────────────────────────────────────────────────
        all_tab_keys = ["t2v", "i2v", "ti2v", "flf2v", "v2v", "vace", "s2v", "animate",
                        "gallery", "settings"]

        def _show_only(key: str):
            return [gr.update(visible=k == key) for k in all_tab_keys]

        for mode_key, _ in MODE_PILLS:
            sidebar[f"mode_{mode_key}"].click(
                fn=lambda k=mode_key: _show_only(k),
                outputs=[tabs[k]["tab"] for k in all_tab_keys],
            )
        sidebar["gallery_btn"].click(
            fn=lambda: _show_only("gallery"),
            outputs=[tabs[k]["tab"] for k in all_tab_keys],
        )
        sidebar["settings_btn"].click(
            fn=lambda: _show_only("settings"),
            outputs=[tabs[k]["tab"] for k in all_tab_keys],
        )
        header["history_btn"].click(
            fn=lambda: _show_only("gallery"),
            outputs=[tabs[k]["tab"] for k in all_tab_keys],
        )
        header["settings_btn"].click(
            fn=lambda: _show_only("settings"),
            outputs=[tabs[k]["tab"] for k in all_tab_keys],
        )

        # ── Generation-aware mode availability ───────────────────────────────
        # (Stubbed: only updates the Settings about block for now. Real disabling
        # of unavailable mode buttons lands in Phase 1.)
        def _refresh_about(generation: str, preset: str) -> str:
            modes = modes_in(generation)  # type: ignore[arg-type]
            preset_str = "fast" if preset.startswith("Fast") else "quality"
            return (
                f"**Wan Studio v0.1** · backend `{backend.label}`\n\n"
                f"**Generation:** {generation}\n\n"
                f"**Available modes in this generation:** {', '.join(modes)}\n\n"
                f"**Preset:** {preset_str}\n\n"
                f"*Phase 0 — UI shell only. Generation handlers wired in Phase 1.*"
            )

        header["generation"].change(
            fn=_refresh_about,
            inputs=[header["generation"], header["preset"]],
            outputs=[tabs["settings"]["about"]],
        )
        header["preset"].change(
            fn=_refresh_about,
            inputs=[header["generation"], header["preset"]],
            outputs=[tabs["settings"]["about"]],
        )

        # Seed the About block on load.
        demo.load(
            fn=_refresh_about,
            inputs=[header["generation"], header["preset"]],
            outputs=[tabs["settings"]["about"]],
        )

    return demo


def main():
    demo = build()
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()
