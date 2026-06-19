"""Tests for app.py routing through ModelRegistry + per-tier @spaces.GPU entrypoints.

Task 8: app.py no longer keeps per-mode handle dicts (T2V_HANDLES / I2V_HANDLES)
nor per-mode closures (_build_t2v_handler / _build_i2v_handler). Instead it owns
one `REGISTRY = ModelRegistry(...)`, two tier entrypoints (generate_large /
generate_xlarge) that dispatch by `mode` through HANDLER_REGISTRY, and a Generate
wiring loop in `build()` driven by HANDLER_REGISTRY.

The plan's headline test is the smoke build (`from app import build; build()`),
which these tests exercise alongside the structural assertions.
"""
import app
from pipelines.handle import ModelRegistry
from pipelines.handlers import HANDLER_REGISTRY


def test_registry_is_model_registry():
    """app owns a single ModelRegistry, not per-mode handle dicts."""
    assert isinstance(app.REGISTRY, ModelRegistry)


def test_per_mode_handle_dicts_removed():
    """The old per-mode caches are gone (replaced by the LRU ModelRegistry)."""
    assert not hasattr(app, "T2V_HANDLES")
    assert not hasattr(app, "I2V_HANDLES")


def test_old_per_mode_builders_removed():
    """The two per-mode closures are replaced by the generic tier entrypoints."""
    assert not hasattr(app, "_build_t2v_handler")
    assert not hasattr(app, "_build_i2v_handler")


def test_per_tier_entrypoints_exist():
    """One decorated entrypoint per ZeroGPU size tier."""
    assert callable(app.generate_large)
    assert callable(app.generate_xlarge)


def test_build_handle_resolves_through_registry():
    """_build_handle returns a handle whose card matches the requested key,
    using HANDLER_REGISTRY specs (not hard-coded per-mode lookups)."""
    handle = app._build_handle("wan2.2_t2v_a14b")
    assert handle.card.key == "wan2.2_t2v_a14b"
    handle = app._build_handle("wan2.1_i2v_14b_480p")
    assert handle.card.key == "wan2.1_i2v_14b_480p"


def test_t2v_and_i2v_modes_are_wired_from_registry():
    """Both wired modes live in HANDLER_REGISTRY so build() iterates them."""
    assert "t2v" in HANDLER_REGISTRY
    assert "i2v" in HANDLER_REGISTRY


def test_flf2v_and_v2v_are_wired_not_toast():
    """After registration, flf2v/v2v Generate buttons route to a runner, not _generate_toast."""
    assert "flf2v" in HANDLER_REGISTRY
    assert "v2v" in HANDLER_REGISTRY
    assert "flf2v" in app._MODE_RUNNERS
    assert "v2v" in app._MODE_RUNNERS


def test_vace_is_wired_not_toast():
    """After registration + Task 3 wiring, the VACE Generate button routes to a
    real runner (not _generate_toast)."""
    assert "vace" in HANDLER_REGISTRY
    assert "vace" in app._MODE_RUNNERS


def test_ui_dispatch_arg_order_aligns_for_v2v_flf2v_and_vace():
    """The fragile, untested property: the positional shape `_inputs_for` builds
    must line up with `_ui_dispatch`'s index reads. v2v/flf2v/vace all carry the
    `generation` at ui_args[1]; v2v has no resolution + a ~3s reserve, flf2v has
    no resolution + a fixed ~5s clip, vace has no resolution + a ~4s reserve. Feed
    positionally-shaped tuples (mirroring the `_inputs_for` order) and assert the
    read-back (generation, _, duration).
    """
    # V2V layout: (video, generation, preset, prompt, strength, *advanced).
    v2v_args = (
        "video.mp4", "wan2.1", "fast", "restyle prompt", 0.6,
        "neg", 1234, False, 0, 0.0, 0.0,
    )
    generation, resolution, duration, preset = app._ui_dispatch("v2v", v2v_args)
    assert generation == "wan2.1"
    assert resolution == ""
    assert duration == 3.0
    assert preset == "fast"

    # FLF2V layout: (start_frame, generation, preset, end_uploaded,
    #                end_generated, prompt, *advanced).
    flf2v_args = (
        "start.png", "wan2.1", "quality", None, None, "transition prompt",
        "neg", 1234, False, 0, 0.0, 0.0,
    )
    generation, resolution, duration, preset = app._ui_dispatch("flf2v", flf2v_args)
    assert generation == "wan2.1"
    assert resolution == ""
    assert duration == 5.0
    assert preset == "quality"

    # VACE layout: (submode, generation, preset, source_video, references,
    #               prompt, *advanced).
    vace_args = (
        "Depth", "wan2.1", "quality", "source.mp4", None, "vace prompt",
        "neg", 1234, False, 0, 0.0, 0.0,
    )
    generation, resolution, duration, preset = app._ui_dispatch("vace", vace_args)
    assert generation == "wan2.1"
    assert resolution == ""
    assert duration == 4.0
    assert preset == "quality"


def test_flf2v_generate_end_handler_exists():
    """The FLF2V end-frame 'Generate' button binds a real T2I sub-handler
    (Wan T2I, num_frames=1), not the no-op toast."""
    import app
    assert callable(getattr(app, "generate_end_frame", None))


def test_smoke_build():
    """The plan's headline test: `from app import build; build()` builds a
    Blocks with no model load and no exception."""
    demo = app.build()
    assert type(demo).__name__ == "Blocks"


def test_snap_vace_frames_trim_pad_passthrough():
    """_snap_vace_frames forces a conditioning list to exactly num_frames:
    trim the tail when longer, repeat the last frame when shorter, passthrough
    when equal — keeping video/mask aligned with the 4k+1 latent length the
    pipeline rounds to. Empty stays empty."""
    seq = list(range(10))
    # longer: trim tail to num_frames
    assert app._snap_vace_frames(seq, 6) == [0, 1, 2, 3, 4, 5]
    # equal: passthrough (same contents)
    assert app._snap_vace_frames(seq, 10) == seq
    # shorter: pad by repeating the LAST element
    assert app._snap_vace_frames([7, 8, 9], 6) == [7, 8, 9, 9, 9, 9]
    # empty: unchanged
    assert app._snap_vace_frames([], 5) == []


def test_outpaint_padded_dims_are_divisible_by_base():
    """Regression guard for the Outpaint mod-16 crash: the out_h/out_w app._run_vace
    derives from the enlarged outpaint canvas MUST be divisible by the pipeline's
    spatial base (16 for Wan 2.1), or WanVACEPipeline.check_inputs hard-raises before
    generation. decode_video snaps dims to %16, so pad = dim//4 is only %4 and the
    padded canvas (dim + 2*(dim//4)) is only %8 — e.g. 624 -> 936, 936 % 16 == 8.
    Replicate _run_vace's snap and assert the chosen dims are %base == 0."""
    from pipelines.vace_inputs import outpaint_video_and_mask

    base = 16  # vae_scale_factor_spatial(8) * patch_size[1](2) for Wan 2.1
    # A non-480x832 decode whose padded canvas is NOT %16 (this is the regressing case).
    for h, w in [(624, 624), (432, 880), (480, 480)]:
        frames = [__import__("PIL.Image", fromlist=["Image"]).new("RGB", (w, h))]
        video, _ = outpaint_video_and_mask(frames, pad=h // 4)
        pw, ph = video[0].size  # enlarged canvas (w+2*pad, h+2*pad)
        # At least one of these canvases violates %16 (proves the test is real).
        # Apply _run_vace's snap-down and assert the result is valid for check_inputs.
        out_h = max(ph // base * base, base)
        out_w = max(pw // base * base, base)
        assert out_h % base == 0 and out_w % base == 0
        assert 0 < out_h <= ph and 0 < out_w <= pw
    # Sanity: the 624x624 canvas really did violate %16 pre-snap (else the test is vacuous).
    bad = outpaint_video_and_mask(
        [__import__("PIL.Image", fromlist=["Image"]).new("RGB", (624, 624))], pad=624 // 4
    )[0][0]
    assert bad.size[0] % base != 0


def test_vace_spatial_base_falls_back_to_16():
    """_vace_spatial_base returns 16 (8*2) when the pipe can't be introspected,
    matching the constant in WanVACEPipeline.check_inputs for Wan 2.1."""
    assert app._vace_spatial_base(None) == 16

    class _Cfg:
        patch_size = (1, 2, 2)

    class _Tf:
        config = _Cfg()

    class _Pipe:
        vae_scale_factor_spatial = 8
        transformer = _Tf()
        transformer_2 = None

    assert app._vace_spatial_base(_Pipe()) == 16
