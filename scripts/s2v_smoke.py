#!/usr/bin/env python3
"""S2V MPS smoke — port the upstream wan.WanS2V (speech-to-video) to Apple MPS.

S2V is NOT in diffusers; this vendors the upstream `wan` package and shims it onto
MPS: device cuda→mps, cuda.synchronize/empty_cache→no-op, amp.autocast('cuda')→
nullcontext (model already bf16), wan's SDPA attention→key-chunked flash (the same
MPS long-key fix), dist guarded (wan checks dist.is_initialized()), wav2vec2 stays on
CPU (AudioEncoder default). Loads S2V's OWN checkpoint (audio cross-attn) + its bundled
VAE/UMT5/wav2vec2 via the wan loader. Saves an mp4 + start/mid/end frames.

Usage:
  python scripts/s2v_smoke.py --ckpt ~/.cache/.../Wan2.2-S2V-14B --image X.jpg --audio talk.wav --frames 17
"""
import argparse
import contextlib
import os
import sys

import torch

REPO = os.path.expanduser("~/Projects/llm/wan-studio")
VENDOR = os.path.join(REPO, "vendor/Wan2.2")
sys.path.insert(0, REPO)
sys.path.insert(0, VENDOR)


def _install_mps_shims():
    # 1) device cuda->mps. Replace torch.device with a CALLABLE OBJECT (not a plain
    #    function) that ALSO delegates `|` (so `torch.device | None` annotations,
    #    evaluated lazily by many torch submodules at import, still build a UnionType)
    #    and isinstance() to the real class. This avoids the whack-a-mole of
    #    `function | None` TypeErrors during `import wan` → diffusers → torchao → …
    _orig_device = torch.device

    class _DeviceFactory:
        def __call__(self, spec=None, *a, **k):
            if isinstance(spec, str) and spec.startswith("cuda"):
                return _orig_device("mps")
            if spec is None:
                return _orig_device(*a, **k)
            return _orig_device(spec, *a, **k)

        def __or__(self, other):       # torch.device | None  (annotation use)
            return _orig_device | other

        def __ror__(self, other):      # None | torch.device
            return other | _orig_device

        def __instancecheck__(self, obj):   # isinstance(x, torch.device)
            return isinstance(obj, _orig_device)

    torch.device = _DeviceFactory()

    # 1b) MPS has NO float64. wan uses float64 heavily for RoPE/complex precision
    #     (model.py, model_s2v.py, motioner.py, s2v_utils.py). Downcast every float64
    #     request to float32 — precision loss is negligible for inference. Patch BEFORE
    #     the .cuda() shim so its self.to() routes through here too.
    _t_to = torch.Tensor.to

    def _to(self, *a, **k):
        a = tuple(torch.float32 if x is torch.float64 else x for x in a)
        if k.get("dtype") is torch.float64:
            k = {**k, "dtype": torch.float32}
        return _t_to(self, *a, **k)
    torch.Tensor.to = _to

    _t_type = torch.Tensor.type

    def _type(self, dtype=None, *a, **k):
        if dtype in (torch.float64, torch.DoubleTensor, "torch.DoubleTensor"):
            dtype = torch.float32
        if dtype is None:
            return _t_type(self)
        return _t_type(self, dtype, *a, **k)
    torch.Tensor.type = _type
    torch.Tensor.double = lambda self, *a, **k: _to(self, torch.float32)

    # 1c) defensive .cuda() -> .to(mps) on tensors + modules
    _mps = _orig_device("mps")
    torch.Tensor.cuda = lambda self, *a, **k: _to(self, _mps)
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(_mps)

    # 2) cuda.synchronize / empty_cache / set_device -> no-op (or mps)
    torch.cuda.synchronize = lambda *a, **k: (torch.mps.synchronize()
                                              if torch.backends.mps.is_available() else None)
    torch.cuda.empty_cache = lambda *a, **k: (torch.mps.empty_cache()
                                              if torch.backends.mps.is_available() else None)
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.is_available = lambda: True  # wan gates some paths on this; we map to mps
    torch.cuda.current_device = lambda: 0   # used as default-arg at class-def time
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_name = lambda *a, **k: "mps"
    torch.cuda.get_device_capability = lambda *a, **k: (0, 0)

    # 3) autocast: wan relies on `torch.amp.autocast('cuda', dtype=param_dtype)` to
    #    harmonize fp32↔bf16 across the whole forward (e.g. fp32 wav2vec2 audio emb into
    #    bf16 audio-cross-attn). Map that to a REAL `autocast('mps', dtype=bf16)` so ops
    #    get cast exactly like on CUDA. The float64 RoPE-precision blocks
    #    (@amp.autocast(dtype=float64)) → DISABLED autocast (full fp32; MPS has no fp64).
    #    enabled=False decorators → no-op. Must be BOTH a context manager AND a decorator.
    import functools
    _orig_autocast = torch.amp.autocast

    def _resolve_ctx(args, kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None:
            for x in args:
                if isinstance(x, torch.dtype):
                    dtype = x
                    break
        enabled = kwargs.get("enabled", True)
        mps_ok = torch.backends.mps.is_available()
        # fp32/fp64 autocast request = "I want higher precision than ambient" → DISABLE
        # autocast for the block (full fp32; MPS has no fp64). Covers the time-embedding
        # block `with amp.autocast(dtype=float32)` which asserts e.dtype==float32, and the
        # RoPE-freq `with amp.autocast(dtype=float64)` blocks.
        if dtype in (torch.float64, torch.float32):
            return _orig_autocast("mps", enabled=False) if mps_ok else contextlib.nullcontext()
        if enabled and dtype in (torch.float16, torch.bfloat16) and mps_ok:
            # cache_enabled=False: the autocast casted-weight cache accumulates across
            # the sampling loop on MPS → grows into swap → OOM at ~step N. We don't need
            # it (weights are already bf16; nothing to cache).
            return _orig_autocast("mps", dtype=dtype, cache_enabled=False)
        return contextlib.nullcontext()

    class _AutocastShim:
        def __init__(self, *a, **k):
            self._a, self._k, self._ctx = a, k, None

        def __enter__(self):
            self._ctx = _resolve_ctx(self._a, self._k)
            return self._ctx.__enter__()

        def __exit__(self, *e):
            return self._ctx.__exit__(*e)

        def __call__(self, func):   # decorator use
            @functools.wraps(func)
            def wrapper(*aa, **kk):
                with _resolve_ctx(self._a, self._k):
                    return func(*aa, **kk)
            return wrapper
    torch.amp.autocast = _AutocastShim
    torch.autocast = _AutocastShim
    import torch.cuda.amp as _camp
    _camp.autocast = _AutocastShim
    torch.cuda.amp.autocast = _AutocastShim
    # stash the real autocast so the chunked-flash attention can DISABLE autocast
    # around its fp32 online-softmax (else autocast downcasts those matmuls to bf16
    # → the MPSNDArrayMatrixMultiplication "different datatype" assertion).
    torch._ORIG_AUTOCAST = _orig_autocast


def _trace_matmuls():
    """Log dtype-mismatched matmul/bmm/baddbmm/linear/einsum with a short stack, flushed
    before the op runs — the last line before the Metal abort pinpoints the culprit."""
    import traceback as _tb

    def _fdt(x):
        return x.dtype if torch.is_tensor(x) else None

    def _wan_frame():
        for fr in reversed(_tb.extract_stack()[:-2]):
            if "/wan/" in fr.filename:
                return f"{fr.filename.split('/wan/')[-1]}:{fr.lineno} {fr.name}() | {fr.line}"
        return "<no wan frame>"

    def _wrap(name, fn, dtype_args):
        def w(*a, **k):
            dts = [_fdt(a[i]) for i in dtype_args if i < len(a)]
            dts = [d for d in dts if d is not None and d.is_floating_point]
            if len(set(dts)) > 1:
                print(f"MATMUL_MISMATCH {name} {dts}  @ {_wan_frame()}", flush=True)
            return fn(*a, **k)
        return w
    torch.matmul = _wrap("matmul", torch.matmul, (0, 1))
    torch.bmm = _wrap("bmm", torch.bmm, (0, 1))
    torch.baddbmm = _wrap("baddbmm", torch.baddbmm, (0, 1, 2))
    torch.einsum = _wrap("einsum", torch.einsum, (1, 2))  # einsum(eq, a, b, ...)
    torch.nn.functional.linear = _wrap("linear", torch.nn.functional.linear, (0, 1, 2))
    torch.Tensor.__matmul__ = _wrap("@", torch.Tensor.__matmul__, (0, 1))
    print("[s2v] matmul dtype-mismatch tracer ON (incl @ / einsum)", flush=True)


def _patch_wan_attention():
    """Replace wan's attention()/flash_attention() with an MPS implementation. The s2v
    DiT + motioner call flash_attention() DIRECTLY (which asserts q.device=='cuda'), so
    we must patch flash_attention in EVERY importing namespace, not just the dispatcher.
    wan layout: q,k,v are [B, L, H, D]. Handles q_scale + per-batch k_lens masking; uses
    SDPA for short key seqs and key-chunked online-softmax (fp32) for long ones (the MPS
    fused-SDPA long-key bug). No GQA in s2v (Nq==Nk)."""
    KEY_THR = int(os.getenv("WAN_STUDIO_MPS_ATTN_CHUNK_THRESHOLD", "8192"))
    QC = int(os.getenv("WAN_STUDIO_MPS_ATTN_QUERY_CHUNK", "2048"))
    KC = int(os.getenv("WAN_STUDIO_MPS_ATTN_KEY_CHUNK", "2048"))
    F = torch.nn.functional

    def _chunked(qd, kd, vd, scale):
        _disable = getattr(torch, "_ORIG_AUTOCAST", None)
        _ac = _disable("mps", enabled=False) if _disable else contextlib.nullcontext()
        B, H, Lq, D = qd.shape
        Lk = kd.shape[2]
        tiles = []
        with _ac:
            for qi in range(0, Lq, QC):
                qt = qd[:, :, qi:qi + QC].float()
                cq = qt.shape[2]
                m = torch.full((B, H, cq, 1), float("-inf"), device=qd.device, dtype=torch.float32)
                l = torch.zeros((B, H, cq, 1), device=qd.device, dtype=torch.float32)
                acc = torch.zeros((B, H, cq, D), device=qd.device, dtype=torch.float32)
                for ki in range(0, Lk, KC):
                    kt = kd[:, :, ki:ki + KC].float()
                    vt = vd[:, :, ki:ki + KC].float()
                    s = torch.matmul(qt, kt.transpose(-1, -2)) * scale
                    mn = torch.maximum(m, s.amax(-1, keepdim=True))
                    p = torch.exp(s - mn)
                    al = torch.exp(m - mn)
                    l = l * al + p.sum(-1, keepdim=True)
                    acc = acc * al + torch.matmul(p, vt)
                    m = mn
                tiles.append((acc / l).to(qd.dtype))
        return torch.cat(tiles, dim=2)

    def _core(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
              q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
              dtype=torch.bfloat16, **kw):
        out_dtype = q.dtype
        qd = q.to(dtype)
        if q_scale is not None:
            qd = qd * q_scale
        kd = k.to(dtype)
        vd = v.to(dtype)
        # [B, L, H, D] -> [B, H, L, D]
        qd = qd.transpose(1, 2)
        kd = kd.transpose(1, 2)
        vd = vd.transpose(1, 2)
        B, H, Lq, D = qd.shape
        Lk = kd.shape[2]
        scale = softmax_scale if softmax_scale is not None else (D ** -0.5)
        # per-batch key-padding mask from k_lens
        attn_mask = None
        if k_lens is not None:
            idx = torch.arange(Lk, device=kd.device)
            keep = idx[None, :] < k_lens.to(kd.device).view(B, 1)  # [B, Lk]
            if not bool(keep.all()):
                attn_mask = torch.zeros(B, 1, 1, Lk, device=kd.device, dtype=qd.dtype)
                attn_mask.masked_fill_(~keep[:, None, None, :], float("-inf"))
        if attn_mask is not None or Lk <= KEY_THR or causal:
            out = F.scaled_dot_product_attention(
                qd, kd, vd, attn_mask=attn_mask, is_causal=causal,
                dropout_p=dropout_p, scale=scale)
        else:
            out = _chunked(qd, kd, vd, scale)
        return out.transpose(1, 2).contiguous().to(out_dtype)

    import wan.modules.attention as _wa
    import wan.modules.model as _wm
    import wan.modules.s2v.model_s2v as _ms2v
    import wan.modules.s2v.motioner as _mot
    for mod in (_wa, _wm, _ms2v, _mot):
        if hasattr(mod, "attention"):
            mod.attention = _core
        if hasattr(mod, "flash_attention"):
            mod.flash_attention = _core
    print("[s2v] wan attention + flash_attention -> MPS impl (q_scale + k_lens + chunked)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--prompt", default="a person talking, natural motion, cinematic")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--out", default=os.path.join(REPO, "tests/outputs/verify_wan2.2_s2v_14b_firstrun.mp4"))
    a = ap.parse_args()

    sys.path.insert(0, "scripts")
    from memcheck import SerialLock

    _install_mps_shims()
    if os.getenv("WAN_S2V_TRACE_MATMUL") == "1":
        _trace_matmuls()
    # decord has no reliable arm64-macOS wheel; speech2video imports VideoReader at
    # module top-level but only uses it for pose_video (we pass None). Stub it.
    import types as _types
    if "decord" not in sys.modules:
        _decord = _types.ModuleType("decord")
        _decord.VideoReader = object
        _decord.cpu = lambda *a, **k: None
        _decord.gpu = lambda *a, **k: None
        sys.modules["decord"] = _decord
    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    _patch_wan_attention()

    cfg = WAN_CONFIGS["s2v-14B"]
    with SerialLock():
        print("[s2v] building WanS2V on MPS…", flush=True)
        s2v = wan.WanS2V(config=cfg, checkpoint_dir=a.ckpt, device_id=0, rank=0,
                         t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
                         convert_model_dtype=True)
        # Wan2_1_VAE sets self.dtype=bf16 (mean/std bf16) but NEVER casts self.model
        # weights — they stay fp32 from the checkpoint, relying on CUDA autocast to
        # harmonize (dead on MPS). Cast VAE conv weights to bf16, AND wrap encode/decode
        # to cast their inputs to bf16 (generate() feeds a mix of fp32/bf16 from several
        # call sites; autocast normally papered over it).
        s2v.vae.model = s2v.vae.model.to(torch.bfloat16)
        # The wrapper's mean/std/scale must also be bf16: decode does `z / scale[1] +
        # scale[0]`; if scale is fp32 it upcasts the bf16 latent to fp32 → conv2(fp32)
        # vs bf16 weight crash. Force the whole VAE dtype regime to bf16.
        s2v.vae.dtype = torch.bfloat16
        s2v.vae.mean = s2v.vae.mean.to(torch.bfloat16)
        s2v.vae.std = s2v.vae.std.to(torch.bfloat16)
        s2v.vae.scale = [s2v.vae.mean, 1.0 / s2v.vae.std]
        _bf16 = torch.bfloat16

        def _cast_list(x):
            if isinstance(x, (list, tuple)):
                return [t.to(_bf16) if torch.is_tensor(t) else t for t in x]
            return x.to(_bf16) if torch.is_tensor(x) else x
        _oenc, _odec = s2v.vae.encode, s2v.vae.decode
        # encode INPUT and OUTPUT cast to bf16: the wrapper's .float() latents would
        # otherwise hit the bf16 DiT patch_embedding (autocast normally re-cast them).
        s2v.vae.encode = lambda videos, *a, **k: _cast_list(_oenc(_cast_list(videos), *a, **k))

        # decode: cast input to bf16; leave fp32 output (final pixels) as-is. CRITICAL:
        # empty_cache FIRST — s2v decodes motion-context+target (~23 latent → ~89 pixel
        # frames), a large allocation. After 20 sampling steps the MPS pool is fragmented;
        # without a pre-clear the decode tips into swap (~111GB). Clearing first lets the
        # decode start from a clean pool (~70-90GB) so it fits regardless of step count.
        def _decode_wrap(zs, *a, **k):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return _odec(_cast_list(zs), *a, **k)
        s2v.vae.decode = _decode_wrap

        # Precision-sensitive modulation layers: the model computes the time embedding
        # under `with amp.autocast(dtype=float32)` and ASSERTS e/e0 are fp32. On CUDA that
        # autocast upcasts the bf16 weights to fp32; MPS autocast supports ONLY bf16/fp16,
        # so we instead keep these two layers in fp32 (their block runs autocast-disabled
        # → fp32 input @ fp32 weight → fp32 e). Other layers stay bf16.
        s2v.noise_model.time_embedding.float()
        s2v.noise_model.time_projection.float()
        # Head_S2V.forward computes self.head(...) INSIDE `with amp.autocast(float32)`
        # (asserts e fp32). Its Linear weight must be fp32 too (MPS can't upcast via
        # autocast). WanLayerNorm has no affine params, so .float() only touches the
        # head Linear + modulation. Attention blocks wrap ONLY element-wise modulation
        # in fp32 (their Linears run outside, under bf16 autocast) → no conversion needed.
        s2v.noise_model.head.float()

        # Bound MPS allocator growth across the sampling loop: empty_cache after every
        # DiT forward (cond + uncond, 2×/step). Upstream only empties conditionally
        # (offload_model). Without this the allocator pool grows into swap over ~20 steps
        # → OOM (4 steps fit, 20 didn't). Cheap vs. the OOM.
        _orig_fwd = s2v.noise_model.forward

        def _fwd_ec(*a, **k):
            out = _orig_fwd(*a, **k)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return out
        s2v.noise_model.forward = _fwd_ec
        print("[s2v] VAE model -> bf16 + encode/decode input-cast; generating…", flush=True)
        video = s2v.generate(
            input_prompt=a.prompt, ref_image_path=a.image, audio_path=a.audio,
            enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None,
            num_repeat=1, pose_video=None, max_area=MAX_AREA_CONFIGS[a.size],
            infer_frames=a.frames, shift=5.0, sample_solver="unipc",
            sampling_steps=a.steps, guide_scale=4.5, seed=42,
            # offload_model=True: move the 14B DiT (~28GB) to CPU before the VAE decode
            # of ~89 frames — the decode peaked at 107GB with the DiT resident. Upstream's
            # intended mechanism for exactly this. Our .cpu()/device shims handle the move.
            offload_model=True,
            init_first_frame=True,
        )
    # video: tensor [C, T, H, W] in [-1,1] (wan convention)
    import numpy as np
    from PIL import Image
    v = video.detach().float().cpu()
    v = ((v.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()  # T,H,W,C
    import imageio
    imageio.mimsave(a.out, list(v), fps=16, codec="libx264", output_params=["-pix_fmt", "yuv420p"], macro_block_size=8)
    rawdir = a.out.replace(".mp4", "_RAW")
    os.makedirs(rawdir, exist_ok=True)
    for lbl, i in (("start", 0), ("mid", len(v) // 2), ("end", len(v) - 1)):
        Image.fromarray(v[i]).save(os.path.join(rawdir, f"{lbl}.png"))
    print(f"S2V_SMOKE_DONE frames={len(v)} -> {a.out} (raw {rawdir})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
