"""Runtime MPS correctness patches for diffusers Wan models.

Imported once at pipeline-package load (pipelines/__init__) so production (app.py)
AND local_verify both get them. **Gated to MPS** — a no-op on CUDA/ZeroGPU/CPU, so
it never changes the deployed-Space behaviour.

THE 720p NOISE BUG (root-caused 2026-06-21/22 via scripts/probe_720p.py)
------------------------------------------------------------------------
Every 720p-native Wan mode decoded to confetti on MPS while 480p was clean. An
early-abort per-step probe showed the latent magnitude blowing up at the FIRST
denoise step (std ±5→±13, NO NaN) ONLY at the 720p grid. Localized by elimination:
  - RoPE strided-assign fix (contiguous interleave) → BYTE-IDENTICAL → NOT it.
  - even-grid (44×80) → still noise → NOT odd-patch-parity → it's SCALE.
  - per-block trace: block0.attn1(self) injects magnitude; only self-attention
    scales with the grid (norms/AdaLN are resolution-independent, cross-attn is
    query-parallel over 512 text tokens).
  - query-chunking → byte-identical noise → NOT a total-size trigger.
  - KEY-chunked online-softmax → latent std 0.74 (healthy, matches 480p) → FIXED.
So: the fused MPS SDPA returns INCORRECT output when the softmax reduces over a long
KEY dimension (~14k). The fix is a manual flash-attention (online softmax) that tiles
BOTH q and k so every exp/sum reduces over only _KEY_CHUNK keys (MPS-safe) and stays
memory-bounded (no OOM). Exact (fp32 accumulation). Verified end-to-end: i2v_720p and
flf2v_720p clean+adherent at native 720p; cross-validated vs the 480p baseline.

We patch the SHARED Wan self-attention processor in BOTH the base transformer AND the
Animate transformer (which ships its own copied WanAttnProcessor) — gated to MPS AND
key-seq > THRESHOLD so 480p / short sequences keep the fast fused SDPA (no regression).
"""
from __future__ import annotations

import os

import torch

_APPLIED = False

_KEY_SEQ_THRESHOLD = int(os.getenv("WAN_STUDIO_MPS_ATTN_CHUNK_THRESHOLD", "8192"))
_QUERY_CHUNK = int(os.getenv("WAN_STUDIO_MPS_ATTN_QUERY_CHUNK", "2048"))
_KEY_CHUNK = int(os.getenv("WAN_STUDIO_MPS_ATTN_KEY_CHUNK", "2048"))


def _make_patched_call(MOD):
    """Build a WanAttnProcessor.__call__ that uses MOD's own helpers
    (_get_qkv_projections / _get_added_kv_projections / dispatch_attention_fn) and
    routes the long-key MAIN self-attention through key-chunked manual flash."""

    def _chunked_dispatch(query, key, value, **kw):
        kseq = key.shape[1]
        if (kseq <= _KEY_SEQ_THRESHOLD or kw.get("attn_mask") is not None
                or kw.get("is_causal")):
            return MOD.dispatch_attention_fn(query, key, value, **kw)
        # manual flash (key-chunked online softmax), fp32 accumulation. q/k/v [B,S,H,D].
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        B, H, Sq, D = q.shape
        Sk = k.shape[2]
        scale = D ** -0.5
        out_tiles = []
        for qi in range(0, Sq, _QUERY_CHUNK):
            qt = q[:, :, qi:qi + _QUERY_CHUNK].float()
            cq = qt.shape[2]
            m = torch.full((B, H, cq, 1), float("-inf"), device=q.device, dtype=torch.float32)
            l = torch.zeros((B, H, cq, 1), device=q.device, dtype=torch.float32)
            acc = torch.zeros((B, H, cq, D), device=q.device, dtype=torch.float32)
            for ki in range(0, Sk, _KEY_CHUNK):
                kt = k[:, :, ki:ki + _KEY_CHUNK].float()
                vt = v[:, :, ki:ki + _KEY_CHUNK].float()
                s = torch.matmul(qt, kt.transpose(-1, -2)) * scale
                m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
                p = torch.exp(s - m_new)
                alpha = torch.exp(m - m_new)
                l = l * alpha + p.sum(dim=-1, keepdim=True)
                acc = acc * alpha + torch.matmul(p, vt)
                m = m_new
            out_tiles.append((acc / l).to(query.dtype))
        return torch.cat(out_tiles, dim=2).transpose(1, 2).contiguous()

    def _patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                      attention_mask=None, rotary_emb=None):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = MOD._get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(hs, freqs_cos, freqs_sin):
                x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out_even = x1 * cos - x2 * sin
                out_odd = x1 * sin + x2 * cos
                out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
                return out.type_as(hs)
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = MOD._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = MOD.dispatch_attention_fn(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = _chunked_dispatch(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    return _patched_call


def apply_mps_patches() -> bool:
    """Idempotently install the MPS Wan patches. Returns True if applied."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        if not torch.backends.mps.is_available():
            return False
    except Exception:
        return False

    patched = []
    from diffusers.models.transformers import transformer_wan as TW
    mods = [("transformer_wan", TW)]
    try:
        from diffusers.models.transformers import transformer_wan_animate as TWA
        mods.append(("transformer_wan_animate", TWA))
    except Exception:
        pass

    for name, MOD in mods:
        if hasattr(MOD, "WanAttnProcessor"):
            MOD.WanAttnProcessor.__call__ = _make_patched_call(MOD)
            patched.append(name)

    _APPLIED = True
    print(f"[mps_patches] Wan attention KEY-CHUNKED flash on MPS (key-seq>{_KEY_SEQ_THRESHOLD}, "
          f"q={_QUERY_CHUNK},k={_KEY_CHUNK}) + contiguous RoPE — patched: {patched}", flush=True)
    return True
