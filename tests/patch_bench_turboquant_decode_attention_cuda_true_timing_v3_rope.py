#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

old = '''def _safe_apply_rope(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_ids: Optional[torch.Tensor],
    position_embeddings: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = sin = None

    if position_embeddings is not None:
        if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]

    if cos is None or sin is None:
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            return q, k

        if position_ids is not None:
            try:
                out = rotary(v, position_ids)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
            except Exception:
                pass

        if cos is None or sin is None:
            try:
                out = rotary(v, seq_len=int(k.shape[-2]))
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
            except Exception:
                pass

        if cos is None or sin is None:
            try:
                out = rotary(v)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
            except Exception:
                pass

    if cos is None or sin is None:
        return q, k

    if hf_apply_rotary_pos_emb is not None:
        try:
            if position_ids is not None:
                return hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        except Exception:
            pass
        try:
            return hf_apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            pass

    return _apply_rope_fallback(q, k, cos, sin)
'''

new = '''def _safe_apply_rope(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_ids: Optional[torch.Tensor],
    position_embeddings: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE without double-indexing cos/sin.

    Newer LLaMA paths often provide `position_embeddings=(cos, sin)`, or
    obtain them through `rotary_emb(v, position_ids)`. In both cases, cos/sin
    are already selected for the current position(s).

    Older paths may produce a full cos/sin table and require
    `apply_rotary_pos_emb(..., position_ids)` to select rows.

    The previous version always tried `apply_rotary_pos_emb(..., position_ids)`
    first, even when cos/sin were already position-selected. In decode with
    position_ids around 2048 and cos/sin length 1, this triggers CUDA
    IndexKernel out-of-bounds asserts.
    """
    cos = sin = None
    cos_sin_already_positioned = False

    if position_embeddings is not None:
        if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            cos_sin_already_positioned = True

    if cos is None or sin is None:
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            return q, k

        if position_ids is not None:
            try:
                out = rotary(v, position_ids)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
                    cos_sin_already_positioned = True
            except TypeError:
                pass

        if cos is None or sin is None:
            try:
                out = rotary(v, seq_len=int(k.shape[-2]))
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
                    cos_sin_already_positioned = False
            except TypeError:
                pass

        if cos is None or sin is None:
            try:
                out = rotary(v)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
                    cos_sin_already_positioned = True
            except TypeError:
                pass

    if cos is None or sin is None:
        return q, k

    if hf_apply_rotary_pos_emb is not None:
        if (not cos_sin_already_positioned) and position_ids is not None:
            try:
                return hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            except TypeError:
                pass

        try:
            return hf_apply_rotary_pos_emb(q, k, cos, sin)
        except TypeError:
            pass

    return _apply_rope_fallback(q, k, cos, sin)
'''

if old not in s:
    raise SystemExit("[FAIL] Could not find the existing _safe_apply_rope() block. Source drifted.")

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("[OK] Patched _safe_apply_rope(): avoid double position_ids indexing for already-positioned cos/sin.")
