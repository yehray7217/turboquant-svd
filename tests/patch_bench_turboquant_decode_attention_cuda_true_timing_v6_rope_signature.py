#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

start = s.find("def _safe_apply_rope(")
if start < 0:
    raise SystemExit("[FAIL] Could not find _safe_apply_rope().")

end_marker = "\n\ndef _get_attn_config("
end = s.find(end_marker, start)
if end < 0:
    raise SystemExit("[FAIL] Could not find end marker after _safe_apply_rope().")

new_func = '''def _safe_apply_rope(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_ids: Optional[torch.Tensor],
    position_embeddings: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply LLaMA RoPE while respecting the local Transformers API variant.

    Two common APIs exist:

    1) Newer form:
         rotary_emb(x, position_ids) -> (cos, sin)
         apply_rotary_pos_emb(q, k, cos, sin)

       Here cos/sin are already selected for the current positions.

    2) Older form:
         rotary_emb(x, seq_len=kv_seq_len) -> (cos_table, sin_table)
         apply_rotary_pos_emb(q, k, cos_table, sin_table, position_ids)

       Here cos/sin are full tables and position_ids must select rows.

    The previous decode patch blindly tried rotary_emb(v, position_ids).
    On an older forward(x, seq_len=...) implementation, that passes the
    position-id tensor as positional seq_len, which produced a full-length
    RoPE table and broadcasted q/k from Q=1 to Q=2048.
    """
    cos = sin = None
    rope_mode = None

    if position_embeddings is not None:
        if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            rope_mode = "already_positioned"

    rotary = getattr(module, "rotary_emb", None)

    if (cos is None or sin is None) and rotary is not None:
        try:
            rotary_sig = inspect.signature(rotary.forward)
            rotary_param_names = set(rotary_sig.parameters.keys())
        except Exception:
            rotary_param_names = set()

        # Newer HF API: forward(x, position_ids)
        if "position_ids" in rotary_param_names and position_ids is not None:
            out = rotary(v, position_ids)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                cos, sin = out[0], out[1]
                rope_mode = "already_positioned"

        # Older HF API: forward(x, seq_len=...)
        elif "seq_len" in rotary_param_names:
            if position_ids is not None and torch.is_tensor(position_ids) and position_ids.numel() > 0:
                kv_seq_len = int(position_ids.detach().max().item()) + 1
            else:
                kv_seq_len = int(k.shape[-2])

            out = rotary(v, seq_len=max(int(k.shape[-2]), int(kv_seq_len)))
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                cos, sin = out[0], out[1]
                rope_mode = "full_table"

        elif position_ids is not None:
            try:
                out = rotary(v, position_ids)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
                    rope_mode = "already_positioned"
            except TypeError:
                out = rotary(v)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
                    rope_mode = "already_positioned"
        else:
            out = rotary(v)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                cos, sin = out[0], out[1]
                rope_mode = "already_positioned"

    if cos is None or sin is None:
        return q, k

    if hf_apply_rotary_pos_emb is not None:
        if rope_mode == "full_table" and position_ids is not None:
            try:
                return hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            except TypeError:
                pass

        try:
            return hf_apply_rotary_pos_emb(q, k, cos, sin)
        except TypeError:
            pass

    if rope_mode == "full_table" and position_ids is not None:
        if cos.ndim == 2:
            cos = cos[position_ids]
            sin = sin[position_ids]
        elif cos.ndim == 3:
            batch = cos.shape[0]
            if batch == 1 and position_ids.shape[0] > 1:
                cos = cos.expand(position_ids.shape[0], -1, -1)
                sin = sin.expand(position_ids.shape[0], -1, -1)
            gather_idx = position_ids.to(device=cos.device, dtype=torch.long).unsqueeze(-1).expand(-1, -1, cos.shape[-1])
            cos = torch.gather(cos, dim=1, index=gather_idx)
            sin = torch.gather(sin, dim=1, index=gather_idx)

    return _apply_rope_fallback(q, k, cos, sin)
'''

s = s[:start] + new_func + s[end:]
p.write_text(s, encoding="utf-8")
print("[OK] Patched _safe_apply_rope() with signature-aware old/new Transformers RoPE handling.")
