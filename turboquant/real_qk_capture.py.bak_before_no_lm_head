from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CapturedQK:
    queries: torch.Tensor          # [B,Hq,Q,D]
    keys: torch.Tensor             # [B,Hq,T,D], K/V heads expanded to query heads if needed
    input_ids: torch.Tensor        # [B,T]
    layer_idx: int
    rope_applied: bool
    rope_detail: str
    num_attention_heads: int
    num_key_value_heads: int
    key_heads_expanded: bool


def _resolve_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """
    Resolve the decoder backbone for common AutoModelForCausalLM wrappers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise ValueError(
        "Could not resolve decoder backbone with `.layers`. "
        "This benchmark currently targets Llama-family style models."
    )


def _resolve_attention_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    backbone = _resolve_backbone(model)
    layers = backbone.layers
    if not (0 <= int(layer_idx) < len(layers)):
        raise IndexError(f"layer_idx={layer_idx} outside [0, {len(layers) - 1}].")
    layer = layers[int(layer_idx)]
    if not hasattr(layer, "self_attn"):
        raise ValueError(f"Layer {layer_idx} has no `.self_attn`.")
    return layer.self_attn


def _extract_layer_input_hidden(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    layer_idx: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Use `output_hidden_states=True` and return the hidden state entering layer_idx.
    Hugging Face convention: hidden_states[0] = embeddings, hidden_states[i]
    enters decoder layer i for standard decoder-only stacks.
    """
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    outputs = model(**kwargs)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states.")
    if int(layer_idx) >= len(hidden_states):
        raise RuntimeError(
            f"hidden_states has {len(hidden_states)} entries; cannot read layer_idx={layer_idx}."
        )
    return hidden_states[int(layer_idx)]


def _attention_shape(
    attn: torch.nn.Module,
    model: torch.nn.Module,
) -> tuple[int, int, int]:
    cfg = getattr(model, "config", None)

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and cfg is not None:
        num_heads = getattr(cfg, "num_attention_heads", None)

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None and cfg is not None:
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None or num_heads is None:
            raise ValueError("Could not infer attention head_dim.")
        head_dim = int(hidden_size) // int(num_heads)

    if num_heads is None or num_kv_heads is None:
        raise ValueError("Could not infer attention head counts.")

    return int(num_heads), int(num_kv_heads), int(head_dim)


def _repeat_kv_to_query_heads(
    keys: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, bool]:
    if num_attention_heads == num_key_value_heads:
        return keys, False
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads; "
            f"got {num_attention_heads}, {num_key_value_heads}."
        )
    repeat = num_attention_heads // num_key_value_heads
    return keys.repeat_interleave(repeat, dim=1).contiguous(), True


def _try_apply_rope(
    model: torch.nn.Module,
    attn: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, str]:
    """
    Best-effort RoPE application for Llama-family Transformers versions.

    The API changed across Transformers releases. We try several known call
    patterns and return a detailed status string. Failure is explicit rather
    than silent.
    """
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except Exception as exc:
        return q, k, False, f"apply_rotary_pos_emb import failed: {type(exc).__name__}: {exc}"

    rope_candidates = []
    if hasattr(attn, "rotary_emb"):
        rope_candidates.append(("attn.rotary_emb", attn.rotary_emb))
    backbone = _resolve_backbone(model)
    if hasattr(backbone, "rotary_emb"):
        rope_candidates.append(("backbone.rotary_emb", backbone.rotary_emb))

    if not rope_candidates:
        return q, k, False, "No rotary_emb attribute found on attention or backbone."

    call_errors: list[str] = []
    for rope_name, rope in rope_candidates:
        patterns = [
            ("rope(k, position_ids)", lambda: rope(k, position_ids)),
            ("rope(k, seq_len=T)", lambda: rope(k, seq_len=k.shape[-2])),
            ("rope(k)", lambda: rope(k)),
        ]
        for pattern_name, call in patterns:
            try:
                result = call()
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    call_errors.append(f"{rope_name}:{pattern_name} returned non-(cos,sin).")
                    continue
                cos, sin = result

                # Transformers versions differ:
                # - older Llama-2 stacks often require:
                #     apply_rotary_pos_emb(q, k, cos, sin, position_ids)
                # - newer variants may accept:
                #     apply_rotary_pos_emb(q, k, cos, sin)
                #
                # Try the Llama-2-compatible path first, then fall back.
                apply_errors: list[str] = []
                for apply_name, apply_call in [
                    (
                        "apply_rotary_pos_emb(..., position_ids)",
                        lambda: apply_rotary_pos_emb(q, k, cos, sin, position_ids),
                    ),
                    (
                        "apply_rotary_pos_emb(...)",
                        lambda: apply_rotary_pos_emb(q, k, cos, sin),
                    ),
                ]:
                    try:
                        q_rope, k_rope = apply_call()
                        return (
                            q_rope.contiguous(),
                            k_rope.contiguous(),
                            True,
                            f"Applied via {rope_name}:{pattern_name}:{apply_name}",
                        )
                    except Exception as apply_exc:
                        apply_errors.append(
                            f"{apply_name} failed: {type(apply_exc).__name__}: {apply_exc}"
                        )

                call_errors.extend(
                    [f"{rope_name}:{pattern_name}:{msg}" for msg in apply_errors]
                )
                continue
            except Exception as exc:
                call_errors.append(
                    f"{rope_name}:{pattern_name} failed: {type(exc).__name__}: {exc}"
                )

    return q, k, False, " | ".join(call_errors[-6:])


@torch.no_grad()
def capture_llama_style_qk(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    layer_idx: int,
    num_query_tokens: int = 1,
    apply_rope: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> CapturedQK:
    """
    Capture attention q/k states for a Llama-family layer.

    Output queries are the last `num_query_tokens` query vectors. Keys span the
    full sequence. For GQA/MQA models, key heads are expanded to query-head
    count so dense qK^T logits and TurboQuant reference logits share [B,H,Q,T].
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B,T], got {tuple(input_ids.shape)}.")
    if num_query_tokens <= 0:
        raise ValueError("num_query_tokens must be positive.")
    if num_query_tokens > input_ids.shape[1]:
        raise ValueError("num_query_tokens cannot exceed sequence length.")

    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    hidden = _extract_layer_input_hidden(
        model,
        input_ids,
        layer_idx=int(layer_idx),
        attention_mask=attention_mask,
    )

    attn = _resolve_attention_layer(model, int(layer_idx))
    num_heads, num_kv_heads, head_dim = _attention_shape(attn, model)

    if not hasattr(attn, "q_proj") or not hasattr(attn, "k_proj"):
        raise ValueError("Attention module must expose q_proj and k_proj.")

    B, T, _ = hidden.shape
    q_full = attn.q_proj(hidden).view(B, T, num_heads, head_dim).transpose(1, 2)
    k_full = attn.k_proj(hidden).view(B, T, num_kv_heads, head_dim).transpose(1, 2)

    rope_applied = False
    rope_detail = "RoPE disabled by flag."
    if apply_rope:
        position_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
        q_full, k_full, rope_applied, rope_detail = _try_apply_rope(
            model,
            attn,
            q_full,
            k_full,
            position_ids=position_ids,
        )

    k_expanded, expanded = _repeat_kv_to_query_heads(
        k_full,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )
    q_selected = q_full[:, :, -int(num_query_tokens):, :].contiguous()

    return CapturedQK(
        queries=q_selected.to(torch.float32).contiguous(),
        keys=k_expanded.to(torch.float32).contiguous(),
        input_ids=input_ids.detach().contiguous(),
        layer_idx=int(layer_idx),
        rope_applied=bool(rope_applied),
        rope_detail=str(rope_detail),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        key_heads_expanded=bool(expanded),
    )
