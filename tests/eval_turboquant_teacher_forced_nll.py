#!/usr/bin/env python3
from __future__ import annotations

"""
Decode-only end-to-end timing for TurboQuant CUDA attention logits.

Scope
-----
This benchmark targets the intended accelerated regime already validated by the
project's readiness benchmarks:

  B=1, Q=1, D=128 decode attention logits

Benchmark policy
----------------
- Prefill keeps the model's original attention implementation.
- Decode steps with q_len == 1 and an existing KV cache are replaced by a
  TurboQuant CUDA nonfactor-combined logits path.
- The first replacement decode step is treated as a *prime/build* step:
    * per-layer rotation / sketch creation
    * per-layer scalar codebook fitting on the prefetched cache
    * full prefix K encoding into compressed TurboQuant state
  This step is reported separately and excluded from steady-state decode timing.
- Timed replacement decode steps append only the newly generated key's
  compressed representation, then run the CUDA logits kernel.

This gives a much more relevant runtime signal than the earlier full-sequence
PPL replacement timing, while still being honest that:
- the compressed cache append path is currently Python/PyTorch glue,
- packed-state concatenation is not yet an optimized allocator,
- this is a research integration benchmark, not a final serving engine.
"""

import argparse
import os
import inspect
import json
import math
import re
import statistics
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

try:
    from modules.svd_hf_registry import register_svdllama_auto_classes
    register_svdllama_auto_classes()
except Exception:
    pass

from turboquant import (
    make_random_orthogonal_rotation,
    fit_lloyd_scalar_codebook,
    make_gaussian_sketch,
    encode_turboquant_prod_keys,
)
from turboquant.qjl import qjl_project_query, qjl_residual_logits, qjl_encode_residual
from turboquant.rotation import rotate, inverse_rotate
from turboquant.scalar_quant import scalar_quantize, scalar_dequantize
from turboquant.scalar_lane_layout import pack_scalar_codes_lane_word_4bit, unpack_scalar_codes_lane_word_4bit
from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble, unpack_qjl_signs_lane_nibble
from turboquant.decode_pack_cuda_fastpath import DecodePackCudaFastPath, pack_qjl_signs_1bit_cuda, fused_residual_qjl256_pack_cuda, fused_scalar_quant_pack_4bit_cuda
from turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_combined_reduction_nonfactor_qjl512_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl512_combined_reduction_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_combined_reduction_nonfactor_qjl256_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl256_combined_reduction_logits_b1q1_d128_cuda,
)

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
except Exception:
    hf_apply_rotary_pos_emb = None


def _pack_qjl_signs_fast(signs: torch.Tensor) -> torch.Tensor:
    if signs.is_cuda and int(signs.shape[-1]) in {128, 256, 512}:
        return pack_qjl_signs_1bit_cuda(signs.contiguous(), 1).contiguous()
    return pack_qjl_signs_lane_nibble(signs).contiguous()


def _parse_dtype(name: str) -> torch.dtype | str:
    name = str(name).lower()
    if name == "auto":
        return "auto"
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    x = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
    if sin.ndim == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _apply_rope_decode_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode-only RoPE apply for q_len=1.

    Expected q/k:
      q: [B, H,   1, D]
      k: [B, Hkv, 1, D]

    Handles:
      full-table cos/sin: [seq_len, D] or [1, seq_len, D]
      already-positioned cos/sin: [B, 1, D], [1, D], or [D]
    """
    if int(q.shape[-2]) != 1 or int(k.shape[-2]) != 1:
        return _apply_rope_fallback(q, k, cos, sin)

    local_cos = cos
    local_sin = sin

    # Full-table old HF path.
    if position_ids is not None and torch.is_tensor(position_ids):
        if local_cos.ndim == 2 and local_cos.shape[0] > 1:
            # position_ids shape usually [B, 1].
            local_cos = local_cos[position_ids].squeeze(1)
            local_sin = local_sin[position_ids].squeeze(1)
        elif local_cos.ndim == 3 and local_cos.shape[-2] > 1:
            # [B or 1, seq_len, D]
            local_cos = local_cos[:, position_ids.reshape(-1), :].squeeze(-2)
            local_sin = local_sin[:, position_ids.reshape(-1), :].squeeze(-2)

    # Normalize to [B_or_1, 1, 1, D] for q/k broadcast.
    if local_cos.ndim == 1:
        local_cos = local_cos.view(1, 1, 1, -1)
        local_sin = local_sin.view(1, 1, 1, -1)
    elif local_cos.ndim == 2:
        local_cos = local_cos.unsqueeze(1).unsqueeze(1)
        local_sin = local_sin.unsqueeze(1).unsqueeze(1)
    elif local_cos.ndim == 3:
        local_cos = local_cos.unsqueeze(1)
        local_sin = local_sin.unsqueeze(1)
    elif local_cos.ndim == 4:
        pass
    else:
        return _apply_rope_fallback(q, k, cos, sin)

    return (
        (q * local_cos) + (_rotate_half(q) * local_sin),
        (k * local_cos) + (_rotate_half(k) * local_sin),
    )


def _safe_apply_rope_decode_fast(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_ids: Optional[torch.Tensor],
    position_embeddings: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Production decode-only RoPE fast path.

    This mirrors the verified TQ_PROFILE_ROPE_BREAKDOWN=1 fast-apply path,
    but removes profiling hooks. It keeps old/new HF rotary API handling and
    replaces generic HF apply_rotary_pos_emb with _apply_rope_decode_fast.
    """
    if int(q.shape[-2]) != 1 or int(k.shape[-2]) != 1:
        return _safe_apply_rope(
            module,
            q,
            k,
            v,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

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

        if "seq_len" in rotary_param_names:
            if position_ids is not None and torch.is_tensor(position_ids) and position_ids.numel() > 0:
                kv_seq_len = int(position_ids.detach().max().item()) + 1
            else:
                kv_seq_len = int(k.shape[-2])

            out = rotary(v, seq_len=max(int(k.shape[-2]), int(kv_seq_len)))
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                cos, sin = out[0], out[1]
                rope_mode = "full_table"

        elif "position_ids" in rotary_param_names and position_ids is not None:
            out = rotary(v, position_ids)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                cos, sin = out[0], out[1]
                rope_mode = "already_positioned"

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

    return _apply_rope_decode_fast(
        q,
        k,
        cos,
        sin,
        position_ids if rope_mode == "full_table" else None,
    )


def _safe_apply_rope(
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


def _get_attn_config(module: torch.nn.Module) -> tuple[int, int, int, int]:
    num_heads = getattr(module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(module, "num_attention_heads", None)
    if num_heads is None and hasattr(module, "config"):
        num_heads = getattr(module.config, "num_attention_heads", None)

    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None and hasattr(module, "config"):
        num_kv_heads = getattr(module.config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_heads

    head_dim = getattr(module, "head_dim", None)
    if head_dim is None and hasattr(module, "config"):
        hidden_size = int(getattr(module.config, "hidden_size"))
        head_dim = hidden_size // int(num_heads)

    num_kv_groups = getattr(module, "num_key_value_groups", None)
    if num_kv_groups is None:
        num_kv_groups = int(num_heads) // int(num_kv_heads)

    if num_heads is None or num_kv_heads is None or head_dim is None:
        raise RuntimeError(f"Could not infer attention layout for module={type(module).__name__}")

    return int(num_heads), int(num_kv_heads), int(num_kv_groups), int(head_dim)


def _reshape_projected(
    x: torch.Tensor,
    *,
    bsz: int,
    seqlen: int,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    return x.view(bsz, seqlen, heads, head_dim).transpose(1, 2).contiguous()


def _extract_hidden_and_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    if args:
        hidden_states = args[0]
    else:
        hidden_states = kwargs.get("hidden_states", None)
    if not torch.is_tensor(hidden_states):
        raise RuntimeError("Could not locate hidden_states in attention forward.")
    return hidden_states, kwargs


def _extract_past(kwargs: dict[str, Any]) -> Any:
    if "past_key_value" in kwargs:
        return kwargs.get("past_key_value")
    return kwargs.get("past_key_values")


def _cache_update(
    *,
    module: torch.nn.Module,
    past: Any,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    kwargs: dict[str, Any],
    cos: Any = None,
    sin: Any = None,
    profile_components: bool = False,
    state: Optional["PackedLayerState"] = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    # Cache-update root-cause diagnostics:
    # - past cache type
    # - whether cache_position is present
    # - which update branch is used
    # - inner CUDA time of the actual update/cat branch
    if state is not None:
        state.add_cache_past_type(type(past).__name__ if past is not None else "None")
        if kwargs.get("cache_position", None) is None:
            state.cache_position_absent_count += 1
        else:
            state.cache_position_present_count += 1

    if past is None:
        if state is not None:
            state.add_cache_update_path("past_none_return")
        return key_states, value_states, None

    if hasattr(past, "update"):
        if state is not None:
            state.add_cache_update_path("cache_object_update")

        layer_idx = int(getattr(module, "layer_idx", 0))
        cache_kwargs = {}
        if cos is not None:
            cache_kwargs["cos"] = cos
        if sin is not None:
            cache_kwargs["sin"] = sin
        cache_position = kwargs.get("cache_position", None)
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position

        def _do_cache_object_update():
            try:
                return past.update(key_states, value_states, layer_idx, cache_kwargs)
            except TypeError:
                if state is not None:
                    state.add_cache_update_path("cache_object_update_typeerror_fallback")
                return past.update(key_states, value_states, layer_idx)

        def _probe_dynamic_cache_cat_breakdown():
            """
            Diagnostic only. For this Transformers DynamicCache layout,
            old K/V tensors live in:
                past.key_cache[layer_idx]
                past.value_cache[layer_idx]
            Time equivalent K/V torch.cat separately without mutating cache.
            """
            if not bool(profile_components):
                return

            old_k = None
            old_v = None

            try:
                old_k = past.key_cache[int(getattr(state, "layer_idx", -1))]
                old_v = past.value_cache[int(getattr(state, "layer_idx", -1))]
            except Exception:
                pass

            if old_k is None or old_v is None:
                if state is not None:
                    state.add_component_ms("dynamic_cache_probe_key_cat", 0.0)
                    state.add_component_ms("dynamic_cache_probe_value_cat", 0.0)
                return

            _, key_cat_ms = _profile_cuda_ms(
                True,
                lambda: torch.cat([old_k, key_states], dim=-2),
            )
            _, value_cat_ms = _profile_cuda_ms(
                True,
                lambda: torch.cat([old_v, value_states], dim=-2),
            )

            if state is not None:
                state.add_component_ms("dynamic_cache_probe_key_cat", key_cat_ms)
                state.add_component_ms("dynamic_cache_probe_value_cat", value_cat_ms)

        _probe_dynamic_cache_cat_breakdown()


        pre_cache_update_sync_ms = None
        if bool(profile_components):
            _pre_cache_update_t0 = time.perf_counter()
            torch.cuda.synchronize()
            pre_cache_update_sync_ms = (
                time.perf_counter() - _pre_cache_update_t0
            ) * 1000.0

        if state is not None:
            state.add_component_ms(
                "pre_cache_update_cuda_synchronize_wall",
                pre_cache_update_sync_ms,
            )

        (key_states, value_states), ms = _profile_cuda_ms(
            bool(profile_components),
            _do_cache_object_update,
        )
        if state is not None:
            state.add_component_ms("cache_object_past_update_inner", ms)
        return key_states, value_states, past

    if isinstance(past, (tuple, list)) and len(past) >= 2:
        if state is not None:
            state.add_cache_update_path("tuple_or_list_cat")

        def _tuple_cat_update():
            key_states_local = torch.cat([past[0], key_states], dim=-2)
            value_states_local = torch.cat([past[1], value_states], dim=-2)
            return key_states_local, value_states_local

        (key_states, value_states), ms = _profile_cuda_ms(
            bool(profile_components),
            _tuple_cat_update,
        )
        if state is not None:
            state.add_component_ms("tuple_cache_cat_inner", ms)
        return key_states, value_states, (key_states, value_states)

    if state is not None:
        state.add_cache_update_path("unsupported_cache_type")
    raise RuntimeError(f"Unsupported past_key_value type: {type(past)!r}")


def _normalize_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=device)
    if mask.dtype != dtype:
        mask = mask.to(dtype)
    if mask.ndim == 4:
        return mask[..., :q_len, :kv_len]
    if mask.ndim == 2:
        keep = mask[:, None, None, :kv_len].to(torch.bool)
        out = torch.zeros((mask.shape[0], 1, q_len, kv_len), device=device, dtype=dtype)
        return out.masked_fill(~keep, torch.finfo(dtype).min)
    return None


def _event_time_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


def _summary_ms(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "p90_ms": float("nan"),
            "times_ms": [],
        }
    vals = sorted(float(x) for x in xs)
    p90_idx = min(len(vals) - 1, max(0, math.ceil(0.9 * len(vals)) - 1))
    return {
        "count": int(len(xs)),
        "mean_ms": float(statistics.mean(xs)),
        "median_ms": float(statistics.median(xs)),
        "min_ms": float(min(xs)),
        "max_ms": float(max(xs)),
        "p90_ms": float(vals[p90_idx]),
        "times_ms": [float(x) for x in xs],
    }



def _reference_tq_logits_from_packed(
    *,
    rotated_queries: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    centroids: torch.Tensor,
    scalar_lane_words: torch.Tensor,
    qjl_lane_nibbles: torch.Tensor,
    residual_norms: torch.Tensor,
    active_kv_len: int | None = None,
) -> torch.Tensor:
    """
    Slow PyTorch reference TQ logits for qjl_dim > 128 quality validation.

    Expected:
      rotated_queries:        [1,H,1,D]
      qjl_projected_queries:  [1,H,1,M]
      scalar_lane_words:      [1,H,T,D/2] packed 4-bit lane-word codes
      qjl_lane_nibbles:       [1,H,T,M/8]
      residual_norms:         [1,H,T]
      centroids:              [16]
    """
    T = int(scalar_lane_words.shape[-2])
    if active_kv_len is not None:
        T = min(T, int(active_kv_len))

    scalar_packed = scalar_lane_words[..., :T, :].contiguous()
    qjl_packed = qjl_lane_nibbles[..., :T, :].contiguous()
    norms = residual_norms[..., :T].contiguous()

    # Unpack scalar codes to [1,H,T,D].
    codes = unpack_scalar_codes_lane_word_4bit(scalar_packed).to(torch.long)
    deq = centroids.to(torch.float32)[codes].to(torch.float32)

    q_rot = rotated_queries.to(torch.float32)
    scalar_logits = torch.einsum("bhqd,bhtd->bhqt", q_rot, deq)

    qjl_dim = int(qjl_projected_queries.shape[-1])
    signs = unpack_qjl_signs_lane_nibble(qjl_packed, qjl_dim=qjl_dim).to(torch.float32)

    # qjl_residual_logits expects projected query and residual signs/norms.
    residual_logits = qjl_residual_logits(
        qjl_projected_queries.to(torch.float32),
        signs,
        norms.to(torch.float32),
    )

    return (scalar_logits + residual_logits).contiguous()


def _try_call_nonfactor_kernel(
    *,
    rotated_queries: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    centroids: torch.Tensor,
    scalar_lane_words: torch.Tensor,
    qjl_lane_nibbles: torch.Tensor,
    residual_norms: torch.Tensor,    active_kv_len: int | None = None,

) -> tuple[torch.Tensor, str]:
    """
    Call the project-local nonfactor CUDA wrapper.

    The CUDA wrapper is keyword-only in the current repo state. The earlier
    decode integration only had a narrow name mapping; once those aliases missed,
    it fell back to positional calls and failed with:

      takes 0 positional arguments but 6 were given

    This adapter:
      1) introspects the Python signature when possible,
      2) maps parameter names to the six tensors using robust heuristics,
      3) tries a small set of explicit keyword layouts used by project variants,
      4) emits the discovered signature and all failures if none match.
    """
    qjl_m = int(qjl_projected_queries.shape[-1])
    if qjl_m == 128:
        fn = turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda
    elif qjl_m == 256:
        fn = turboquant_full_4bit_lane_word_lane_nibble_qjl256_combined_reduction_logits_b1q1_d128_cuda
    elif qjl_m == 512:
        fn = turboquant_full_4bit_lane_word_lane_nibble_qjl512_combined_reduction_logits_b1q1_d128_cuda
    else:
        raise RuntimeError(f"Unsupported qjl_dim for nonfactor CUDA kernel: {qjl_m}")

    try:
        sig = inspect.signature(fn)
        sig_text = str(sig)
        params = list(sig.parameters.values())
    except Exception as e:
        sig = None
        sig_text = f"<inspect.signature failed: {type(e).__name__}: {e}>"
        params = []

    def choose_tensor(param_name: str):
        n = param_name.lower()

        if "active" in n and "kv" in n:
            return active_kv_len

        if ("rot" in n or "rotated" in n) and ("quer" in n or n.startswith("q_")):
            return rotated_queries

        if ("project" in n or "proj" in n) and ("quer" in n or "qjl" in n):
            return qjl_projected_queries

        if "centroid" in n or "codebook" in n:
            return centroids

        if "norm" in n:
            return residual_norms

        if (
            ("scalar" in n or "code" in n)
            and ("lane" in n or "word" in n or "packed" in n)
            and "qjl" not in n
        ):
            return scalar_lane_words

        if (
            "qjl" in n
            and ("lane" in n or "nibble" in n or "packed" in n or "sign" in n)
        ):
            return qjl_lane_nibbles

        if "sign" in n and "scalar" not in n:
            return qjl_lane_nibbles

        return None

    errors: list[str] = []

    # First: signature-driven heuristic mapping.
    if params:
        heuristic_kwargs: dict[str, torch.Tensor] = {}
        unresolved: list[str] = []
        keyword_compatible = True

        for p in params:
            if p.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                keyword_compatible = False
                unresolved.append(f"{p.name}:{p.kind}")
                continue

            t = choose_tensor(p.name)
            if t is None:
                unresolved.append(p.name)
            else:
                heuristic_kwargs[p.name] = t

        if keyword_compatible and not unresolved and heuristic_kwargs:
            try:
                return fn(**heuristic_kwargs), "signature_heuristic_kwargs:" + ",".join(heuristic_kwargs.keys())
            except Exception as e:
                errors.append(
                    "signature_heuristic_kwargs "
                    + repr(list(heuristic_kwargs.keys()))
                    + f": {type(e).__name__}: {e}"
                )
        else:
            errors.append(
                "signature_heuristic_unresolved "
                + f"signature={sig_text} unresolved={unresolved} mapped={list(heuristic_kwargs.keys())}"
            )

    # Second: explicit known/plausible keyword layouts.
    keyword_candidates: list[tuple[str, dict[str, torch.Tensor]]] = [
        (
            "kw_rot_qjl_centroids_packed_scalar_packed_qjl_norms",
            {
                "rotated_queries": rotated_queries,
                "qjl_projected_queries": qjl_projected_queries,
                "centroids": centroids,
                "packed_scalar_codes_lane_word": scalar_lane_words,
                "packed_qjl_signs_lane_nibble": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_rot_qjl_centroids_scalar_lane_qjl_lane_norms",
            {
                "rotated_queries": rotated_queries,
                "qjl_projected_queries": qjl_projected_queries,
                "centroids": centroids,
                "scalar_lane_words": scalar_lane_words,
                "qjl_lane_nibbles": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_rot_qjl_scalar_centroids_packed_scalar_packed_qjl_norms",
            {
                "rotated_queries": rotated_queries,
                "qjl_projected_queries": qjl_projected_queries,
                "scalar_centroids": centroids,
                "packed_scalar_codes_lane_word": scalar_lane_words,
                "packed_qjl_signs_lane_nibble": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_rot_proj_centroids_packed_codes_packed_signs_norms",
            {
                "rotated_queries": rotated_queries,
                "projected_queries": qjl_projected_queries,
                "centroids": centroids,
                "packed_codes_lane_word": scalar_lane_words,
                "packed_signs_lane_nibble": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_qrot_qproj_centroids_packed_codes_packed_signs_norms",
            {
                "q_rot": rotated_queries,
                "qjl_projected_query": qjl_projected_queries,
                "centroids": centroids,
                "packed_scalar_codes": scalar_lane_words,
                "packed_qjl_signs": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
    ]

    for name, kwargs in keyword_candidates:
        try:
            return fn(**kwargs), name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not call TurboQuant nonfactor combined CUDA kernel with keyword adapter. "
        f"Discovered signature: {sig_text}. Adapter attempts:\n"
        + "\n".join(errors)
    )


def _profile_component_enabled(name: str, global_enabled: bool) -> bool:
    if not bool(global_enabled):
        return False

    only = os.environ.get("TQ_PROFILE_ONLY", "").strip()
    if not only:
        return True

    allowed = {
        x.strip()
        for x in only.split(",")
        if x.strip()
    }
    return name in allowed


def _profile_cuda_ms(enabled: bool, fn):
    # Diagnostic-only synchronous CUDA event timing.
    # This intentionally synchronizes after each component so attribution is clear.
    # Do not use profiling-mode total runtime as a production latency number.
    if not enabled:
        return fn(), None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


@dataclass
class PackedLayerState:
    name: str
    layer_idx: int
    rotation: Optional[torch.Tensor] = None
    sketch: Optional[torch.Tensor] = None
    centroids: Optional[torch.Tensor] = None
    scalar_lane_words: Optional[torch.Tensor] = None
    qjl_lane_nibbles: Optional[torch.Tensor] = None
    residual_norms: Optional[torch.Tensor] = None
    scalar_lane_words_storage: Optional[torch.Tensor] = None
    qjl_lane_nibbles_storage: Optional[torch.Tensor] = None
    residual_norms_storage: Optional[torch.Tensor] = None
    compressed_cache_len: int = 0
    compressed_cache_capacity: int = 0
    dense_key_storage: Optional[torch.Tensor] = None
    dense_value_storage: Optional[torch.Tensor] = None
    dense_cache_len: int = 0
    dense_cache_capacity: int = 0
    build_calls: int = 0
    append_calls: int = 0
    decode_calls: int = 0
    kernel_adapter: Optional[str] = None
    last_kv_len: int = 0
    component_ms_total: dict[str, float] = field(default_factory=dict)
    component_ms_count: dict[str, int] = field(default_factory=dict)
    cache_update_path_counts: dict[str, int] = field(default_factory=dict)
    cache_update_past_type_counts: dict[str, int] = field(default_factory=dict)
    cache_position_present_count: int = 0
    cache_position_absent_count: int = 0

    def add_component_ms(self, name: str, ms: Optional[float]) -> None:
        if ms is None:
            return
        self.component_ms_total[name] = float(self.component_ms_total.get(name, 0.0) + float(ms))
        self.component_ms_count[name] = int(self.component_ms_count.get(name, 0) + 1)

    def add_cache_update_path(self, name: str) -> None:
        self.cache_update_path_counts[name] = int(self.cache_update_path_counts.get(name, 0) + 1)

    def add_cache_past_type(self, name: str) -> None:
        self.cache_update_past_type_counts[name] = int(self.cache_update_past_type_counts.get(name, 0) + 1)

    def ready(self) -> bool:
        return (
            self.rotation is not None
            and self.sketch is not None
            and self.centroids is not None
            and self.scalar_lane_words is not None
            and self.qjl_lane_nibbles is not None
            and self.residual_norms is not None
        )



class TQStaticValueCache(Cache):
    """
    Minimal HF-compatible cache for long-context TQ experiments.

    Purpose:
      - avoid DynamicCache torch.cat for dense V during decode
      - do not preallocate dense K
      - keep prefix dense K only for first compressed-prefix build
      - append V by slice-write into preallocated storage

    This is intentionally benchmark-local and only targets the LLaMA decode path
    used by tests/bench_turboquant_decode_attention_cuda_true_timing.py.
    """

    def __init__(self, *, reserve_tokens: int) -> None:
        self.key_cache = []
        self.value_cache = []
        self.value_storage = []
        self.cache_len = []
        self.cache_capacity = []
        self.reserve_tokens = max(0, int(reserve_tokens))
        self.seen_tokens = 0

    def __len__(self) -> int:
        return len(self.value_cache)

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= int(layer_idx):
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.value_storage.append(None)
            self.cache_len.append(0)
            self.cache_capacity.append(0)

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        self._ensure_layer(int(layer_idx))
        return int(self.cache_len[int(layer_idx)])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._ensure_layer(int(layer_idx))
        return int(self.cache_len[int(layer_idx)])

    def get_max_length(self) -> int:
        if not self.cache_capacity:
            return 0
        return int(max(self.cache_capacity))

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        self._ensure_layer(int(layer_idx))
        layer_i = int(layer_idx)

        add = int(value_states.shape[-2])

        # First call for this layer: prefill.
        if self.value_storage[layer_i] is None:
            prefix_len = int(value_states.shape[-2])
            capacity = prefix_len + int(self.reserve_tokens)

            v_shape = list(value_states.shape)
            v_shape[-2] = capacity

            storage = torch.empty(
                v_shape,
                dtype=value_states.dtype,
                device=value_states.device,
            )
            storage[..., :prefix_len, :].copy_(value_states)

            self.value_storage[layer_i] = storage
            self.cache_len[layer_i] = prefix_len
            self.cache_capacity[layer_i] = capacity

            # Keep prefix dense K only. Do not allocate expanded dense K.
            self.key_cache[layer_i] = key_states
            self.value_cache[layer_i] = storage[..., :prefix_len, :]

            if layer_i == 0:
                self.seen_tokens = prefix_len

            return self.key_cache[layer_i], self.value_cache[layer_i]

        # Decode update: append V by slice-write, do not append dense K.
        start = int(self.cache_len[layer_i])
        end = start + add
        capacity = int(self.cache_capacity[layer_i])

        if end > capacity:
            raise RuntimeError(
                "TQStaticValueCache capacity exceeded: "
                f"need end={end}, capacity={capacity}. "
                "Increase --compressed_cache_reserve_tokens."
            )

        self.value_storage[layer_i][..., start:end, :].copy_(value_states)
        self.cache_len[layer_i] = end

        # Keep dense K as prefix-only. TQ compressed K handles appended K.
        active_k = self.key_cache[layer_i]
        active_v = self.value_storage[layer_i][..., :end, :]

        self.value_cache[layer_i] = active_v

        if layer_i == 0:
            self.seen_tokens = end

        return active_k, active_v

    def to_legacy_cache(self):
        return tuple(
            (self.key_cache[i], self.value_cache[i])
            for i in range(len(self.value_cache))
        )


class TurboQuantDecodeAttentionPatcher:
    def __init__(
        self,
        *,
        scalar_bits: int,
        qjl_dim: int,
        lloyd_iters: int,
        max_codebook_samples: int,
        rotation_seed: int,
        sketch_seed: int,
        codebook_seed: int,
        verbose_build: bool,
        profile_components: bool,
        use_pack_cuda_fastpath: bool,
        compressed_cache_reserve_tokens: int,
        profile_append_inner: bool = False,
    ) -> None:
        if int(scalar_bits) != 4:
            raise ValueError("Current CUDA decode integration expects scalar_bits=4.")
        if int(qjl_dim) not in {128, 256, 512}:
            raise ValueError("Current CUDA decode integration currently expects qjl_dim in {128, 256, 512}.")
        self.scalar_bits = int(scalar_bits)
        self.qjl_dim = int(qjl_dim)
        self.lloyd_iters = int(lloyd_iters)
        self.max_codebook_samples = int(max_codebook_samples)
        self.rotation_seed = int(rotation_seed)
        self.sketch_seed = int(sketch_seed)
        self.codebook_seed = int(codebook_seed)
        self.verbose_build = bool(verbose_build)
        self.profile_components = bool(profile_components)
        self.use_pack_cuda_fastpath = bool(use_pack_cuda_fastpath) and int(qjl_dim) == 128
        self.pack_cuda_fastpath = None
        self.compressed_cache_reserve_tokens = max(0, int(compressed_cache_reserve_tokens))
        self.profile_append_inner = bool(profile_append_inner)
        self.states: dict[int, PackedLayerState] = {}
        self.original_forwards: list[tuple[torch.nn.Module, Any]] = []

    def _ensure_pack_cuda_fastpath(self, device: torch.device):
        if not self.use_pack_cuda_fastpath:
            return None
        if self.pack_cuda_fastpath is None:
            self.pack_cuda_fastpath = DecodePackCudaFastPath(
                scalar_ref_fn=pack_scalar_codes_lane_word_4bit,
                qjl_ref_fn=pack_qjl_signs_lane_nibble,
                device=device,
            )
            print(
                json.dumps({"pack_cuda_fastpath": self.pack_cuda_fastpath.summary()}),
                flush=True,
            )
        return self.pack_cuda_fastpath

    def _install_preallocated_compressed_cache(
        self,
        *,
        state: PackedLayerState,
        scalar_lane_words: torch.Tensor,
        qjl_lane_nibbles: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> None:
        prefix_len = int(scalar_lane_words.shape[-2])
        capacity = prefix_len + max(0, int(self.compressed_cache_reserve_tokens))

        scalar_shape = list(scalar_lane_words.shape)
        qjl_shape = list(qjl_lane_nibbles.shape)
        norm_shape = list(residual_norms.shape)

        scalar_shape[-2] = capacity
        qjl_shape[-2] = capacity
        norm_shape[-1] = capacity

        scalar_storage = torch.empty(
            scalar_shape,
            dtype=scalar_lane_words.dtype,
            device=scalar_lane_words.device,
        )
        qjl_storage = torch.empty(
            qjl_shape,
            dtype=qjl_lane_nibbles.dtype,
            device=qjl_lane_nibbles.device,
        )
        norm_storage = torch.empty(
            norm_shape,
            dtype=residual_norms.dtype,
            device=residual_norms.device,
        )

        scalar_storage[..., :prefix_len, :].copy_(scalar_lane_words)
        qjl_storage[..., :prefix_len, :].copy_(qjl_lane_nibbles)
        norm_storage[..., :prefix_len].copy_(residual_norms)

        state.scalar_lane_words_storage = scalar_storage
        state.qjl_lane_nibbles_storage = qjl_storage
        state.residual_norms_storage = norm_storage

        state.compressed_cache_len = prefix_len
        state.compressed_cache_capacity = capacity

        state.scalar_lane_words = scalar_storage[..., :prefix_len, :]
        state.qjl_lane_nibbles = qjl_storage[..., :prefix_len, :]
        state.residual_norms = norm_storage[..., :prefix_len]

    def _append_preallocated_compressed_cache(
        self,
        *,
        state: PackedLayerState,
        scalar_new: torch.Tensor,
        qjl_new: torch.Tensor,
        norms_new: torch.Tensor,
    ) -> None:
        if (
            state.scalar_lane_words_storage is None
            or state.qjl_lane_nibbles_storage is None
            or state.residual_norms_storage is None
        ):
            assert state.scalar_lane_words is not None
            assert state.qjl_lane_nibbles is not None
            assert state.residual_norms is not None

            self._install_preallocated_compressed_cache(
                state=state,
                scalar_lane_words=state.scalar_lane_words,
                qjl_lane_nibbles=state.qjl_lane_nibbles,
                residual_norms=state.residual_norms,
            )

        assert state.scalar_lane_words_storage is not None
        assert state.qjl_lane_nibbles_storage is not None
        assert state.residual_norms_storage is not None

        if os.environ.get("TQ_DEBUG_APPEND_FUNCTION_ENTRY", "0") == "1":
            try:
                print("[DEBUG append function entry]", {
                    "layer_idx": int(getattr(state, "layer_idx", -1)),
                    "append_calls": int(getattr(state, "append_calls", -1)),
                    "compressed_cache_len_before": int(getattr(state, "compressed_cache_len", -1)),
                    "scalar_new_shape": list(scalar_new.shape),
                    "qjl_new_shape": list(qjl_new.shape),
                    "norms_new_shape": list(norms_new.shape),
                    "scalar_new_dtype": str(scalar_new.dtype),
                    "qjl_new_dtype": str(qjl_new.dtype),
                    "norms_new_dtype": str(norms_new.dtype),
                }, flush=True)
            except Exception as e:
                print("[DEBUG append function entry ERROR]", repr(e), flush=True)

        if os.environ.get("TQ_SKIP_COMPRESSED_K_APPEND", "0") == "1":
            try:
                print("[DEBUG skip compressed K append]", {
                    "layer_idx": int(getattr(state, "layer_idx", -1)),
                    "compressed_cache_len": int(getattr(state, "compressed_cache_len", -1)),
                    "scalar_new_shape": list(scalar_new.shape),
                }, flush=True)
            except Exception:
                pass
            return

        start = int(state.compressed_cache_len)
        add = int(scalar_new.shape[-2])
        end = start + add

        if end > int(state.compressed_cache_capacity):
            raise RuntimeError(
                "Preallocated compressed cache capacity exceeded: "
                f"need end={end}, capacity={state.compressed_cache_capacity}. "
                "Increase --compressed_cache_reserve_tokens."
            )

        state.scalar_lane_words_storage[..., start:end, :].copy_(scalar_new)
        state.qjl_lane_nibbles_storage[..., start:end, :].copy_(qjl_new)
        state.residual_norms_storage[..., start:end].copy_(norms_new)

        state.compressed_cache_len = end
        state.scalar_lane_words = state.scalar_lane_words_storage[..., :end, :]
        state.qjl_lane_nibbles = state.qjl_lane_nibbles_storage[..., :end, :]
        state.residual_norms = state.residual_norms_storage[..., :end]

        if state.layer_idx == 0 and state.append_calls == 0:
            print(
                "[DEBUG prealloc contiguous]",
                "scalar=", state.scalar_lane_words.is_contiguous(),
                "qjl=", state.qjl_lane_nibbles.is_contiguous(),
                "norm=", state.residual_norms.is_contiguous(),
                "scalar_stride=", tuple(state.scalar_lane_words.stride()),
                "qjl_stride=", tuple(state.qjl_lane_nibbles.stride()),
                "norm_stride=", tuple(state.residual_norms.stride()),
                flush=True,
            )

    def _install_preallocated_dense_kv_cache(
        self,
        *,
        state: PackedLayerState,
        past: Any,
        full_keys_kv: torch.Tensor,
        full_values_kv: torch.Tensor,
    ) -> None:
        """
        Install full-capacity dense K/V cache storage for HF DynamicCache.

        full_keys_kv / full_values_kv shape:
            [B, num_kv_heads, T, head_dim]
        """
        if not hasattr(past, "key_cache") or not hasattr(past, "value_cache"):
            return

        layer_i = int(state.layer_idx)
        prefix_len = int(full_values_kv.shape[-2])
        capacity = prefix_len + max(0, int(self.compressed_cache_reserve_tokens))

        dense_cache_mode = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()
        if dense_cache_mode not in ("kv", "v_only", "dynamic_v", "dynamic_v_no_k"):
            raise RuntimeError(
                "Invalid TQ_DENSE_CACHE_MODE. Expected 'kv', 'v_only', 'dynamic_v', or 'dynamic_v_no_k', "
                f"got {dense_cache_mode!r}"
            )

        if dense_cache_mode == "dynamic_v":
            # No duplicate dense K/V storage. Keep HF DynamicCache tensors as-is.
            # This is a low-memory long-context mode; append falls back to
            # _cache_update / DynamicCache.update.
            state.dense_key_storage = None
            state.dense_value_storage = None
            state.dense_cache_len = prefix_len
            state.dense_cache_capacity = prefix_len
            past.key_cache[layer_i] = full_keys_kv
            past.value_cache[layer_i] = full_values_kv
            return

        v_shape = list(full_values_kv.shape)
        v_shape[-2] = capacity

        v_storage = torch.empty(
            v_shape,
            dtype=full_values_kv.dtype,
            device=full_values_kv.device,
        )
        v_storage[..., :prefix_len, :].copy_(full_values_kv)

        if dense_cache_mode == "kv":
            k_shape = list(full_keys_kv.shape)
            k_shape[-2] = capacity
            k_storage = torch.empty(
                k_shape,
                dtype=full_keys_kv.dtype,
                device=full_keys_kv.device,
            )
            k_storage[..., :prefix_len, :].copy_(full_keys_kv)
            state.dense_key_storage = k_storage
            past.key_cache[layer_i] = k_storage[..., :prefix_len, :]
        else:
            # TurboQuant logits use compressed K; keep the existing active dense K
            # view only for HF cache bookkeeping and avoid allocating a second
            # preallocated dense K storage.
            state.dense_key_storage = None
            past.key_cache[layer_i] = full_keys_kv

        state.dense_value_storage = v_storage
        state.dense_cache_len = prefix_len
        state.dense_cache_capacity = capacity

        past.value_cache[layer_i] = v_storage[..., :prefix_len, :]

    def _append_preallocated_dense_kv_cache(
        self,
        *,
        state: PackedLayerState,
        past: Any,
        key_states_new: torch.Tensor,
        value_states_new: torch.Tensor,
    ):
        """
        Append one/new token K/V into preallocated dense cache storage.

        Keeps key_cache/value_cache lengths consistent while avoiding
        DynamicCache.update's per-token torch.cat.
        """
        if not hasattr(past, "key_cache") or not hasattr(past, "value_cache"):
            return None

        layer_i = int(state.layer_idx)

        dense_cache_mode = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()
        if dense_cache_mode not in ("kv", "v_only", "dynamic_v", "dynamic_v_no_k"):
            raise RuntimeError(
                "Invalid TQ_DENSE_CACHE_MODE. Expected 'kv', 'v_only', 'dynamic_v', or 'dynamic_v_no_k', "
                f"got {dense_cache_mode!r}"
            )

        need_install = (
            state.dense_value_storage is None
            or int(state.dense_cache_capacity) <= 0
            or (dense_cache_mode == "kv" and state.dense_key_storage is None)
        )

        if need_install:
            try:
                old_k = past.key_cache[layer_i]
                old_v = past.value_cache[layer_i]
            except Exception:
                return None

            if old_k is None or old_v is None:
                return None

            self._install_preallocated_dense_kv_cache(
                state=state,
                past=past,
                full_keys_kv=old_k,
                full_values_kv=old_v,
            )

        assert state.dense_value_storage is not None
        if dense_cache_mode == "kv":
            assert state.dense_key_storage is not None

        start = int(state.dense_cache_len)
        add = int(value_states_new.shape[-2])
        end = start + add

        if end > int(state.dense_cache_capacity):
            raise RuntimeError(
                "Preallocated dense K/V cache capacity exceeded: "
                f"need end={end}, capacity={state.dense_cache_capacity}. "
                "Increase --compressed_cache_reserve_tokens."
            )

        state.dense_value_storage[..., start:end, :].copy_(value_states_new)

        state.dense_cache_len = end

        active_v = state.dense_value_storage[..., :end, :]

        if dense_cache_mode == "kv":
            state.dense_key_storage[..., start:end, :].copy_(key_states_new)
            active_k = state.dense_key_storage[..., :end, :]
        else:
            # Do not allocate preallocated dense K. Keep key cache logically
            # consistent using the existing active key view plus the new token.
            old_k = past.key_cache[layer_i]
            if old_k is None:
                active_k = key_states_new
            else:
                active_k = torch.cat([old_k, key_states_new], dim=-2)

        past.key_cache[layer_i] = active_k
        past.value_cache[layer_i] = active_v

        return active_k, active_v, past

    def _active_compressed_cache_inputs(
        self,
        state: PackedLayerState,
    ):
        """
        Return full contiguous compressed storage + logical active KV length
        when preallocation is active. Fall back to active tensors otherwise.
        """
        if (
            state.scalar_lane_words_storage is not None
            and state.qjl_lane_nibbles_storage is not None
            and state.residual_norms_storage is not None
            and int(state.compressed_cache_len) > 0
        ):
            return (
                state.scalar_lane_words_storage,
                state.qjl_lane_nibbles_storage,
                state.residual_norms_storage,
                int(state.compressed_cache_len),
            )

        assert state.scalar_lane_words is not None
        assert state.qjl_lane_nibbles is not None
        assert state.residual_norms is not None
        return (
            state.scalar_lane_words,
            state.qjl_lane_nibbles,
            state.residual_norms,
            int(state.scalar_lane_words.shape[-2]),
        )

    def _fit_state_from_full_k(
        self,
        *,
        state: PackedLayerState,
        full_keys: torch.Tensor,
        head_dim: int,
    ) -> None:
        device = full_keys.device
        # Build random matrices on CPU first, then move to CUDA.
        # torch.linalg.qr on CUDA can obscure earlier async CUDA failures by
        # reporting device-side assert at the rotation construction line.
        state.rotation = make_random_orthogonal_rotation(
            int(head_dim),
            seed=self.rotation_seed + int(state.layer_idx),
            device=torch.device("cpu"),
        ).to(device=device, dtype=torch.float32).contiguous()
        state.sketch = make_gaussian_sketch(
            int(head_dim),
            int(self.qjl_dim),
            seed=self.sketch_seed + int(state.layer_idx),
            device=torch.device("cpu"),
        ).to(device=device, dtype=torch.float32).contiguous()

        prefix_chunk_tokens = int(os.environ.get("TQ_PREFIX_BUILD_CHUNK_TOKENS", "0") or "0")

        if prefix_chunk_tokens > 0 and int(full_keys.shape[-2]) > prefix_chunk_tokens:
            # Avoid full-prefix rotated_key_samples transient allocation.
            # We only need samples for Lloyd codebook fitting, so collect bounded
            # scalar samples chunk by chunk.
            seq_len = int(full_keys.shape[-2])
            max_samples = int(self.max_codebook_samples)
            sample_chunks = []
            collected = 0

            for start in range(0, seq_len, prefix_chunk_tokens):
                end = min(seq_len, start + prefix_chunk_tokens)
                chunk_keys = full_keys[..., start:end, :].contiguous()

                rotated_chunk = torch.matmul(
                    chunk_keys.reshape(-1, int(head_dim)).to(torch.float32),
                    state.rotation.T.to(torch.float32),
                ).contiguous()

                flat = rotated_chunk.reshape(-1)
                remaining = max(0, max_samples - collected)
                if remaining <= 0:
                    del rotated_chunk, flat, chunk_keys
                    break

                take = min(int(flat.numel()), int(remaining))
                sample_chunks.append(flat[:take].contiguous())
                collected += int(take)

                del rotated_chunk, flat, chunk_keys

            if not sample_chunks:
                raise RuntimeError("No samples collected for chunked prefix centroid fitting.")

            rotated_key_samples = torch.cat(sample_chunks, dim=0).contiguous()

            if self.verbose_build:
                print(
                    json.dumps(
                        {
                            "tq_prefix_centroid_sample_chunked": True,
                            "layer_idx": int(state.layer_idx),
                            "seq_len": int(seq_len),
                            "chunk_tokens": int(prefix_chunk_tokens),
                            "sample_scalars": int(rotated_key_samples.numel()),
                        }
                    ),
                    flush=True,
                )

        else:
            rotated_key_samples = torch.matmul(
                full_keys.reshape(-1, int(head_dim)).to(torch.float32),
                state.rotation.T.to(torch.float32),
            ).contiguous()

        state.centroids = fit_lloyd_scalar_codebook(
            rotated_key_samples.reshape(-1),
            num_levels=1 << int(self.scalar_bits),
            max_iters=int(self.lloyd_iters),
            max_samples=int(self.max_codebook_samples),
            seed=self.codebook_seed + int(state.layer_idx),
        ).contiguous()

        del rotated_key_samples

        prefix_chunk_tokens = int(os.environ.get("TQ_PREFIX_BUILD_CHUNK_TOKENS", "0") or "0")

        if prefix_chunk_tokens > 0 and int(full_keys.shape[-2]) > prefix_chunk_tokens:
            scalar_chunks = []
            qjl_chunks = []
            norm_chunks = []

            seq_len = int(full_keys.shape[-2])
            for start in range(0, seq_len, prefix_chunk_tokens):
                end = min(seq_len, start + prefix_chunk_tokens)
                chunk_keys = full_keys[..., start:end, :].contiguous()

                encoding = encode_turboquant_prod_keys(
                    chunk_keys,
                    rotation=state.rotation,
                    centroids=state.centroids,
                    sketch=state.sketch,
                )

                scalar_chunks.append(
                    pack_scalar_codes_lane_word_4bit(encoding.codes).contiguous()
                )
                qjl_chunks.append(
                    pack_qjl_signs_lane_nibble(encoding.residual_signs).contiguous()
                )
                norm_chunks.append(encoding.residual_norms.contiguous())

                # Release large transient references before next chunk.
                del encoding, chunk_keys

            state.scalar_lane_words = torch.cat(scalar_chunks, dim=-2).contiguous()
            state.qjl_lane_nibbles = torch.cat(qjl_chunks, dim=-2).contiguous()
            state.residual_norms = torch.cat(norm_chunks, dim=-1).contiguous()

            if self.verbose_build:
                print(
                    json.dumps(
                        {
                            "tq_prefix_build_chunked": True,
                            "layer_idx": int(state.layer_idx),
                            "seq_len": int(seq_len),
                            "chunk_tokens": int(prefix_chunk_tokens),
                            "num_chunks": int(len(scalar_chunks)),
                        }
                    ),
                    flush=True,
                )

        else:
            encoding = encode_turboquant_prod_keys(
                full_keys,
                rotation=state.rotation,
                centroids=state.centroids,
                sketch=state.sketch,
            )
            state.scalar_lane_words = pack_scalar_codes_lane_word_4bit(encoding.codes).contiguous()
            state.qjl_lane_nibbles = pack_qjl_signs_lane_nibble(encoding.residual_signs).contiguous()
            state.residual_norms = encoding.residual_norms.contiguous()
        state.build_calls += 1
        state.last_kv_len = int(full_keys.shape[-2])

        if self.verbose_build:
            print(
                json.dumps(
                    {
                        "tq_decode_build": True,
                        "layer_idx": int(state.layer_idx),
                        "name": state.name,
                        "full_keys_shape": list(full_keys.shape),
                        "centroids_shape": list(state.centroids.shape),
                        "packed_scalar_shape": list(state.scalar_lane_words.shape),
                        "packed_qjl_shape": list(state.qjl_lane_nibbles.shape),
                        "residual_norm_shape": list(state.residual_norms.shape),
                    }
                ),
                flush=True,
            )

    def _append_new_k(
        self,
        *,
        state: PackedLayerState,
        new_keys: torch.Tensor,
    ) -> None:
        assert state.rotation is not None
        assert state.sketch is not None
        assert state.centroids is not None
        assert state.scalar_lane_words is not None
        assert state.qjl_lane_nibbles is not None
        assert state.residual_norms is not None

        inner_profile = bool(self.profile_components and self.profile_append_inner)

        use_fused_append_compressed_k = (
            os.environ.get("TQ_DISABLE_FUSED_APPEND_COMPRESSED_K", "").strip() != "1"
            and os.environ.get("TQ_RESIDUAL_MODE", "full").strip().lower() == "full"
            and state.scalar_lane_words_storage is not None
            and state.qjl_lane_nibbles_storage is not None
            and state.residual_norms_storage is not None
            and int(state.compressed_cache_len) > 0
        )

        if use_fused_append_compressed_k:
            from turboquant.decode_pack_cuda_fastpath import fused_append_compressed_k_cuda

            start = int(state.compressed_cache_len)
            add = int(new_keys.shape[-2])
            if add != 1:
                raise RuntimeError("fused append compressed K currently expects T=1.")
            end = start + add
            if end > int(state.compressed_cache_capacity):
                raise RuntimeError(
                    "Fused append compressed K capacity exceeded: "
                    f"need end={end}, capacity={state.compressed_cache_capacity}."
                )

            _, ms = _profile_cuda_ms(
                inner_profile,
                lambda: fused_append_compressed_k_cuda(
                    new_keys,
                    state.centroids,
                    state.rotation,
                    state.sketch,
                    state.scalar_lane_words_storage,
                    state.qjl_lane_nibbles_storage,
                    state.residual_norms_storage,
                    start,
                ),
            )
            state.add_component_ms("fused_append_compressed_k_cuda", ms)

            state.compressed_cache_len = end
            state.scalar_lane_words = state.scalar_lane_words_storage[..., :end, :]
            state.qjl_lane_nibbles = state.qjl_lane_nibbles_storage[..., :end, :]
            state.residual_norms = state.residual_norms_storage[..., :end]

            state.append_calls += 1
            state.last_kv_len = int(state.scalar_lane_words.shape[-2])
            return

        # -----------------------------------------------------------------
        # Encode new K, split using the same sequence as:
        # turboquant.turboquant_prod.encode_turboquant_prod_keys()
        # -----------------------------------------------------------------
        residual_mode = os.environ.get("TQ_RESIDUAL_MODE", "full").strip().lower()
        if residual_mode not in ("full", "rotated", "original"):
            raise RuntimeError(
                "Invalid TQ_RESIDUAL_MODE. Expected one of: full, rotated, original; "
                f"got {residual_mode!r}"
            )

        use_fused_rotate_quant_residual_qjl = (
            residual_mode == "full"
            and int(self.qjl_dim) == 128
            and os.environ.get("TQ_DISABLE_FUSED_ROTATE_QUANT_RESIDUAL_QJL", "").strip() != "1"
        )

        if use_fused_rotate_quant_residual_qjl:
            from turboquant.decode_pack_cuda_fastpath import fused_rotate_quant_residual_qjl_cuda

            (codes_raw, residual_signs_raw, residual_norms_raw), ms = _profile_cuda_ms(
                inner_profile,
                lambda: fused_rotate_quant_residual_qjl_cuda(
                    new_keys,
                    state.centroids,
                    state.rotation,
                    state.sketch,
                ),
            )
            state.add_component_ms("fused_rotate_quant_residual_qjl_new_k_cuda", ms)
            state.add_component_ms("encode_residual_mode_full_fused_rotate_quant", 0.0)

        else:
            if os.environ.get("TQ_DEBUG_APPEND_SHAPES", "").strip() == "1" and int(state.layer_idx) == 0:
                print(
                    "[DEBUG append shapes before rotate]",
                    "new_keys", tuple(new_keys.shape), new_keys.dtype, new_keys.stride(),
                    "rotation", tuple(state.rotation.shape), state.rotation.dtype, state.rotation.stride(),
                    "centroids", tuple(state.centroids.shape), state.centroids.dtype, state.centroids.stride(),
                    "sketch", tuple(state.sketch.shape), state.sketch.dtype, state.sketch.stride(),
                    flush=True,
                )

            rotated_keys, ms = _profile_cuda_ms(
                inner_profile,
                lambda: rotate(new_keys, state.rotation).to(torch.float32),
            )
            state.add_component_ms("encode_rotate_new_k", ms)

            if os.environ.get("TQ_DEBUG_APPEND_SHAPES", "").strip() == "1" and int(state.layer_idx) == 0:
                print(
                    "[DEBUG append shapes after rotate]",
                    "rotated_keys", tuple(rotated_keys.shape), rotated_keys.dtype, rotated_keys.stride(),
                    flush=True,
                )

            use_scalar_quant_cuda = (
                os.environ.get("TQ_ENABLE_SCALAR_QUANT_CUDA", "").strip() == "1"
                and int(rotated_keys.shape[-1]) == 128
                and state.centroids is not None
                and int(state.centroids.numel()) == 16
            )

            use_fused_scalar_quant_pack = (
                int(self.qjl_dim) == 256
                and os.environ.get("TQ_ENABLE_FUSED_SCALAR_QUANT_PACK", "1").strip() != "0"
            )

            if use_fused_scalar_quant_pack:
                (codes_raw, scalar_new_fused), ms = _profile_cuda_ms(
                    inner_profile,
                    lambda: fused_scalar_quant_pack_4bit_cuda(rotated_keys, state.centroids),
                )
                state.add_component_ms("fused_scalar_quant_pack_new_k", ms)
            elif use_scalar_quant_cuda:
                from turboquant.decode_pack_cuda_fastpath import scalar_quantize_16_cuda

                codes_raw, ms = _profile_cuda_ms(
                    inner_profile,
                    lambda: scalar_quantize_16_cuda(rotated_keys, state.centroids),
                )
                state.add_component_ms("encode_scalar_quantize_new_k_cuda", ms)
                scalar_new_fused = None
            else:
                codes_raw, ms = _profile_cuda_ms(
                    inner_profile,
                    lambda: scalar_quantize(rotated_keys, state.centroids),
                )
                state.add_component_ms("encode_scalar_quantize_new_k", ms)
                scalar_new_fused = None

            use_fused_residual_qjl = (
                residual_mode == "full"
                and int(self.qjl_dim) == 128
                and os.environ.get("TQ_DISABLE_FUSED_RESIDUAL_QJL", "").strip() != "1"
            )

            if use_fused_residual_qjl:
                from turboquant.decode_pack_cuda_fastpath import fused_dequant_residual_qjl_cuda

                (residual_signs_raw, residual_norms_raw), ms = _profile_cuda_ms(
                    inner_profile,
                    lambda: fused_dequant_residual_qjl_cuda(
                        new_keys,
                        codes_raw,
                        state.centroids,
                        state.rotation,
                        state.sketch,
                    ),
                )
                state.add_component_ms("fused_dequant_residual_qjl_new_k_cuda", ms)
                state.add_component_ms("encode_residual_mode_full_fused", 0.0)

            else:
                if residual_mode == "full":
                    reconstructed_rotated, ms = _profile_cuda_ms(
                        inner_profile,
                        lambda: scalar_dequantize(codes_raw, state.centroids),
                    )
                    state.add_component_ms("encode_scalar_dequantize_new_k", ms)

                    reconstructed_keys, ms = _profile_cuda_ms(
                        inner_profile,
                        lambda: inverse_rotate(reconstructed_rotated, state.rotation).to(torch.float32),
                    )
                    state.add_component_ms("encode_inverse_rotate_reconstruct_new_k", ms)

                    residual, ms = _profile_cuda_ms(
                        inner_profile,
                        lambda: new_keys.to(torch.float32) - reconstructed_keys,
                    )
                    state.add_component_ms("encode_residual_subtract_new_k", ms)

                elif residual_mode == "rotated":
                    state.add_component_ms("encode_scalar_dequantize_new_k_skipped", 0.0)
                    state.add_component_ms("encode_inverse_rotate_reconstruct_new_k_skipped", 0.0)
                    state.add_component_ms("encode_residual_subtract_new_k_skipped", 0.0)
                    residual = rotated_keys

                else:
                    state.add_component_ms("encode_scalar_dequantize_new_k_skipped", 0.0)
                    state.add_component_ms("encode_inverse_rotate_reconstruct_new_k_skipped", 0.0)
                    state.add_component_ms("encode_residual_subtract_new_k_skipped", 0.0)
                    residual = new_keys.to(torch.float32)

                if (
                    int(self.qjl_dim) == 256
                    and residual_mode == "full"
                    and os.environ.get("TQ_ENABLE_FUSED_RESIDUAL_QJL256_PACK", "1").strip() != "0"
                ):
                    (qjl_new_fused, residual_norms_raw), ms = _profile_cuda_ms(
                        inner_profile,
                        lambda: fused_residual_qjl256_pack_cuda(residual, state.sketch),
                    )
                    state.add_component_ms("fused_residual_qjl256_pack_new_k", ms)

                    if (
                        os.environ.get("TQ_DEBUG_FUSED_QJL256_APPEND_PARITY", "").strip() == "1"
                        and int(state.layer_idx) in {0, 2}
                        and int(state.append_calls) < 2
                    ):
                        (residual_signs_ref, residual_norms_ref) = qjl_encode_residual(
                            residual, state.sketch
                        )
                        qjl_new_ref = _pack_qjl_signs_fast(residual_signs_ref)
                        print(
                            "[DEBUG fused qjl256 append parity]",
                            {
                                "layer_idx": int(state.layer_idx),
                                "append_calls": int(state.append_calls),
                                "residual_shape": list(residual.shape),
                                "qjl_fused_shape": list(qjl_new_fused.shape),
                                "qjl_ref_shape": list(qjl_new_ref.shape),
                                "qjl_same": bool(torch.equal(qjl_new_fused, qjl_new_ref)),
                                "qjl_max_abs": float((qjl_new_fused.to(torch.int16) - qjl_new_ref.to(torch.int16)).abs().max().item()),
                                "qjl_mean_abs": float((qjl_new_fused.to(torch.int16) - qjl_new_ref.to(torch.int16)).abs().float().mean().item()),
                                "norm_max_abs": float((residual_norms_raw - residual_norms_ref).abs().max().item()),
                                "norm_mean_abs": float((residual_norms_raw - residual_norms_ref).abs().mean().item()),
                            },
                            flush=True,
                        )

                    residual_signs_raw = None
                else:
                    (residual_signs_raw, residual_norms_raw), ms = _profile_cuda_ms(
                        inner_profile,
                        lambda: qjl_encode_residual(residual, state.sketch),
                    )
                    state.add_component_ms("encode_qjl_encode_residual_new_k", ms)
                    qjl_new_fused = None
                state.add_component_ms(f"encode_residual_mode_{residual_mode}", 0.0)

        codes, ms = _profile_cuda_ms(
            inner_profile,
            lambda: codes_raw.contiguous(),
        )
        state.add_component_ms("encode_contiguous_codes_new_k", ms)

        if residual_signs_raw is None:
            residual_signs = None
            state.add_component_ms("encode_contiguous_residual_signs_new_k_skipped_fused", 0.0)
        else:
            residual_signs, ms = _profile_cuda_ms(
                inner_profile,
                lambda: residual_signs_raw.contiguous(),
            )
            state.add_component_ms("encode_contiguous_residual_signs_new_k", ms)

        norms_new, ms = _profile_cuda_ms(
            inner_profile,
            lambda: residual_norms_raw.contiguous(),
        )
        state.add_component_ms("encode_contiguous_residual_norms_new_k", ms)

        # -----------------------------------------------------------------
        # Pack + append compressed state.
        # These names are kept compatible with the prior v2 breakdown.
        # -----------------------------------------------------------------
        pack_fastpath = self._ensure_pack_cuda_fastpath(codes.device)

        if scalar_new_fused is not None:
            scalar_new = scalar_new_fused
            state.add_component_ms("pack_scalar_codes_new_k_skipped_fused", 0.0)
        else:
            scalar_new, ms = _profile_cuda_ms(
                inner_profile,
                (
                    (lambda: pack_fastpath.pack_scalar(codes))
                    if pack_fastpath is not None
                    else (lambda: pack_scalar_codes_lane_word_4bit(codes).contiguous())
                ),
            )
            state.add_component_ms(
                "pack_scalar_codes_new_k_cuda_fastpath"
                if pack_fastpath is not None
                else "pack_scalar_codes_new_k",
                ms,
            )

        if qjl_new_fused is not None:
            qjl_new = qjl_new_fused
            state.add_component_ms("pack_qjl_signs_new_k_skipped_fused", 0.0)
        else:
            qjl_new, ms = _profile_cuda_ms(
                inner_profile,
                (
                    (lambda: pack_fastpath.pack_qjl(residual_signs))
                    if pack_fastpath is not None
                    else (lambda: _pack_qjl_signs_fast(residual_signs))
                ),
            )
            state.add_component_ms(
                "pack_qjl_signs_new_k_cuda_fastpath"
                if pack_fastpath is not None
                else "pack_qjl_signs_new_k",
                ms,
            )

        if (
            os.environ.get("TQ_DEBUG_COMPRESSED_WRITE_SHAPES", "").strip() == "1"
            and int(state.layer_idx) == 0
        ):
            print(
                "[DEBUG compressed write shapes]",
                "codes", tuple(codes.shape), codes.dtype, codes.stride(),
                "residual_signs", tuple(residual_signs.shape), residual_signs.dtype, residual_signs.stride(),
                "norms_new", tuple(norms_new.shape), norms_new.dtype, norms_new.stride(),
                "scalar_new", tuple(scalar_new.shape), scalar_new.dtype, scalar_new.stride(),
                "qjl_new", tuple(qjl_new.shape), qjl_new.dtype, qjl_new.stride(),
                "scalar_storage",
                tuple(state.scalar_lane_words_storage.shape) if state.scalar_lane_words_storage is not None else None,
                state.scalar_lane_words_storage.dtype if state.scalar_lane_words_storage is not None else None,
                state.scalar_lane_words_storage.stride() if state.scalar_lane_words_storage is not None else None,
                "qjl_storage",
                tuple(state.qjl_lane_nibbles_storage.shape) if state.qjl_lane_nibbles_storage is not None else None,
                state.qjl_lane_nibbles_storage.dtype if state.qjl_lane_nibbles_storage is not None else None,
                state.qjl_lane_nibbles_storage.stride() if state.qjl_lane_nibbles_storage is not None else None,
                "norm_storage",
                tuple(state.residual_norms_storage.shape) if state.residual_norms_storage is not None else None,
                state.residual_norms_storage.dtype if state.residual_norms_storage is not None else None,
                state.residual_norms_storage.stride() if state.residual_norms_storage is not None else None,
                "start", int(state.compressed_cache_len),
                "capacity", int(state.compressed_cache_capacity),
                flush=True,
            )

        def _slice_write_preallocated_cache():
            if os.environ.get("TQ_DEBUG_APPEND_ENCODE_PARITY", "0") == "1":
                try:
                    # Recompute reference encode for the just-appended RoPE-applied key
                    # using the exact same rotation/sketch/centroids as the cache state.
                    assert state.rotation is not None
                    assert state.sketch is not None
                    assert state.centroids is not None

                    k_ref = new_keys.detach().to(torch.float32)
                    rot_ref = rotate(k_ref, state.rotation)
                    codes_ref = scalar_quantize(rot_ref, state.centroids)
                    deq_ref = scalar_dequantize(codes_ref, state.centroids)
                    residual_ref = rot_ref - deq_ref
                    signs_ref, norms_ref = qjl_encode_residual(residual_ref, state.sketch)
                    scalar_ref = pack_scalar_codes_lane_word_4bit(codes_ref).contiguous()
                    qjl_ref = pack_qjl_signs_lane_nibble(signs_ref).contiguous()

                    def _mdiff(a, b):
                        if a.shape != b.shape:
                            return {"shape_a": list(a.shape), "shape_b": list(b.shape), "same_shape": False}
                        da = (a.detach().to(torch.float32) - b.detach().to(torch.float32)).abs()
                        return {
                            "same_shape": True,
                            "max_abs": float(da.max().item()) if da.numel() else 0.0,
                            "mean_abs": float(da.mean().item()) if da.numel() else 0.0,
                        }

                    if int(getattr(state, "layer_idx", -1)) in {0, 7, 28}:
                        print("[DEBUG append encode parity]", {
                            "layer_idx": int(getattr(state, "layer_idx", -1)),
                            "new_keys_shape": list(new_keys.shape),
                            "new_keys_norm": float(new_keys.detach().float().norm().item()),
                            "scalar_new_vs_ref": _mdiff(scalar_new, scalar_ref),
                            "qjl_new_vs_ref": _mdiff(qjl_new, qjl_ref),
                            "norms_new_vs_ref": _mdiff(norms_new, norms_ref),
                        }, flush=True)
                except Exception as e:
                    print("[DEBUG append encode parity ERROR]", repr(e), flush=True)

            self._append_preallocated_compressed_cache(
                state=state,
                scalar_new=scalar_new,
                qjl_new=(qjl_ref if (os.environ.get("TQ_FORCE_APPEND_QJL_REF_FINAL", "0") == "1" and "qjl_ref" in locals()) else qjl_new),
                norms_new=norms_new,
            )
            return None

        _, ms = _profile_cuda_ms(
            inner_profile,
            _slice_write_preallocated_cache,
        )
        state.add_component_ms("preallocated_compressed_cache_slice_write", ms)

        state.append_calls += 1
        state.last_kv_len = int(state.scalar_lane_words.shape[-2])

    def _make_forward(self, module: torch.nn.Module, state: PackedLayerState, original_forward: Any):
        patcher = self

        @torch.no_grad()
        def tq_decode_forward(self_module: torch.nn.Module, *f_args: Any, **f_kwargs: Any):
            hidden_states, kwargs = _extract_hidden_and_kwargs(f_args, f_kwargs)
            past = _extract_past(kwargs)

            # Prefill or no-cache path: keep original attention exactly.
            if int(hidden_states.shape[1]) != 1 or past is None:
                return original_forward(*f_args, **f_kwargs)

            bsz, q_len, hidden_size = hidden_states.shape
            if int(bsz) != 1:
                raise RuntimeError("Current CUDA decode integration expects batch size B=1.")
            num_heads, num_kv_heads, num_kv_groups, head_dim = _get_attn_config(self_module)
            if int(head_dim) != 128:
                raise RuntimeError(f"Current CUDA decode integration expects D=128, got {head_dim}.")

            def _project_qkv():
                query_states_local = _reshape_projected(
                    self_module.q_proj(hidden_states),
                    bsz=bsz,
                    seqlen=q_len,
                    heads=num_heads,
                    head_dim=head_dim,
                )
                key_states_new_local = _reshape_projected(
                    self_module.k_proj(hidden_states),
                    bsz=bsz,
                    seqlen=q_len,
                    heads=num_kv_heads,
                    head_dim=head_dim,
                )
                value_states_new_local = _reshape_projected(
                    self_module.v_proj(hidden_states),
                    bsz=bsz,
                    seqlen=q_len,
                    heads=num_kv_heads,
                    head_dim=head_dim,
                )
                return query_states_local, key_states_new_local, value_states_new_local

            (query_states, key_states_new, value_states_new), ms = _profile_cuda_ms(
                patcher.profile_components,
                _project_qkv,
            )
            state.add_component_ms("qkv_projection", ms)

            position_ids = kwargs.get("position_ids", None)
            position_embeddings = kwargs.get("position_embeddings", None)

            if (
                os.environ.get("TQ_DEBUG_POSITION_CACHE", "").strip() == "1"
                and int(state.layer_idx) == 0
            ):
                cache_position_dbg = kwargs.get("cache_position", None)
                try:
                    past_len_dbg = past.get_seq_length(int(state.layer_idx)) if hasattr(past, "get_seq_length") else None
                except Exception as e:
                    past_len_dbg = f"ERR:{repr(e)}"
                try:
                    key_len_dbg = (
                        int(past.key_cache[int(state.layer_idx)].shape[-2])
                        if hasattr(past, "key_cache") and past.key_cache[int(state.layer_idx)] is not None
                        else None
                    )
                except Exception as e:
                    key_len_dbg = f"ERR:{repr(e)}"
                try:
                    value_len_dbg = (
                        int(past.value_cache[int(state.layer_idx)].shape[-2])
                        if hasattr(past, "value_cache") and past.value_cache[int(state.layer_idx)] is not None
                        else None
                    )
                except Exception as e:
                    value_len_dbg = f"ERR:{repr(e)}"
                print(
                    "[DEBUG position cache]",
                    "position_ids=", (
                        position_ids.detach().cpu().tolist()
                        if torch.is_tensor(position_ids) else position_ids
                    ),
                    "cache_position=", (
                        cache_position_dbg.detach().cpu().tolist()
                        if torch.is_tensor(cache_position_dbg) else cache_position_dbg
                    ),
                    "past_len=", past_len_dbg,
                    "key_len=", key_len_dbg,
                    "value_len=", value_len_dbg,
                    "compressed_len=", int(state.compressed_cache_len),
                    flush=True,
                )

            def _rope_apply_only():
                # Default production path:
                # use decode-only RoPE apply fast path for q_len=1, unless
                # explicitly disabled for A/B debugging.
                if os.environ.get("TQ_DISABLE_ROPE_FAST_APPLY", "").strip() != "1":
                    state.add_component_ms("rope_fast_default_path_calls", 0.0)
                    return _safe_apply_rope_decode_fast(
                        self_module,
                        query_states,
                        key_states_new,
                        value_states_new,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )

                state.add_component_ms("rope_fast_disabled_path_calls", 0.0)
                return _safe_apply_rope(
                    self_module,
                    query_states,
                    key_states_new,
                    value_states_new,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

            def _rope_apply_breakdown_only():
                """
                Diagnostic path for old rotary_emb(seq_len=...) APIs.
                It preserves correctness but separately times:
                  - position_ids.max().item()
                  - rotary(v, seq_len=...)
                  - apply_rotary_pos_emb / fallback apply
                Falls back to _safe_apply_rope for other API variants.
                """
                cos = sin = None
                rope_mode = None

                if position_embeddings is not None:
                    if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
                        cos, sin = position_embeddings[0], position_embeddings[1]
                        rope_mode = "already_positioned"

                rotary = getattr(self_module, "rotary_emb", None)
                if (cos is None or sin is None) and rotary is not None:
                    try:
                        rotary_sig = inspect.signature(rotary.forward)
                        rotary_param_names = set(rotary_sig.parameters.keys())
                    except Exception:
                        rotary_param_names = set()

                    if "seq_len" not in rotary_param_names:
                        return _rope_apply_only()

                    if position_ids is not None and torch.is_tensor(position_ids) and position_ids.numel() > 0:
                        def _pos_item():
                            return int(position_ids.detach().max().item()) + 1

                        kv_seq_len, pos_ms = _profile_cuda_ms(
                            _profile_component_enabled("rope_position_item", patcher.profile_components),
                            _pos_item,
                        )
                        state.add_component_ms("rope_position_item", pos_ms)
                    else:
                        kv_seq_len = int(key_states_new.shape[-2])

                    def _rotary_table():
                        return rotary(
                            value_states_new,
                            seq_len=max(int(key_states_new.shape[-2]), int(kv_seq_len)),
                        )

                    out, table_ms = _profile_cuda_ms(
                        _profile_component_enabled("rope_rotary_table", patcher.profile_components),
                        _rotary_table,
                    )
                    state.add_component_ms("rope_rotary_table", table_ms)

                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        cos, sin = out[0], out[1]
                        rope_mode = "full_table"

                if cos is None or sin is None:
                    return query_states, key_states_new

                def _apply_rope_pos():
                    # Decode-only fast path. This avoids HF generic
                    # apply_rotary_pos_emb broadcast/shape handling overhead.
                    if int(query_states.shape[-2]) == 1 and int(key_states_new.shape[-2]) == 1:
                        return _apply_rope_decode_fast(
                            query_states,
                            key_states_new,
                            cos,
                            sin,
                            position_ids if rope_mode == "full_table" else None,
                        )

                    if hf_apply_rotary_pos_emb is not None:
                        if rope_mode == "full_table" and position_ids is not None:
                            try:
                                return hf_apply_rotary_pos_emb(
                                    query_states,
                                    key_states_new,
                                    cos,
                                    sin,
                                    position_ids,
                                )
                            except TypeError:
                                pass

                        try:
                            return hf_apply_rotary_pos_emb(
                                query_states,
                                key_states_new,
                                cos,
                                sin,
                            )
                        except TypeError:
                            pass

                    if rope_mode == "full_table" and position_ids is not None:
                        local_cos = cos
                        local_sin = sin
                        if local_cos.ndim == 2:
                            local_cos = local_cos[position_ids].squeeze(1)
                        if local_sin.ndim == 2:
                            local_sin = local_sin[position_ids].squeeze(1)
                        return _apply_rope_fallback(
                            query_states,
                            key_states_new,
                            local_cos,
                            local_sin,
                        )

                    return _apply_rope_fallback(query_states, key_states_new, cos, sin)

                out, apply_ms = _profile_cuda_ms(
                    _profile_component_enabled("rope_apply_pos_emb", patcher.profile_components),
                    _apply_rope_pos,
                )
                state.add_component_ms("rope_apply_pos_emb", apply_ms)
                return out

            if os.environ.get("TQ_PROFILE_ROPE_BREAKDOWN", "").strip() == "1":
                (query_states, key_states_new), ms = _profile_cuda_ms(
                    _profile_component_enabled("rope_apply", patcher.profile_components),
                    _rope_apply_breakdown_only,
                )
            else:
                (query_states, key_states_new), ms = _profile_cuda_ms(
                    _profile_component_enabled("rope_apply", patcher.profile_components),
                    _rope_apply_only,
                )
            state.add_component_ms("rope_apply", ms)

            def _cache_update_only():
                # Safe experimental steady-state path:
                # keep both dense K and dense V cache lengths consistent, but
                # avoid DynamicCache.update's per-token torch.cat by writing
                # into preallocated storage once state is ready.
                dense_cache_mode = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()

                if (
                    dense_cache_mode not in ("dynamic_v", "dynamic_v_no_k", "dynamic_k_no_vtail")
                    and state.ready()
                    and hasattr(past, "key_cache")
                    and hasattr(past, "value_cache")
                ):
                    out = patcher._append_preallocated_dense_kv_cache(
                        state=state,
                        past=past,
                        key_states_new=key_states_new,
                        value_states_new=value_states_new,
                    )
                    if out is not None:
                        return out

                if dense_cache_mode == "dynamic_k_no_vtail":
                    # Fair V-tail baseline:
                    # K path matches dynamic_v_no_k: dense K remains prefix-only and
                    # generated K is appended only into compressed K.
                    # V path remains dense-cat, so the only difference versus
                    # dynamic_v_no_k is dense V cat vs V-tail no-cat.
                    layer_i = int(state.layer_idx)
                    if not hasattr(past, "value_cache"):
                        raise RuntimeError("dynamic_k_no_vtail requires HF-style value_cache.")

                    old_v = past.value_cache[layer_i]
                    if old_v is None:
                        active_v = value_states_new
                    else:
                        active_v = torch.cat([old_v, value_states_new], dim=-2)
                    past.value_cache[layer_i] = active_v

                    old_k = past.key_cache[layer_i]
                    if old_k is None:
                        full_keys_prefix = key_states_new[..., :0, :]
                    else:
                        full_keys_prefix = old_k

                    return full_keys_prefix, active_v, past

                if dense_cache_mode == "dynamic_v_no_k":
                    # Low-memory TurboQuant mode:
                    # Use TQStaticValueCache.update when available. It appends V
                    # by slice-write and keeps dense K prefix-only.
                    layer_i = int(state.layer_idx)

                    if hasattr(past, "update") and past.__class__.__name__ == "TQStaticValueCache":
                        full_keys_prefix, active_v = past.update(
                            key_states_new,
                            value_states_new,
                            layer_i,
                            kwargs.get("cache_kwargs", None),
                        )
                        return full_keys_prefix, active_v, past

                    # Fallback for non-static cache: this may still OOM at 16K
                    # because DynamicCache uses torch.cat.
                    if not hasattr(past, "value_cache"):
                        raise RuntimeError("dynamic_v_no_k requires HF-style value_cache.")

                    old_v = past.value_cache[layer_i]

                    # dynamic_v_no_k_tail_cache_no_cat:
                    # Avoid materializing torch.cat([old_v, value_states_new])
                    # during long-context decode. Keep the large prefill V as
                    # base_v and accumulate only generated/teacher-forced V in
                    # a small per-layer tail. The later probs @ V path handles
                    # (base_v, tail_v) without concatenating.
                    if old_v is None:
                        active_v = value_states_new
                        dynamic_v_tail = None
                    else:
                        prev_tail = getattr(state, "dynamic_v_tail", None)
                        if prev_tail is None:
                            dynamic_v_tail = value_states_new
                        else:
                            dynamic_v_tail = torch.cat([prev_tail, value_states_new], dim=-2)
                        setattr(state, "dynamic_v_tail", dynamic_v_tail)
                        active_v = old_v


                    if os.environ.get("TQ_DEBUG_VTAIL_DENSE_COMPARE", "").strip() == "1":
                        # Debug-only dense reference. This intentionally materializes dense V,
                        # so use only for short eval/debug, not 32k timing.
                        debug_dense_v = getattr(state, "debug_dense_v", None)
                        if debug_dense_v is None:
                            if old_v is None:
                                debug_dense_v = value_states_new
                            else:
                                debug_dense_v = torch.cat([old_v, value_states_new], dim=-2)
                        else:
                            debug_dense_v = torch.cat([debug_dense_v, value_states_new], dim=-2)
                        setattr(state, "debug_dense_v", debug_dense_v)

                        if old_v is not None:
                            if dynamic_v_tail is None:
                                recon_v = active_v
                            else:
                                recon_v = torch.cat([active_v, dynamic_v_tail], dim=-2)

                            same_shape = list(debug_dense_v.shape) == list(recon_v.shape)
                            if same_shape:
                                diff = (debug_dense_v - recon_v).float().abs()
                                max_abs = float(diff.max().item())
                                mean_abs = float(diff.mean().item())
                            else:
                                max_abs = None
                                mean_abs = None

                            if int(state.layer_idx) == 0 and int(state.append_calls) < 8:
                                print(
                                    "[DEBUG vtail dense_compare]",
                                    {
                                        "layer": int(state.layer_idx),
                                        "append_calls": int(state.append_calls),
                                        "debug_dense_v": list(debug_dense_v.shape),
                                        "recon_v": list(recon_v.shape),
                                        "same_shape": same_shape,
                                        "max_abs": max_abs,
                                        "mean_abs": mean_abs,
                                    },
                                    flush=True,
                                )

                    old_k = past.key_cache[layer_i]
                    if old_k is None:
                        full_keys_prefix = key_states_new[..., :0, :]
                    else:
                        full_keys_prefix = old_k

                    return full_keys_prefix, (active_v, dynamic_v_tail), past

                return _cache_update(
                    module=self_module,
                    past=past,
                    key_states=key_states_new,
                    value_states=value_states_new,
                    kwargs=kwargs,
                    profile_components=patcher.profile_components,
                    state=state,
                )

            (full_keys_kv, full_values_kv, present), ms = _profile_cuda_ms(
                patcher.profile_components,
                _cache_update_only,
            )
            state.add_component_ms("cache_update_only", ms)

            if state.ready() and state.dense_key_storage is not None:
                state.add_component_ms("preallocated_dense_kv_cache_update", ms)

            def _repeat_kv_states():
                dense_cache_mode = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()

                if dense_cache_mode in ("dynamic_v_no_k", "dynamic_k_no_vtail"):
                    # Low-memory long-context mode:
                    # - full_keys_kv is only needed for first compressed prefix build.
                    # - num_kv_groups is 1 for the current LLaMA config, so avoid an
                    #   unnecessary contiguous duplicate of full prefix K.
                    # - TQ logits use compressed K after build; dense full_keys is not
                    #   needed for logits.
                    if int(num_kv_groups) != 1:
                        full_keys_local = _repeat_kv(full_keys_kv, num_kv_groups)
                    else:
                        full_keys_local = full_keys_kv

                    if isinstance(full_values_kv, tuple):
                        base_v, tail_v = full_values_kv
                        if int(num_kv_groups) != 1:
                            base_v = _repeat_kv(base_v, num_kv_groups).contiguous()
                            if tail_v is not None:
                                tail_v = _repeat_kv(tail_v, num_kv_groups).contiguous()
                        full_values_local = (base_v, tail_v)
                    else:
                        if int(num_kv_groups) != 1:
                            full_values_local = _repeat_kv(full_values_kv, num_kv_groups).contiguous()
                        else:
                            full_values_local = full_values_kv
                    new_keys_expanded_local = _repeat_kv(key_states_new, num_kv_groups).contiguous()
                    return full_keys_local, full_values_local, new_keys_expanded_local

                full_keys_local = _repeat_kv(full_keys_kv, num_kv_groups).contiguous()
                full_values_local = _repeat_kv(full_values_kv, num_kv_groups).contiguous()
                new_keys_expanded_local = _repeat_kv(key_states_new, num_kv_groups).contiguous()
                return full_keys_local, full_values_local, new_keys_expanded_local

            (full_keys, full_values, new_keys_expanded), ms = _profile_cuda_ms(
                patcher.profile_components,
                _repeat_kv_states,
            )
            state.add_component_ms("repeat_kv_materialize", ms)

            if (
                os.environ.get("TQ_DEBUG_DUMP_NEW_KEYS", "").strip() == "1"
                and int(state.layer_idx) == 0
                and int(state.decode_calls) < int(os.environ.get("TQ_DEBUG_DUMP_STEPS", "4"))
            ):
                dump_dir = Path(os.environ.get("TQ_DEBUG_DUMP_NEW_KEYS_DIR", "runs/svd_uniform_08/eval/debug_new_keys"))
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / (
                    f"layer{int(state.layer_idx):02d}_"
                    f"append{int(state.append_calls):05d}_"
                    f"decode{int(state.decode_calls):05d}.pt"
                )
                torch.save(
                    {
                        "layer_idx": int(state.layer_idx),
                        "append_calls": int(state.append_calls),
                        "decode_calls": int(state.decode_calls),
                        "full_keys_len": int(full_keys.shape[-2]),
                        "new_keys_expanded": new_keys_expanded.detach().float().cpu(),
                    },
                    dump_path,
                )
                print("[DEBUG dump new_keys]", str(dump_path), flush=True)

            if not state.ready():
                dense_cache_mode_for_install = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()
                if (
                    dense_cache_mode_for_install not in ("dynamic_v", "dynamic_v_no_k", "dynamic_k_no_vtail")
                    and hasattr(present, "key_cache")
                    and hasattr(present, "value_cache")
                    and state.dense_key_storage is None
                ):
                    _, dense_install_ms = _profile_cuda_ms(
                        patcher.profile_components,
                        lambda: patcher._install_preallocated_dense_kv_cache(
                            state=state,
                            past=present,
                            full_keys_kv=full_keys_kv,
                            full_values_kv=full_values_kv,
                        ),
                    )
                    state.add_component_ms(
                        "install_preallocated_dense_kv_cache",
                        dense_install_ms,
                    )

                if (
                    os.environ.get("TQ_STATIC_V_BUILD_ONLY_TIMED", "").strip() == "1"
                    and os.environ.get("TQ_STATIC_V_CACHE", "").strip() == "1"
                    and os.environ.get("TQ_CURRENT_DECODE_PHASE", "") == "prime"
                ):
                    return original_forward(*f_args, **f_kwargs)

                dense_cache_mode_for_fit = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()
                fit_dynamic_v_no_k_with_current_k = (
                    dense_cache_mode_for_fit == "dynamic_v_no_k"
                    and os.environ.get("TQ_DEBUG_VTAIL_FIT_WITH_CURRENT_K", "").strip() == "1"
                )
                if fit_dynamic_v_no_k_with_current_k:
                    # Debug-only parity path:
                    # match normal path's initial compressed-K build, which sees prefix+current K.
                    # This materializes a temporary K cat and is NOT for 32k timing.
                    full_keys_for_fit = torch.cat([full_keys, new_keys_expanded], dim=-2)
                    if int(state.layer_idx) == 0:
                        print(
                            "[DEBUG fit_with_current_k]",
                            {
                                "layer": int(state.layer_idx),
                                "full_keys": list(full_keys.shape),
                                "new_keys_expanded": list(new_keys_expanded.shape),
                                "full_keys_for_fit": list(full_keys_for_fit.shape),
                            },
                            flush=True,
                        )
                else:
                    full_keys_for_fit = full_keys

                _, ms = _profile_cuda_ms(
                    patcher.profile_components,
                    lambda: patcher._fit_state_from_full_k(
                        state=state,
                        full_keys=full_keys_for_fit,
                        head_dim=head_dim,
                    ),
                )
                state.add_component_ms("prime_build_full_prefix_state", ms)

                if os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower() in ("dynamic_v_no_k", "dynamic_k_no_vtail") and not fit_dynamic_v_no_k_with_current_k:
                    # prime_build_dynamic_v_no_k_append_current:
                    # The low-memory cache update returned only prefix dense K
                    # to avoid key_cache torch.cat. Append the current token K
                    # to compressed K here so compressed K length matches V.
                    _, ms = _profile_cuda_ms(
                        patcher.profile_components,
                        lambda: patcher._append_new_k(
                            state=state,
                            new_keys=new_keys_expanded,
                        ),
                    )
                    state.add_component_ms("prime_build_append_current_k_dynamic_v_no_k", ms)
            else:
                _, ms = _profile_cuda_ms(
                    patcher.profile_components,
                    lambda: patcher._append_new_k(
                        state=state,
                        new_keys=new_keys_expanded,
                    ),
                )
                state.add_component_ms("append_encode_pack_cat_new_k", ms)

            assert state.rotation is not None
            assert state.sketch is not None
            assert state.centroids is not None
            assert state.scalar_lane_words is not None
            assert state.qjl_lane_nibbles is not None
            assert state.residual_norms is not None

            def _prepare_query_factors():
                rotated_queries_local = rotate(query_states, state.rotation).to(torch.float32).contiguous()
                qjl_projected_queries_local = qjl_project_query(query_states, state.sketch).to(torch.float32).contiguous()
                return rotated_queries_local, qjl_projected_queries_local

            (rotated_queries, qjl_projected_queries), ms = _profile_cuda_ms(
                patcher.profile_components,
                _prepare_query_factors,
            )
            state.add_component_ms("rotate_q_and_qjl_project_q", ms)

            (
                active_scalar_lane_words,
                active_qjl_lane_nibbles,
                active_residual_norms,
                active_kv_len,
            ) = patcher._active_compressed_cache_inputs(state)

            def _kernel_call():
                if os.environ.get("TQ_FORCE_REFERENCE_TQ_LOGITS", "0") == "1":
                    return (_reference_tq_logits_from_packed(
                    rotated_queries=rotated_queries,
                    qjl_projected_queries=qjl_projected_queries,
                    centroids=state.centroids,
                    scalar_lane_words=active_scalar_lane_words,
                    qjl_lane_nibbles=active_qjl_lane_nibbles,
                    residual_norms=active_residual_norms,
                    active_kv_len=active_kv_len,
                    ), "reference_torch_logits")
                return _try_call_nonfactor_kernel(
                    rotated_queries=rotated_queries,
                    qjl_projected_queries=qjl_projected_queries,
                    centroids=state.centroids,
                    scalar_lane_words=active_scalar_lane_words,
                    qjl_lane_nibbles=active_qjl_lane_nibbles,
                    residual_norms=active_residual_norms,
                    active_kv_len=active_kv_len,
                )

            (tq_logits, adapter), ms = _profile_cuda_ms(
                patcher.profile_components,
                _kernel_call,
            )
            state.add_component_ms("turboquant_cuda_logits_kernel_wrapper", ms)
            state.kernel_adapter = adapter

            if (
                os.environ.get("TQ_DEBUG_DUMP_TQ_LOGITS", "").strip() == "1"
                and int(state.layer_idx) == 0
                and int(state.append_calls) < int(os.environ.get("TQ_DEBUG_DUMP_STEPS", "4"))
            ):
                dump_dir = Path(os.environ.get("TQ_DEBUG_DUMP_DIR", "runs/svd_uniform_08/eval/debug_logits_dumps"))
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / (
                    f"layer{int(state.layer_idx):02d}_"
                    f"append{int(state.append_calls):05d}_"
                    f"decode{int(state.decode_calls):05d}.pt"
                )
                torch.save(
                    {
                        "layer_idx": int(state.layer_idx),
                        "append_calls": int(state.append_calls),
                        "decode_calls": int(state.decode_calls),
                        "active_kv_len": int(active_kv_len),
                        "full_keys_len": int(full_keys.shape[-2]),
                        "tq_logits": tq_logits.detach().float().cpu(),
                        "attn_logits": attn_logits.detach().float().cpu() if "attn_logits" in locals() else None,
                    },
                    dump_path,
                )
                print("[DEBUG dump tq_logits]", str(dump_path), flush=True)

            if (
                os.environ.get("TQ_DEBUG_KLOGITS", "").strip() == "1"
                and int(state.layer_idx) == 0
                and int(state.append_calls) < 8
            ):
                def _shape_or_none(x):
                    try:
                        return list(x.shape)
                    except Exception:
                        return None

                print(
                    "[DEBUG klogits]",
                    {
                        "layer": int(state.layer_idx),
                        "append_calls": int(state.append_calls),
                        "decode_calls": int(state.decode_calls),
                        "active_kv_len": int(active_kv_len),
                        "full_keys_len": int(full_keys.shape[-2]),
                        "new_keys_expanded": _shape_or_none(new_keys_expanded),
                        "tq_logits": list(tq_logits.shape),
                        "scalar_lane_words": _shape_or_none(state.scalar_lane_words),
                        "qjl_lane_nibbles": _shape_or_none(state.qjl_lane_nibbles),
                        "residual_norms": _shape_or_none(state.residual_norms),
                        "last_kv_len": int(getattr(state, "last_kv_len", -1)),
                    },
                    flush=True,
                )

            scale = float(getattr(self_module, "scaling", 1.0 / math.sqrt(float(head_dim))))
            attn_logits = tq_logits.to(torch.float32) * float(scale)

            def _mask_build_only():
                dense_cache_mode_for_mask = os.environ.get("TQ_DENSE_CACHE_MODE", "kv").strip().lower()
                if dense_cache_mode_for_mask in ("dynamic_v_no_k", "dynamic_k_no_vtail"):
                    # In dynamic_v_no_k, dense full_keys is prefix-only and does not
                    # include generated/teacher-forced K tail. The logits kernel uses
                    # compressed K with active_kv_len, so the attention mask must match
                    # logits length, not dense full_keys.shape[-2].
                    mask_kv_len = int(active_kv_len)
                else:
                    mask_kv_len = int(full_keys.shape[-2])

                if (
                    os.environ.get("TQ_DEBUG_KLOGITS", "").strip() == "1"
                    and int(state.layer_idx) == 0
                    and int(state.append_calls) < 8
                ):
                    print(
                        "[DEBUG mask_kv_len]",
                        {
                            "layer": int(state.layer_idx),
                            "append_calls": int(state.append_calls),
                            "mode": dense_cache_mode_for_mask,
                            "mask_kv_len": int(mask_kv_len),
                            "active_kv_len": int(active_kv_len),
                            "full_keys_len": int(full_keys.shape[-2]),
                            "attn_logits_len": int(attn_logits.shape[-1]),
                        },
                        flush=True,
                    )

                return _normalize_mask(
                    kwargs.get("attention_mask", None),
                    q_len=q_len,
                    kv_len=mask_kv_len,
                    dtype=attn_logits.dtype,
                    device=attn_logits.device,
                )

            mask, ms = _profile_cuda_ms(
                patcher.profile_components,
                _mask_build_only,
            )
            state.add_component_ms("post_logits_mask_build", ms)

            def _mask_add_only():
                if mask is None:
                    return attn_logits
                return attn_logits + mask

            attn_logits_masked, ms = _profile_cuda_ms(
                patcher.profile_components,
                _mask_add_only,
            )
            state.add_component_ms("post_logits_mask_add", ms)

            def _softmax_only():
                return torch.softmax(
                    attn_logits_masked,
                    dim=-1,
                    dtype=torch.float32,
                )

            pre_softmax_sync_ms = None
            if patcher.profile_components:
                _pre_softmax_t0 = time.perf_counter()
                torch.cuda.synchronize()
                pre_softmax_sync_ms = (
                    time.perf_counter() - _pre_softmax_t0
                ) * 1000.0

            state.add_component_ms(
                "pre_softmax_cuda_synchronize_wall",
                pre_softmax_sync_ms,
            )

            attn_probs_fp32, ms = _profile_cuda_ms(
                patcher.profile_components,
                _softmax_only,
            )
            state.add_component_ms("post_logits_softmax_fp32", ms)

            if (
                patcher.profile_components
                and ms is not None
                and float(ms) > 100.0
            ):
                dump_dir = Path(
                    "runs/svd_uniform_08/eval/softmax_spike_dumps"
                )
                dump_dir.mkdir(parents=True, exist_ok=True)

                dump_path = dump_dir / (
                    f"softmax_spike_layer{int(state.layer_idx):02d}_"
                    f"decodecall{int(state.decode_calls):05d}_"
                    f"{float(ms):.3f}ms.pt"
                )

                torch.save(
                    {
                        "layer_idx": int(state.layer_idx),
                        "decode_calls": int(state.decode_calls),
                        "softmax_ms": float(ms),
                        "shape": tuple(attn_logits_masked.shape),
                        "dtype": str(attn_logits_masked.dtype),
                        "device": str(attn_logits_masked.device),
                        "attn_logits_masked_cpu": (
                            attn_logits_masked.detach().to("cpu")
                        ),
                    },
                    dump_path,
                )

                print(
                    f"[DEBUG softmax spike dump] {dump_path} "
                    f"ms={float(ms):.3f} "
                    f"shape={tuple(attn_logits_masked.shape)}",
                    flush=True,
                )

            def _value_dtype(values_obj):
                if isinstance(values_obj, tuple):
                    base_v, _tail_v = values_obj
                    return base_v.dtype
                return values_obj.dtype

            def _cast_values_to_dtype(values_obj, dtype):
                if isinstance(values_obj, tuple):
                    base_v, tail_v = values_obj
                    base_v = base_v.to(dtype)
                    if tail_v is not None:
                        tail_v = tail_v.to(dtype)
                    return base_v, tail_v
                return values_obj.to(dtype)

            def _matmul_probs_v_dynamic_tail(attn_probs_local, values_obj):
                if isinstance(values_obj, tuple):
                    base_v, tail_v = values_obj

                    if (
                        os.environ.get("TQ_DEBUG_VTAIL_SHAPES", "").strip() == "1"
                        and int(state.layer_idx) == 0
                        and int(state.append_calls) < 8
                    ):
                        base_len_dbg = int(base_v.shape[-2])
                        tail_len_dbg = 0 if tail_v is None else int(tail_v.shape[-2])
                        probs_len_dbg = int(attn_probs_local.shape[-1])
                        print(
                            "[DEBUG vtail matmul]",
                            {
                                "layer": int(state.layer_idx),
                                "append_calls": int(state.append_calls),
                                "attn_probs": list(attn_probs_local.shape),
                                "base_v": list(base_v.shape),
                                "tail_v": None if tail_v is None else list(tail_v.shape),
                                "base_len": base_len_dbg,
                                "tail_len": tail_len_dbg,
                                "sum_len": base_len_dbg + tail_len_dbg,
                                "probs_len": probs_len_dbg,
                                "ok": probs_len_dbg == base_len_dbg + tail_len_dbg,
                            },
                            flush=True,
                        )

                    base_len = int(base_v.shape[-2])
                    out = torch.matmul(attn_probs_local[..., :base_len], base_v)

                    if tail_v is not None and int(tail_v.shape[-2]) > 0:
                        tail_len = int(tail_v.shape[-2])
                        out = out + torch.matmul(
                            attn_probs_local[..., base_len:base_len + tail_len],
                            tail_v,
                        )

                    if (
                        os.environ.get("TQ_DEBUG_VTAIL_PARITY", "").strip() == "1"
                        and int(state.layer_idx) == 0
                        and int(state.append_calls) < 6
                        and tail_v is not None
                    ):
                        ref_v = torch.cat([base_v, tail_v], dim=-2)
                        ref = torch.matmul(attn_probs_local, ref_v)
                        diff = (out - ref).float().abs()
                        print(
                            "[DEBUG vtail parity]",
                            {
                                "layer": int(state.layer_idx),
                                "append_calls": int(state.append_calls),
                                "out_shape": list(out.shape),
                                "ref_shape": list(ref.shape),
                                "max_abs": float(diff.max().item()),
                                "mean_abs": float(diff.mean().item()),
                                "out_norm": float(out.float().norm().item()),
                                "ref_norm": float(ref.float().norm().item()),
                            },
                            flush=True,
                        )

                    return out

                return torch.matmul(attn_probs_local, values_obj)

            if (
                os.environ.get("TQ_DEBUG_VTAIL_SHAPES", "").strip() == "1"
                and int(state.layer_idx) == 0
                and int(state.append_calls) < 8
            ):
                print(
                    "[DEBUG full_values type]",
                    {
                        "layer": int(state.layer_idx),
                        "append_calls": int(state.append_calls),
                        "type": type(full_values).__name__,
                        "is_tuple": isinstance(full_values, tuple),
                        "shape": None if isinstance(full_values, tuple) else list(full_values.shape),
                    },
                    flush=True,
                )

            def _probs_cast_only():
                return attn_probs_fp32.to(_value_dtype(full_values))

            attn_probs, ms = _profile_cuda_ms(
                patcher.profile_components,
                _probs_cast_only,
            )
            state.add_component_ms("post_logits_probs_cast_to_v_dtype", ms)

            def _values_cast_only():
                return _cast_values_to_dtype(full_values, attn_probs.dtype)

            full_values_for_matmul, ms = _profile_cuda_ms(
                patcher.profile_components,
                _values_cast_only,
            )
            state.add_component_ms("post_logits_values_cast_to_probs_dtype", ms)

            def _matmul_probs_v_only():
                return _matmul_probs_v_dynamic_tail(attn_probs, full_values_for_matmul)

            attn_output_heads, ms = _profile_cuda_ms(
                patcher.profile_components,
                _matmul_probs_v_only,
            )
            state.add_component_ms("post_logits_matmul_probs_v", ms)

            def _reshape_only():
                return (
                    attn_output_heads
                    .transpose(1, 2)
                    .contiguous()
                    .reshape(bsz, q_len, hidden_size)
                )

            attn_output_reshaped, ms = _profile_cuda_ms(
                patcher.profile_components,
                _reshape_only,
            )
            state.add_component_ms("post_logits_transpose_contiguous_reshape", ms)

            def _o_proj_input_cast_only():
                return attn_output_reshaped.to(hidden_states.dtype)

            attn_output_for_o_proj, ms = _profile_cuda_ms(
                patcher.profile_components,
                _o_proj_input_cast_only,
            )
            state.add_component_ms("post_logits_o_proj_input_cast", ms)

            def _o_proj_only():
                return self_module.o_proj(attn_output_for_o_proj)

            attn_output, ms = _profile_cuda_ms(
                patcher.profile_components,
                _o_proj_only,
            )
            state.add_component_ms("post_logits_o_proj", ms)

            state.decode_calls += 1
            state.last_kv_len = int(full_keys.shape[-2])

            # User's current stack accepts this LLaMA attention return structure.
            return attn_output, None, present

        @torch.no_grad()
        def profiled_tq_decode_forward(self_module: torch.nn.Module, *f_args: Any, **f_kwargs: Any):
            if not bool(patcher.profile_components):
                return tq_decode_forward(self_module, *f_args, **f_kwargs)

            try:
                hidden_states_probe, kwargs_probe = _extract_hidden_and_kwargs(f_args, f_kwargs)
                past_probe = _extract_past(kwargs_probe)
                is_replacement_decode = int(hidden_states_probe.shape[1]) == 1 and past_probe is not None
            except Exception:
                is_replacement_decode = False

            if not is_replacement_decode:
                return tq_decode_forward(self_module, *f_args, **f_kwargs)

            total_key = (
                "replacement_attention_forward_total_steady"
                if state.ready()
                else "replacement_attention_forward_total_prime_build"
            )
            out, ms = _profile_cuda_ms(
                True,
                lambda: tq_decode_forward(self_module, *f_args, **f_kwargs),
            )
            state.add_component_ms(total_key, ms)
            return out

        return types.MethodType(profiled_tq_decode_forward, module)

    def install(self, model: torch.nn.Module, replace_layers: Optional[set[int]]) -> list[dict[str, Any]]:
        installed = []
        layer_re = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
        fallback_counter = 0

        for name, module in model.named_modules():
            has_attn_proj = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))
            if not has_attn_proj:
                continue

            m = layer_re.search(name + ".")
            if m:
                layer_idx = int(m.group(1))
            else:
                layer_idx = fallback_counter
                fallback_counter += 1

            if replace_layers is not None and layer_idx not in replace_layers:
                continue

            state = PackedLayerState(name=name, layer_idx=layer_idx)
            self.states[layer_idx] = state
            original_forward = module.forward
            self.original_forwards.append((module, original_forward))
            module.forward = self._make_forward(module, state, original_forward)

            installed.append(
                {
                    "layer_idx": int(getattr(state, "layer_idx", -1)),
                    "name": name,
                    "module_type": type(module).__name__,
                }
            )

        if not installed:
            raise RuntimeError("No attention modules were patched for decode replacement.")
        return installed

    def restore(self) -> None:
        for module, original_forward in self.original_forwards:
            module.forward = original_forward
        self.original_forwards.clear()

    def summary(self) -> dict[str, Any]:
        return {
            str(k): {
                "name": v.name,
                "layer_idx": int(v.layer_idx),
                "build_calls": int(v.build_calls),
                "append_calls": int(v.append_calls),
                "decode_calls": int(v.decode_calls),
                "kernel_adapter": v.kernel_adapter,
                "last_kv_len": int(v.last_kv_len),
                "scalar_packed_shape": list(v.scalar_lane_words.shape) if v.scalar_lane_words is not None else None,
                "qjl_packed_shape": list(v.qjl_lane_nibbles.shape) if v.qjl_lane_nibbles is not None else None,
                "residual_norm_shape": list(v.residual_norms.shape) if v.residual_norms is not None else None,
                "component_ms_total": dict(v.component_ms_total),
                "component_ms_count": dict(v.component_ms_count),
                "component_ms_mean": {
                    name: (
                        float(v.component_ms_total[name] / v.component_ms_count[name])
                        if int(v.component_ms_count.get(name, 0)) > 0 else None
                    )
                    for name in v.component_ms_total
                },
                "cache_update_path_counts": dict(v.cache_update_path_counts),
                "cache_update_past_type_counts": dict(v.cache_update_past_type_counts),
                "cache_position_present_count": int(v.cache_position_present_count),
                "cache_position_absent_count": int(v.cache_position_absent_count),
            }
            for k, v in sorted(self.states.items(), key=lambda kv: kv[0])
        }


@dataclass
class BaselineAttentionLayerProfile:
    name: str
    layer_idx: int
    total_ms: float = 0.0
    count: int = 0

    def add_ms(self, ms: Optional[float]) -> None:
        if ms is None:
            return
        self.total_ms += float(ms)
        self.count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_idx": int(self.layer_idx),
            "baseline_attention_forward_total_ms": float(self.total_ms),
            "baseline_attention_forward_total_calls": int(self.count),
            "baseline_attention_forward_total_mean_ms": (
                float(self.total_ms / self.count) if self.count > 0 else None
            ),
        }


class BaselineDecodeAttentionProfiler:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.states: dict[int, BaselineAttentionLayerProfile] = {}
        self.original_forwards: list[tuple[torch.nn.Module, Any]] = []

    def _make_forward(
        self,
        module: torch.nn.Module,
        state: BaselineAttentionLayerProfile,
        original_forward: Any,
    ):
        profiler = self

        @torch.no_grad()
        def profiled_baseline_forward(self_module: torch.nn.Module, *f_args: Any, **f_kwargs: Any):
            if not profiler.enabled:
                return original_forward(*f_args, **f_kwargs)

            try:
                hidden_states_probe, kwargs_probe = _extract_hidden_and_kwargs(f_args, f_kwargs)
                past_probe = _extract_past(kwargs_probe)
                is_decode_attention = int(hidden_states_probe.shape[1]) == 1 and past_probe is not None
            except Exception:
                is_decode_attention = False

            if not is_decode_attention:
                return original_forward(*f_args, **f_kwargs)

            out, ms = _profile_cuda_ms(
                True,
                lambda: original_forward(*f_args, **f_kwargs),
            )
            state.add_ms(ms)
            return out

        return types.MethodType(profiled_baseline_forward, module)

    def install(self, model: torch.nn.Module, profile_layers: Optional[set[int]]) -> list[dict[str, Any]]:
        installed = []
        layer_re = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
        fallback_counter = 0

        for name, module in model.named_modules():
            has_attn_proj = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))
            if not has_attn_proj:
                continue

            m = layer_re.search(name + ".")
            if m:
                layer_idx = int(m.group(1))
            else:
                layer_idx = fallback_counter
                fallback_counter += 1

            if profile_layers is not None and layer_idx not in profile_layers:
                continue

            state = BaselineAttentionLayerProfile(name=name, layer_idx=layer_idx)
            self.states[layer_idx] = state
            original_forward = module.forward
            self.original_forwards.append((module, original_forward))
            module.forward = self._make_forward(module, state, original_forward)

            installed.append(
                {
                    "layer_idx": int(getattr(state, "layer_idx", -1)),
                    "name": name,
                    "module_type": type(module).__name__,
                }
            )

        return installed

    def restore(self) -> None:
        for module, original_forward in self.original_forwards:
            module.forward = original_forward
        self.original_forwards.clear()

    def summary(self) -> dict[str, Any]:
        return {
            str(k): v.summary()
            for k, v in sorted(self.states.items(), key=lambda kv: kv[0])
        }


def _parse_replace_layers(raw: str) -> Optional[set[int]]:
    raw = str(raw).strip().lower()
    if raw in {"all", "*", ""}:
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _build_prompt_ids(tokenizer, *, prompt_len: int, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    if ids.numel() == 0:
        raise RuntimeError("Tokenizer produced no ids.")
    chunks = []
    total = 0
    while total < int(prompt_len):
        chunks.append(ids)
        total += int(ids.numel())
    full = torch.cat(chunks, dim=0)[: int(prompt_len)]
    return full.view(1, -1).to(device=device, dtype=torch.long)


def _extract_logits_and_past(outputs: Any) -> tuple[torch.Tensor, Any]:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    past = getattr(outputs, "past_key_values", None)
    if past is None and isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        past = outputs[1]
    return logits, past


@torch.no_grad()
def _manual_decode_run(
    model: torch.nn.Module,
    *,
    prompt_ids: torch.Tensor,
    timed_decode_tokens: int,
    prime_decode_tokens: int,
    label: str,
    component_patcher: Any | None = None,
) -> dict[str, Any]:
    if prompt_ids.device.type != "cuda":
        raise RuntimeError("Decode timing benchmark expects CUDA prompt ids.")

    def _snapshot_component_totals() -> dict[str, float] | None:
        if component_patcher is None:
            return None

        totals: dict[str, float] = {}
        for state in component_patcher.states.values():
            for name, value in state.component_ms_total.items():
                totals[name] = totals.get(name, 0.0) + float(value)
        return totals

    def _snapshot_layer_cache_update_totals() -> dict[int, dict[str, float]] | None:
        if component_patcher is None:
            return None

        out: dict[int, dict[str, float]] = {}
        for state in component_patcher.states.values():
            layer_idx = int(state.layer_idx)
            out[layer_idx] = {
                "cache_update_only": float(
                    state.component_ms_total.get("cache_update_only", 0.0)
                ),
                "cache_object_past_update_inner": float(
                    state.component_ms_total.get("cache_object_past_update_inner", 0.0)
                ),
            }
        return out

    def _snapshot_layer_post_logits_totals() -> dict[int, dict[str, float]] | None:
        if component_patcher is None:
            return None

        out: dict[int, dict[str, float]] = {}
        for state in component_patcher.states.values():
            layer_idx = int(state.layer_idx)
            out[layer_idx] = {
                "post_logits_softmax_fp32": float(
                    state.component_ms_total.get("post_logits_softmax_fp32", 0.0)
                ),
                "post_logits_matmul_probs_v": float(
                    state.component_ms_total.get("post_logits_matmul_probs_v", 0.0)
                ),
                "post_logits_o_proj": float(
                    state.component_ms_total.get("post_logits_o_proj", 0.0)
                ),
            }
        return out

    torch.cuda.synchronize()
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()
    static_v_cache = None
    if os.environ.get("TQ_STATIC_V_CACHE", "").strip() == "1":
        reserve = int(os.environ.get("TQ_STATIC_V_CACHE_RESERVE_TOKENS", "0") or "0")
        static_v_cache = TQStaticValueCache(reserve_tokens=reserve)
        prefill_outputs = model(
            input_ids=prompt_ids,
            past_key_values=static_v_cache,
            use_cache=True,
        )
    else:
        prefill_outputs = model(input_ids=prompt_ids, use_cache=True)
    prefill_end.record()
    torch.cuda.synchronize()
    prefill_ms = float(prefill_start.elapsed_time(prefill_end))

    logits, past = _extract_logits_and_past(prefill_outputs)
    if os.environ.get("TQ_DEBUG_TOPK_LOGITS", "").strip() == "1":
        # Infer the logits variable used immediately before argmax.
        _debug_logits = logits
        if _debug_logits.ndim == 3:
            _debug_logits = _debug_logits[:, -1, :]
        elif _debug_logits.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected logits shape for top-k debug: {tuple(_debug_logits.shape)}")
        topk = min(10, int(_debug_logits.shape[-1]))
        vals, ids = torch.topk(_debug_logits[0], k=topk, dim=-1)
        print(
            "[DEBUG topk logits]",
            "top_ids=", ids.detach().cpu().tolist(),
            "top_vals=", [float(x) for x in vals.detach().cpu().tolist()],
            flush=True,
        )
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens: list[int] = []
    prime_ms: list[float] = []

    for _ in range(int(prime_decode_tokens)):
        def _prime_call():
            os.environ["TQ_CURRENT_DECODE_PHASE"] = "prime"
            return model(input_ids=next_token, past_key_values=past, use_cache=True)
        outputs, dt = _event_time_ms(_prime_call)
        prime_ms.append(float(dt))
        logits, past = _extract_logits_and_past(outputs)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    decode_ms: list[float] = []
    timed_token_component_breakdown: list[dict[str, Any]] = []
    timed_token_layer_cache_update_breakdown: list[dict[str, Any]] = []
    timed_token_layer_post_logits_breakdown: list[dict[str, Any]] = []

    for timed_token_idx in range(int(timed_decode_tokens)):
        component_before = _snapshot_component_totals()
        layer_cache_before = _snapshot_layer_cache_update_totals()
        layer_post_logits_before = _snapshot_layer_post_logits_totals()

        def _decode_call():
            os.environ["TQ_CURRENT_DECODE_PHASE"] = "timed"
            return model(input_ids=next_token, past_key_values=past, use_cache=True)

        outputs, dt = _event_time_ms(_decode_call)
        decode_ms.append(float(dt))

        component_after = _snapshot_component_totals()
        layer_cache_after = _snapshot_layer_cache_update_totals()
        layer_post_logits_after = _snapshot_layer_post_logits_totals()

        if component_before is not None and component_after is not None:
            names = sorted(set(component_before) | set(component_after))
            component_delta_ms = {
                name: float(component_after.get(name, 0.0) - component_before.get(name, 0.0))
                for name in names
            }
            timed_token_component_breakdown.append(
                {
                    "token_idx": int(timed_token_idx),
                    "decode_ms": float(dt),
                    "component_delta_ms": component_delta_ms,
                }
            )

        if layer_cache_before is not None and layer_cache_after is not None:
            layer_rows = []
            layer_ids = sorted(set(layer_cache_before) | set(layer_cache_after))
            for layer_idx in layer_ids:
                before = layer_cache_before.get(layer_idx, {})
                after = layer_cache_after.get(layer_idx, {})
                layer_rows.append(
                    {
                        "layer_idx": int(getattr(state, "layer_idx", -1)),
                        "cache_update_only_ms": float(
                            after.get("cache_update_only", 0.0)
                            - before.get("cache_update_only", 0.0)
                        ),
                        "cache_object_past_update_inner_ms": float(
                            after.get("cache_object_past_update_inner", 0.0)
                            - before.get("cache_object_past_update_inner", 0.0)
                        ),
                    }
                )

            timed_token_layer_cache_update_breakdown.append(
                {
                    "token_idx": int(timed_token_idx),
                    "decode_ms": float(dt),
                    "layers": layer_rows,
                }
            )

        if layer_post_logits_before is not None and layer_post_logits_after is not None:
            layer_rows = []
            layer_ids = sorted(set(layer_post_logits_before) | set(layer_post_logits_after))
            for layer_idx in layer_ids:
                before = layer_post_logits_before.get(layer_idx, {})
                after = layer_post_logits_after.get(layer_idx, {})
                layer_rows.append(
                    {
                        "layer_idx": int(getattr(state, "layer_idx", -1)),
                        "post_logits_softmax_fp32_ms": float(
                            after.get("post_logits_softmax_fp32", 0.0)
                            - before.get("post_logits_softmax_fp32", 0.0)
                        ),
                        "post_logits_matmul_probs_v_ms": float(
                            after.get("post_logits_matmul_probs_v", 0.0)
                            - before.get("post_logits_matmul_probs_v", 0.0)
                        ),
                        "post_logits_o_proj_ms": float(
                            after.get("post_logits_o_proj", 0.0)
                            - before.get("post_logits_o_proj", 0.0)
                        ),
                    }
                )

            timed_token_layer_post_logits_breakdown.append(
                {
                    "token_idx": int(timed_token_idx),
                    "decode_ms": float(dt),
                    "layers": layer_rows,
                }
            )

        logits, past = _extract_logits_and_past(outputs)

        if os.environ.get("TQ_DEBUG_TIMED_TOPK_LOGITS", "").strip() == "1":
            _debug_logits = logits[:, -1, :]
            topk = min(10, int(_debug_logits.shape[-1]))
            vals, ids = torch.topk(_debug_logits[0], k=topk, dim=-1)
            print(
                "[DEBUG timed topk logits]",
                "token_idx=", int(timed_token_idx),
                "top_ids=", ids.detach().cpu().tolist(),
                "top_vals=", [float(x) for x in vals.detach().cpu().tolist()],
                "margin_0_1=", (
                    float(vals[0].detach().cpu().item() - vals[1].detach().cpu().item())
                    if topk > 1 else None
                ),
                flush=True,
            )

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(int(next_token.item()))

    decode_summary = _summary_ms(decode_ms)
    tok_per_sec = float(1000.0 / decode_summary["mean_ms"]) if decode_summary["mean_ms"] > 0 else None

    return {
        "label": label,
        "prompt_len": int(prompt_ids.shape[1]),
        "prime_decode_tokens": int(prime_decode_tokens),
        "timed_decode_tokens": int(timed_decode_tokens),
        "prefill_ms": float(prefill_ms),
        "prime_decode_ms": _summary_ms(prime_ms),
        "decode_ms_per_token": decode_summary,
        "tokens_per_sec": tok_per_sec,
        "generated_token_ids": generated_tokens,
        "timed_token_component_breakdown": timed_token_component_breakdown,
        "timed_token_layer_cache_update_breakdown": timed_token_layer_cache_update_breakdown,
        "timed_token_layer_post_logits_breakdown": timed_token_layer_post_logits_breakdown,
    }



@torch.no_grad()
def _teacher_forced_nll_run(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    context_len: int,
    eval_tokens: int,
    label: str,
    component_patcher: Any | None = None,
) -> dict[str, Any]:
    """
    Teacher-forced decode NLL.

    The model is first prefilling input_ids[:, :context_len].
    Then for eval_tokens steps, it feeds the ground-truth token at each step,
    and computes NLL for the next ground-truth token.

    This avoids autoregressive trajectory divergence.
    """
    if input_ids.device.type != "cuda":
        raise RuntimeError("Teacher-forced timing expects CUDA input ids.")

    total_needed = int(context_len) + int(eval_tokens) + 1
    if int(input_ids.shape[1]) < total_needed:
        raise RuntimeError(
            f"Need at least context_len + eval_tokens + 1 = {total_needed} tokens, "
            f"got {input_ids.shape[1]}"
        )

    prompt_ids = input_ids[:, : int(context_len)].contiguous()

    torch.cuda.synchronize()
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
    if os.environ.get("TQ_STATIC_V_CACHE", "").strip() == "1":
        reserve = int(os.environ.get("TQ_STATIC_V_CACHE_RESERVE_TOKENS", "0") or "0")
        static_v_cache = TQStaticValueCache(reserve_tokens=reserve)
        outputs = model(
            input_ids=prompt_ids,
            past_key_values=static_v_cache,
            use_cache=True,
        )
    else:
        outputs = model(input_ids=prompt_ids, use_cache=True)
    prefill_end.record()
    torch.cuda.synchronize()

    prefill_ms = float(prefill_start.elapsed_time(prefill_end))
    logits, past = _extract_logits_and_past(outputs)

    # NLL for first target after prefill, using last prefill logits.
    losses = []
    top1_hits = []
    top5_hits = []

    def _accumulate(
        logits_step: torch.Tensor,
        target: torch.Tensor,
        *,
        token_idx: int,
        feed_pos: int | None,
        target_pos: int,
        teacher_token: torch.Tensor | None,
    ):
        # logits_step: [1, vocab], target: [1]
        log_probs = torch.log_softmax(logits_step.float(), dim=-1)
        nll = -log_probs.gather(-1, target.view(-1, 1)).squeeze(-1)
        losses.append(float(nll.detach().cpu().item()))

        top5 = torch.topk(logits_step, k=min(5, logits_step.shape[-1]), dim=-1).indices
        top1_hits.append(int(top5[:, 0].eq(target).detach().cpu().item()))
        top5_hits.append(int(top5.eq(target.view(-1, 1)).any(dim=-1).detach().cpu().item()))

        if os.environ.get("TQ_DEBUG_TEACHER_STEP_ALIGNMENT", "0") == "1":
            try:
                tq_summary = {}
                patcher_obj = component_patcher if component_patcher is not None else None
                states = getattr(patcher_obj, "states", {}) if patcher_obj is not None else {}
                for li in [0, 7, 28, 31]:
                    st = states.get(li) if isinstance(states, dict) else None
                    if st is not None:
                        tq_summary[str(li)] = {
                            "compressed_cache_len": int(getattr(st, "compressed_cache_len", -1)),
                            "append_calls": int(getattr(st, "append_calls", -1)),
                            "last_kv_len": int(getattr(st, "last_kv_len", -1)),
                        }

                print("[DEBUG teacher step alignment]", {
                    "label": str(label),
                    "token_idx": int(token_idx),
                    "feed_pos": None if feed_pos is None else int(feed_pos),
                    "target_pos": int(target_pos),
                    "teacher_token": (
                        None if teacher_token is None
                        else int(teacher_token.detach().view(-1)[0].item())
                    ),
                    "target_token": int(target.detach().view(-1)[0].item()),
                    "loss": float(nll.detach().cpu().item()),
                    "top1_hit": int(top1_hits[-1]),
                    "top5_hit": int(top5_hits[-1]),
                    "tq_summary": tq_summary,
                }, flush=True)
            except Exception as e:
                print("[DEBUG teacher step alignment ERROR]", repr(e), flush=True)

    first_target = input_ids[:, int(context_len)]
    _accumulate(logits[:, -1, :], first_target, token_idx=0, feed_pos=None, target_pos=int(context_len), teacher_token=None)

    step_ms = []

    # Feed ground-truth tokens, not generated tokens.
    for i in range(int(eval_tokens) - 1):
        feed_pos = int(context_len) + i
        target_pos = feed_pos + 1

        teacher_token = input_ids[:, feed_pos:feed_pos + 1].contiguous()
        target = input_ids[:, target_pos].contiguous()

        def _call():
            os.environ["TQ_CURRENT_DECODE_PHASE"] = "timed"
            return model(input_ids=teacher_token, past_key_values=past, use_cache=True)

        outputs, dt = _event_time_ms(_call)
        step_ms.append(float(dt))

        logits, past = _extract_logits_and_past(outputs)
        _accumulate(logits[:, -1, :], target, token_idx=i + 1, feed_pos=feed_pos, target_pos=target_pos, teacher_token=teacher_token)

    mean_nll = float(sum(losses) / max(1, len(losses)))
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())

    return {
        "label": label,
        "context_len": int(context_len),
        "eval_tokens": int(eval_tokens),
        "prefill_ms": float(prefill_ms),
        "decode_step_ms": _summary_ms(step_ms),
        "mean_nll": mean_nll,
        "ppl": ppl,
        "top1_acc": float(sum(top1_hits) / max(1, len(top1_hits))),
        "top5_acc": float(sum(top5_hits) / max(1, len(top5_hits))),
        "losses": losses,
        "top1_hits": top1_hits,
        "top5_hits": top5_hits,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode-only end-to-end timing for TurboQuant CUDA attention-logits replacement."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt_len", type=int, default=2048)
    p.add_argument("--timed_decode_tokens", type=int, default=128)
    p.add_argument("--prime_decode_tokens", type=int, default=1)
    p.add_argument("--context_len", type=int, default=None)
    p.add_argument("--eval_tokens", type=int, default=128)
    p.add_argument("--replace_layers", default="all")
    p.add_argument("--scalar_bits", type=int, default=4)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--lloyd_iters", type=int, default=10)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument("--rotation_seed", type=int, default=101)
    p.add_argument("--sketch_seed", type=int, default=202)
    p.add_argument("--codebook_seed", type=int, default=303)
    p.add_argument("--quiet_build", action="store_true")
    p.add_argument(
        "--use_pack_cuda_fastpath",
        action="store_true",
        help="Use validated CUDA fast path for scalar/QJL new-K packing.",
    )
    p.add_argument(
        "--compressed_cache_reserve_tokens",
        type=int,
        default=0,
        help=(
            "Reserve compressed KV append capacity per layer. "
            "0 means auto = prime_decode_tokens + timed_decode_tokens + 8."
        ),
    )
    p.add_argument(
        "--profile_components",
        action="store_true",
        help="Synchronous CUDA-event profiling of replacement subcomponents. Diagnostic only; do not use total runtime as a performance result.",
    )
    p.add_argument(
        "--profile_append_inner",
        action="store_true",
        help=(
            "When --profile_components is set, also profile nested _append_new_k sub-steps. "
            "Default off avoids nested CUDA event synchronization overhead."
        ),
    )
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Run replacement only. Useful after a baseline run, and avoids carrying any prior CUDA error state into replacement.",
    )
    p.add_argument(
        "--skip_replacement",
        action="store_true",
        help="Run baseline only and skip installing/running the TurboQuant replacement.",
    )
    p.add_argument(
        "--text",
        default="TurboQuant decode-time integration benchmark. This text is repeated to build a deterministic prompt. ",
    )
    p.add_argument(
        "--prompt_ids_pt",
        default=None,
        help="Optional .pt file containing input_ids tensor/list/dict to use directly as prompt ids.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This benchmark is intended for CUDA; pass --device cuda:0.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_parse_dtype(args.torch_dtype),
        trust_remote_code=bool(args.trust_remote_code),
        device_map=None,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()

    if os.environ.get("TQ_DEBUG_BASELINE_ATTENTION_BACKEND", "").strip() == "1":
        cfg = getattr(model, "config", None)
        print("[DEBUG baseline backend] model_class =", type(model), flush=True)
        print("[DEBUG baseline backend] config_class =", type(cfg), flush=True)
        print("[DEBUG baseline backend] model_type =", getattr(cfg, "model_type", None), flush=True)
        print("[DEBUG baseline backend] _attn_implementation =", getattr(cfg, "_attn_implementation", None), flush=True)
        print("[DEBUG baseline backend] attn_implementation =", getattr(cfg, "attn_implementation", None), flush=True)

        for name, m in model.named_modules():
            lname = name.lower()
            if "attn" in lname or "attention" in lname:
                print("[DEBUG baseline backend] first_attention_name =", name, flush=True)
                print("[DEBUG baseline backend] first_attention_class =", type(m), flush=True)
                for k in [
                    "layer_idx",
                    "num_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "num_key_value_groups",
                    "attention_dropout",
                ]:
                    if hasattr(m, k):
                        print(f"[DEBUG baseline backend] {k} =", getattr(m, k), flush=True)
                try:
                    src = inspect.getsource(m.forward)
                    print("[DEBUG baseline backend] forward_source_head_BEGIN", flush=True)
                    print("\n".join(src.splitlines()[:120]), flush=True)
                    print("[DEBUG baseline backend] forward_source_head_END", flush=True)
                except Exception as e:
                    print("[DEBUG baseline backend] could_not_get_forward_source =", repr(e), flush=True)
                break

    if args.prompt_ids_pt is not None:
        loaded_prompt = torch.load(args.prompt_ids_pt, map_location="cpu")
        if isinstance(loaded_prompt, dict):
            loaded_prompt = loaded_prompt.get("input_ids", loaded_prompt.get("prompt_ids"))
        if isinstance(loaded_prompt, (list, tuple)):
            loaded_prompt = torch.tensor(loaded_prompt, dtype=torch.long)
        if not torch.is_tensor(loaded_prompt):
            raise RuntimeError(
                f"--prompt_ids_pt must contain a tensor/list/dict with input_ids, got {type(loaded_prompt)}"
            )
        if loaded_prompt.ndim == 1:
            loaded_prompt = loaded_prompt.view(1, -1)
        if loaded_prompt.ndim != 2:
            raise RuntimeError(
                f"--prompt_ids_pt tensor must be rank 1 or 2, got shape={tuple(loaded_prompt.shape)}"
            )
        prompt_ids = loaded_prompt[:, : int(args.prompt_len)].to(
            device=device,
            dtype=torch.long,
        ).contiguous()
        if int(prompt_ids.shape[1]) < int(args.prompt_len):
            raise RuntimeError(
                f"--prompt_ids_pt has only {prompt_ids.shape[1]} tokens, "
                f"requested prompt_len={args.prompt_len}"
            )
    else:
        prompt_ids = _build_prompt_ids(
            tokenizer,
            prompt_len=int(args.prompt_len),
            text=str(args.text),
            device=device,
        )

    print("========== Decode-only TurboQuant CUDA attention replacement timing ==========")
    print(f"model_name          = {args.model_name}")
    print(f"device              = {device}")
    print(f"prompt_len          = {args.prompt_len}")
    print(f"prime_decode_tokens = {args.prime_decode_tokens}")
    print(f"timed_decode_tokens = {args.timed_decode_tokens}")
    print(f"replace_layers      = {args.replace_layers}")

    baseline = None
    baseline_attention_profiler = BaselineDecodeAttentionProfiler(
        enabled=bool(args.profile_components) and not bool(args.skip_baseline)
    )
    baseline_attention_profile_installed = []

    if not bool(args.skip_baseline):
        if bool(args.profile_components):
            baseline_attention_profile_installed = baseline_attention_profiler.install(
                model,
                profile_layers=_parse_replace_layers(args.replace_layers),
            )

        context_len = int(args.context_len or args.prompt_len)
        baseline = _teacher_forced_nll_run(
            model,
            input_ids=prompt_ids,
            context_len=context_len,
            eval_tokens=int(args.eval_tokens),
            label="baseline_original_attention_teacher_forced_nll",
            component_patcher=None,
        )

        if bool(args.profile_components):
            baseline_attention_profiler.restore()

        print(json.dumps({"baseline_decode": baseline}, indent=2))
    else:
        print(json.dumps({"baseline_decode": "skipped"}, indent=2))

    if bool(getattr(args, "skip_replacement", False)):
        out = {
            "baseline_decode": baseline if baseline is not None else "skipped",
            "replacement_decode": "skipped",
            "patcher_summary": {},
            "latency_comparison": {
                "baseline_mean_ms_per_token": (
                    float(baseline["decode_step_ms"]["mean_ms"])
                    if baseline is not None else None
                ),
                "replacement_mean_ms_per_token": None,
                "replacement_over_baseline": None,
                "baseline_over_replacement_speedup": None,
                "baseline_tokens_per_sec": (
                    (float(1000.0 / baseline["decode_step_ms"]["mean_ms"]) if baseline is not None and float(baseline["decode_step_ms"]["mean_ms"]) > 0 else None)
                ),
                "replacement_tokens_per_sec": None,
            },
            "token_comparison": {
                "aligned_tokens": None,
                "exact_match_count": None,
                "exact_match_ratio": None,
            },
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps({"save": args.out}, indent=2))
        return

    patcher = TurboQuantDecodeAttentionPatcher(
        scalar_bits=int(args.scalar_bits),
        qjl_dim=int(args.qjl_dim),
        lloyd_iters=int(args.lloyd_iters),
        max_codebook_samples=int(args.max_codebook_samples),
        rotation_seed=int(args.rotation_seed),
        sketch_seed=int(args.sketch_seed),
        codebook_seed=int(args.codebook_seed),
        verbose_build=not bool(args.quiet_build),
        profile_components=bool(args.profile_components),
        use_pack_cuda_fastpath=bool(args.use_pack_cuda_fastpath),
        compressed_cache_reserve_tokens=(
            int(args.compressed_cache_reserve_tokens)
            if int(args.compressed_cache_reserve_tokens) > 0
            else int(args.prime_decode_tokens) + int(args.timed_decode_tokens) + 8
        ),
        profile_append_inner=bool(args.profile_append_inner),
    )
    installed = patcher.install(model, replace_layers=_parse_replace_layers(args.replace_layers))
    print(json.dumps({"installed_decode_replacement_forwards": installed}, indent=2))

    context_len = int(args.context_len or args.prompt_len)
    replacement = _teacher_forced_nll_run(
        model,
        input_ids=prompt_ids,
        context_len=context_len,
        eval_tokens=int(args.eval_tokens),
        label="turboquant_cuda_teacher_forced_nll_replacement",
        component_patcher=patcher,
    )
    patcher_summary = patcher.summary()
    print(json.dumps({"replacement_decode": replacement}, indent=2))
    print(json.dumps({"patcher_summary": patcher_summary}, indent=2))
    patcher.restore()

    if baseline is not None:
        comparisons = {
            "teacher_forced_nll": {
                "baseline_mean_nll": float(baseline["mean_nll"]),
                "replacement_mean_nll": float(replacement["mean_nll"]),
                "delta_nll": float(replacement["mean_nll"] - baseline["mean_nll"]),
                "baseline_ppl": float(baseline["ppl"]),
                "replacement_ppl": float(replacement["ppl"]),
                "ppl_ratio": float(replacement["ppl"] / baseline["ppl"]),
                "baseline_top1_acc": float(baseline["top1_acc"]),
                "replacement_top1_acc": float(replacement["top1_acc"]),
                "baseline_top5_acc": float(baseline["top5_acc"]),
                "replacement_top5_acc": float(replacement["top5_acc"]),
            },
            "scope": {
                "metric": "teacher_forced_next_token_nll",
                "trajectory": "ground_truth_tokens",
                "baseline_skipped": False,
            },
        }
    else:
        comparisons = {
            "teacher_forced_nll": {
                "baseline_mean_nll": None,
                "replacement_mean_nll": float(replacement["mean_nll"]),
                "delta_nll": None,
                "baseline_ppl": None,
                "replacement_ppl": float(replacement["ppl"]),
                "ppl_ratio": None,
                "baseline_top1_acc": None,
                "replacement_top1_acc": float(replacement["top1_acc"]),
                "baseline_top5_acc": None,
                "replacement_top5_acc": float(replacement["top5_acc"]),
            },
            "scope": {
                "metric": "teacher_forced_next_token_nll",
                "trajectory": "ground_truth_tokens",
                "baseline_skipped": True,
            },
        }

    print(json.dumps({"comparisons": comparisons}, indent=2))

    payload = {
        "benchmark": "teacher_forced_turboquant_nll",
        "config": vars(args),
        "baseline": baseline,
        "replacement": replacement,
        "installed": installed,
        "baseline_attention_profile_installed": baseline_attention_profile_installed,
        "baseline_attention_profile_summary": baseline_attention_profiler.summary(),
        "patcher_summary": patcher_summary,
        "pack_cuda_fastpath_summary": (
            patcher.pack_cuda_fastpath.summary()
            if patcher.pack_cuda_fastpath is not None
            else {"enabled": False}
        ),
        "comparisons": comparisons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
