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
from turboquant.qjl import qjl_project_query, qjl_encode_residual
from turboquant.rotation import rotate, inverse_rotate
from turboquant.scalar_quant import scalar_quantize, scalar_dequantize
from turboquant.scalar_lane_layout import pack_scalar_codes_lane_word_4bit
from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble
from turboquant.decode_pack_cuda_fastpath import DecodePackCudaFastPath
from turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
except Exception:
    hf_apply_rotary_pos_emb = None


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
                old_k = past.key_cache[int(layer_idx)]
                old_v = past.value_cache[int(layer_idx)]
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
    fn = turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda

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
        if int(qjl_dim) != 128:
            raise ValueError("Current CUDA decode integration expects qjl_dim=128.")
        self.scalar_bits = int(scalar_bits)
        self.qjl_dim = int(qjl_dim)
        self.lloyd_iters = int(lloyd_iters)
        self.max_codebook_samples = int(max_codebook_samples)
        self.rotation_seed = int(rotation_seed)
        self.sketch_seed = int(sketch_seed)
        self.codebook_seed = int(codebook_seed)
        self.verbose_build = bool(verbose_build)
        self.profile_components = bool(profile_components)
        self.use_pack_cuda_fastpath = bool(use_pack_cuda_fastpath)
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

        k_shape = list(full_keys_kv.shape)
        v_shape = list(full_values_kv.shape)
        k_shape[-2] = capacity
        v_shape[-2] = capacity

        k_storage = torch.empty(
            k_shape,
            dtype=full_keys_kv.dtype,
            device=full_keys_kv.device,
        )
        v_storage = torch.empty(
            v_shape,
            dtype=full_values_kv.dtype,
            device=full_values_kv.device,
        )

        k_storage[..., :prefix_len, :].copy_(full_keys_kv)
        v_storage[..., :prefix_len, :].copy_(full_values_kv)

        state.dense_key_storage = k_storage
        state.dense_value_storage = v_storage
        state.dense_cache_len = prefix_len
        state.dense_cache_capacity = capacity

        past.key_cache[layer_i] = k_storage[..., :prefix_len, :]
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

        if (
            state.dense_key_storage is None
            or state.dense_value_storage is None
            or int(state.dense_cache_capacity) <= 0
        ):
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

        assert state.dense_key_storage is not None
        assert state.dense_value_storage is not None

        start = int(state.dense_cache_len)
        add = int(value_states_new.shape[-2])
        end = start + add

        if end > int(state.dense_cache_capacity):
            raise RuntimeError(
                "Preallocated dense K/V cache capacity exceeded: "
                f"need end={end}, capacity={state.dense_cache_capacity}. "
                "Increase --compressed_cache_reserve_tokens."
            )

        state.dense_key_storage[..., start:end, :].copy_(key_states_new)
        state.dense_value_storage[..., start:end, :].copy_(value_states_new)

        state.dense_cache_len = end

        active_k = state.dense_key_storage[..., :end, :]
        active_v = state.dense_value_storage[..., :end, :]

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

        # -----------------------------------------------------------------
        # Encode new K, split using the same sequence as:
        # turboquant.turboquant_prod.encode_turboquant_prod_keys()
        # -----------------------------------------------------------------
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

        if use_scalar_quant_cuda:
            from turboquant.decode_pack_cuda_fastpath import scalar_quantize_16_cuda

            codes_raw, ms = _profile_cuda_ms(
                inner_profile,
                lambda: scalar_quantize_16_cuda(rotated_keys, state.centroids),
            )
            state.add_component_ms("encode_scalar_quantize_new_k_cuda", ms)
        else:
            codes_raw, ms = _profile_cuda_ms(
                inner_profile,
                lambda: scalar_quantize(rotated_keys, state.centroids),
            )
            state.add_component_ms("encode_scalar_quantize_new_k", ms)

        residual_mode = os.environ.get("TQ_RESIDUAL_MODE", "full").strip().lower()
        if residual_mode not in ("full", "rotated", "original"):
            raise RuntimeError(
                "Invalid TQ_RESIDUAL_MODE. Expected one of: full, rotated, original; "
                f"got {residual_mode!r}"
            )

        use_fused_residual_qjl = (
            residual_mode == "full"
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

            (residual_signs_raw, residual_norms_raw), ms = _profile_cuda_ms(
                inner_profile,
                lambda: qjl_encode_residual(residual, state.sketch),
            )
            state.add_component_ms("encode_qjl_encode_residual_new_k", ms)
            state.add_component_ms(f"encode_residual_mode_{residual_mode}", 0.0)

        codes, ms = _profile_cuda_ms(
            inner_profile,
            lambda: codes_raw.contiguous(),
        )
        state.add_component_ms("encode_contiguous_codes_new_k", ms)

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

        qjl_new, ms = _profile_cuda_ms(
            inner_profile,
            (
                (lambda: pack_fastpath.pack_qjl(residual_signs))
                if pack_fastpath is not None
                else (lambda: pack_qjl_signs_lane_nibble(residual_signs).contiguous())
            ),
        )
        state.add_component_ms(
            "pack_qjl_signs_new_k_cuda_fastpath"
            if pack_fastpath is not None
            else "pack_qjl_signs_new_k",
            ms,
        )

        def _slice_write_preallocated_cache():
            self._append_preallocated_compressed_cache(
                state=state,
                scalar_new=scalar_new,
                qjl_new=qjl_new,
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
                if state.ready() and hasattr(past, "key_cache") and hasattr(past, "value_cache"):
                    out = patcher._append_preallocated_dense_kv_cache(
                        state=state,
                        past=past,
                        key_states_new=key_states_new,
                        value_states_new=value_states_new,
                    )
                    if out is not None:
                        return out

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
                full_keys_local = _repeat_kv(full_keys_kv, num_kv_groups).contiguous()
                full_values_local = _repeat_kv(full_values_kv, num_kv_groups).contiguous()
                new_keys_expanded_local = _repeat_kv(key_states_new, num_kv_groups).contiguous()
                return full_keys_local, full_values_local, new_keys_expanded_local

            (full_keys, full_values, new_keys_expanded), ms = _profile_cuda_ms(
                patcher.profile_components,
                _repeat_kv_states,
            )
            state.add_component_ms("repeat_kv_materialize", ms)

            if not state.ready():
                if (
                    hasattr(present, "key_cache")
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

                _, ms = _profile_cuda_ms(
                    patcher.profile_components,
                    lambda: patcher._fit_state_from_full_k(
                        state=state,
                        full_keys=full_keys,
                        head_dim=head_dim,
                    ),
                )
                state.add_component_ms("prime_build_full_prefix_state", ms)
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

            scale = float(getattr(self_module, "scaling", 1.0 / math.sqrt(float(head_dim))))
            attn_logits = tq_logits.to(torch.float32) * float(scale)

            def _mask_build_only():
                return _normalize_mask(
                    kwargs.get("attention_mask", None),
                    q_len=q_len,
                    kv_len=int(full_keys.shape[-2]),
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

            def _probs_cast_only():
                return attn_probs_fp32.to(full_values.dtype)

            attn_probs, ms = _profile_cuda_ms(
                patcher.profile_components,
                _probs_cast_only,
            )
            state.add_component_ms("post_logits_probs_cast_to_v_dtype", ms)

            def _values_cast_only():
                return full_values.to(attn_probs.dtype)

            full_values_for_matmul, ms = _profile_cuda_ms(
                patcher.profile_components,
                _values_cast_only,
            )
            state.add_component_ms("post_logits_values_cast_to_probs_dtype", ms)

            def _matmul_probs_v_only():
                return torch.matmul(attn_probs, full_values_for_matmul)

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
                    "layer_idx": int(layer_idx),
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
                    "layer_idx": int(layer_idx),
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
    prefill_outputs = model(input_ids=prompt_ids, use_cache=True)
    prefill_end.record()
    torch.cuda.synchronize()
    prefill_ms = float(prefill_start.elapsed_time(prefill_end))

    logits, past = _extract_logits_and_past(prefill_outputs)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens: list[int] = []
    prime_ms: list[float] = []

    for _ in range(int(prime_decode_tokens)):
        def _prime_call():
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
                        "layer_idx": int(layer_idx),
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
                        "layer_idx": int(layer_idx),
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
        "--text",
        default="TurboQuant decode-time integration benchmark. This text is repeated to build a deterministic prompt. ",
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

        baseline = _manual_decode_run(
            model,
            prompt_ids=prompt_ids,
            timed_decode_tokens=int(args.timed_decode_tokens),
            prime_decode_tokens=int(args.prime_decode_tokens),
            label="baseline_original_attention_decode",
        )

        if bool(args.profile_components):
            baseline_attention_profiler.restore()

        print(json.dumps({"baseline_decode": baseline}, indent=2))
    else:
        print(json.dumps({"baseline_decode": "skipped"}, indent=2))

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

    replacement = _manual_decode_run(
        model,
        prompt_ids=prompt_ids,
        timed_decode_tokens=int(args.timed_decode_tokens),
        prime_decode_tokens=int(args.prime_decode_tokens),
        label="turboquant_cuda_decode_attention_replacement",
        component_patcher=patcher,
    )
    patcher_summary = patcher.summary()
    print(json.dumps({"replacement_decode": replacement}, indent=2))
    print(json.dumps({"patcher_summary": patcher_summary}, indent=2))
    patcher.restore()

    r_mean = float(replacement["decode_ms_per_token"]["mean_ms"])
    if baseline is not None:
        b_mean = float(baseline["decode_ms_per_token"]["mean_ms"])
        baseline_tokens = baseline["generated_token_ids"]
        replacement_tokens = replacement["generated_token_ids"]
        aligned = min(len(baseline_tokens), len(replacement_tokens))
        exact_matches = sum(int(baseline_tokens[i] == replacement_tokens[i]) for i in range(aligned))
        latency_cmp = {
            "baseline_mean_ms_per_token": b_mean,
            "replacement_mean_ms_per_token": r_mean,
            "replacement_over_baseline": float(r_mean / b_mean) if b_mean > 0 else None,
            "baseline_over_replacement_speedup": float(b_mean / r_mean) if r_mean > 0 else None,
            "baseline_tokens_per_sec": baseline["tokens_per_sec"],
            "replacement_tokens_per_sec": replacement["tokens_per_sec"],
        }
        token_cmp = {
            "aligned_tokens": int(aligned),
            "exact_match_count": int(exact_matches),
            "exact_match_ratio": float(exact_matches / aligned) if aligned > 0 else None,
        }
    else:
        latency_cmp = {
            "baseline_mean_ms_per_token": None,
            "replacement_mean_ms_per_token": r_mean,
            "replacement_over_baseline": None,
            "baseline_over_replacement_speedup": None,
            "baseline_tokens_per_sec": None,
            "replacement_tokens_per_sec": replacement["tokens_per_sec"],
        }
        token_cmp = {
            "aligned_tokens": None,
            "exact_match_count": None,
            "exact_match_ratio": None,
        }

    comparisons = {
        "steady_state_decode_latency": latency_cmp,
        "generated_token_agreement": token_cmp,
        "scope": {
            "prefill_attention": "original_attention",
            "timed_decode_attention": "turboquant_cuda_nonfactor_combined_logits",
            "prime_step_excluded_from_steady_state": True,
            "compressed_cache_append": "Python/PyTorch glue with packed tensor concatenation",
            "baseline_skipped": bool(args.skip_baseline),
        },
    }
    print(json.dumps({"comparisons": comparisons}, indent=2))

    payload = {
        "benchmark": "turboquant_decode_attention_cuda_true_timing",
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
