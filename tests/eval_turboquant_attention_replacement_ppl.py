#!/usr/bin/env python3
from __future__ import annotations

"""
End-to-end PPL prototype with TurboQuant attention-logits replacement.

Scope
-----
This is a quality/integration prototype, not a speed benchmark:
- It leaves the original Hugging Face / SVD-LLaMA attention forward in place.
- A forward hook recomputes attention output using TurboQuant approximate qK^T logits
  and replaces the module's attention output tensor.
- Therefore runtime is intentionally slower than the original model.

Why a hook?
-----------
It is much less invasive than rewriting every Transformers / SVD-LLaMA attention
class, and it is sufficient to answer the immediate research question:
  "If TurboQuant logits replace dense attention logits end-to-end, what PPL do we get?"

Codebook policy
---------------
For the first integration test, each replaced layer lazily fits one shared scalar
codebook on the first batch it sees, then freezes and reuses it for the rest of
the PPL run. This is deliberately simple:
- per-layer shared scalar codebook [16] for 4-bit
- random orthogonal rotation per layer
- Gaussian QJL sketch per layer

This is NOT claimed as the final deployment policy. It is a first end-to-end
quality probe.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
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
    turboquant_prod_reference_logits,
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
    """
    Equivalent to Hugging Face LLaMA repeat_kv.
    Input:  [B, H_kv, T, D]
    Output: [B, H_kv*n_rep, T, D]
    """
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
    """
    Fallback for older/newer helper signature differences.
    Expected cos/sin before unsqueeze: [B,T,D] or [T,D].
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
    if sin.ndim == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


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
    Apply RoPE with best-effort compatibility across common LLaMA attention variants.
    """
    cos = sin = None

    if position_embeddings is not None:
        if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]

    if cos is None or sin is None:
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            return q, k

        # Newer Transformers style: rotary_emb(x, position_ids)
        if position_ids is not None:
            try:
                out = rotary(v, position_ids)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
            except TypeError:
                pass
            except Exception:
                pass

        # Older style: rotary_emb(x, seq_len=...)
        if cos is None or sin is None:
            try:
                out = rotary(v, seq_len=int(k.shape[-2]))
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    cos, sin = out[0], out[1]
            except TypeError:
                pass
            except Exception:
                pass

        # Another older style: rotary_emb(v)
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
        # Old versions often expect position_ids; newer helpers do not.
        try:
            if position_ids is not None:
                return hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        except TypeError:
            pass
        except Exception:
            pass

        try:
            return hf_apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            pass

    return _apply_rope_fallback(q, k, cos, sin)


def _get_attn_config(module: torch.nn.Module) -> tuple[int, int, int, int]:
    """
    Return: num_heads, num_kv_heads, num_kv_groups, head_dim.
    """
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


def _build_causal_mask(
    *,
    bsz: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Additive causal mask [B,1,Q,K], 0 for allowed, -inf for masked.
    """
    mask = torch.full((q_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1 + kv_len - q_len)
    return mask.view(1, 1, q_len, kv_len).expand(bsz, 1, q_len, kv_len)


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    bsz: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return additive mask [B,1,Q,K].
    For PPL use_cache=False, a standard causal mask is enough when no explicit mask arrives.
    """
    if attention_mask is None:
        return _build_causal_mask(
            bsz=bsz,
            q_len=q_len,
            kv_len=kv_len,
            device=device,
            dtype=dtype,
        )

    mask = attention_mask
    if mask.device != device:
        mask = mask.to(device)
    if mask.dtype != dtype:
        mask = mask.to(dtype)

    # Common HF eager format: [B,1,Q,K]
    if mask.ndim == 4:
        return mask[..., :q_len, :kv_len].contiguous()

    # 2D padding mask [B,K], convert to additive + causal.
    if mask.ndim == 2:
        causal = _build_causal_mask(
            bsz=bsz,
            q_len=q_len,
            kv_len=kv_len,
            device=device,
            dtype=dtype,
        )
        keep = mask[:, None, None, :kv_len].to(torch.bool)
        pad = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=dtype)
        pad = pad.masked_fill(~keep, torch.finfo(dtype).min)
        return causal + pad

    # Conservative fallback.
    return _build_causal_mask(
        bsz=bsz,
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype=dtype,
    )


def _replace_first_output(original_output: Any, replacement: torch.Tensor) -> Any:
    """
    Preserve the original attention forward return structure, replacing only the attn_output tensor.
    """
    if torch.is_tensor(original_output):
        return replacement

    if isinstance(original_output, tuple):
        if len(original_output) == 0:
            return original_output
        return (replacement, *original_output[1:])

    if isinstance(original_output, list):
        if len(original_output) == 0:
            return original_output
        return [replacement, *original_output[1:]]

    return replacement


@dataclass
class LayerTQState:
    name: str
    layer_idx: int
    rotation: Optional[torch.Tensor] = None
    sketch: Optional[torch.Tensor] = None
    centroids: Optional[torch.Tensor] = None
    fit_calls: int = 0
    replace_calls: int = 0
    last_info: Optional[dict[str, Any]] = None


class TurboQuantAttentionReplacement:
    def __init__(
        self,
        *,
        scalar_bits: int,
        qjl_dim: int,
        lloyd_iters: int,
        max_codebook_samples: int,
        codebook_fit_batch_limit: int,
        rotation_seed: int,
        sketch_seed: int,
        codebook_seed: int,
        verbose_first_fit: bool,
    ) -> None:
        self.scalar_bits = int(scalar_bits)
        self.qjl_dim = int(qjl_dim)
        self.lloyd_iters = int(lloyd_iters)
        self.max_codebook_samples = int(max_codebook_samples)
        self.codebook_fit_batch_limit = int(codebook_fit_batch_limit)
        self.rotation_seed = int(rotation_seed)
        self.sketch_seed = int(sketch_seed)
        self.codebook_seed = int(codebook_seed)
        self.verbose_first_fit = bool(verbose_first_fit)
        self.states: dict[int, LayerTQState] = {}
        self.handles = []

    def _ensure_state(
        self,
        *,
        state: LayerTQState,
        keys_for_fit: torch.Tensor,
        head_dim: int,
    ) -> None:
        device = keys_for_fit.device
        if state.rotation is None:
            state.rotation = make_random_orthogonal_rotation(
                int(head_dim),
                seed=self.rotation_seed + int(state.layer_idx),
                device=device,
            )
        if state.sketch is None:
            state.sketch = make_gaussian_sketch(
                int(head_dim),
                int(self.qjl_dim),
                seed=self.sketch_seed + int(state.layer_idx),
                device=device,
            )

        if state.centroids is None:
            levels = 1 << int(self.scalar_bits)
            rotation = state.rotation
            rotated_keys = torch.matmul(
                keys_for_fit.reshape(-1, int(head_dim)).to(torch.float32),
                rotation.T.to(torch.float32),
            )
            if self.codebook_fit_batch_limit > 0 and rotated_keys.shape[0] > self.codebook_fit_batch_limit:
                rotated_keys = rotated_keys[: self.codebook_fit_batch_limit]
            state.centroids = fit_lloyd_scalar_codebook(
                rotated_keys.reshape(-1),
                num_levels=int(levels),
                max_iters=int(self.lloyd_iters),
                max_samples=int(self.max_codebook_samples),
                seed=self.codebook_seed + int(state.layer_idx),
            ).contiguous()
            state.fit_calls += 1

            if self.verbose_first_fit:
                print(
                    json.dumps(
                        {
                            "tq_first_fit": True,
                            "layer_idx": int(state.layer_idx),
                            "name": state.name,
                            "centroids_shape": list(state.centroids.shape),
                            "centroids_min": float(state.centroids.min().item()),
                            "centroids_max": float(state.centroids.max().item()),
                            "fit_key_rows": int(rotated_keys.shape[0]),
                            "scalar_bits": int(self.scalar_bits),
                            "qjl_dim": int(self.qjl_dim),
                        }
                    ),
                    flush=True,
                )

    @torch.no_grad()
    def make_hook(self, state: LayerTQState):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], original_output: Any):
            # Llama attention may receive hidden_states positionally or as a keyword,
            # depending on the model wrapper / Transformers version.
            if args:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.get("hidden_states", None)

            if not torch.is_tensor(hidden_states):
                return original_output
            if hidden_states.ndim != 3:
                return original_output

            bsz, q_len, hidden_size = hidden_states.shape
            num_heads, num_kv_heads, num_kv_groups, head_dim = _get_attn_config(module)

            # First PPL integration only supports no KV cache / full-sequence eval.
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                raise RuntimeError(
                    "TurboQuantAttentionReplacement prototype currently expects use_cache=False / no past_key_value."
                )

            query_states = _reshape_projected(
                module.q_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_heads,
                head_dim=head_dim,
            )
            key_states = _reshape_projected(
                module.k_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_kv_heads,
                head_dim=head_dim,
            )
            value_states = _reshape_projected(
                module.v_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_kv_heads,
                head_dim=head_dim,
            )

            position_ids = kwargs.get("position_ids", None)
            position_embeddings = kwargs.get("position_embeddings", None)
            query_states, key_states = _safe_apply_rope(
                module,
                query_states,
                key_states,
                value_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            key_states = _repeat_kv(key_states, num_kv_groups)
            value_states = _repeat_kv(value_states, num_kv_groups)
            kv_len = int(key_states.shape[-2])

            self._ensure_state(
                state=state,
                keys_for_fit=key_states,
                head_dim=head_dim,
            )

            assert state.rotation is not None
            assert state.sketch is not None
            assert state.centroids is not None

            encoding = encode_turboquant_prod_keys(
                key_states,
                rotation=state.rotation,
                centroids=state.centroids,
                sketch=state.sketch,
            )
            tq_logits = turboquant_prod_reference_logits(
                query_states,
                encoding,
                rotation=state.rotation,
                centroids=state.centroids,
                sketch=state.sketch,
            )

            scale = float(getattr(module, "scaling", 1.0 / math.sqrt(float(head_dim))))
            attn_logits = tq_logits.to(torch.float32) * float(scale)

            attention_mask = kwargs.get("attention_mask", None)
            additive_mask = _normalize_attention_mask(
                attention_mask,
                bsz=bsz,
                q_len=q_len,
                kv_len=kv_len,
                device=attn_logits.device,
                dtype=attn_logits.dtype,
            )
            attn_logits = attn_logits + additive_mask

            attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_output = torch.matmul(attn_probs, value_states.to(attn_probs.dtype))
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_size)
            attn_output = module.o_proj(attn_output.to(hidden_states.dtype))

            state.replace_calls += 1
            state.last_info = {
                "last_hidden_shape": list(hidden_states.shape),
                "last_key_shape": list(key_states.shape),
                "last_logits_shape": list(tq_logits.shape),
                "last_scale": float(scale),
            }

            return _replace_first_output(original_output, attn_output)

        return hook

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

            state = LayerTQState(name=name, layer_idx=layer_idx)
            self.states[layer_idx] = state

            try:
                handle = module.register_forward_hook(self.make_hook(state), with_kwargs=True)
            except TypeError as e:
                raise RuntimeError(
                    "This prototype requires torch.nn.Module.register_forward_hook(..., with_kwargs=True). "
                    "The current PyTorch build does not expose that hook form."
                ) from e

            self.handles.append(handle)
            installed.append({"layer_idx": int(layer_idx), "name": name, "module_type": type(module).__name__})

        if not installed:
            raise RuntimeError("No attention modules with q_proj/k_proj/v_proj/o_proj were found for replacement.")
        return installed

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any]:
        return {
            str(k): {
                "name": v.name,
                "layer_idx": int(v.layer_idx),
                "fit_calls": int(v.fit_calls),
                "replace_calls": int(v.replace_calls),
                "centroids_shape": list(v.centroids.shape) if v.centroids is not None else None,
                "last_info": v.last_info,
            }
            for k, v in sorted(self.states.items(), key=lambda kv: kv[0])
        }


@torch.no_grad()
def evaluate_wikitext2_ppl(
    model: torch.nn.Module,
    tokenizer,
    *,
    seqlen: int,
    max_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0].to(torch.long)

    total_available = int(ids.numel() // int(seqlen))
    nsamples = total_available if int(max_samples) <= 0 else min(total_available, int(max_samples))

    nll_sum = 0.0
    token_count = 0
    per_sample_nll = []
    t0 = time.time()

    model.eval()
    for i in range(nsamples):
        batch = ids[i * int(seqlen) : (i + 1) * int(seqlen)].unsqueeze(0).to(device)
        outputs = model(input_ids=batch, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll_val = float(nll.item())
        tok = int(shift_labels.numel())
        nll_sum += nll_val
        token_count += tok
        per_sample_nll.append(nll_val / max(1, tok))

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            json.dumps(
                {
                    "eval_progress": True,
                    "sample": int(i + 1),
                    "nsamples": int(nsamples),
                    "mean_nll_so_far": float(nll_sum / max(1, token_count)),
                    "ppl_so_far": float(math.exp(nll_sum / max(1, token_count))),
                }
            ),
            flush=True,
        )

    mean_nll = float(nll_sum / max(1, token_count))
    ppl = float(math.exp(mean_nll))
    return {
        "dataset": "wikitext2",
        "seqlen": int(seqlen),
        "nsamples": int(nsamples),
        "total_available_samples": int(total_available),
        "tokens": int(token_count),
        "mean_nll": mean_nll,
        "ppl": ppl,
        "elapsed_sec": float(time.time() - t0),
        "per_sample_mean_nll": per_sample_nll,
    }


def _parse_replace_layers(raw: str) -> Optional[set[int]]:
    raw = str(raw).strip().lower()
    if raw in {"all", "*", ""}:
        return None
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end Wikitext2 PPL with TurboQuant attention-logits replacement via attention forward hooks."
    )
    p.add_argument("--model_name", required=True, help="HF model id or local saved compressed model directory.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max_eval_samples", type=int, default=8, help="Use 0 for full Wikitext2.")
    p.add_argument("--run_dense_baseline_first", action="store_true")
    p.add_argument("--replace_layers", default="all", help="'all' or comma-separated layer indices, e.g. 0,15,31.")
    p.add_argument("--scalar_bits", type=int, default=4)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--lloyd_iters", type=int, default=10)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument(
        "--codebook_fit_batch_limit",
        type=int,
        default=0,
        help="Optional cap on rotated key rows used for first-fit codebook; 0 = all rows from first batch.",
    )
    p.add_argument("--rotation_seed", type=int, default=101)
    p.add_argument("--sketch_seed", type=int, default=202)
    p.add_argument("--codebook_seed", type=int, default=303)
    p.add_argument("--quiet_first_fit", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = _parse_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=None,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()

    payload: dict[str, Any] = {
        "benchmark": "turboquant_attention_replacement_end_to_end_ppl",
        "config": vars(args),
        "baseline": None,
        "replacement": None,
    }

    print("========== TurboQuant attention replacement: end-to-end Wikitext2 PPL ==========")
    print(f"model_name       = {args.model_name}")
    print(f"device           = {device}")
    print(f"torch_dtype      = {args.torch_dtype}")
    print(f"seqlen           = {args.seqlen}")
    print(f"max_eval_samples = {args.max_eval_samples}")
    print(f"replace_layers   = {args.replace_layers}")
    print(f"scalar_bits      = {args.scalar_bits}")
    print(f"qjl_dim          = {args.qjl_dim}")

    if bool(args.run_dense_baseline_first):
        print("[Baseline] Evaluating original attention PPL...")
        payload["baseline"] = evaluate_wikitext2_ppl(
            model,
            tokenizer,
            seqlen=int(args.seqlen),
            max_samples=int(args.max_eval_samples),
            device=device,
        )
        print(json.dumps({"baseline_result": payload["baseline"]}, indent=2))

    replacer = TurboQuantAttentionReplacement(
        scalar_bits=int(args.scalar_bits),
        qjl_dim=int(args.qjl_dim),
        lloyd_iters=int(args.lloyd_iters),
        max_codebook_samples=int(args.max_codebook_samples),
        codebook_fit_batch_limit=int(args.codebook_fit_batch_limit),
        rotation_seed=int(args.rotation_seed),
        sketch_seed=int(args.sketch_seed),
        codebook_seed=int(args.codebook_seed),
        verbose_first_fit=not bool(args.quiet_first_fit),
    )

    replace_layers = _parse_replace_layers(args.replace_layers)
    installed = replacer.install(model, replace_layers=replace_layers)
    print(json.dumps({"installed_turboquant_replacement_hooks": installed}, indent=2))

    print("[Replacement] Evaluating TurboQuant-replaced attention PPL...")
    replacement_eval = evaluate_wikitext2_ppl(
        model,
        tokenizer,
        seqlen=int(args.seqlen),
        max_samples=int(args.max_eval_samples),
        device=device,
    )
    payload["replacement"] = {
        "ppl_eval": replacement_eval,
        "hook_summary": replacer.summary(),
        "installed": installed,
    }
    print(json.dumps({"replacement_result": payload["replacement"]}, indent=2))

    replacer.remove()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
