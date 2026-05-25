#!/usr/bin/env python3
from __future__ import annotations

"""
Correct timing benchmark for TurboQuant attention replacement.

This script differs from the earlier hook-based PPL prototype:

Previous prototype:
  original attention forward runs
  + hook recomputes TurboQuant attention output
  = useful for PPL quality, invalid for runtime

This benchmark:
  baseline path: original attention only
  replacement path: monkey-patched TurboQuant attention forward only
  = valid for timing the current replacement implementation

Important scope:
- The replacement forward currently uses the Python/PyTorch TurboQuant reference path.
- It is NOT the optimized decode-only CUDA kernel path.
- Therefore these timings answer:
    "What is the real cost of the current full-sequence replacement implementation?"
  not:
    "What will the final optimized decode kernel end-to-end speedup be?"
"""

import argparse
import json
import math
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


def _safe_apply_rope(
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
        raise RuntimeError(f"Could not infer attention layout for {type(module).__name__}")

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

    if mask.ndim == 4:
        return mask[..., :q_len, :kv_len].contiguous()

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

    return _build_causal_mask(
        bsz=bsz,
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype=dtype,
    )


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


class TurboQuantForwardPatcher:
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
        self.original_forwards: list[tuple[torch.nn.Module, Any]] = []

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

    def _make_forward(self, module: torch.nn.Module, state: LayerTQState):
        patcher = self

        @torch.no_grad()
        def tq_forward(self_module: torch.nn.Module, *f_args: Any, **f_kwargs: Any):
            if f_args:
                hidden_states = f_args[0]
            else:
                hidden_states = f_kwargs.get("hidden_states", None)

            if not torch.is_tensor(hidden_states):
                raise RuntimeError("TurboQuant replacement forward could not locate hidden_states.")
            if hidden_states.ndim != 3:
                raise RuntimeError(f"Expected hidden_states [B,T,C], got {tuple(hidden_states.shape)}")

            past_key_value = f_kwargs.get("past_key_value", None)
            if past_key_value is not None:
                raise RuntimeError("This benchmark expects use_cache=False / no past_key_value.")

            output_attentions = bool(f_kwargs.get("output_attentions", False))
            if output_attentions:
                raise RuntimeError("This benchmark expects output_attentions=False.")

            bsz, q_len, hidden_size = hidden_states.shape
            num_heads, num_kv_heads, num_kv_groups, head_dim = _get_attn_config(self_module)

            query_states = _reshape_projected(
                self_module.q_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_heads,
                head_dim=head_dim,
            )
            key_states = _reshape_projected(
                self_module.k_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_kv_heads,
                head_dim=head_dim,
            )
            value_states = _reshape_projected(
                self_module.v_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_kv_heads,
                head_dim=head_dim,
            )

            position_ids = f_kwargs.get("position_ids", None)
            position_embeddings = f_kwargs.get("position_embeddings", None)
            query_states, key_states = _safe_apply_rope(
                self_module,
                query_states,
                key_states,
                value_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            key_states = _repeat_kv(key_states, num_kv_groups)
            value_states = _repeat_kv(value_states, num_kv_groups)
            kv_len = int(key_states.shape[-2])

            patcher._ensure_state(
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

            scale = float(getattr(self_module, "scaling", 1.0 / math.sqrt(float(head_dim))))
            attn_logits = tq_logits.to(torch.float32) * float(scale)

            attention_mask = f_kwargs.get("attention_mask", None)
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
            attn_output = self_module.o_proj(attn_output.to(hidden_states.dtype))

            state.replace_calls += 1
            state.last_info = {
                "last_hidden_shape": list(hidden_states.shape),
                "last_key_shape": list(key_states.shape),
                "last_logits_shape": list(tq_logits.shape),
                "last_scale": float(scale),
            }

            # LLaMA decoder layers consume a 3-tuple:
            # (attn_output, attn_weights_or_None, present_key_value_or_None)
            return attn_output, None, None

        return types.MethodType(tq_forward, module)

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
            self.original_forwards.append((module, module.forward))
            module.forward = self._make_forward(module, state)

            installed.append({
                "layer_idx": int(layer_idx),
                "name": name,
                "module_type": type(module).__name__,
            })

        if not installed:
            raise RuntimeError("No attention modules with q_proj/k_proj/v_proj/o_proj were patched.")
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
                "fit_calls": int(v.fit_calls),
                "replace_calls": int(v.replace_calls),
                "centroids_shape": list(v.centroids.shape) if v.centroids is not None else None,
                "last_info": v.last_info,
            }
            for k, v in sorted(self.states.items(), key=lambda kv: kv[0])
        }


def _parse_replace_layers(raw: str) -> Optional[set[int]]:
    raw = str(raw).strip().lower()
    if raw in {"all", "*", ""}:
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _load_wikitext2_tokens(tokenizer) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt")["input_ids"][0].to(torch.long)


def _segment_tokens(ids: torch.Tensor, *, seqlen: int, max_samples: int) -> list[torch.Tensor]:
    total_available = int(ids.numel() // int(seqlen))
    nsamples = total_available if int(max_samples) <= 0 else min(total_available, int(max_samples))
    return [ids[i * int(seqlen) : (i + 1) * int(seqlen)].unsqueeze(0) for i in range(nsamples)]


@torch.no_grad()
def _evaluate_ppl_from_segments(
    model: torch.nn.Module,
    segments: list[torch.Tensor],
    *,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_sample_mean_nll = []
    t0 = time.time()

    for i, seg in enumerate(segments):
        batch = seg.to(device)
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
        total_nll += nll_val
        total_tokens += tok
        per_sample_mean_nll.append(nll_val / max(1, tok))

        print(
            json.dumps(
                {
                    "ppl_progress": label,
                    "sample": int(i + 1),
                    "nsamples": int(len(segments)),
                    "mean_nll_so_far": float(total_nll / max(1, total_tokens)),
                    "ppl_so_far": float(math.exp(total_nll / max(1, total_tokens))),
                }
            ),
            flush=True,
        )

        del batch, outputs, logits, shift_logits, shift_labels, nll
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_nll = float(total_nll / max(1, total_tokens))
    return {
        "nsamples": int(len(segments)),
        "tokens": int(total_tokens),
        "mean_nll": mean_nll,
        "ppl": float(math.exp(mean_nll)),
        "elapsed_sec": float(time.time() - t0),
        "per_sample_mean_nll": per_sample_mean_nll,
    }


@torch.no_grad()
def _prime_replacement_codebooks(
    model: torch.nn.Module,
    first_segment: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    batch = first_segment.to(device)
    outputs = model(input_ids=batch, use_cache=False)
    del outputs, batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return {"prime_elapsed_sec": float(time.time() - t0)}


@torch.no_grad()
def _bench_forward_latency_ms(
    model: torch.nn.Module,
    batch_cpu: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    label: str,
) -> dict[str, Any]:
    model.eval()
    batch = batch_cpu.to(device)

    for _ in range(int(warmup)):
        outputs = model(input_ids=batch, use_cache=False)
        del outputs
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms = []
    if device.type == "cuda":
        for _ in range(int(iters)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model(input_ids=batch, use_cache=False)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
            del outputs
    else:
        for _ in range(int(iters)):
            t0 = time.perf_counter()
            outputs = model(input_ids=batch, use_cache=False)
            dt = (time.perf_counter() - t0) * 1000.0
            times_ms.append(float(dt))
            del outputs

    if device.type == "cuda":
        torch.cuda.empty_cache()

    vals = torch.tensor(times_ms, dtype=torch.float64)
    return {
        "label": label,
        "warmup": int(warmup),
        "iters": int(iters),
        "times_ms": times_ms,
        "mean_ms": float(vals.mean().item()),
        "median_ms": float(vals.median().item()),
        "min_ms": float(vals.min().item()),
        "max_ms": float(vals.max().item()),
        "std_ms": float(vals.std(unbiased=False).item()) if len(times_ms) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correct runtime benchmark: original attention vs true TurboQuant replacement attention forward."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max_eval_samples", type=int, default=8, help="0 = full Wikitext2.")
    p.add_argument("--skip_ppl", action="store_true", help="Measure latency only.")
    p.add_argument("--replace_layers", default="all")
    p.add_argument("--scalar_bits", type=int, default=4)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--lloyd_iters", type=int, default=10)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument("--codebook_fit_batch_limit", type=int, default=0)
    p.add_argument("--rotation_seed", type=int, default=101)
    p.add_argument("--sketch_seed", type=int, default=202)
    p.add_argument("--codebook_seed", type=int, default=303)
    p.add_argument("--quiet_first_fit", action="store_true")
    p.add_argument("--timing_warmup", type=int, default=3)
    p.add_argument("--timing_iters", type=int, default=10)
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

    ids = _load_wikitext2_tokens(tokenizer)
    segments = _segment_tokens(
        ids,
        seqlen=int(args.seqlen),
        max_samples=int(args.max_eval_samples),
    )
    if not segments:
        raise RuntimeError("No Wikitext2 segments available for the requested seqlen.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=None,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()

    payload: dict[str, Any] = {
        "benchmark": "turboquant_attention_replacement_correct_timing",
        "config": vars(args),
        "baseline": {},
        "replacement": {},
        "comparisons": {},
    }

    print("========== Correct timing: original attention vs true TurboQuant replacement ==========")
    print(f"model_name       = {args.model_name}")
    print(f"device           = {device}")
    print(f"torch_dtype      = {args.torch_dtype}")
    print(f"seqlen           = {args.seqlen}")
    print(f"max_eval_samples = {args.max_eval_samples}")
    print(f"replace_layers   = {args.replace_layers}")
    print(f"scalar_bits      = {args.scalar_bits}")
    print(f"qjl_dim          = {args.qjl_dim}")

    # Baseline original-attention latency.
    print("[Baseline] Forward latency benchmark...")
    baseline_latency = _bench_forward_latency_ms(
        model,
        segments[0],
        device=device,
        warmup=int(args.timing_warmup),
        iters=int(args.timing_iters),
        label="baseline_original_attention",
    )
    payload["baseline"]["latency"] = baseline_latency
    print(json.dumps({"baseline_latency": baseline_latency}, indent=2))

    if not bool(args.skip_ppl):
        print("[Baseline] PPL evaluation...")
        baseline_ppl = _evaluate_ppl_from_segments(
            model,
            segments,
            device=device,
            label="baseline",
        )
        payload["baseline"]["ppl_eval"] = baseline_ppl
        print(json.dumps({"baseline_ppl": baseline_ppl}, indent=2))

    # Patch to true replacement forward.
    patcher = TurboQuantForwardPatcher(
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
    installed = patcher.install(model, replace_layers=_parse_replace_layers(args.replace_layers))
    payload["replacement"]["installed"] = installed
    print(json.dumps({"installed_replacement_forwards": installed}, indent=2))

    # Prime codebooks outside timed latency/PPL.
    print("[Replacement] Priming per-layer codebooks on the first segment...")
    prime = _prime_replacement_codebooks(
        model,
        segments[0],
        device=device,
    )
    payload["replacement"]["prime"] = prime
    print(json.dumps({"replacement_prime": prime}, indent=2))

    print("[Replacement] Forward latency benchmark...")
    replacement_latency = _bench_forward_latency_ms(
        model,
        segments[0],
        device=device,
        warmup=int(args.timing_warmup),
        iters=int(args.timing_iters),
        label="turboquant_replacement_attention",
    )
    payload["replacement"]["latency"] = replacement_latency
    print(json.dumps({"replacement_latency": replacement_latency}, indent=2))

    if not bool(args.skip_ppl):
        print("[Replacement] PPL evaluation...")
        replacement_ppl = _evaluate_ppl_from_segments(
            model,
            segments,
            device=device,
            label="replacement",
        )
        payload["replacement"]["ppl_eval"] = replacement_ppl
        print(json.dumps({"replacement_ppl": replacement_ppl}, indent=2))

    payload["replacement"]["forward_summary"] = patcher.summary()

    b_mean = float(payload["baseline"]["latency"]["mean_ms"])
    r_mean = float(payload["replacement"]["latency"]["mean_ms"])
    payload["comparisons"]["latency"] = {
        "baseline_mean_ms": b_mean,
        "replacement_mean_ms": r_mean,
        "replacement_over_baseline": float(r_mean / b_mean) if b_mean > 0 else None,
        "baseline_over_replacement_speedup": float(b_mean / r_mean) if r_mean > 0 else None,
    }

    if "ppl_eval" in payload["baseline"] and "ppl_eval" in payload["replacement"]:
        b_ppl = float(payload["baseline"]["ppl_eval"]["ppl"])
        r_ppl = float(payload["replacement"]["ppl_eval"]["ppl"])
        b_nll = float(payload["baseline"]["ppl_eval"]["mean_nll"])
        r_nll = float(payload["replacement"]["ppl_eval"]["mean_nll"])
        payload["comparisons"]["ppl"] = {
            "baseline_ppl": b_ppl,
            "replacement_ppl": r_ppl,
            "delta_ppl": float(r_ppl - b_ppl),
            "relative_ppl_increase": float((r_ppl - b_ppl) / b_ppl) if b_ppl > 0 else None,
            "baseline_mean_nll": b_nll,
            "replacement_mean_nll": r_nll,
            "delta_mean_nll": float(r_nll - b_nll),
        }

    print(json.dumps({"comparisons": payload["comparisons"]}, indent=2))

    patcher.restore()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
