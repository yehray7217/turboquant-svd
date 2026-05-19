#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant import (
    make_random_orthogonal_rotation,
    fit_lloyd_scalar_codebook,
    make_gaussian_sketch,
    encode_turboquant_prod_keys,
    turboquant_prod_reference_logits,
    dense_fp32_logits,
)
from turboquant.real_qk_capture import capture_llama_style_qk


DEFAULT_TEXT = (
    "TurboQuant quality benchmark. This fixed deterministic prompt is repeated "
    "to build a real language-model activation sequence. It is not intended to "
    "measure downstream task quality; it validates true TurboQuant logits on "
    "actual attention q/k activations. "
)


def parse_dtype(name: str) -> torch.dtype | str:
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


def build_fixed_input_ids(
    tokenizer,
    *,
    seq_len: int,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    if not text:
        raise ValueError("text must be non-empty.")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    if ids.numel() == 0:
        raise ValueError("Tokenizer produced no tokens.")

    pieces = []
    total = 0
    while total < int(seq_len):
        pieces.append(ids)
        total += int(ids.numel())
    joined = torch.cat(pieces, dim=0)[: int(seq_len)]
    return joined.view(1, -1).to(device=device, dtype=torch.long)


@torch.no_grad()
def logits_quality_metrics(
    approx: torch.Tensor,
    dense: torch.Tensor,
    *,
    topk: int,
) -> dict[str, float]:
    approx = approx.to(torch.float32)
    dense = dense.to(torch.float32)
    diff = approx - dense

    k = min(int(topk), int(dense.shape[-1]))
    approx_idx = torch.topk(approx, k=k, dim=-1).indices
    dense_idx = torch.topk(dense, k=k, dim=-1).indices

    approx_mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(
        dim=-1,
        index=approx_idx,
        value=True,
    )
    overlap = torch.gather(approx_mask, dim=-1, index=dense_idx).float().mean()
    top1 = (torch.argmax(approx, dim=-1) == torch.argmax(dense, dim=-1)).float().mean()

    dense_probs = torch.softmax(dense, dim=-1)
    approx_probs = torch.softmax(approx, dim=-1)
    candidate_mass = torch.gather(dense_probs, dim=-1, index=approx_idx).sum(dim=-1).mean()
    kl_dense_to_approx = torch.sum(
        dense_probs * (torch.log(dense_probs.clamp_min(1e-12)) - torch.log(approx_probs.clamp_min(1e-12))),
        dim=-1,
    ).mean()

    return {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        f"top{k}_overlap_vs_dense": float(overlap.item()),
        "top1_agreement_vs_dense": float(top1.item()),
        f"dense_softmax_mass_on_approx_top{k}": float(candidate_mass.item()),
        "softmax_kl_dense_to_approx": float(kl_dense_to_approx.item()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Real LLM Q/K activation benchmark for true TurboQuant_prod reference logits."
        )
    )
    p.add_argument("--model_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_query_tokens", type=int, default=1)
    p.add_argument("--scalar_bits", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--qjl_dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--quality_topk", type=int, default=32)
    p.add_argument("--lloyd_iters", type=int, default=20)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--text_file", default=None)
    p.add_argument("--no_rope", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rotation_seed", type=int, default=101)
    p.add_argument("--sketch_seed", type=int, default=202)
    p.add_argument("--codebook_seed", type=int, default=303)
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    text = str(args.text)
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")

    model_dtype = parse_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    input_ids = build_fixed_input_ids(
        tokenizer,
        seq_len=int(args.seq_len),
        text=text,
        device=device,
    )
    qk = capture_llama_style_qk(
        model,
        input_ids,
        layer_idx=int(args.layer_idx),
        num_query_tokens=int(args.num_query_tokens),
        apply_rope=not bool(args.no_rope),
    )

    D = int(qk.queries.shape[-1])
    rotation = make_random_orthogonal_rotation(
        D,
        seed=int(args.rotation_seed),
        device=device,
    )
    rotated_key_samples = torch.matmul(
        qk.keys.reshape(-1, D),
        rotation.T,
    ).contiguous()

    dense = dense_fp32_logits(qk.queries, qk.keys)

    print("========== True TurboQuant real Q/K activation benchmark ==========")
    print(f"model_id             = {args.model_id}")
    print(f"layer_idx            = {args.layer_idx}")
    print(f"seq_len              = {args.seq_len}")
    print(f"query_tokens         = {args.num_query_tokens}")
    print(f"heads q/kv           = {qk.num_attention_heads}/{qk.num_key_value_heads}")
    print(f"key_heads_expanded   = {qk.key_heads_expanded}")
    print(f"rope_applied         = {qk.rope_applied}")
    print(f"rope_detail          = {qk.rope_detail}")
    print(f"fixed seeds          = seed={args.seed}, rotation={args.rotation_seed}, sketch={args.sketch_seed}, codebook={args.codebook_seed}")

    results: list[dict[str, Any]] = []
    for scalar_bits in args.scalar_bits:
        levels = 1 << int(scalar_bits)
        centroids = fit_lloyd_scalar_codebook(
            rotated_key_samples,
            num_levels=levels,
            max_iters=int(args.lloyd_iters),
            max_samples=int(args.max_codebook_samples),
            seed=int(args.codebook_seed) + int(scalar_bits),
        )

        for qjl_dim in args.qjl_dims:
            sketch = make_gaussian_sketch(
                D,
                int(qjl_dim),
                seed=int(args.sketch_seed) + int(qjl_dim),
                device=device,
            )
            encoding = encode_turboquant_prod_keys(
                qk.keys,
                rotation=rotation,
                centroids=centroids,
                sketch=sketch,
            )
            approx = turboquant_prod_reference_logits(
                qk.queries,
                encoding,
                rotation=rotation,
                centroids=centroids,
                sketch=sketch,
            )
            metrics = logits_quality_metrics(
                approx,
                dense,
                topk=int(args.quality_topk),
            )
            item = {
                "scalar_bits": int(scalar_bits),
                "qjl_dim": int(qjl_dim),
                "quality_vs_dense_fp32_qkt": metrics,
                "codebook_centroids": [float(x) for x in centroids.detach().cpu()],
                "encoding_shapes": {
                    "codes": list(encoding.codes.shape),
                    "residual_signs": list(encoding.residual_signs.shape),
                    "residual_norms": list(encoding.residual_norms.shape),
                },
            }
            print(json.dumps(item, indent=2))
            results.append(item)

    payload = {
        "benchmark": "true_turboquant_real_qk_reference",
        "config": {
            **vars(args),
            "torch_dtype": str(args.torch_dtype),
        },
        "qk_capture": {
            "queries_shape": list(qk.queries.shape),
            "keys_shape": list(qk.keys.shape),
            "layer_idx": int(qk.layer_idx),
            "rope_applied": bool(qk.rope_applied),
            "rope_detail": str(qk.rope_detail),
            "num_attention_heads": int(qk.num_attention_heads),
            "num_key_value_heads": int(qk.num_key_value_heads),
            "key_heads_expanded": bool(qk.key_heads_expanded),
        },
        "fixed_seeds": {
            "seed": int(args.seed),
            "rotation_seed": int(args.rotation_seed),
            "sketch_seed": int(args.sketch_seed),
            "codebook_seed": int(args.codebook_seed),
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
