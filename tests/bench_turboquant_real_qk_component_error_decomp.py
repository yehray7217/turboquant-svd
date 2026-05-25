#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    dense_fp32_logits,
)
from turboquant.real_qk_capture import capture_llama_style_qk
from turboquant.rotation import rotate
from turboquant.scalar_quant import scalar_dequantize
from turboquant.qjl import qjl_project_query, qjl_residual_logits


DEFAULT_TEXT = (
    "TurboQuant component error benchmark. This deterministic prompt is repeated "
    "to build a real language-model activation sequence. "
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


def build_fixed_input_ids(tokenizer, *, seq_len: int, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    if ids.numel() == 0:
        raise ValueError("Tokenizer produced zero tokens.")
    pieces = []
    total = 0
    while total < int(seq_len):
        pieces.append(ids)
        total += int(ids.numel())
    joined = torch.cat(pieces, dim=0)[: int(seq_len)]
    return joined.view(1, -1).to(device=device, dtype=torch.long)


@torch.no_grad()
def metrics(approx: torch.Tensor, target: torch.Tensor, *, topk: int | None = None) -> dict[str, float]:
    approx = approx.to(torch.float32)
    target = target.to(torch.float32)
    diff = approx - target

    out = {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "target_absmax": float(target.abs().max().item()),
        "approx_absmax": float(approx.abs().max().item()),
    }

    if topk is not None:
        k = min(int(topk), int(target.shape[-1]))
        approx_idx = torch.topk(approx, k=k, dim=-1).indices
        target_idx = torch.topk(target, k=k, dim=-1).indices

        approx_mask = torch.zeros_like(target, dtype=torch.bool).scatter_(
            dim=-1,
            index=approx_idx,
            value=True,
        )
        overlap = torch.gather(approx_mask, dim=-1, index=target_idx).float().mean()
        top1 = (torch.argmax(approx, dim=-1) == torch.argmax(target, dim=-1)).float().mean()

        target_probs = torch.softmax(target, dim=-1)
        approx_probs = torch.softmax(approx, dim=-1)
        candidate_mass = torch.gather(target_probs, dim=-1, index=approx_idx).sum(dim=-1).mean()
        kl = torch.sum(
            target_probs * (
                torch.log(target_probs.clamp_min(1e-12))
                - torch.log(approx_probs.clamp_min(1e-12))
            ),
            dim=-1,
        ).mean()

        out.update({
            f"top{k}_overlap_vs_target": float(overlap.item()),
            "top1_agreement_vs_target": float(top1.item()),
            f"target_softmax_mass_on_approx_top{k}": float(candidate_mass.item()),
            "softmax_kl_target_to_approx": float(kl.item()),
        })
    return out


@torch.no_grad()
def tensor_stats(x: torch.Tensor) -> dict[str, float | int]:
    xf = x.to(torch.float32)
    finite = torch.isfinite(xf)
    safe = torch.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "finite_ratio": float(finite.float().mean().item()),
        "nan_count": int(torch.isnan(xf).sum().item()),
        "posinf_count": int(torch.isposinf(xf).sum().item()),
        "neginf_count": int(torch.isneginf(xf).sum().item()),
        "absmax_finite_safe": float(safe.abs().max().item()),
        "mean_finite_safe": float(safe.mean().item()),
        "std_finite_safe": float(safe.std().item()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Q/K TurboQuant component error decomposition: scalar vs QJL residual."
    )
    p.add_argument("--model_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--num_query_tokens", type=int, default=1)
    p.add_argument("--scalar_bits", type=int, nargs="+", default=[4])
    p.add_argument("--qjl_dims", type=int, nargs="+", default=[128])
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

    payload: dict[str, Any] = {
        "benchmark": "true_turboquant_real_qk_component_error_decomposition",
        "config": {
            "model_id": str(args.model_id),
            "revision": args.revision,
            "device": str(device),
            "torch_dtype": str(args.torch_dtype),
            "layers": [int(x) for x in args.layers],
            "seq_len": int(args.seq_len),
            "num_query_tokens": int(args.num_query_tokens),
            "scalar_bits": [int(x) for x in args.scalar_bits],
            "qjl_dims": [int(x) for x in args.qjl_dims],
            "quality_topk": int(args.quality_topk),
            "lloyd_iters": int(args.lloyd_iters),
            "max_codebook_samples": int(args.max_codebook_samples),
            "rope_applied": not bool(args.no_rope),
        },
        "results": [],
    }

    print("========== TurboQuant real-Q/K component error decomposition ==========")
    print(f"model_id        = {args.model_id}")
    print(f"seq_len         = {args.seq_len}")
    print(f"layers          = {args.layers}")
    print(f"scalar_bits     = {args.scalar_bits}")
    print(f"qjl_dims        = {args.qjl_dims}")

    for layer_idx in args.layers:
        print(f"[Layer] {layer_idx}")
        qk = capture_llama_style_qk(
            model,
            input_ids,
            layer_idx=int(layer_idx),
            num_query_tokens=int(args.num_query_tokens),
            apply_rope=not bool(args.no_rope),
        )

        q_stats = tensor_stats(qk.queries)
        k_stats = tensor_stats(qk.keys)
        print(json.dumps({
            "layer_idx": int(layer_idx),
            "queries_stats": q_stats,
            "keys_stats": k_stats,
        }, indent=2))

        if q_stats["finite_ratio"] < 1.0 or k_stats["finite_ratio"] < 1.0:
            item = {
                "layer_idx": int(layer_idx),
                "skipped_due_to_nonfinite_qk": True,
                "queries_stats": q_stats,
                "keys_stats": k_stats,
            }
            print(json.dumps(item, indent=2))
            payload["results"].append(item)
            del qk
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

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

                rotated_queries = rotate(qk.queries, rotation).to(torch.float32)
                reconstructed_rotated_keys = scalar_dequantize(encoding.codes, centroids)
                scalar_logits = torch.einsum(
                    "bhqd,bhtd->bhqt",
                    rotated_queries,
                    reconstructed_rotated_keys.to(torch.float32),
                )

                query_projected = qjl_project_query(qk.queries, sketch)
                qjl_residual = qjl_residual_logits(
                    query_projected,
                    encoding.residual_signs,
                    encoding.residual_norms,
                )
                full_tq = scalar_logits + qjl_residual
                true_residual = dense - scalar_logits

                scalar_metrics = metrics(scalar_logits, dense, topk=int(args.quality_topk))
                residual_metrics = metrics(qjl_residual, true_residual, topk=None)
                full_metrics = metrics(full_tq, dense, topk=int(args.quality_topk))

                scalar_rmse = float(scalar_metrics["rmse"])
                full_rmse = float(full_metrics["rmse"])

                item = {
                    "layer_idx": int(layer_idx),
                    "seq_len": int(args.seq_len),
                    "scalar_bits": int(scalar_bits),
                    "qjl_dim": int(qjl_dim),
                    "component_quality": {
                        "scalar_only_vs_dense": scalar_metrics,
                        "qjl_residual_vs_true_residual": residual_metrics,
                        "full_turboquant_vs_dense": full_metrics,
                    },
                    "component_energy": {
                        "true_residual_rmse": float(torch.sqrt(torch.mean(true_residual.to(torch.float32).square())).item()),
                        "qjl_residual_rmse": float(torch.sqrt(torch.mean(qjl_residual.to(torch.float32).square())).item()),
                    },
                    "residual_correction_effect": {
                        "scalar_only_rmse_vs_dense": scalar_rmse,
                        "full_turboquant_rmse_vs_dense": full_rmse,
                        "rmse_reduction_absolute": scalar_rmse - full_rmse,
                        "rmse_reduction_ratio": (
                            (scalar_rmse - full_rmse) / scalar_rmse if scalar_rmse > 0 else 0.0
                        ),
                    },
                }
                print(json.dumps(item, indent=2))
                payload["results"].append(item)

                del sketch, encoding, scalar_logits, qjl_residual, full_tq, true_residual
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del qk, rotation, rotated_key_samples, dense
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
