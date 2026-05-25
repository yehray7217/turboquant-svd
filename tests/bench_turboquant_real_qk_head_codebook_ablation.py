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
    dense_fp32_logits,
)
from turboquant.real_qk_capture import capture_llama_style_qk
from turboquant.rotation import rotate, inverse_rotate
from turboquant.qjl import qjl_encode_residual, qjl_project_query, qjl_residual_logits
from turboquant.scalar_quant import scalar_quantize, scalar_dequantize


DEFAULT_TEXT = (
    "TurboQuant head-specific scalar codebook ablation. "
    "This deterministic prompt is repeated to build a real language-model "
    "activation sequence. "
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


@torch.no_grad()
def quality_metrics(approx: torch.Tensor, target: torch.Tensor, *, topk: int | None = None) -> dict[str, float]:
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
def fit_head_specific_codebooks(
    rotated_keys: torch.Tensor,
    *,
    num_levels: int,
    lloyd_iters: int,
    max_codebook_samples: int,
    seed: int,
) -> torch.Tensor:
    """
    rotated_keys: [B,H,T,D]
    returns: [H,K]
    """
    if rotated_keys.ndim != 4:
        raise ValueError("rotated_keys must be [B,H,T,D].")
    B, H, T, D = rotated_keys.shape
    codebooks = []
    for h in range(H):
        samples = rotated_keys[:, h, :, :].reshape(-1)
        c = fit_lloyd_scalar_codebook(
            samples,
            num_levels=int(num_levels),
            max_iters=int(lloyd_iters),
            max_samples=int(max_codebook_samples),
            seed=int(seed) + int(h),
        )
        if c.ndim != 1 or int(c.numel()) != int(num_levels):
            raise RuntimeError(f"Expected shared per-head codebook [K], got {tuple(c.shape)}")
        codebooks.append(c)
    return torch.stack(codebooks, dim=0).contiguous()  # [H,K]


@torch.no_grad()
def head_scalar_quantize(
    rotated_keys: torch.Tensor,
    head_centroids: torch.Tensor,
    *,
    chunk_rows: int,
) -> torch.Tensor:
    """
    rotated_keys: [B,H,T,D]
    head_centroids: [H,K]
    returns codes: [B,H,T,D] uint8
    """
    if rotated_keys.ndim != 4:
        raise ValueError("rotated_keys must be [B,H,T,D].")
    if head_centroids.ndim != 2:
        raise ValueError("head_centroids must be [H,K].")
    B, H, T, D = rotated_keys.shape
    if int(head_centroids.shape[0]) != H:
        raise ValueError(f"Head mismatch: keys H={H}, centroids H={head_centroids.shape[0]}")

    out = torch.empty((B, H, T, D), device=rotated_keys.device, dtype=torch.uint8)
    chunk_rows = max(1, int(chunk_rows))
    for h in range(H):
        x = rotated_keys[:, h, :, :].reshape(-1, D).to(torch.float32)  # [B*T,D]
        c = head_centroids[h].to(device=rotated_keys.device, dtype=torch.float32).view(1, 1, -1)  # [1,1,K]
        chunks = []
        for begin in range(0, int(x.shape[0]), chunk_rows):
            xx = x[begin: begin + chunk_rows]
            dist = torch.abs(xx.unsqueeze(-1) - c)  # [C,D,K]
            codes = torch.argmin(dist, dim=-1).to(torch.uint8)
            chunks.append(codes)
        head_codes = torch.cat(chunks, dim=0).reshape(B, T, D)
        out[:, h, :, :] = head_codes
    return out.contiguous()


@torch.no_grad()
def head_scalar_dequantize(codes: torch.Tensor, head_centroids: torch.Tensor) -> torch.Tensor:
    """
    codes: [B,H,T,D] uint8
    head_centroids: [H,K]
    returns: [B,H,T,D] float32
    """
    if codes.ndim != 4:
        raise ValueError("codes must be [B,H,T,D].")
    if head_centroids.ndim != 2:
        raise ValueError("head_centroids must be [H,K].")
    B, H, T, D = codes.shape
    if int(head_centroids.shape[0]) != H:
        raise ValueError(f"Head mismatch: codes H={H}, centroids H={head_centroids.shape[0]}")

    out = torch.empty((B, H, T, D), device=codes.device, dtype=torch.float32)
    for h in range(H):
        table = head_centroids[h].to(device=codes.device, dtype=torch.float32)
        out[:, h, :, :] = table[codes[:, h, :, :].to(torch.long)]
    return out.contiguous()


@torch.no_grad()
def build_shared_encoding(
    keys: torch.Tensor,
    *,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rotated_keys = rotate(keys, rotation).to(torch.float32)
    codes = scalar_quantize(rotated_keys, centroids)
    reconstructed_rotated = scalar_dequantize(codes, centroids)
    reconstructed_keys = inverse_rotate(reconstructed_rotated, rotation).to(torch.float32)
    residual = keys.to(torch.float32) - reconstructed_keys
    residual_signs, residual_norms = qjl_encode_residual(residual, sketch)
    return {
        "codes": codes.contiguous(),
        "reconstructed_rotated": reconstructed_rotated.contiguous(),
        "residual_signs": residual_signs.contiguous(),
        "residual_norms": residual_norms.contiguous(),
    }


@torch.no_grad()
def build_head_encoding(
    keys: torch.Tensor,
    *,
    rotation: torch.Tensor,
    head_centroids: torch.Tensor,
    sketch: torch.Tensor,
    quant_chunk_rows: int,
) -> dict[str, torch.Tensor]:
    rotated_keys = rotate(keys, rotation).to(torch.float32)
    codes = head_scalar_quantize(rotated_keys, head_centroids, chunk_rows=int(quant_chunk_rows))
    reconstructed_rotated = head_scalar_dequantize(codes, head_centroids)
    reconstructed_keys = inverse_rotate(reconstructed_rotated, rotation).to(torch.float32)
    residual = keys.to(torch.float32) - reconstructed_keys
    residual_signs, residual_norms = qjl_encode_residual(residual, sketch)
    return {
        "codes": codes.contiguous(),
        "reconstructed_rotated": reconstructed_rotated.contiguous(),
        "residual_signs": residual_signs.contiguous(),
        "residual_norms": residual_norms.contiguous(),
    }


@torch.no_grad()
def evaluate_encoding(
    *,
    queries: torch.Tensor,
    dense: torch.Tensor,
    rotation: torch.Tensor,
    sketch: torch.Tensor,
    encoding: dict[str, torch.Tensor],
    topk: int,
) -> dict[str, Any]:
    rotated_queries = rotate(queries, rotation).to(torch.float32)
    scalar_logits = torch.einsum(
        "bhqd,bhtd->bhqt",
        rotated_queries,
        encoding["reconstructed_rotated"].to(torch.float32),
    )
    query_projected = qjl_project_query(queries, sketch)
    residual_logits = qjl_residual_logits(
        query_projected,
        encoding["residual_signs"],
        encoding["residual_norms"],
    )
    full_logits = scalar_logits + residual_logits
    true_residual_logits = dense - scalar_logits

    scalar_metrics = quality_metrics(scalar_logits, dense, topk=int(topk))
    residual_metrics = quality_metrics(residual_logits, true_residual_logits, topk=None)
    full_metrics = quality_metrics(full_logits, dense, topk=int(topk))

    scalar_rmse = float(scalar_metrics["rmse"])
    full_rmse = float(full_metrics["rmse"])

    return {
        "component_quality": {
            "scalar_only_vs_dense": scalar_metrics,
            "qjl_residual_vs_true_residual": residual_metrics,
            "full_turboquant_vs_dense": full_metrics,
        },
        "component_energy": {
            "true_residual_rmse": float(torch.sqrt(torch.mean(true_residual_logits.to(torch.float32).square())).item()),
            "qjl_residual_rmse": float(torch.sqrt(torch.mean(residual_logits.to(torch.float32).square())).item()),
        },
        "residual_correction_effect": {
            "scalar_only_rmse_vs_dense": scalar_rmse,
            "full_turboquant_rmse_vs_dense": full_rmse,
            "rmse_reduction_absolute": scalar_rmse - full_rmse,
            "rmse_reduction_ratio": (
                (scalar_rmse - full_rmse) / scalar_rmse if scalar_rmse > 0 else 0.0
            ),
        },
        "encoding_shapes": {
            "codes": list(encoding["codes"].shape),
            "residual_signs": list(encoding["residual_signs"].shape),
            "residual_norms": list(encoding["residual_norms"].shape),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare shared [K] vs head-specific [H,K] TurboQuant scalar codebooks on real Q/K."
    )
    p.add_argument("--model_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--num_query_tokens", type=int, default=1)
    p.add_argument("--scalar_bits", type=int, default=4)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--quality_topk", type=int, default=32)
    p.add_argument("--lloyd_iters", type=int, default=20)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument("--quant_chunk_rows", type=int, default=4096)
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
        "benchmark": "turboquant_real_qk_head_specific_codebook_ablation",
        "config": {
            "model_id": str(args.model_id),
            "revision": args.revision,
            "device": str(device),
            "torch_dtype": str(args.torch_dtype),
            "layers": [int(x) for x in args.layers],
            "seq_len": int(args.seq_len),
            "num_query_tokens": int(args.num_query_tokens),
            "scalar_bits": int(args.scalar_bits),
            "qjl_dim": int(args.qjl_dim),
            "quality_topk": int(args.quality_topk),
            "lloyd_iters": int(args.lloyd_iters),
            "max_codebook_samples": int(args.max_codebook_samples),
            "quant_chunk_rows": int(args.quant_chunk_rows),
            "rope_applied": not bool(args.no_rope),
        },
        "results": [],
    }

    print("========== TurboQuant shared vs head-specific scalar codebook ablation ==========")
    print(f"model_id        = {args.model_id}")
    print(f"seq_len         = {args.seq_len}")
    print(f"layers          = {args.layers}")
    print(f"scalar_bits     = {args.scalar_bits}")
    print(f"qjl_dim         = {args.qjl_dim}")

    levels = 1 << int(args.scalar_bits)

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
        sketch = make_gaussian_sketch(
            D,
            int(args.qjl_dim),
            seed=int(args.sketch_seed) + int(args.qjl_dim),
            device=device,
        )
        dense = dense_fp32_logits(qk.queries, qk.keys)
        rotated_keys = rotate(qk.keys, rotation).to(torch.float32)

        shared_samples = rotated_keys.reshape(-1)
        shared_centroids = fit_lloyd_scalar_codebook(
            shared_samples,
            num_levels=int(levels),
            max_iters=int(args.lloyd_iters),
            max_samples=int(args.max_codebook_samples),
            seed=int(args.codebook_seed),
        )
        head_centroids = fit_head_specific_codebooks(
            rotated_keys,
            num_levels=int(levels),
            lloyd_iters=int(args.lloyd_iters),
            max_codebook_samples=int(args.max_codebook_samples),
            seed=int(args.codebook_seed),
        )

        shared_encoding = build_shared_encoding(
            qk.keys,
            rotation=rotation,
            centroids=shared_centroids,
            sketch=sketch,
        )
        head_encoding = build_head_encoding(
            qk.keys,
            rotation=rotation,
            head_centroids=head_centroids,
            sketch=sketch,
            quant_chunk_rows=int(args.quant_chunk_rows),
        )

        shared_eval = evaluate_encoding(
            queries=qk.queries,
            dense=dense,
            rotation=rotation,
            sketch=sketch,
            encoding=shared_encoding,
            topk=int(args.quality_topk),
        )
        head_eval = evaluate_encoding(
            queries=qk.queries,
            dense=dense,
            rotation=rotation,
            sketch=sketch,
            encoding=head_encoding,
            topk=int(args.quality_topk),
        )

        shared_full_rmse = float(shared_eval["component_quality"]["full_turboquant_vs_dense"]["rmse"])
        head_full_rmse = float(head_eval["component_quality"]["full_turboquant_vs_dense"]["rmse"])
        shared_full_kl = float(shared_eval["component_quality"]["full_turboquant_vs_dense"]["softmax_kl_target_to_approx"])
        head_full_kl = float(head_eval["component_quality"]["full_turboquant_vs_dense"]["softmax_kl_target_to_approx"])

        item = {
            "layer_idx": int(layer_idx),
            "seq_len": int(args.seq_len),
            "scalar_bits": int(args.scalar_bits),
            "qjl_dim": int(args.qjl_dim),
            "queries_stats": q_stats,
            "keys_stats": k_stats,
            "shared_codebook": {
                "centroids": [float(x) for x in shared_centroids.detach().cpu()],
                **shared_eval,
            },
            "head_specific_codebook": {
                "centroids_shape": list(head_centroids.shape),
                "centroids_per_head": [
                    [float(x) for x in row] for row in head_centroids.detach().cpu()
                ],
                **head_eval,
            },
            "head_specific_vs_shared": {
                "full_rmse_shared": shared_full_rmse,
                "full_rmse_head_specific": head_full_rmse,
                "full_rmse_reduction_absolute": shared_full_rmse - head_full_rmse,
                "full_rmse_reduction_ratio": (
                    (shared_full_rmse - head_full_rmse) / shared_full_rmse
                    if shared_full_rmse > 0 else 0.0
                ),
                "full_kl_shared": shared_full_kl,
                "full_kl_head_specific": head_full_kl,
                "full_kl_reduction_absolute": shared_full_kl - head_full_kl,
            },
        }
        print(json.dumps(item, indent=2))
        payload["results"].append(item)

        del (
            qk,
            rotation,
            sketch,
            dense,
            rotated_keys,
            shared_samples,
            shared_centroids,
            head_centroids,
            shared_encoding,
            head_encoding,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
