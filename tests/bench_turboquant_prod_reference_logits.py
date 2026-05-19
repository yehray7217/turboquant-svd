#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from turboquant import (
    make_random_orthogonal_rotation,
    fit_lloyd_scalar_codebook,
    make_gaussian_sketch,
    encode_turboquant_prod_keys,
    turboquant_prod_reference_logits,
    dense_fp32_logits,
)


def ensure_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return dev


def maybe_unit_normalize(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def quality_metrics(
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

    dense_prob = torch.softmax(dense, dim=-1)
    mass = torch.gather(dense_prob, dim=-1, index=approx_idx).sum(dim=-1).mean()

    return {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        f"top{k}_overlap_vs_dense": float(overlap.item()),
        "top1_agreement_vs_dense": float(top1.item()),
        f"dense_softmax_mass_on_approx_top{k}": float(mass.item()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "True TurboQuant_prod PyTorch reference benchmark: "
            "dense FP32 qK^T vs rotated scalar quantization + QJL residual logits."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 4096])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--num_queries", type=int, default=1)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--scalar_bits", type=int, default=3)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--n_calib_vectors", type=int, default=8192)
    p.add_argument("--lloyd_iters", type=int, default=20)
    p.add_argument("--quality_topk", type=int, default=32)
    p.add_argument("--unit_normalize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    B = int(args.batch_size)
    H = int(args.num_heads)
    Q = int(args.num_queries)
    D = int(args.head_dim)
    M = int(args.qjl_dim)
    levels = 1 << int(args.scalar_bits)

    rotation = make_random_orthogonal_rotation(
        D,
        seed=int(args.seed) + 101,
        device=device,
    )
    sketch = make_gaussian_sketch(
        D,
        M,
        seed=int(args.seed) + 202,
        device=device,
    )

    calib = maybe_unit_normalize(
        torch.randn(int(args.n_calib_vectors), D, device=device, dtype=torch.float32),
        bool(args.unit_normalize),
    )
    calib_rotated = torch.matmul(calib, rotation.T)
    centroids = fit_lloyd_scalar_codebook(
        calib_rotated,
        num_levels=levels,
        max_iters=int(args.lloyd_iters),
        seed=int(args.seed) + 303,
    )

    print("========== True TurboQuant_prod PyTorch reference benchmark ==========")
    print(f"device          = {device}")
    print(f"B,H,Q,D         = {B},{H},{Q},{D}")
    print(f"scalar_bits     = {args.scalar_bits}")
    print(f"qjl_dim         = {M}")
    print(f"unit_normalize  = {bool(args.unit_normalize)}")
    print(f"seq_lens        = {list(args.seq_lens)}")
    print("[Note] This is an algorithm/reference benchmark, not a CUDA speed benchmark.")

    results = []
    for T in args.seq_lens:
        print("=" * 78)
        print(f"[Reference] T={int(T)}")
        print("=" * 78)

        queries = maybe_unit_normalize(
            torch.randn(B, H, Q, D, device=device, dtype=torch.float32),
            bool(args.unit_normalize),
        )
        keys = maybe_unit_normalize(
            torch.randn(B, H, int(T), D, device=device, dtype=torch.float32),
            bool(args.unit_normalize),
        )

        encoding = encode_turboquant_prod_keys(
            keys,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )
        dense = dense_fp32_logits(queries, keys)
        approx = turboquant_prod_reference_logits(
            queries,
            encoding,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        metrics = quality_metrics(
            approx,
            dense,
            topk=int(args.quality_topk),
        )
        result = {
            "seq_len": int(T),
            "quality_vs_dense_fp32_qkt": metrics,
            "encoding_shapes": {
                "codes": list(encoding.codes.shape),
                "residual_signs": list(encoding.residual_signs.shape),
                "residual_norms": list(encoding.residual_norms.shape),
            },
            "theoretical_storage_components": {
                "scalar_code_bits_per_coordinate": int(args.scalar_bits),
                "qjl_sign_bits_per_key": int(M),
                "residual_norm_scalar_per_key": 1,
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

    payload = {
        "benchmark": "true_turboquant_prod_reference_logits",
        "method": (
            "random_orthogonal_rotation_then_scalar_quantization_then_"
            "qjl_residual_inner_product_estimator"
        ),
        "config": vars(args),
        "centroids": [float(x) for x in centroids.detach().cpu()],
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
