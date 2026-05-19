#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from turboquant.turboquant_logits_baseline_cuda import (
    dense_fp32_qkt_b1q1_d128_cuda,
    turboquant_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_component_ablation_cuda import (
    turboquant_scalar_only_4bit_logits_b1q1_d128_cuda,
    turboquant_qjl_only_qjl128_logits_b1q1_d128_cuda,
    turboquant_full_4bit_qjl128_logits_b1q1_d128_cuda,
)


def sync() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def bench_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(int(warmup)):
        out = fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(iters)):
        out = fn()
    end.record()
    sync()
    if out is None:
        raise RuntimeError("Benchmark function produced no output.")
    return float(start.elapsed_time(end) / int(iters)), out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "True TurboQuant CUDA component ablation: dense vs scalar-only vs "
            "QJL-only vs full scalar+QJL."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    H = int(args.num_heads)
    D = 128
    M = 128

    print("========== True TurboQuant CUDA component ablation ==========")
    print(f"device       = {device}")
    print(f"H,D,M        = {H},{D},{M}")
    print(f"seq_lens     = {list(args.seq_lens)}")
    print("[Layout] 4-bit packed scalar codes + 1-bit packed QJL128 signs + FP32 residual norms.")

    results = []
    for T in args.seq_lens:
        T = int(T)
        print("=" * 78)
        print(f"[Benchmark] T={T}")
        print("=" * 78)

        queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        keys = torch.randn(1, H, T, D, device=device, dtype=torch.float32)
        rotated_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        qjl_projected_queries = torch.randn(1, H, 1, M, device=device, dtype=torch.float32)
        packed_codes = torch.randint(
            0, 256, (1, H, T, D // 2), device=device, dtype=torch.uint8
        )
        packed_signs = torch.randint(
            0, 256, (1, H, T, M // 8), device=device, dtype=torch.uint8
        )
        residual_norms = torch.rand(1, H, T, device=device, dtype=torch.float32)
        centroids = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)

        dense_ms, _ = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(queries, keys),
            warmup=args.warmup,
            iters=args.iters,
        )
        scalar_ms, scalar_out = bench_ms(
            lambda: turboquant_scalar_only_4bit_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                packed_scalar_codes=packed_codes,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        qjl_ms, qjl_out = bench_ms(
            lambda: turboquant_qjl_only_qjl128_logits_b1q1_d128_cuda(
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=packed_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        full_ablation_ms, full_ablation_out = bench_ms(
            lambda: turboquant_full_4bit_qjl128_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                packed_scalar_codes=packed_codes,
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=packed_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        full_baseline_ms, full_baseline_out = bench_ms(
            lambda: turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                packed_scalar_codes=packed_codes,
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=packed_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        full_parity = (full_ablation_out - full_baseline_out).abs()
        algebra_parity = (full_ablation_out - (scalar_out + qjl_out)).abs()

        result = {
            "seq_len": T,
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "scalar_only_4bit_cuda_ms": float(scalar_ms),
                "qjl_only_qjl128_cuda_ms": float(qjl_ms),
                "full_component_ablation_cuda_ms": float(full_ablation_ms),
                "full_previous_baseline_cuda_ms": float(full_baseline_ms),
            },
            "speedup_vs_dense": {
                "scalar_only_over_dense": float(dense_ms / scalar_ms),
                "qjl_only_over_dense": float(dense_ms / qjl_ms),
                "full_component_over_dense": float(dense_ms / full_ablation_ms),
                "full_previous_baseline_over_dense": float(dense_ms / full_baseline_ms),
            },
            "full_kernel_checks": {
                "full_component_vs_previous_baseline_max_abs_diff": float(full_parity.max().item()),
                "full_component_vs_previous_baseline_mean_abs_diff": float(full_parity.mean().item()),
                "full_component_vs_scalar_plus_qjl_max_abs_diff": float(algebra_parity.max().item()),
                "full_component_vs_scalar_plus_qjl_mean_abs_diff": float(algebra_parity.mean().item()),
            },
            "naive_sum_of_component_times_ms": float(scalar_ms + qjl_ms),
            "full_vs_naive_component_sum_ratio": float(full_ablation_ms / (scalar_ms + qjl_ms)),
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del queries, keys, rotated_queries, qjl_projected_queries
        del packed_codes, packed_signs, residual_norms, centroids
        del scalar_out, qjl_out, full_ablation_out, full_baseline_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_cuda_component_ablation",
        "method": "dense_vs_scalar_only_vs_qjl_only_vs_full",
        "config": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
