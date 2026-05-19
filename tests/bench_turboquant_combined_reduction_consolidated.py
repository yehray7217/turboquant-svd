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
)
from turboquant.turboquant_best_full_cuda import (
    turboquant_best_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_full_cuda import (
    turboquant_factor_lut_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_combined_reduction_full_cuda import (
    build_turboquant_scalar_factor_lut_fp32,
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.scalar_lane_layout import pack_scalar_codes_lane_word_4bit
from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble


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


@torch.no_grad()
def bench_lut_build_ms(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    return bench_ms(
        lambda: build_turboquant_scalar_factor_lut_fp32(
            rotated_queries,
            centroids,
        ),
        warmup=warmup,
        iters=iters,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Consolidated true TurboQuant logits benchmark after combined-reduction optimization."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--build_warmup", type=int, default=20)
    p.add_argument("--build_iters", type=int, default=200)
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

    print("========== Consolidated true TurboQuant combined-reduction benchmark ==========")
    print(f"device             = {device}")
    print(f"H,D,M              = {H},{D},{M}")
    print(f"seq_lens           = {list(args.seq_lens)}")
    print(f"kernel warmup/iter = {args.warmup}/{args.iters}")
    print(f"build warmup/iter  = {args.build_warmup}/{args.build_iters}")
    print("[Compare]")
    print("  dense FP32 qK^T")
    print("  non-factor-LUT best full")
    print("  factor-LUT full with separate scalar/QJL reductions")
    print("  factor-LUT full with combined scalar/QJL reduction")
    print("[Effective]")
    print("  factor-LUT variants report build+kernel time in addition to kernel-only time.")

    results = []

    for T in args.seq_lens:
        T = int(T)
        print("=" * 78)
        print(f"[Benchmark] T={T}")
        print("=" * 78)

        dense_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        dense_keys = torch.randn(1, H, T, D, device=device, dtype=torch.float32)

        rotated_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        qjl_projected_queries = torch.randn(1, H, 1, M, device=device, dtype=torch.float32)
        residual_norms = torch.rand(1, H, T, device=device, dtype=torch.float32)
        centroids = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)

        scalar_codes = torch.randint(
            0, 16, (1, H, T, D), device=device, dtype=torch.uint8
        )
        lane_word_scalar_codes = pack_scalar_codes_lane_word_4bit(scalar_codes)

        sign_values = torch.randint(
            0, 2, (1, H, T, M), device=device, dtype=torch.int8
        )
        sign_values = torch.where(
            sign_values > 0,
            torch.ones((), device=device, dtype=torch.int8),
            -torch.ones((), device=device, dtype=torch.int8),
        ).contiguous()
        lane_nibble_signs = pack_qjl_signs_lane_nibble(sign_values)

        factor_lut_build_ms, scalar_factor_lut = bench_lut_build_ms(
            rotated_queries,
            centroids,
            warmup=args.build_warmup,
            iters=args.build_iters,
        )

        dense_ms, _ = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(
                dense_queries,
                dense_keys,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        non_factor_ms, non_factor_out = bench_ms(
            lambda: turboquant_best_4bit_qjl128_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        factor_ms, factor_out = bench_ms(
            lambda: turboquant_factor_lut_4bit_qjl128_logits_b1q1_d128_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        combined_ms, combined_out = bench_ms(
            lambda: turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        factor_effective_ms = factor_lut_build_ms + factor_ms
        combined_effective_ms = factor_lut_build_ms + combined_ms

        factor_vs_non_factor = (factor_out - non_factor_out).abs()
        combined_vs_factor = (combined_out - factor_out).abs()
        combined_vs_non_factor = (combined_out - non_factor_out).abs()

        result = {
            "seq_len": T,
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "non_factor_lut_best_full_cuda_ms": float(non_factor_ms),
                "factor_lut_build_ms": float(factor_lut_build_ms),
                "factor_lut_separate_reduction_full_kernel_ms": float(factor_ms),
                "factor_lut_combined_reduction_full_kernel_ms": float(combined_ms),
                "factor_lut_separate_reduction_effective_build_plus_kernel_ms": float(factor_effective_ms),
                "factor_lut_combined_reduction_effective_build_plus_kernel_ms": float(combined_effective_ms),
            },
            "speedup_vs_dense": {
                "non_factor_lut_best_full_over_dense": float(dense_ms / non_factor_ms),
                "factor_lut_separate_reduction_kernel_only_over_dense": float(dense_ms / factor_ms),
                "factor_lut_combined_reduction_kernel_only_over_dense": float(dense_ms / combined_ms),
                "factor_lut_separate_reduction_effective_over_dense": float(dense_ms / factor_effective_ms),
                "factor_lut_combined_reduction_effective_over_dense": float(dense_ms / combined_effective_ms),
            },
            "speedup_vs_prior_turboquant": {
                "factor_lut_separate_kernel_over_non_factor_lut_best": float(non_factor_ms / factor_ms),
                "factor_lut_combined_kernel_over_non_factor_lut_best": float(non_factor_ms / combined_ms),
                "factor_lut_combined_kernel_over_factor_lut_separate_kernel": float(factor_ms / combined_ms),
                "factor_lut_combined_effective_over_non_factor_lut_best": float(non_factor_ms / combined_effective_ms),
                "factor_lut_combined_effective_over_factor_lut_separate_effective": float(factor_effective_ms / combined_effective_ms),
            },
            "parity": {
                "factor_lut_separate_vs_non_factor_lut_best": {
                    "max_abs_diff": float(factor_vs_non_factor.max().item()),
                    "mean_abs_diff": float(factor_vs_non_factor.mean().item()),
                },
                "factor_lut_combined_vs_factor_lut_separate": {
                    "max_abs_diff": float(combined_vs_factor.max().item()),
                    "mean_abs_diff": float(combined_vs_factor.mean().item()),
                },
                "factor_lut_combined_vs_non_factor_lut_best": {
                    "max_abs_diff": float(combined_vs_non_factor.max().item()),
                    "mean_abs_diff": float(combined_vs_non_factor.mean().item()),
                },
            },
            "factor_lut": {
                "shape": [1, H, D, 16],
                "storage_bytes_per_head": int(D * 16 * 4),
                "storage_bytes_all_heads": int(H * D * 16 * 4),
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del dense_queries, dense_keys
        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, lane_word_scalar_codes
        del sign_values, lane_nibble_signs
        del scalar_factor_lut
        del non_factor_out, factor_out, combined_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_combined_reduction_consolidated",
        "method": (
            "dense_vs_non_factor_best_vs_factor_lut_separate_"
            "vs_factor_lut_combined_reduction"
        ),
        "config": vars(args),
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
