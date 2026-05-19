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
from turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_combined_reduction_full_cuda import (
    build_turboquant_scalar_factor_lut_fp32,
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_dual_path_dispatch import (
    DEFAULT_FACTOR_LUT_THRESHOLD_T,
    turboquant_dual_path_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.scalar_lane_layout import pack_scalar_codes_lane_word_4bit
from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble


def sync() -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def bench_ms(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, object]:
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
def bench_factor_lut_build_ms(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    ms, out = bench_ms(
        lambda: build_turboquant_scalar_factor_lut_fp32(rotated_queries, centroids),
        warmup=warmup,
        iters=iters,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("factor LUT builder returned a non-tensor.")
    return ms, out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dual-path dispatch benchmark for the current best true-TurboQuant logits policy."
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
    p.add_argument("--factor_lut_threshold_t", type=int, default=DEFAULT_FACTOR_LUT_THRESHOLD_T)
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
    threshold = int(args.factor_lut_threshold_t)

    print("========== True TurboQuant dual-path dispatch benchmark ==========")
    print(f"device                 = {device}")
    print(f"H,D,M                  = {H},{D},{M}")
    print(f"seq_lens               = {list(args.seq_lens)}")
    print(f"kernel warmup/iter     = {args.warmup}/{args.iters}")
    print(f"build warmup/iter      = {args.build_warmup}/{args.build_iters}")
    print(f"factor LUT threshold T = {threshold}")
    print("[Policy]")
    print(f"  T <  {threshold}: non-factor combined reduction")
    print(f"  T >= {threshold}: factor-LUT combined reduction")
    print("[Compare]")
    print("  dense FP32 qK^T")
    print("  explicit non-factor combined")
    print("  explicit factor-LUT combined, effective build+kernel")
    print("  dispatched dual-path policy")

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

        factor_lut_build_ms, scalar_factor_lut = bench_factor_lut_build_ms(
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

        nonfactor_ms, nonfactor_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
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

        factor_kernel_ms, factor_out = bench_ms(
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

        factor_effective_ms = factor_lut_build_ms + factor_kernel_ms

        dispatch_ms, dispatch_result = bench_ms(
            lambda: turboquant_dual_path_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
                factor_lut_threshold_t=threshold,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        dispatch_out, dispatch_path = dispatch_result
        if not isinstance(dispatch_out, torch.Tensor):
            raise TypeError("Dispatch output is not a tensor.")

        expected_path = (
            "factor_lut_combined"
            if T >= threshold
            else "nonfactor_combined"
        )
        if dispatch_path != expected_path:
            raise RuntimeError(
                f"Dispatch path mismatch for T={T}: "
                f"got {dispatch_path}, expected {expected_path}."
            )

        chosen_reference = (
            factor_out
            if expected_path == "factor_lut_combined"
            else nonfactor_out
        )
        dispatch_diff = (dispatch_out - chosen_reference).abs()

        policy_reference_ms = (
            factor_effective_ms
            if expected_path == "factor_lut_combined"
            else nonfactor_ms
        )

        result = {
            "seq_len": T,
            "selected_path": dispatch_path,
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "nonfactor_combined_kernel_ms": float(nonfactor_ms),
                "factor_lut_build_ms": float(factor_lut_build_ms),
                "factor_lut_combined_kernel_ms": float(factor_kernel_ms),
                "factor_lut_combined_effective_build_plus_kernel_ms": float(factor_effective_ms),
                "dual_path_dispatch_measured_ms": float(dispatch_ms),
                "policy_reference_ms": float(policy_reference_ms),
            },
            "speedup_vs_dense": {
                "nonfactor_combined_over_dense": float(dense_ms / nonfactor_ms),
                "factor_lut_combined_kernel_only_over_dense": float(dense_ms / factor_kernel_ms),
                "factor_lut_combined_effective_over_dense": float(dense_ms / factor_effective_ms),
                "dual_path_dispatch_measured_over_dense": float(dense_ms / dispatch_ms),
                "dual_path_policy_reference_over_dense": float(dense_ms / policy_reference_ms),
            },
            "dispatch_vs_policy_reference": {
                "latency_ratio_policy_reference_over_dispatch_measured": float(
                    policy_reference_ms / dispatch_ms
                ),
                "output_max_abs_diff": float(dispatch_diff.max().item()),
                "output_mean_abs_diff": float(dispatch_diff.mean().item()),
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del dense_queries, dense_keys
        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, lane_word_scalar_codes
        del sign_values, lane_nibble_signs
        del scalar_factor_lut, nonfactor_out, factor_out, dispatch_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_dual_path_dispatch",
        "method": "threshold_dispatch_nonfactor_combined_vs_factor_lut_combined",
        "config": vars(args),
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
