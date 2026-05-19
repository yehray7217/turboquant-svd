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
from turboquant.turboquant_full_lane_nibble_early_norm_cuda import (
    turboquant_full_4bit_lane_nibble_qjl128_early_norm_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_best_full_cuda import (
    turboquant_best_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.packing import pack_scalar_codes_4bit
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


def pack_standard_qjl_signs(signs: torch.Tensor) -> torch.Tensor:
    if signs.shape[-1] != 128:
        raise ValueError("Expected QJL signs last dim = 128.")
    shifts = torch.arange(8, device=signs.device, dtype=torch.uint8)
    reshaped = (signs > 0).to(torch.uint8).reshape(*signs.shape[:-1], 16, 8)
    return torch.sum(
        torch.bitwise_left_shift(reshaped, shifts),
        dim=-1,
        dtype=torch.int64,
    ).to(torch.uint8).contiguous()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Consolidated true TurboQuant full logits benchmark: dense FP32 vs "
            "initial full baseline vs QJL lane-nibble full vs current best full."
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

    print("========== Consolidated true TurboQuant full logits benchmark ==========")
    print(f"device       = {device}")
    print(f"H,D,M        = {H},{D},{M}")
    print(f"seq_lens     = {list(args.seq_lens)}")
    print("[Compare]")
    print("  dense FP32 qK^T")
    print("  initial full baseline: standard scalar + standard QJL signs")
    print("  QJL-optimized full: standard scalar + lane-nibble QJL signs")
    print("  current best full: scalar lane-word + lane-nibble QJL signs")
    print("[Storage]")
    print("  all TurboQuant full variants = 64B scalar + 16B signs + 4B residual norm = 84B/token/head")

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
        standard_packed_scalar_codes = pack_scalar_codes_4bit(scalar_codes)
        lane_word_scalar_codes = pack_scalar_codes_lane_word_4bit(scalar_codes)

        sign_values = torch.randint(
            0, 2, (1, H, T, M), device=device, dtype=torch.int8
        )
        sign_values = torch.where(
            sign_values > 0,
            torch.ones((), device=device, dtype=torch.int8),
            -torch.ones((), device=device, dtype=torch.int8),
        ).contiguous()
        standard_packed_signs = pack_standard_qjl_signs(sign_values)
        lane_nibble_signs = pack_qjl_signs_lane_nibble(sign_values)

        dense_ms, _ = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(dense_queries, dense_keys),
            warmup=args.warmup,
            iters=args.iters,
        )

        initial_full_ms, initial_full_out = bench_ms(
            lambda: turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                packed_scalar_codes=standard_packed_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=standard_packed_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        qjl_optimized_full_ms, qjl_optimized_full_out = bench_ms(
            lambda: turboquant_full_4bit_lane_nibble_qjl128_early_norm_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                packed_scalar_codes=standard_packed_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        best_full_ms, best_full_out = bench_ms(
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

        qjl_vs_initial = (qjl_optimized_full_out - initial_full_out).abs()
        best_vs_qjl = (best_full_out - qjl_optimized_full_out).abs()
        best_vs_initial = (best_full_out - initial_full_out).abs()

        result = {
            "seq_len": T,
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "initial_full_turboquant_cuda_ms": float(initial_full_ms),
                "qjl_lane_nibble_full_cuda_ms": float(qjl_optimized_full_ms),
                "best_scalar_lane_word_qjl_lane_nibble_full_cuda_ms": float(best_full_ms),
            },
            "speedup_vs_dense": {
                "initial_full_over_dense": float(dense_ms / initial_full_ms),
                "qjl_lane_nibble_full_over_dense": float(dense_ms / qjl_optimized_full_ms),
                "best_full_over_dense": float(dense_ms / best_full_ms),
            },
            "speedup_vs_prior_turboquant_variants": {
                "qjl_lane_nibble_full_over_initial_full": float(initial_full_ms / qjl_optimized_full_ms),
                "best_full_over_qjl_lane_nibble_full": float(qjl_optimized_full_ms / best_full_ms),
                "best_full_over_initial_full": float(initial_full_ms / best_full_ms),
            },
            "parity": {
                "qjl_lane_nibble_full_vs_initial_full": {
                    "max_abs_diff": float(qjl_vs_initial.max().item()),
                    "mean_abs_diff": float(qjl_vs_initial.mean().item()),
                },
                "best_full_vs_qjl_lane_nibble_full": {
                    "max_abs_diff": float(best_vs_qjl.max().item()),
                    "mean_abs_diff": float(best_vs_qjl.mean().item()),
                },
                "best_full_vs_initial_full": {
                    "max_abs_diff": float(best_vs_initial.max().item()),
                    "mean_abs_diff": float(best_vs_initial.mean().item()),
                },
            },
            "storage_model_bytes_per_token_head": {
                "dense_fp32_k": 512,
                "turboquant_scalar_codes": 64,
                "turboquant_qjl_signs": 16,
                "turboquant_residual_norm_fp32": 4,
                "turboquant_total": 84,
                "dense_key_over_turboquant_total": float(512 / 84),
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del dense_queries, dense_keys
        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, standard_packed_scalar_codes, lane_word_scalar_codes
        del sign_values, standard_packed_signs, lane_nibble_signs
        del initial_full_out, qjl_optimized_full_out, best_full_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_best_full_consolidated",
        "method": (
            "dense_fp32_vs_initial_full_vs_qjl_lane_nibble_full_vs_"
            "scalar_lane_word_qjl_lane_nibble_best_full"
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
