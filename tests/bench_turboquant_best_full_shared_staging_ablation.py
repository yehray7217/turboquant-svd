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

from turboquant.turboquant_best_full_cuda import (
    turboquant_best_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_best_full_shared_staging_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_cuda,
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_cuda,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Best full TurboQuant shared staging ablation: current best vs "
            "shared queries vs shared queries+centroids."
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

    print("========== Best full TurboQuant shared staging ablation ==========")
    print(f"device       = {device}")
    print(f"H,D,M        = {H},{D},{M}")
    print(f"seq_lens     = {list(args.seq_lens)}")
    print("[Variants]")
    print("  current_best")
    print("  shared_queries")
    print("  shared_queries_centroids")
    print("[Shared bytes/block]")
    print("  shared_queries           = 1024 bytes")
    print("  shared_queries_centroids = 1088 bytes")

    results = []

    for T in args.seq_lens:
        T = int(T)
        print("=" * 78)
        print(f"[Benchmark] T={T}")
        print("=" * 78)

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

        current_ms, current_out = bench_ms(
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

        shared_q_ms, shared_q_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_cuda(
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

        shared_qc_ms, shared_qc_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_cuda(
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

        shared_q_diff = (shared_q_out - current_out).abs()
        shared_qc_diff = (shared_qc_out - current_out).abs()

        result = {
            "seq_len": T,
            "timing_ms": {
                "current_best_ms": float(current_ms),
                "shared_queries_ms": float(shared_q_ms),
                "shared_queries_centroids_ms": float(shared_qc_ms),
            },
            "speedup_vs_current_best": {
                "shared_queries_over_current_best": float(current_ms / shared_q_ms),
                "shared_queries_centroids_over_current_best": float(current_ms / shared_qc_ms),
            },
            "parity_vs_current_best": {
                "shared_queries": {
                    "max_abs_diff": float(shared_q_diff.max().item()),
                    "mean_abs_diff": float(shared_q_diff.mean().item()),
                },
                "shared_queries_centroids": {
                    "max_abs_diff": float(shared_qc_diff.max().item()),
                    "mean_abs_diff": float(shared_qc_diff.mean().item()),
                },
            },
            "shared_bytes_per_block": {
                "shared_queries": 1024,
                "shared_queries_centroids": 1088,
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, lane_word_scalar_codes, sign_values, lane_nibble_signs
        del current_out, shared_q_out, shared_qc_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_best_full_shared_staging_ablation",
        "method": "current_best_vs_shared_queries_vs_shared_queries_centroids",
        "config": vars(args),
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
