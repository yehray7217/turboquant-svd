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

from turboquant.turboquant_component_ablation_cuda import (
    turboquant_qjl_only_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.qjl_sign_layout import (
    pack_qjl_signs_lane_nibble,
)
from turboquant.turboquant_qjl_sign_layout_ablation_cuda import (
    turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda,
    turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda,
    turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda,
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
            "True TurboQuant QJL-only ablation: baseline vs early norm vs "
            "lane-nibble sign layout vs lane-nibble + early norm."
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
    M = 128

    print("========== True TurboQuant QJL sign-layout / early-norm ablation ==========")
    print(f"device       = {device}")
    print(f"H,M          = {H},{M}")
    print(f"seq_lens     = {list(args.seq_lens)}")
    print("[Variants] baseline, early_norm, lane_nibble_signs, lane_nibble_signs+early_norm.")
    print("[Storage] standard signs and lane-nibble signs both use 16 bytes / token / head.")

    results = []
    for T in args.seq_lens:
        T = int(T)
        print("=" * 78)
        print(f"[Benchmark] T={T}")
        print("=" * 78)

        qjl_projected_queries = torch.randn(1, H, 1, M, device=device, dtype=torch.float32)
        standard_signs = torch.randint(
            0, 2, (1, H, T, M), device=device, dtype=torch.int8
        )
        standard_signs = torch.where(
            standard_signs > 0,
            torch.ones((), device=device, dtype=torch.int8),
            -torch.ones((), device=device, dtype=torch.int8),
        ).contiguous()

        # Standard packed bit layout as used by the baseline QJL kernel.
        bit_shifts = torch.arange(8, device=device, dtype=torch.uint8)
        reshaped = (standard_signs > 0).to(torch.uint8).reshape(1, H, T, M // 8, 8)
        packed_standard_signs = torch.sum(
            torch.bitwise_left_shift(reshaped, bit_shifts),
            dim=-1,
            dtype=torch.int64,
        ).to(torch.uint8).contiguous()

        lane_nibble_signs = pack_qjl_signs_lane_nibble(standard_signs)
        residual_norms = torch.rand(1, H, T, device=device, dtype=torch.float32)

        baseline_ms, baseline_out = bench_ms(
            lambda: turboquant_qjl_only_qjl128_logits_b1q1_d128_cuda(
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=packed_standard_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        early_norm_ms, early_norm_out = bench_ms(
            lambda: turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda(
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=packed_standard_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        lane_ms, lane_out = bench_ms(
            lambda: turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda(
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        lane_early_ms, lane_early_out = bench_ms(
            lambda: turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda(
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        early_diff = (early_norm_out - baseline_out).abs()
        lane_diff = (lane_out - baseline_out).abs()
        lane_early_diff = (lane_early_out - baseline_out).abs()

        result = {
            "seq_len": T,
            "timing_ms": {
                "baseline_qjl_only_ms": float(baseline_ms),
                "early_norm_ms": float(early_norm_ms),
                "lane_nibble_signs_ms": float(lane_ms),
                "lane_nibble_signs_early_norm_ms": float(lane_early_ms),
            },
            "speedup_vs_baseline": {
                "early_norm_over_baseline": float(baseline_ms / early_norm_ms),
                "lane_nibble_over_baseline": float(baseline_ms / lane_ms),
                "lane_nibble_early_norm_over_baseline": float(baseline_ms / lane_early_ms),
            },
            "parity_vs_baseline": {
                "early_norm": {
                    "max_abs_diff": float(early_diff.max().item()),
                    "mean_abs_diff": float(early_diff.mean().item()),
                },
                "lane_nibble_signs": {
                    "max_abs_diff": float(lane_diff.max().item()),
                    "mean_abs_diff": float(lane_diff.mean().item()),
                },
                "lane_nibble_signs_early_norm": {
                    "max_abs_diff": float(lane_early_diff.max().item()),
                    "mean_abs_diff": float(lane_early_diff.mean().item()),
                },
            },
            "sign_storage_bytes_per_token_head": {
                "standard_packed": 16,
                "lane_nibble": 16,
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del qjl_projected_queries, standard_signs, packed_standard_signs
        del lane_nibble_signs, residual_norms
        del baseline_out, early_norm_out, lane_out, lane_early_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_qjl_sign_layout_early_norm_ablation",
        "method": "baseline_vs_early_norm_vs_lane_nibble_vs_lane_nibble_early_norm",
        "config": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
