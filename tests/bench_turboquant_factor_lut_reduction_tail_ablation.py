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

from turboquant.factor_lut import build_scalar_factor_lut_fp32
from turboquant.turboquant_factor_lut_combined_reduction_full_cuda import (
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_reduction_tail_ablation_cuda import (
    turboquant_factor_lut_combined_reduction_tail_down_cuda,
    turboquant_factor_lut_combined_reduction_tail_xor_cuda,
    turboquant_factor_lut_combined_reduction_tail_halfwidth_cuda,
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
            "Factor-LUT combined reduction tail ablation: "
            "current production path vs down/xor/halfwidth reduction variants."
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

    print("========== True TurboQuant factor-LUT reduction-tail ablation ==========")
    print(f"device             = {device}")
    print(f"H,D,M              = {H},{D},{M}")
    print(f"seq_lens           = {list(args.seq_lens)}")
    print(f"kernel warmup/iter = {args.warmup}/{args.iters}")
    print(f"build warmup/iter  = {args.build_warmup}/{args.build_iters}")
    print("[Compare]")
    print("  production_current:")
    print("    current factor-LUT combined reduction mainline")
    print("  tail_down:")
    print("    standalone reproduction using standard SHFL.DOWN tree")
    print("  tail_xor:")
    print("    SHFL.XOR all-reduce tail")
    print("  tail_halfwidth:")
    print("    merge upper 16 lanes, then width-16 SHFL.DOWN tail")
    print("[Note]")
    print("  LUT loads, QJL path, scalar math, and combined reduction formulation are unchanged.")
    print("  tail_down validates that the ablation kernel matches the current production tail.")

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

        build_ms, scalar_factor_lut = bench_ms(
            lambda: build_scalar_factor_lut_fp32(rotated_queries, centroids),
            warmup=args.build_warmup,
            iters=args.build_iters,
        )

        production_ms, production_out = bench_ms(
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

        down_ms, down_out = bench_ms(
            lambda: turboquant_factor_lut_combined_reduction_tail_down_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        xor_ms, xor_out = bench_ms(
            lambda: turboquant_factor_lut_combined_reduction_tail_xor_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        half_ms, half_out = bench_ms(
            lambda: turboquant_factor_lut_combined_reduction_tail_halfwidth_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        variants = {
            "production_current": (production_ms, production_out),
            "tail_down": (down_ms, down_out),
            "tail_xor": (xor_ms, xor_out),
            "tail_halfwidth": (half_ms, half_out),
        }
        best_name, (best_ms, _) = min(variants.items(), key=lambda kv: kv[1][0])

        parity = {}
        for name, (_, out) in variants.items():
            diff = (out - production_out).abs()
            parity[name] = {
                "max_abs_diff_vs_production": float(diff.max().item()),
                "mean_abs_diff_vs_production": float(diff.mean().item()),
            }

        result = {
            "seq_len": T,
            "timing_ms": {
                "factor_lut_build_ms": float(build_ms),
                **{name: float(ms) for name, (ms, _) in variants.items()},
            },
            "effective_build_plus_kernel_ms": {
                name: float(build_ms + ms)
                for name, (ms, _) in variants.items()
            },
            "speedup_vs_production_current": {
                name: float(production_ms / ms)
                for name, (ms, _) in variants.items()
            },
            "best_kernel_variant": {
                "name": best_name,
                "ms": float(best_ms),
            },
            "parity": parity,
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, lane_word_scalar_codes
        del sign_values, lane_nibble_signs
        del scalar_factor_lut
        del production_out, down_out, xor_out, half_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_factor_lut_reduction_tail_ablation",
        "method": "production_vs_down_vs_xor_vs_halfwidth_reduction_tails",
        "config": vars(args),
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
