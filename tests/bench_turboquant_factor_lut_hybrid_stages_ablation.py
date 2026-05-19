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
from turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_combined_reduction_full_cuda import (
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_hybrid_stages_ablation_cuda import (
    turboquant_full_4bit_lane_word_factor_lut_hybrid1_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
    turboquant_full_4bit_lane_word_factor_lut_hybrid2_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
    turboquant_full_4bit_lane_word_factor_lut_hybrid3_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
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
            "Factor-LUT hybrid stage ablation: "
            "0/1/2/3/4 LUT scalar stages under combined reduction."
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

    print("========== True TurboQuant factor-LUT hybrid stage ablation ==========")
    print(f"device             = {device}")
    print(f"H,D,M              = {H},{D},{M}")
    print(f"seq_lens           = {list(args.seq_lens)}")
    print(f"kernel warmup/iter = {args.warmup}/{args.iters}")
    print(f"build warmup/iter  = {args.build_warmup}/{args.build_iters}")
    print("[Compare]")
    print("  LUT stages = 0: non-factor combined")
    print("  LUT stages = 1: hybrid1")
    print("  LUT stages = 2: hybrid2")
    print("  LUT stages = 3: hybrid3")
    print("  LUT stages = 4: factor-LUT combined")
    print("[Goal]")
    print("  Find whether reducing the number of scatter factor-LUT loads beats both endpoints.")

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

        s0_ms, s0_out = bench_ms(
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

        s1_ms, s1_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_factor_lut_hybrid1_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        s2_ms, s2_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_factor_lut_hybrid2_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        s3_ms, s3_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_factor_lut_hybrid3_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        s4_ms, s4_out = bench_ms(
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

        variants = {
            "lut0_nonfactor_combined": (s0_ms, s0_out),
            "lut1_hybrid": (s1_ms, s1_out),
            "lut2_hybrid": (s2_ms, s2_out),
            "lut3_hybrid": (s3_ms, s3_out),
            "lut4_factor_lut_combined": (s4_ms, s4_out),
        }
        best_name, (best_ms, _) = min(variants.items(), key=lambda kv: kv[1][0])

        diffs = {}
        for name, (_, out_tensor) in variants.items():
            diff = (out_tensor - s0_out).abs()
            diffs[name] = {
                "max_abs_diff_vs_lut0": float(diff.max().item()),
                "mean_abs_diff_vs_lut0": float(diff.mean().item()),
            }

        result = {
            "seq_len": T,
            "factor_lut_build_ms": float(build_ms),
            "timing_ms": {
                name: float(ms)
                for name, (ms, _) in variants.items()
            },
            "effective_build_plus_kernel_ms": {
                "lut0_nonfactor_combined": float(s0_ms),
                "lut1_hybrid": float(build_ms + s1_ms),
                "lut2_hybrid": float(build_ms + s2_ms),
                "lut3_hybrid": float(build_ms + s3_ms),
                "lut4_factor_lut_combined": float(build_ms + s4_ms),
            },
            "speedup_vs_lut0_kernel": {
                name: float(s0_ms / ms)
                for name, (ms, _) in variants.items()
            },
            "speedup_vs_lut4_kernel": {
                name: float(s4_ms / ms)
                for name, (ms, _) in variants.items()
            },
            "best_kernel_variant": {
                "name": best_name,
                "ms": float(best_ms),
            },
            "parity_vs_lut0": diffs,
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, lane_word_scalar_codes
        del sign_values, lane_nibble_signs, scalar_factor_lut
        del s0_out, s1_out, s2_out, s3_out, s4_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_factor_lut_hybrid_stages_ablation",
        "method": "combined_reduction_scalar_lut_stage_count_0_to_4",
        "config": vars(args),
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
