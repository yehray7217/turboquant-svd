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
from turboquant.factor_lut_layout import build_scalar_factor_lut_warp_code_major_fp32
from turboquant.turboquant_factor_lut_full_cuda import (
    turboquant_factor_lut_4bit_qjl128_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_layout_ablation_cuda import (
    turboquant_full_4bit_lane_word_factor_lut_warp_code_major_lane_nibble_qjl128_logits_b1q1_d128_cuda,
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
            "True TurboQuant factor-LUT layout ablation: current [H,128,16] "
            "vs warp-code-major [H,4,16,32]."
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

    print("========== True TurboQuant factor-LUT layout ablation ==========")
    print(f"device             = {device}")
    print(f"H,D,M              = {H},{D},{M}")
    print(f"seq_lens           = {list(args.seq_lens)}")
    print(f"kernel warmup/iter = {args.warmup}/{args.iters}")
    print(f"build warmup/iter  = {args.build_warmup}/{args.build_iters}")
    print("[Compare]")
    print("  current factor LUT:       [1,H,128,16]")
    print("  warp-code-major factor LUT [1,H,4,16,32]")
    print("[Goal]")
    print("  Test whether [stage,code,lane] physical layout reduces the factor-LUT access cost.")

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

        current_build_ms, current_factor_lut = bench_ms(
            lambda: build_scalar_factor_lut_fp32(rotated_queries, centroids),
            warmup=args.build_warmup,
            iters=args.build_iters,
        )
        warp_build_ms, warp_factor_lut = bench_ms(
            lambda: build_scalar_factor_lut_warp_code_major_fp32(rotated_queries, centroids),
            warmup=args.build_warmup,
            iters=args.build_iters,
        )

        current_ms, current_out = bench_ms(
            lambda: turboquant_factor_lut_4bit_qjl128_logits_b1q1_d128_cuda(
                scalar_factor_lut=current_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        warp_ms, warp_out = bench_ms(
            lambda: turboquant_full_4bit_lane_word_factor_lut_warp_code_major_lane_nibble_qjl128_logits_b1q1_d128_cuda(
                warp_code_major_factor_lut=warp_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        parity = (warp_out - current_out).abs()
        current_effective_ms = current_build_ms + current_ms
        warp_effective_ms = warp_build_ms + warp_ms

        result = {
            "seq_len": T,
            "timing_ms": {
                "current_factor_lut_build_ms": float(current_build_ms),
                "warp_code_major_factor_lut_build_ms": float(warp_build_ms),
                "current_factor_lut_full_kernel_ms": float(current_ms),
                "warp_code_major_factor_lut_full_kernel_ms": float(warp_ms),
                "current_factor_lut_effective_build_plus_kernel_ms": float(current_effective_ms),
                "warp_code_major_factor_lut_effective_build_plus_kernel_ms": float(warp_effective_ms),
            },
            "speedup": {
                "warp_code_major_kernel_over_current_factor_lut_kernel": float(current_ms / warp_ms),
                "warp_code_major_effective_over_current_factor_lut_effective": float(
                    current_effective_ms / warp_effective_ms
                ),
            },
            "parity_warp_code_major_vs_current_factor_lut": {
                "max_abs_diff": float(parity.max().item()),
                "mean_abs_diff": float(parity.mean().item()),
            },
            "factor_lut_shapes": {
                "current": [1, H, D, 16],
                "warp_code_major": [1, H, 4, 16, 32],
            },
            "factor_lut_storage_bytes_all_heads": int(H * D * 16 * 4),
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del scalar_codes, lane_word_scalar_codes
        del sign_values, lane_nibble_signs
        del current_factor_lut, warp_factor_lut, current_out, warp_out
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_factor_lut_layout_ablation",
        "method": "factor_lut_h_coord_code_vs_warp_code_major_h_stage_code_lane",
        "config": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
