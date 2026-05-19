#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from turboquant.factor_lut import build_scalar_factor_lut_fp32
from turboquant.turboquant_logits_baseline_cuda import (
    dense_fp32_qkt_b1q1_d128_cuda,
)
from turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)
from turboquant.turboquant_factor_lut_combined_reduction_full_cuda import (
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
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
        raise RuntimeError("benchmark function produced no output.")
    return float(start.elapsed_time(end) / int(iters)), out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full synthetic timing sweep for the three current logits paths: "
            "dense FP32 qK^T, non-factor combined, and factor-LUT combined."
        )
    )
    p.add_argument("--seq_start", type=int, default=1024)
    p.add_argument("--seq_stop", type=int, default=131072)
    p.add_argument("--seq_step", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--build_warmup", type=int, default=20)
    p.add_argument("--build_iters", type=int, default=200)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--tsv_out",
        default=None,
        help="Optional TSV summary path. Defaults to <out stem>.tsv.",
    )
    return p.parse_args()


def make_seq_lens(start: int, stop: int, step: int) -> list[int]:
    if start <= 0 or stop <= 0 or step <= 0:
        raise ValueError("seq_start, seq_stop, and seq_step must be positive.")
    if start > stop:
        raise ValueError("seq_start must be <= seq_stop.")
    seq_lens = list(range(int(start), int(stop) + 1, int(step)))
    if not seq_lens or seq_lens[-1] != int(stop):
        raise ValueError(
            "seq_stop must be exactly reachable by seq_start + k*seq_step. "
            f"Got start={start}, stop={stop}, step={step}."
        )
    return seq_lens


def write_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_len",
        "dense_fp32_qkt_cuda_ms",
        "nonfactor_combined_cuda_ms",
        "factor_lut_build_ms",
        "factor_lut_combined_kernel_ms",
        "factor_lut_combined_effective_ms",
        "nonfactor_speedup_vs_dense",
        "factor_lut_kernel_speedup_vs_dense",
        "factor_lut_effective_speedup_vs_dense",
        "fastest_method_effective",
        "fastest_time_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in results:
            t = r["timing_ms"]
            s = r["speedup_vs_dense"]
            best = r["fastest_method_effective"]
            writer.writerow({
                "seq_len": r["seq_len"],
                "dense_fp32_qkt_cuda_ms": t["dense_fp32_qkt_cuda_ms"],
                "nonfactor_combined_cuda_ms": t["nonfactor_combined_cuda_ms"],
                "factor_lut_build_ms": t["factor_lut_build_ms"],
                "factor_lut_combined_kernel_ms": t["factor_lut_combined_kernel_ms"],
                "factor_lut_combined_effective_ms": t["factor_lut_combined_effective_build_plus_kernel_ms"],
                "nonfactor_speedup_vs_dense": s["nonfactor_combined_over_dense"],
                "factor_lut_kernel_speedup_vs_dense": s["factor_lut_combined_kernel_only_over_dense"],
                "factor_lut_effective_speedup_vs_dense": s["factor_lut_combined_effective_over_dense"],
                "fastest_method_effective": best["name"],
                "fastest_time_ms": best["ms"],
            })


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    seq_lens = make_seq_lens(args.seq_start, args.seq_stop, args.seq_step)
    H = int(args.num_heads)
    D = 128
    M = 128

    print("========== True TurboQuant three-method full sweep ==========")
    print(f"device             = {device}")
    print(f"H,D,M              = {H},{D},{M}")
    print(f"seq_start/stop/step= {args.seq_start}/{args.seq_stop}/{args.seq_step}")
    print(f"num_points         = {len(seq_lens)}")
    print(f"kernel warmup/iter = {args.warmup}/{args.iters}")
    print(f"build warmup/iter  = {args.build_warmup}/{args.build_iters}")
    print("[Methods]")
    print("  dense_fp32_qkt_cuda")
    print("  nonfactor_combined_cuda")
    print("  factor_lut_combined_cuda")
    print("    factor-LUT reported as both kernel-only and effective build+kernel")
    print("[Synthetic compressed input]")
    print("  Packed scalar codes are generated directly as random uint8 lane-word payloads.")
    print("  Packed QJL signs are generated directly as random uint8 lane-nibble payloads.")
    print("  This avoids giant unpacked temporary tensors and keeps the 128K sweep memory-safe.")
    print("-" * 150)
    print(
        f"{'T':>7} | {'dense ms':>10} | {'nonfactor ms':>13} | {'factor kernel':>13} | "
        f"{'LUT build':>9} | {'factor effective':>16} | {'NF x':>7} | {'Factor eff x':>12} | {'best':>18}"
    )
    print("-" * 150)

    results: list[dict[str, Any]] = []

    for idx, T in enumerate(seq_lens, start=1):
        T = int(T)

        dense_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        dense_keys = torch.randn(1, H, T, D, device=device, dtype=torch.float32)

        rotated_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        qjl_projected_queries = torch.randn(1, H, 1, M, device=device, dtype=torch.float32)
        residual_norms = torch.rand(1, H, T, device=device, dtype=torch.float32)
        centroids = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)

        # Directly create already-packed payloads:
        # - lane_word scalar codes: [1,H,T,64] uint8; each lane's 2 bytes encode 4 4-bit scalar codes.
        # - lane_nibble QJL signs: [1,H,T,16] uint8; each byte stores two 4-bit lane nibbles.
        lane_word_scalar_codes = torch.randint(
            0, 256, (1, H, T, 64), device=device, dtype=torch.uint8
        )
        lane_nibble_signs = torch.randint(
            0, 256, (1, H, T, 16), device=device, dtype=torch.uint8
        )

        factor_lut_build_ms, scalar_factor_lut = bench_ms(
            lambda: build_scalar_factor_lut_fp32(rotated_queries, centroids),
            warmup=int(args.build_warmup),
            iters=int(args.build_iters),
        )

        dense_ms, _ = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(
                dense_queries,
                dense_keys,
            ),
            warmup=int(args.warmup),
            iters=int(args.iters),
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
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        factor_ms, factor_out = bench_ms(
            lambda: turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        factor_effective_ms = float(factor_lut_build_ms + factor_ms)
        parity_diff = (factor_out - nonfactor_out).abs()

        effective_candidates = {
            "dense_fp32_qkt_cuda": float(dense_ms),
            "nonfactor_combined_cuda": float(nonfactor_ms),
            "factor_lut_combined_effective": float(factor_effective_ms),
        }
        best_name, best_ms = min(effective_candidates.items(), key=lambda kv: kv[1])

        result = {
            "seq_len": T,
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "nonfactor_combined_cuda_ms": float(nonfactor_ms),
                "factor_lut_build_ms": float(factor_lut_build_ms),
                "factor_lut_combined_kernel_ms": float(factor_ms),
                "factor_lut_combined_effective_build_plus_kernel_ms": float(factor_effective_ms),
            },
            "speedup_vs_dense": {
                "nonfactor_combined_over_dense": float(dense_ms / nonfactor_ms),
                "factor_lut_combined_kernel_only_over_dense": float(dense_ms / factor_ms),
                "factor_lut_combined_effective_over_dense": float(dense_ms / factor_effective_ms),
            },
            "parity_factor_lut_vs_nonfactor": {
                "max_abs_diff": float(parity_diff.max().item()),
                "mean_abs_diff": float(parity_diff.mean().item()),
                "rmse": float(torch.sqrt((parity_diff * parity_diff).mean()).item()),
            },
            "fastest_method_effective": {
                "name": best_name,
                "ms": float(best_ms),
            },
        }
        results.append(result)

        print(
            f"{T:7d} | {dense_ms:10.4f} | {nonfactor_ms:13.4f} | {factor_ms:13.4f} | "
            f"{factor_lut_build_ms:9.4f} | {factor_effective_ms:16.4f} | "
            f"{dense_ms/nonfactor_ms:7.3f} | {dense_ms/factor_effective_ms:12.3f} | {best_name:>18}"
        )

        del dense_queries, dense_keys
        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del lane_word_scalar_codes, lane_nibble_signs, scalar_factor_lut
        del nonfactor_out, factor_out, parity_diff
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tsv_out = Path(args.tsv_out) if args.tsv_out else out.with_suffix(".tsv")
    write_tsv(tsv_out, results)

    payload = {
        "benchmark": "true_turboquant_three_method_full_sweep",
        "method": (
            "synthetic_1k_step_sweep_dense_fp32_vs_nonfactor_combined_vs_"
            "factor_lut_combined_effective"
        ),
        "config": vars(args),
        "seq_lens": seq_lens,
        "results": results,
        "tsv_summary": str(tsv_out),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 150)
    print(f"[Save JSON] {out}")
    print(f"[Save TSV ] {tsv_out}")


if __name__ == "__main__":
    main()
