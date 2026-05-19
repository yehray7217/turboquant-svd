#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

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


def percentile_nearest(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of empty list.")
    vals = sorted(float(v) for v in values)
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    idx = round((len(vals) - 1) * q)
    return vals[int(idx)]


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    vals = [float(x) for x in samples]
    if not vals:
        raise ValueError("samples must be non-empty.")
    return {
        "samples_ms": vals,
        "min_ms": float(min(vals)),
        "p25_ms": float(percentile_nearest(vals, 0.25)),
        "median_ms": float(statistics.median(vals)),
        "p75_ms": float(percentile_nearest(vals, 0.75)),
        "max_ms": float(max(vals)),
        "mean_ms": float(statistics.fmean(vals)),
    }


@torch.no_grad()
def bench_one_ms(
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


@torch.no_grad()
def bench_repeat_ms(
    fn: Callable[[], torch.Tensor],
    *,
    repeats: int,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    samples: list[float] = []
    last_out = None
    for _ in range(int(repeats)):
        ms, out = bench_one_ms(fn, warmup=warmup, iters=iters)
        samples.append(float(ms))
        last_out = out
    if last_out is None:
        raise RuntimeError("Repeated benchmark produced no output.")
    return summarize_samples(samples), last_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ultra-long robust median sweep for dense FP32, non-factor combined, "
            "and factor-LUT combined TurboQuant logits."
        )
    )
    p.add_argument("--seq_start", type=int, default=131072)
    p.add_argument("--seq_stop", type=int, default=262144)
    p.add_argument("--seq_step", type=int, default=4096)

    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)

    p.add_argument("--build_repeats", type=int, default=15)
    p.add_argument("--build_warmup", type=int, default=20)
    p.add_argument("--build_iters", type=int, default=200)
    p.add_argument("--build_probe_heads", type=int, default=32)

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


def gib(n_bytes: int | float) -> float:
    return float(n_bytes) / (1024.0 ** 3)


def estimate_tensor_bytes(*, H: int, T: int, D: int = 128, M: int = 128) -> dict[str, float]:
    dense_keys = 1 * H * T * D * 4
    dense_query = 1 * H * 1 * D * 4
    packed_scalar = 1 * H * T * 64
    packed_signs = 1 * H * T * 16
    residual_norms = 1 * H * T * 4
    qjl_query = 1 * H * 1 * M * 4
    rotated_query = 1 * H * 1 * D * 4
    factor_lut = 1 * H * D * 16 * 4
    outputs = 3 * 1 * H * 1 * T * 4
    approximate_total = (
        dense_keys
        + dense_query
        + packed_scalar
        + packed_signs
        + residual_norms
        + qjl_query
        + rotated_query
        + factor_lut
        + outputs
    )
    return {
        "dense_keys_gib": gib(dense_keys),
        "packed_scalar_gib": gib(packed_scalar),
        "packed_signs_gib": gib(packed_signs),
        "residual_norms_gib": gib(residual_norms),
        "approx_live_tensor_total_gib": gib(approximate_total),
    }


def write_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_len",
        "dense_median_ms",
        "dense_p25_ms",
        "dense_p75_ms",
        "nonfactor_median_ms",
        "nonfactor_p25_ms",
        "nonfactor_p75_ms",
        "factor_kernel_median_ms",
        "factor_kernel_p25_ms",
        "factor_kernel_p75_ms",
        "global_factor_lut_build_median_ms",
        "factor_effective_median_ms",
        "dense_over_nonfactor_median",
        "dense_over_factor_kernel_median",
        "dense_over_factor_effective_median",
        "effective_winner",
        "effective_winner_median_ms",
        "factor_effective_minus_nonfactor_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in results:
            d = r["timing_summary_ms"]
            s = r["speedup_vs_dense_median"]
            w = r["fastest_method_effective_median"]
            writer.writerow({
                "seq_len": r["seq_len"],
                "dense_median_ms": d["dense_fp32_qkt_cuda"]["median_ms"],
                "dense_p25_ms": d["dense_fp32_qkt_cuda"]["p25_ms"],
                "dense_p75_ms": d["dense_fp32_qkt_cuda"]["p75_ms"],
                "nonfactor_median_ms": d["nonfactor_combined_cuda"]["median_ms"],
                "nonfactor_p25_ms": d["nonfactor_combined_cuda"]["p25_ms"],
                "nonfactor_p75_ms": d["nonfactor_combined_cuda"]["p75_ms"],
                "factor_kernel_median_ms": d["factor_lut_combined_kernel_cuda"]["median_ms"],
                "factor_kernel_p25_ms": d["factor_lut_combined_kernel_cuda"]["p25_ms"],
                "factor_kernel_p75_ms": d["factor_lut_combined_kernel_cuda"]["p75_ms"],
                "global_factor_lut_build_median_ms": r["global_factor_lut_build_median_ms"],
                "factor_effective_median_ms": d["factor_lut_combined_effective_global_build_plus_kernel"]["median_ms"],
                "dense_over_nonfactor_median": s["nonfactor_combined_over_dense"],
                "dense_over_factor_kernel_median": s["factor_lut_combined_kernel_only_over_dense"],
                "dense_over_factor_effective_median": s["factor_lut_combined_effective_over_dense"],
                "effective_winner": w["name"],
                "effective_winner_median_ms": w["median_ms"],
                "factor_effective_minus_nonfactor_ms": r["factor_effective_minus_nonfactor_median_ms"],
            })


def infer_ultralong_factor_policy(
    results: list[dict[str, Any]],
    *,
    hysteresis_ratio: float = 0.01,
) -> dict[str, Any]:
    """
    On the ultra-long tail, infer:
      - first factor-effective win over nonfactor by > hysteresis
      - first stable suffix where factor keeps winning over all later T
      - factor win rate across swept points
    """
    first_factor_win_t = None
    stable_suffix_t = None
    wins = 0

    def factor_is_clear_win(r: dict[str, Any]) -> bool:
        d = r["timing_summary_ms"]
        nf = float(d["nonfactor_combined_cuda"]["median_ms"])
        factor_eff = float(
            d["factor_lut_combined_effective_global_build_plus_kernel"]["median_ms"]
        )
        return factor_eff < nf * (1.0 - hysteresis_ratio)

    for r in results:
        if factor_is_clear_win(r):
            wins += 1
            if first_factor_win_t is None:
                first_factor_win_t = int(r["seq_len"])

    for i, r in enumerate(results):
        if all(factor_is_clear_win(x) for x in results[i:]):
            stable_suffix_t = int(r["seq_len"])
            break

    return {
        "hysteresis_ratio": float(hysteresis_ratio),
        "first_factor_effective_clear_win_t": first_factor_win_t,
        "factor_effective_stable_suffix_t": stable_suffix_t,
        "factor_clear_win_points": int(wins),
        "num_points": int(len(results)),
        "factor_clear_win_rate": float(wins / len(results)) if results else 0.0,
    }


@torch.no_grad()
def measure_global_factor_lut_build(
    *,
    device: torch.device,
    heads: int,
    repeats: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + 91001)
    torch.cuda.manual_seed_all(int(seed) + 91001)

    rotated_queries = torch.randn(1, int(heads), 1, 128, device=device, dtype=torch.float32)
    centroids = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)

    summary, out = bench_repeat_ms(
        lambda: build_scalar_factor_lut_fp32(rotated_queries, centroids),
        repeats=int(repeats),
        warmup=int(warmup),
        iters=int(iters),
    )
    summary["lut_shape"] = list(out.shape)

    del rotated_queries, centroids, out
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if int(args.repeats) < 3 or int(args.repeats) % 2 == 0:
        raise ValueError("--repeats should be an odd integer >= 3 for a robust median.")
    if int(args.build_repeats) < 3 or int(args.build_repeats) % 2 == 0:
        raise ValueError("--build_repeats should be an odd integer >= 3 for a robust median.")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    seq_lens = make_seq_lens(args.seq_start, args.seq_stop, args.seq_step)
    H = int(args.num_heads)
    D = 128
    M = 128

    global_build_summary = measure_global_factor_lut_build(
        device=device,
        heads=int(args.build_probe_heads),
        repeats=int(args.build_repeats),
        warmup=int(args.build_warmup),
        iters=int(args.build_iters),
        seed=int(args.seed),
    )
    global_build_median_ms = float(global_build_summary["median_ms"])
    max_mem_estimate = estimate_tensor_bytes(H=H, T=max(seq_lens), D=D, M=M)

    print("========== True TurboQuant three-method ultra-long sweep, robust median ==========")
    print(f"device                  = {device}")
    print(f"H,D,M                   = {H},{D},{M}")
    print(f"seq_start/stop/step     = {args.seq_start}/{args.seq_stop}/{args.seq_step}")
    print(f"num_points              = {len(seq_lens)}")
    print(f"kernel repeats          = {args.repeats}")
    print(f"kernel warmup/iter      = {args.warmup}/{args.iters}")
    print(f"build repeats           = {args.build_repeats}")
    print(f"build warmup/iter       = {args.build_warmup}/{args.build_iters}")
    print(f"global LUT build median = {global_build_median_ms:.6f} ms")
    print("[Max-T rough live tensor estimate, excluding allocator/workspace overhead]")
    print(json.dumps(max_mem_estimate, indent=2))
    print("[Methods]")
    print("  dense FP32 qK^T CUDA")
    print("  nonfactor combined CUDA")
    print("  factor-LUT combined CUDA")
    print("    effective factor time = global LUT build median + factor kernel median")
    print("-" * 196)
    print(
        f"{'T':>7} | {'dense med':>10} {'[p25,p75]':>21} | "
        f"{'NF med':>10} {'[p25,p75]':>21} | "
        f"{'Factor ker med':>14} {'[p25,p75]':>21} | "
        f"{'Factor eff med':>14} | {'NF x':>7} | {'Fact eff x':>10} | {'winner':>28}"
    )
    print("-" * 196)

    results: list[dict[str, Any]] = []

    for T in seq_lens:
        T = int(T)

        local_seed = int(args.seed) + T
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed_all(local_seed)

        dense_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        dense_keys = torch.randn(1, H, T, D, device=device, dtype=torch.float32)

        rotated_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        qjl_projected_queries = torch.randn(1, H, 1, M, device=device, dtype=torch.float32)
        residual_norms = torch.rand(1, H, T, device=device, dtype=torch.float32)
        centroids = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)

        lane_word_scalar_codes = torch.randint(
            0, 256, (1, H, T, 64), device=device, dtype=torch.uint8
        )
        lane_nibble_signs = torch.randint(
            0, 256, (1, H, T, 16), device=device, dtype=torch.uint8
        )
        scalar_factor_lut = build_scalar_factor_lut_fp32(rotated_queries, centroids)

        dense_summary, _ = bench_repeat_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(
                dense_queries,
                dense_keys,
            ),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        nonfactor_summary, nonfactor_out = bench_repeat_ms(
            lambda: turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        factor_summary, factor_out = bench_repeat_ms(
            lambda: turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
                scalar_factor_lut=scalar_factor_lut,
                lane_word_scalar_codes=lane_word_scalar_codes,
                qjl_projected_queries=qjl_projected_queries,
                lane_nibble_qjl_signs=lane_nibble_signs,
                residual_norms=residual_norms,
            ),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        dense_med = float(dense_summary["median_ms"])
        nonfactor_med = float(nonfactor_summary["median_ms"])
        factor_kernel_med = float(factor_summary["median_ms"])
        factor_effective_med = global_build_median_ms + factor_kernel_med

        factor_effective_summary = {
            "median_ms": float(factor_effective_med),
            "p25_ms": float(global_build_median_ms + factor_summary["p25_ms"]),
            "p75_ms": float(global_build_median_ms + factor_summary["p75_ms"]),
            "global_build_median_ms": float(global_build_median_ms),
            "factor_kernel_median_ms": float(factor_kernel_med),
        }

        parity_diff = (factor_out - nonfactor_out).abs()

        candidates = {
            "dense_fp32_qkt_cuda": dense_med,
            "nonfactor_combined_cuda": nonfactor_med,
            "factor_lut_combined_effective_global_build_plus_kernel": factor_effective_med,
        }
        winner_name, winner_ms = min(candidates.items(), key=lambda kv: kv[1])

        result = {
            "seq_len": T,
            "global_factor_lut_build_median_ms": float(global_build_median_ms),
            "timing_summary_ms": {
                "dense_fp32_qkt_cuda": dense_summary,
                "nonfactor_combined_cuda": nonfactor_summary,
                "factor_lut_combined_kernel_cuda": factor_summary,
                "factor_lut_combined_effective_global_build_plus_kernel": factor_effective_summary,
            },
            "speedup_vs_dense_median": {
                "nonfactor_combined_over_dense": float(dense_med / nonfactor_med),
                "factor_lut_combined_kernel_only_over_dense": float(dense_med / factor_kernel_med),
                "factor_lut_combined_effective_over_dense": float(dense_med / factor_effective_med),
            },
            "factor_effective_minus_nonfactor_median_ms": float(
                factor_effective_med - nonfactor_med
            ),
            "parity_factor_lut_vs_nonfactor": {
                "max_abs_diff": float(parity_diff.max().item()),
                "mean_abs_diff": float(parity_diff.mean().item()),
                "rmse": float(torch.sqrt((parity_diff * parity_diff).mean()).item()),
            },
            "fastest_method_effective_median": {
                "name": winner_name,
                "median_ms": float(winner_ms),
            },
        }
        results.append(result)

        print(
            f"{T:7d} | "
            f"{dense_med:10.4f} [{dense_summary['p25_ms']:.4f},{dense_summary['p75_ms']:.4f}] | "
            f"{nonfactor_med:10.4f} [{nonfactor_summary['p25_ms']:.4f},{nonfactor_summary['p75_ms']:.4f}] | "
            f"{factor_kernel_med:14.4f} [{factor_summary['p25_ms']:.4f},{factor_summary['p75_ms']:.4f}] | "
            f"{factor_effective_med:14.4f} | "
            f"{dense_med/nonfactor_med:7.3f} | "
            f"{dense_med/factor_effective_med:10.3f} | "
            f"{winner_name:>28}"
        )

        del dense_queries, dense_keys
        del rotated_queries, qjl_projected_queries, residual_norms, centroids
        del lane_word_scalar_codes, lane_nibble_signs, scalar_factor_lut
        del nonfactor_out, factor_out, parity_diff
        torch.cuda.empty_cache()

    policy_inference = infer_ultralong_factor_policy(results, hysteresis_ratio=0.01)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tsv_out = Path(args.tsv_out) if args.tsv_out else out.with_suffix(".tsv")
    write_tsv(tsv_out, results)

    payload = {
        "benchmark": "true_turboquant_three_method_ultralong_sweep_median",
        "method": (
            "synthetic_ultralong_robust_median_sweep_dense_vs_nonfactor_vs_factor_lut_"
            "with_global_median_factor_lut_build"
        ),
        "config": vars(args),
        "seq_lens": seq_lens,
        "max_t_tensor_estimate_gib": max_mem_estimate,
        "global_factor_lut_build_summary_ms": global_build_summary,
        "policy_inference": policy_inference,
        "results": results,
        "tsv_summary": str(tsv_out),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 196)
    print("[Policy inference]")
    print(json.dumps(policy_inference, indent=2))
    print(f"[Save JSON] {out}")
    print(f"[Save TSV ] {tsv_out}")


if __name__ == "__main__":
    main()
