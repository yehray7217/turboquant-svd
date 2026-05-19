#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from turboquant import (
    make_random_orthogonal_rotation,
    fit_lloyd_scalar_codebook,
    make_gaussian_sketch,
    encode_turboquant_prod_keys,
    turboquant_prod_reference_logits,
    dense_fp32_logits,
)
from turboquant.qjl import qjl_project_query
from turboquant.packing import (
    pack_scalar_codes_4bit,
    pack_qjl_signs_1bit,
)
from turboquant.turboquant_logits_baseline_cuda import (
    dense_fp32_qkt_b1q1_d128_cuda,
    turboquant_4bit_qjl128_logits_b1q1_d128_cuda,
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


@torch.no_grad()
def parity_smoke(
    *,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    B, H, Q, T, D, M = 1, 2, 1, 64, 128, 128
    queries = torch.randn(B, H, Q, D, device=device, dtype=torch.float32)
    keys = torch.randn(B, H, T, D, device=device, dtype=torch.float32)

    rotation = make_random_orthogonal_rotation(D, seed=int(seed) + 11, device=device)
    sketch = make_gaussian_sketch(D, M, seed=int(seed) + 22, device=device)

    calib = torch.randn(4096, D, device=device, dtype=torch.float32)
    calib_rot = torch.matmul(calib, rotation.T)
    centroids = fit_lloyd_scalar_codebook(
        calib_rot,
        num_levels=16,
        max_iters=8,
        max_samples=200_000,
        seed=int(seed) + 33,
    )

    encoding = encode_turboquant_prod_keys(
        keys,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )
    ref_dense = dense_fp32_logits(queries, keys)
    ref_tq = turboquant_prod_reference_logits(
        queries,
        encoding,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    rotated_queries = torch.matmul(queries, rotation.T).contiguous()
    qjl_projected_queries = qjl_project_query(queries, sketch).contiguous()
    packed_codes = pack_scalar_codes_4bit(encoding.codes)
    packed_signs = pack_qjl_signs_1bit(encoding.residual_signs)

    cuda_dense = dense_fp32_qkt_b1q1_d128_cuda(queries, keys)
    cuda_tq = turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
        rotated_queries=rotated_queries,
        packed_scalar_codes=packed_codes,
        qjl_projected_queries=qjl_projected_queries,
        packed_qjl_signs=packed_signs,
        residual_norms=encoding.residual_norms,
        centroids=centroids,
    )

    dense_diff = (cuda_dense - ref_dense).abs()
    tq_diff = (cuda_tq - ref_tq).abs()
    return {
        "dense_cuda_vs_pytorch_max_abs_diff": float(dense_diff.max().item()),
        "dense_cuda_vs_pytorch_mean_abs_diff": float(dense_diff.mean().item()),
        "turboquant_cuda_vs_reference_max_abs_diff": float(tq_diff.max().item()),
        "turboquant_cuda_vs_reference_mean_abs_diff": float(tq_diff.mean().item()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Unoptimized true TurboQuant CUDA logits baseline speed benchmark: "
            "dense FP32 qK^T vs 4-bit packed scalar-code lookup + packed QJL128 signs."
        )
    )
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_parity", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    H = int(args.num_heads)
    D = 128
    M = 128

    payload: dict = {
        "benchmark": "true_turboquant_cuda_logits_baseline_speed",
        "method": "dense_fp32_cuda_vs_4bit_packed_scalar_qjl128_cuda",
        "config": vars(args),
    }

    if not args.skip_parity:
        print("========== Parity smoke ==========")
        smoke = parity_smoke(device=device, seed=int(args.seed))
        print(json.dumps(smoke, indent=2))
        payload["parity_smoke"] = smoke

    print("========== Speed benchmark ==========")
    print(f"device       = {device}")
    print(f"H,D,M        = {H},{D},{M}")
    print(f"seq_lens     = {list(args.seq_lens)}")
    print("[Note] Timing uses synthetic random tensors to measure kernel throughput.")
    print("[Layout] TurboQuant candidate: 4-bit scalar packed codes + 1-bit packed QJL signs + FP32 residual norm.")

    results = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + 12345)

    for T in args.seq_lens:
        T = int(T)
        print("=" * 78)
        print(f"[Benchmark] T={T}")
        print("=" * 78)

        queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        keys = torch.randn(1, H, T, D, device=device, dtype=torch.float32)

        rotated_queries = torch.randn(1, H, 1, D, device=device, dtype=torch.float32)
        qjl_projected_queries = torch.randn(1, H, 1, M, device=device, dtype=torch.float32)
        packed_codes = torch.randint(
            0,
            256,
            (1, H, T, D // 2),
            device=device,
            dtype=torch.uint8,
        )
        packed_signs = torch.randint(
            0,
            256,
            (1, H, T, M // 8),
            device=device,
            dtype=torch.uint8,
        )
        residual_norms = torch.rand(1, H, T, device=device, dtype=torch.float32)
        centroids = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)

        dense_ms, _ = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(queries, keys),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        turbo_ms, _ = bench_ms(
            lambda: turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
                rotated_queries=rotated_queries,
                packed_scalar_codes=packed_codes,
                qjl_projected_queries=qjl_projected_queries,
                packed_qjl_signs=packed_signs,
                residual_norms=residual_norms,
                centroids=centroids,
            ),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        dense_key_bytes_per_token_head = D * 4
        turbo_bytes_per_token_head = (D // 2) + (M // 8) + 4
        result = {
            "seq_len": T,
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "turboquant_4bit_qjl128_cuda_ms": float(turbo_ms),
            },
            "speedup": {
                "dense_over_turboquant": float(dense_ms / turbo_ms),
            },
            "storage_model_bytes_per_token_head": {
                "dense_key_fp32_bytes": int(dense_key_bytes_per_token_head),
                "turboquant_scalar_codes_4bit_bytes": int(D // 2),
                "turboquant_qjl_signs_1bit_bytes": int(M // 8),
                "turboquant_residual_norm_fp32_bytes": 4,
                "turboquant_total_bytes": int(turbo_bytes_per_token_head),
                "dense_key_over_turboquant_total_bytes": float(
                    dense_key_bytes_per_token_head / turbo_bytes_per_token_head
                ),
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del queries, keys, rotated_queries, qjl_projected_queries
        del packed_codes, packed_signs, residual_norms, centroids
        torch.cuda.empty_cache()

    payload["results"] = results
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
