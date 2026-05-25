#!/usr/bin/env python3
"""
bench_turboquant_paper_attention_logits_speedup.py

Reconstructed benchmark driver for the TurboQuant-SVD project.

Purpose
-------
Paper-style attention-logits microbenchmark:
  dense FP32 qK^T baseline
  vs a TurboQuant-style packed-K attention-logits path

This file intentionally preserves the CLI shape used in the project notes:
  python tests/bench_turboquant_paper_attention_logits_speedup.py \
    --seq_lens 16384 32768 65536 131072 \
    --warmup 20 \
    --iters 100 \
    --out runs/svd_uniform_08/eval/bench_turboquant_decode_b1q1_warp8_meta64.json

Backend policy
--------------
1. If a project-local optimized backend is available, set:
     TURBOQUANT_LOGITS_BACKEND="module:function"
   The callable must accept:
     fn(q, packed_k, meta) -> logits
   where:
     q        : [B,H,Q,D] float32 tensor
     packed_k : uint8 packed 2-bit codes, shape [B,H,S,ceil(D/4)]
     meta     : dict containing "scale", "zero", "num_levels", "group_size"

2. Without TURBOQUANT_LOGITS_BACKEND, the script uses a reference implementation
   that unpacks/dequantizes K and performs qK^T in PyTorch. This checks metric
   plumbing and JSON/reporting, but it is NOT a CUDA-kernel speedup measurement.

The JSON output explicitly records backend_kind and benchmark_valid_for_speedup.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Q", type=int, default=1)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--M", type=int, default=128, help="Retained for compatibility/reporting.")
    ap.add_argument("--num_levels", type=int, default=4, help="4 levels = logical 2-bit codes.")
    ap.add_argument("--group_size", type=int, default=64, help="Metadata group size along sequence dimension.")
    ap.add_argument("--atol", type=float, default=2.5)
    ap.add_argument("--rtol", type=float, default=0.20)
    ap.add_argument("--skip_correctness", action="store_true")
    return ap.parse_args()


def require_cuda(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise SystemExit("This benchmark is intended for CUDA. Pass --device cuda:0.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    torch.cuda.set_device(dev)
    return dev


def sync() -> None:
    torch.cuda.synchronize()


def median_ms(samples_ms: List[float]) -> float:
    return float(statistics.median(samples_ms))


def p90_ms(samples_ms: List[float]) -> float:
    if not samples_ms:
        return float("nan")
    xs = sorted(samples_ms)
    idx = min(len(xs) - 1, math.ceil(0.90 * len(xs)) - 1)
    return float(xs[idx])


def time_cuda_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> Tuple[float, float, float]:
    for _ in range(max(0, warmup)):
        out = fn()
    sync()

    samples = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    sync()
    return median_ms(samples), p90_ms(samples), float(statistics.mean(samples))


def dense_logits(q: torch.Tensor, k_fp32: torch.Tensor) -> torch.Tensor:
    # q: [B,H,Q,D], K: [B,H,S,D] -> [B,H,Q,S]
    return torch.matmul(q, k_fp32.transpose(-1, -2))


def quantize_groupwise_symmetric_2bit(k: torch.Tensor, num_levels: int, group_size: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reference packer:
      - K shape [B,H,S,D], float32
      - group metadata across sequence groups of `group_size`
      - 4 levels encoded as uint8 {0,1,2,3}
      - pack 4 2-bit codes per byte along D
    """
    if num_levels != 4:
        raise ValueError("This reconstructed packer currently supports num_levels=4 only.")
    B, H, S, D = k.shape
    if D % 4 != 0:
        raise ValueError(f"D must be divisible by 4 for 2-bit packing, got D={D}")

    ng = (S + group_size - 1) // group_size
    pad_s = ng * group_size - S
    if pad_s:
        pad = torch.zeros((B, H, pad_s, D), device=k.device, dtype=k.dtype)
        k_pad = torch.cat([k, pad], dim=2)
    else:
        k_pad = k

    kg = k_pad.view(B, H, ng, group_size, D)
    k_min = kg.amin(dim=(3, 4), keepdim=True)
    k_max = kg.amax(dim=(3, 4), keepdim=True)
    denom = torch.clamp(k_max - k_min, min=1e-8)
    scale = denom / float(num_levels - 1)
    zero = k_min

    codes = torch.round((kg - zero) / scale).clamp_(0, num_levels - 1).to(torch.uint8)
    codes = codes.view(B, H, ng * group_size, D)[:, :, :S, :].contiguous()

    # Pack 4 2-bit codes into one byte.
    c0 = codes[..., 0::4]
    c1 = codes[..., 1::4]
    c2 = codes[..., 2::4]
    c3 = codes[..., 3::4]
    packed = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).contiguous()

    meta = {
        "scale": scale.squeeze(3).squeeze(3).contiguous(),  # [B,H,ng]
        "zero": zero.squeeze(3).squeeze(3).contiguous(),    # [B,H,ng]
        "num_levels": torch.tensor(num_levels, device=k.device, dtype=torch.int32),
        "group_size": torch.tensor(group_size, device=k.device, dtype=torch.int32),
        "seq_len": torch.tensor(S, device=k.device, dtype=torch.int32),
        "dim": torch.tensor(D, device=k.device, dtype=torch.int32),
    }
    return packed, meta


def unpack_reference(packed: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    B, H, S, packed_d = packed.shape
    D = int(meta["dim"].item())
    group_size = int(meta["group_size"].item())
    ng = (S + group_size - 1) // group_size

    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03

    codes = torch.empty((B, H, S, D), device=packed.device, dtype=torch.float32)
    codes[..., 0::4] = c0.float()
    codes[..., 1::4] = c1.float()
    codes[..., 2::4] = c2.float()
    codes[..., 3::4] = c3.float()

    group_ids = torch.arange(S, device=packed.device, dtype=torch.long) // group_size
    scale = meta["scale"].index_select(2, group_ids).unsqueeze(-1)  # [B,H,S,1]
    zero = meta["zero"].index_select(2, group_ids).unsqueeze(-1)
    return codes * scale + zero


def reference_turboquant_logits(q: torch.Tensor, packed_k: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    k_deq = unpack_reference(packed_k, meta)
    return dense_logits(q, k_deq)


def load_env_backend() -> Tuple[Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor], str, str]:
    entry = os.environ.get("TURBOQUANT_LOGITS_BACKEND", "").strip()
    if not entry:
        return reference_turboquant_logits, "reference_unpack_dequant_matmul", "reference"
    if ":" not in entry:
        raise SystemExit(
            "TURBOQUANT_LOGITS_BACKEND must be 'module:function', "
            f"got: {entry!r}"
        )
    mod_name, fn_name = entry.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise SystemExit(f"Backend entry is not callable: {entry}")
    return fn, entry, "optimized_env_backend"


def correctness_summary(ref: torch.Tensor, test: torch.Tensor, atol: float, rtol: float) -> Dict[str, object]:
    diff = (ref - test).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ok = bool(torch.allclose(ref, test, atol=atol, rtol=rtol))
    return {
        "allclose": ok,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def bytes_dense_k(B: int, H: int, S: int, D: int) -> int:
    return B * H * S * D * 4  # FP32 K


def bytes_logical_2bit_k(B: int, H: int, S: int, D: int) -> int:
    return B * H * S * D // 4  # 2-bit per element


def bytes_packed_tensor(packed: torch.Tensor) -> int:
    return int(packed.numel() * packed.element_size())


def main() -> None:
    args = parse_args()
    dev = require_cuda(args.device)
    torch.manual_seed(args.seed)

    backend_fn, backend_name, backend_kind = load_env_backend()

    print("========== Paper-style TurboQuant attention-logits benchmark ==========")
    print(f"device       = {dev}")
    print(f"B            = {args.B}")
    print(f"H            = {args.H}")
    print(f"Q            = {args.Q}")
    print(f"D            = {args.D}")
    print(f"M            = {args.M}")
    print(f"num_levels   = {args.num_levels}")
    print(f"group_size   = {args.group_size}")
    print(f"warmup       = {args.warmup}")
    print(f"iters        = {args.iters}")
    print(f"seq_lens     = {args.seq_lens}")
    print(f"backend      = {backend_name}")
    print()
    print("[Note] Dense baseline is FP32 qK^T.")
    print("[Note] Logical compressed-K bit-width is 2-bit when num_levels=4.")
    if backend_kind == "reference":
        print("[Warning] No optimized TurboQuant backend selected; results are plumbing/reference only, not kernel-speedup evidence.")
    print()

    out = {
        "benchmark": "paper_style_turboquant_attention_logits_speedup",
        "device": str(dev),
        "torch_version": torch.__version__,
        "config": {
            "B": args.B, "H": args.H, "Q": args.Q, "D": args.D, "M": args.M,
            "num_levels": args.num_levels, "logical_k_bits": int(round(math.log2(args.num_levels))),
            "group_size": args.group_size, "warmup": args.warmup, "iters": args.iters,
            "seq_lens": list(args.seq_lens), "seed": args.seed,
        },
        "backend_name": backend_name,
        "backend_kind": backend_kind,
        "benchmark_valid_for_speedup": backend_kind != "reference",
        "results": [],
    }

    with torch.inference_mode():
        for S in args.seq_lens:
            q = torch.randn((args.B, args.H, args.Q, args.D), device=dev, dtype=torch.float32)
            k = torch.randn((args.B, args.H, S, args.D), device=dev, dtype=torch.float32)
            packed, meta = quantize_groupwise_symmetric_2bit(k, args.num_levels, args.group_size)

            dense_fn = lambda: dense_logits(q, k)
            tq_fn = lambda: backend_fn(q, packed, meta)

            dense_ms_med, dense_ms_p90, dense_ms_mean = time_cuda_ms(dense_fn, args.warmup, args.iters)
            tq_ms_med, tq_ms_p90, tq_ms_mean = time_cuda_ms(tq_fn, args.warmup, args.iters)

            correctness = None
            if not args.skip_correctness:
                dense_out = dense_fn()
                tq_out = tq_fn()
                sync()
                correctness = correctness_summary(dense_out, tq_out, args.atol, args.rtol)

            speedup = dense_ms_med / tq_ms_med if tq_ms_med > 0 else float("inf")
            row = {
                "seq_len": int(S),
                "dense_fp32_qkt_ms_median": dense_ms_med,
                "dense_fp32_qkt_ms_p90": dense_ms_p90,
                "dense_fp32_qkt_ms_mean": dense_ms_mean,
                "turboquant_logits_ms_median": tq_ms_med,
                "turboquant_logits_ms_p90": tq_ms_p90,
                "turboquant_logits_ms_mean": tq_ms_mean,
                "speedup_dense_over_turboquant_median": speedup,
                "dense_k_bytes_fp32": bytes_dense_k(args.B, args.H, S, args.D),
                "logical_compressed_k_bytes_2bit": bytes_logical_2bit_k(args.B, args.H, S, args.D),
                "packed_tensor_bytes": bytes_packed_tensor(packed),
                "correctness_vs_dense": correctness,
            }
            out["results"].append(row)

            print(
                f"S={S:>6d} | dense={dense_ms_med:>8.4f} ms | "
                f"TQ={tq_ms_med:>8.4f} ms | speedup={speedup:>7.3f}x | "
                f"backend={backend_kind}"
            )
            if correctness is not None:
                print(
                    f"         correctness: allclose={correctness['allclose']} "
                    f"max_abs={correctness['max_abs']:.6f} mean_abs={correctness['mean_abs']:.6f}"
                )

            del q, k, packed, meta
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"[OK] JSON written: {out_path}")


if __name__ == "__main__":
    main()
