#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant import (
    make_random_orthogonal_rotation,
    fit_lloyd_scalar_codebook,
    make_gaussian_sketch,
    encode_turboquant_prod_keys,
    turboquant_prod_reference_logits,
    dense_fp32_logits,
    rotate,
    qjl_project_query,
)
from turboquant.real_qk_capture import capture_llama_style_qk
from turboquant.scalar_lane_layout import pack_scalar_codes_lane_word_4bit
from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble
from turboquant.turboquant_logits_baseline_cuda import (
    dense_fp32_qkt_b1q1_d128_cuda,
)
from turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)


DEFAULT_TEXT = (
    "TurboQuant real-QK short-context crossover benchmark. This deterministic "
    "prompt is repeated to capture real Llama-style attention activations and "
    "measure when dense FP32 qK^T becomes slower than the non-factor combined "
    "TurboQuant logits kernel."
)


def parse_dtype(name: str) -> torch.dtype | str:
    name = str(name).lower()
    if name == "auto":
        return "auto"
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_fixed_input_ids(
    tokenizer,
    *,
    seq_len: int,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    if not text:
        raise ValueError("text must be non-empty.")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    if ids.numel() == 0:
        raise ValueError("Tokenizer produced no tokens.")

    chunks = []
    total = 0
    while total < int(seq_len):
        chunks.append(ids)
        total += int(ids.numel())
    joined = torch.cat(chunks, dim=0)[: int(seq_len)]
    return joined.view(1, -1).to(device=device, dtype=torch.long)


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
def error_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    diff = x.to(torch.float32) - y.to(torch.float32)
    return {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
    }


@torch.no_grad()
def quality_metrics(
    approx: torch.Tensor,
    dense: torch.Tensor,
    *,
    topk: int,
) -> dict[str, float]:
    approx = approx.to(torch.float32)
    dense = dense.to(torch.float32)
    diff = approx - dense

    k = min(int(topk), int(dense.shape[-1]))
    approx_idx = torch.topk(approx, k=k, dim=-1).indices
    dense_idx = torch.topk(dense, k=k, dim=-1).indices

    approx_mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(
        dim=-1,
        index=approx_idx,
        value=True,
    )
    overlap = torch.gather(approx_mask, dim=-1, index=dense_idx).float().mean()
    top1 = (torch.argmax(approx, dim=-1) == torch.argmax(dense, dim=-1)).float().mean()

    dense_probs = torch.softmax(dense, dim=-1)
    approx_probs = torch.softmax(approx, dim=-1)
    dense_mass_on_approx_topk = torch.gather(
        dense_probs,
        dim=-1,
        index=approx_idx,
    ).sum(dim=-1).mean()
    kl = torch.sum(
        dense_probs
        * (
            torch.log(dense_probs.clamp_min(1e-12))
            - torch.log(approx_probs.clamp_min(1e-12))
        ),
        dim=-1,
    ).mean()

    return {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        f"top{k}_overlap_vs_dense": float(overlap.item()),
        "top1_agreement_vs_dense": float(top1.item()),
        f"dense_softmax_mass_on_approx_top{k}": float(dense_mass_on_approx_topk.item()),
        "softmax_kl_dense_to_approx": float(kl.item()),
    }


def format_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(str(row.get(key, "")) for key, _ in columns)
            + " |"
        )
    return "\n".join([header, sep, *body])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Real-Q/K short-context crossover benchmark: dense FP32 qK^T CUDA "
            "vs non-factor combined TurboQuant CUDA logits."
        )
    )
    p.add_argument("--model_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--num_query_tokens", type=int, default=1)
    p.add_argument("--quality_topk", type=int, default=32)
    p.add_argument("--lloyd_iters", type=int, default=20)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--text_file", default=None)
    p.add_argument("--no_rope", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rotation_seed", type=int, default=101)
    p.add_argument("--sketch_seed", type=int, default=202)
    p.add_argument("--codebook_seed", type=int, default=303)
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if int(args.num_query_tokens) != 1:
        raise ValueError("Current CUDA kernels validated here require --num_query_tokens 1.")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    text = str(args.text)
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")

    model_dtype = parse_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    print("========== True TurboQuant real-Q/K short-context crossover ==========")
    print(f"model_id             = {args.model_id}")
    print(f"layer_idx            = {args.layer_idx}")
    print(f"seq_lens             = {list(args.seq_lens)}")
    print(f"query_tokens         = {args.num_query_tokens}")
    print("[Compare]")
    print("  dense FP32 qK^T CUDA")
    print("  non-factor combined TurboQuant CUDA")
    print("[Fixed true-TurboQuant method]")
    print("  scalar bits = 4")
    print("  QJL dim     = 128")
    print(
        f"fixed seeds          = seed={args.seed}, rotation={args.rotation_seed}, "
        f"sketch={args.sketch_seed}, codebook={args.codebook_seed}"
    )

    results: list[dict[str, Any]] = []

    for seq_len in args.seq_lens:
        T = int(seq_len)
        print("=" * 78)
        print(f"[Real-Q/K crossover] T={T}")
        print("=" * 78)

        input_ids = build_fixed_input_ids(
            tokenizer,
            seq_len=T,
            text=text,
            device=device,
        )
        qk = capture_llama_style_qk(
            model,
            input_ids,
            layer_idx=int(args.layer_idx),
            num_query_tokens=1,
            apply_rope=not bool(args.no_rope),
        )

        D = int(qk.queries.shape[-1])
        if D != 128:
            raise ValueError(f"Expected D=128 for current optimized CUDA kernels, got D={D}.")

        rotation = make_random_orthogonal_rotation(
            D,
            seed=int(args.rotation_seed),
            device=device,
        )
        rotated_key_samples = torch.matmul(
            qk.keys.reshape(-1, D),
            rotation.T,
        ).contiguous()
        centroids = fit_lloyd_scalar_codebook(
            rotated_key_samples,
            num_levels=16,
            max_iters=int(args.lloyd_iters),
            max_samples=int(args.max_codebook_samples),
            seed=int(args.codebook_seed),
        )
        sketch = make_gaussian_sketch(
            D,
            128,
            seed=int(args.sketch_seed),
            device=device,
        )
        encoding = encode_turboquant_prod_keys(
            qk.keys,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        dense_ref = dense_fp32_logits(qk.queries, qk.keys)
        tq_ref = turboquant_prod_reference_logits(
            qk.queries,
            encoding,
            rotation=rotation,
            centroids=centroids,
            sketch=sketch,
        )

        rotated_queries = rotate(qk.queries, rotation).to(torch.float32).contiguous()
        qjl_projected_queries = qjl_project_query(qk.queries, sketch).to(torch.float32).contiguous()
        lane_word_scalar_codes = pack_scalar_codes_lane_word_4bit(encoding.codes.contiguous())
        lane_nibble_signs = pack_qjl_signs_lane_nibble(encoding.residual_signs.contiguous())
        residual_norms = encoding.residual_norms.contiguous().to(torch.float32)

        dense_ms, dense_cuda = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(
                qk.queries,
                qk.keys,
            ),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        tq_ms, tq_cuda = bench_ms(
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

        result = {
            "seq_len": T,
            "selected_path_candidate": "nonfactor_combined",
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_ms),
                "nonfactor_combined_cuda_ms": float(tq_ms),
            },
            "speedup_vs_dense_cuda": {
                "nonfactor_combined_over_dense": float(dense_ms / tq_ms),
            },
            "cuda_parity_vs_reference": {
                "dense_cuda_vs_dense_fp32_reference": error_metrics(dense_cuda, dense_ref),
                "nonfactor_combined_vs_turboquant_reference": error_metrics(tq_cuda, tq_ref),
            },
            "quality_vs_dense_fp32_qkt": {
                "pytorch_turboquant_reference": quality_metrics(
                    tq_ref,
                    dense_ref,
                    topk=int(args.quality_topk),
                ),
                "nonfactor_combined_cuda": quality_metrics(
                    tq_cuda,
                    dense_ref,
                    topk=int(args.quality_topk),
                ),
            },
            "qk_capture": {
                "queries_shape": list(qk.queries.shape),
                "keys_shape": list(qk.keys.shape),
                "rope_applied": bool(qk.rope_applied),
                "rope_detail": str(qk.rope_detail),
                "num_attention_heads": int(qk.num_attention_heads),
                "num_key_value_heads": int(qk.num_key_value_heads),
                "key_heads_expanded": bool(qk.key_heads_expanded),
            },
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del input_ids, qk, rotated_key_samples, centroids, sketch, encoding
        del dense_ref, tq_ref, rotated_queries, qjl_projected_queries
        del lane_word_scalar_codes, lane_nibble_signs, residual_norms
        del dense_cuda, tq_cuda
        torch.cuda.empty_cache()

    summary_rows = []
    first_win_t = None
    for r in results:
        speedup = float(r["speedup_vs_dense_cuda"]["nonfactor_combined_over_dense"])
        if first_win_t is None and speedup > 1.0:
            first_win_t = int(r["seq_len"])
        parity = r["cuda_parity_vs_reference"]["nonfactor_combined_vs_turboquant_reference"]
        quality = r["quality_vs_dense_fp32_qkt"]["nonfactor_combined_cuda"]
        overlap_key = next(k for k in quality if k.startswith("top") and k.endswith("_overlap_vs_dense"))
        summary_rows.append({
            "T": r["seq_len"],
            "dense_ms": f'{r["timing_ms"]["dense_fp32_qkt_cuda_ms"]:.4f}',
            "nonfactor_ms": f'{r["timing_ms"]["nonfactor_combined_cuda_ms"]:.4f}',
            "speedup": f'{speedup:.3f}x',
            "max_diff_ref": f'{parity["max_abs_diff"]:.2e}',
            "top32": f'{quality[overlap_key]:.4f}',
            "top1": f'{quality["top1_agreement_vs_dense"]:.4f}',
            "kl": f'{quality["softmax_kl_dense_to_approx"]:.3e}',
        })

    summary_md = format_markdown_table(
        summary_rows,
        [
            ("T", "T"),
            ("dense_ms", "Dense ms"),
            ("nonfactor_ms", "Non-factor ms"),
            ("speedup", "Speedup"),
            ("max_diff_ref", "Max diff vs TQ ref"),
            ("top32", "Top-32 overlap"),
            ("top1", "Top-1"),
            ("kl", "KL"),
        ],
    )

    print("=" * 78)
    print("[Summary: real-Q/K short-context crossover]")
    print("=" * 78)
    print(summary_md)
    print(f"[Crossover] first T with nonfactor_combined speedup > 1.0: {first_win_t}")

    payload = {
        "benchmark": "true_turboquant_real_qk_short_context_crossover",
        "fixed_method": {
            "scalar_bits": 4,
            "qjl_dim": 128,
        },
        "config": {
            **vars(args),
            "torch_dtype": str(args.torch_dtype),
        },
        "first_t_with_nonfactor_speedup_gt_1": first_win_t,
        "results": results,
        "summary_table_markdown": summary_md,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
