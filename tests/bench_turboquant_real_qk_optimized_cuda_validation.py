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

try:
    from modules.svd_hf_registry import register_svdllama_auto_classes
    register_svdllama_auto_classes()
except Exception:
    pass

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


DEFAULT_TEXT = (
    "TurboQuant optimized CUDA real-QK validation benchmark. This fixed prompt "
    "is repeated to capture real language-model Q/K activations. It is not a "
    "downstream accuracy benchmark; it validates CUDA TurboQuant logits against "
    "the PyTorch true-TurboQuant reference on actual attention activations. "
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

    pieces = []
    total = 0
    while total < int(seq_len):
        pieces.append(ids)
        total += int(ids.numel())
    joined = torch.cat(pieces, dim=0)[: int(seq_len)]
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
        raise RuntimeError("benchmark function produced no output.")
    return float(start.elapsed_time(end) / int(iters)), out


@torch.no_grad()
def error_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    diff = x.to(torch.float32) - y.to(torch.float32)
    return {
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
    }


@torch.no_grad()
def logits_quality_metrics(
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
    candidate_mass = torch.gather(dense_probs, dim=-1, index=approx_idx).sum(dim=-1).mean()
    kl_dense_to_approx = torch.sum(
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
        f"dense_softmax_mass_on_approx_top{k}": float(candidate_mass.item()),
        "softmax_kl_dense_to_approx": float(kl_dense_to_approx.item()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate optimized true-TurboQuant CUDA logits on real Llama-style Q/K activations."
        )
    )
    p.add_argument("--model_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 4096])
    p.add_argument("--num_query_tokens", type=int, default=1)
    p.add_argument("--quality_topk", type=int, default=32)
    p.add_argument("--lloyd_iters", type=int, default=20)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--text_file", default=None)
    p.add_argument("--no_rope", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--build_warmup", type=int, default=20)
    p.add_argument("--build_iters", type=int, default=200)
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
        raise ValueError(
            "The optimized CUDA kernels validated here currently require --num_query_tokens 1."
        )

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

    print("========== True TurboQuant optimized CUDA real-Q/K validation ==========")
    print(f"model_id             = {args.model_id}")
    print(f"layer_idx            = {args.layer_idx}")
    print(f"seq_lens             = {list(args.seq_lens)}")
    print(f"query_tokens         = {args.num_query_tokens}")
    print("[Fixed method]")
    print("  scalar_bits = 4")
    print("  qjl_dim     = 128")
    print("[CUDA variants]")
    print("  dense FP32 qK^T CUDA baseline")
    print("  non-factor combined reduction")
    print("  factor-LUT combined reduction")
    print("[Reference]")
    print("  PyTorch true-TurboQuant prod reference logits")
    print(f"fixed seeds          = seed={args.seed}, rotation={args.rotation_seed}, sketch={args.sketch_seed}, codebook={args.codebook_seed}")

    results: list[dict[str, Any]] = []

    for seq_len in args.seq_lens:
        T = int(seq_len)
        print("=" * 78)
        print(f"[Real-Q/K] T={T}")
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
            raise ValueError(f"This optimized CUDA validation expects head dim D=128, got D={D}.")
        if int(qk.queries.shape[0]) != 1 or int(qk.queries.shape[2]) != 1:
            raise ValueError(
                "Expected queries [1,H,1,128], "
                f"got {tuple(qk.queries.shape)}."
            )

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

        factor_lut_build_ms, scalar_factor_lut = bench_ms(
            lambda: build_scalar_factor_lut_fp32(rotated_queries, centroids),
            warmup=int(args.build_warmup),
            iters=int(args.build_iters),
        )

        dense_cuda_ms, dense_cuda = bench_ms(
            lambda: dense_fp32_qkt_b1q1_d128_cuda(
                qk.queries,
                qk.keys,
            ),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

        nonfactor_ms, nonfactor_cuda = bench_ms(
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

        factor_ms, factor_cuda = bench_ms(
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

        factor_effective_ms = factor_lut_build_ms + factor_ms

        item = {
            "seq_len": T,
            "qk_capture": {
                "queries_shape": list(qk.queries.shape),
                "keys_shape": list(qk.keys.shape),
                "layer_idx": int(qk.layer_idx),
                "rope_applied": bool(qk.rope_applied),
                "rope_detail": str(qk.rope_detail),
                "num_attention_heads": int(qk.num_attention_heads),
                "num_key_value_heads": int(qk.num_key_value_heads),
                "key_heads_expanded": bool(qk.key_heads_expanded),
            },
            "timing_ms": {
                "dense_fp32_qkt_cuda_ms": float(dense_cuda_ms),
                "nonfactor_combined_cuda_ms": float(nonfactor_ms),
                "factor_lut_build_ms": float(factor_lut_build_ms),
                "factor_lut_combined_kernel_ms": float(factor_ms),
                "factor_lut_combined_effective_build_plus_kernel_ms": float(factor_effective_ms),
            },
            "speedup_vs_dense_cuda": {
                "nonfactor_combined_over_dense": float(dense_cuda_ms / nonfactor_ms),
                "factor_lut_combined_kernel_only_over_dense": float(dense_cuda_ms / factor_ms),
                "factor_lut_combined_effective_over_dense": float(dense_cuda_ms / factor_effective_ms),
            },
            "cuda_parity_vs_reference": {
                "dense_cuda_vs_dense_fp32_reference": error_metrics(dense_cuda, dense_ref),
                "nonfactor_combined_vs_turboquant_reference": error_metrics(nonfactor_cuda, tq_ref),
                "factor_lut_combined_vs_turboquant_reference": error_metrics(factor_cuda, tq_ref),
                "factor_lut_combined_vs_nonfactor_combined": error_metrics(factor_cuda, nonfactor_cuda),
            },
            "quality_vs_dense_fp32_qkt": {
                "pytorch_turboquant_reference": logits_quality_metrics(
                    tq_ref,
                    dense_ref,
                    topk=int(args.quality_topk),
                ),
                "nonfactor_combined_cuda": logits_quality_metrics(
                    nonfactor_cuda,
                    dense_ref,
                    topk=int(args.quality_topk),
                ),
                "factor_lut_combined_cuda": logits_quality_metrics(
                    factor_cuda,
                    dense_ref,
                    topk=int(args.quality_topk),
                ),
            },
            "encoding_shapes": {
                "codes": list(encoding.codes.shape),
                "residual_signs": list(encoding.residual_signs.shape),
                "residual_norms": list(encoding.residual_norms.shape),
                "lane_word_scalar_codes": list(lane_word_scalar_codes.shape),
                "lane_nibble_qjl_signs": list(lane_nibble_signs.shape),
                "scalar_factor_lut": list(scalar_factor_lut.shape),
            },
            "codebook_centroids": [float(x) for x in centroids.detach().cpu()],
        }
        print(json.dumps(item, indent=2))
        results.append(item)

        del input_ids, qk, rotated_key_samples, centroids, sketch, encoding
        del dense_ref, tq_ref, rotated_queries, qjl_projected_queries
        del lane_word_scalar_codes, lane_nibble_signs, residual_norms, scalar_factor_lut
        del dense_cuda, nonfactor_cuda, factor_cuda
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_real_qk_optimized_cuda_validation",
        "method": (
            "real_llama_qk_activations_dense_cuda_vs_nonfactor_combined_cuda_"
            "vs_factor_lut_combined_cuda_vs_pytorch_true_turboquant_reference"
        ),
        "config": {
            **vars(args),
            "torch_dtype": str(args.torch_dtype),
            "scalar_bits": 4,
            "qjl_dim": 128,
        },
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
