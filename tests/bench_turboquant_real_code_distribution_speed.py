#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Any

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
    rotate,
    qjl_project_query,
)
from turboquant.real_qk_capture import capture_llama_style_qk
from turboquant.factor_lut import build_scalar_factor_lut_fp32
from turboquant.scalar_lane_layout import pack_scalar_codes_lane_word_4bit
from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble
from turboquant.turboquant_factor_lut_combined_reduction_full_cuda import (
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)


DEFAULT_TEXT = (
    "TurboQuant real-code distribution speed benchmark. This deterministic "
    "prompt is repeated to capture real LLM attention activations and scalar "
    "quantizer codes. It is not a downstream-task quality benchmark. "
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


@torch.no_grad()
def repeat_token_axis(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    Repeat dim=2 token axis until it reaches target_t.
    Supports:
      [1,H,T,D] or [1,H,T]
    """
    if x.ndim not in {3, 4}:
        raise ValueError(f"Expected rank-3 or rank-4 tensor, got rank={x.ndim}.")
    source_t = int(x.shape[2])
    if source_t <= 0:
        raise ValueError("source token axis must be non-empty.")
    repeats = (int(target_t) + source_t - 1) // source_t
    reps = [1] * x.ndim
    reps[2] = repeats
    return x.repeat(*reps)[:, :, : int(target_t)].contiguous()


@torch.no_grad()
def make_histogram_scrambled_source(
    real_codes: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """
    Preserve the exact global 16-code histogram while destroying the original
    head/token/coord arrangement at the captured-source size.

    This is not mathematically perfect iid sampling, but it is a clean
    histogram-matched structural control:
      - same global code counts
      - randomized positions
    """
    if real_codes.dtype != torch.uint8:
        raise ValueError("real_codes must be uint8.")
    flat = real_codes.reshape(-1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm_cpu = torch.randperm(flat.numel(), generator=gen, device="cpu")
    perm = perm_cpu.to(device=flat.device, dtype=torch.long)
    return flat[perm].reshape_as(real_codes).contiguous()


@torch.no_grad()
def code_distribution_stats(codes: torch.Tensor) -> dict[str, Any]:
    if codes.dtype != torch.uint8:
        raise ValueError("codes must be uint8.")
    counts = torch.bincount(codes.reshape(-1).to(torch.int64), minlength=16)[:16]
    counts_cpu = counts.detach().cpu().to(torch.int64)
    total = int(counts_cpu.sum().item())
    if total <= 0:
        raise ValueError("empty code tensor.")
    probs = counts_cpu.to(torch.float64) / float(total)
    nonzero = probs[probs > 0]
    entropy_bits = float((-nonzero * torch.log2(nonzero)).sum().item())
    sorted_probs, _ = torch.sort(probs, descending=True)
    return {
        "num_codes": total,
        "counts": [int(x) for x in counts_cpu.tolist()],
        "probs": [float(x) for x in probs.tolist()],
        "entropy_bits": entropy_bits,
        "top1_mass": float(sorted_probs[:1].sum().item()),
        "top4_mass": float(sorted_probs[:4].sum().item()),
        "top8_mass": float(sorted_probs[:8].sum().item()),
    }


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
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(int(iters)):
        out = fn()
    end.record()
    torch.cuda.synchronize()

    if out is None:
        raise RuntimeError("benchmark function produced no output.")
    return float(start.elapsed_time(end) / int(iters)), out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Speed benchmark for factor-LUT combined TurboQuant under "
            "uniform, histogram-scrambled, and real scalar-code distributions."
        )
    )
    p.add_argument("--model_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--capture_seq_len", type=int, default=4096)
    p.add_argument("--num_query_tokens", type=int, default=1)
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    p.add_argument("--quality_text", default=DEFAULT_TEXT)
    p.add_argument("--text_file", default=None)
    p.add_argument("--no_rope", action="store_true")

    # Current CUDA path is 4-bit scalar + QJL128.
    p.add_argument("--scalar_bits", type=int, default=4)
    p.add_argument("--qjl_dim", type=int, default=128)
    p.add_argument("--lloyd_iters", type=int, default=20)
    p.add_argument("--max_codebook_samples", type=int, default=1_000_000)

    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rotation_seed", type=int, default=101)
    p.add_argument("--sketch_seed", type=int, default=202)
    p.add_argument("--codebook_seed", type=int, default=303)
    p.add_argument("--scramble_seed", type=int, default=404)
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.scalar_bits) != 4:
        raise ValueError("This speed kernel currently expects --scalar_bits 4.")
    if int(args.qjl_dim) != 128:
        raise ValueError("This speed kernel currently expects --qjl_dim 128.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    text = str(args.quality_text)
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

    input_ids = build_fixed_input_ids(
        tokenizer,
        seq_len=int(args.capture_seq_len),
        text=text,
        device=device,
    )
    qk = capture_llama_style_qk(
        model,
        input_ids,
        layer_idx=int(args.layer_idx),
        num_query_tokens=int(args.num_query_tokens),
        apply_rope=not bool(args.no_rope),
    )

    D = int(qk.keys.shape[-1])
    if D != 128:
        raise ValueError(f"This benchmark currently expects head dim D=128, got D={D}.")

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

    rotated_queries = rotate(qk.queries, rotation).to(torch.float32).contiguous()
    qjl_projected_queries = qjl_project_query(qk.queries, sketch).to(torch.float32).contiguous()
    scalar_factor_lut = build_scalar_factor_lut_fp32(rotated_queries, centroids)

    real_codes_src = encoding.codes.contiguous()
    hist_scrambled_codes_src = make_histogram_scrambled_source(
        real_codes_src,
        seed=int(args.scramble_seed),
    )
    residual_signs_src = encoding.residual_signs.contiguous()
    residual_norms_src = encoding.residual_norms.contiguous()

    source_stats = {
        "real_codes_capture": code_distribution_stats(real_codes_src),
        "histogram_scrambled_capture": code_distribution_stats(hist_scrambled_codes_src),
    }

    print("========== True TurboQuant real-code distribution speed benchmark ==========")
    print(f"model_id             = {args.model_id}")
    print(f"layer_idx            = {args.layer_idx}")
    print(f"capture_seq_len      = {args.capture_seq_len}")
    print(f"speed_seq_lens       = {list(args.seq_lens)}")
    print(f"query_tokens         = {args.num_query_tokens}")
    print(f"heads q/kv           = {qk.num_attention_heads}/{qk.num_key_value_heads}")
    print(f"key_heads_expanded   = {qk.key_heads_expanded}")
    print(f"rope_applied         = {qk.rope_applied}")
    print(f"rope_detail          = {qk.rope_detail}")
    print("[Kernel]")
    print("  factor-LUT + QJL lane-nibble + combined reduction")
    print("[Code patterns]")
    print("  uniform_random")
    print("  histogram_scrambled_real_capture")
    print("  tiled_real_capture")
    print("[Note]")
    print("  Real and histogram-scrambled capture codes are tiled along token axis to the target speed T.")
    print("  Kernel timings exclude scalar/QJL encoding, packing, model load, and codebook fitting.")

    results: list[dict[str, Any]] = []

    for T in args.seq_lens:
        T = int(T)
        print("=" * 78)
        print(f"[Benchmark] T={T}")
        print("=" * 78)

        residual_signs = repeat_token_axis(residual_signs_src, T)
        residual_norms = repeat_token_axis(residual_norms_src, T)
        lane_nibble_signs = pack_qjl_signs_lane_nibble(residual_signs)

        real_codes = repeat_token_axis(real_codes_src, T)
        hist_codes = repeat_token_axis(hist_scrambled_codes_src, T)
        uniform_codes = torch.randint(
            0,
            16,
            tuple(real_codes.shape),
            device=device,
            dtype=torch.uint8,
        )

        code_variants = {
            "uniform_random": uniform_codes,
            "histogram_scrambled": hist_codes,
            "tiled_real": real_codes,
        }

        timings: dict[str, float] = {}
        outputs: dict[str, torch.Tensor] = {}
        distribution_stats: dict[str, Any] = {}

        for name, codes in code_variants.items():
            distribution_stats[name] = code_distribution_stats(codes)
            lane_word_codes = pack_scalar_codes_lane_word_4bit(codes)
            ms, out = bench_ms(
                lambda lwc=lane_word_codes: turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
                    scalar_factor_lut=scalar_factor_lut,
                    lane_word_scalar_codes=lwc,
                    qjl_projected_queries=qjl_projected_queries,
                    lane_nibble_qjl_signs=lane_nibble_signs,
                    residual_norms=residual_norms,
                ),
                warmup=int(args.warmup),
                iters=int(args.iters),
            )
            timings[name] = float(ms)
            outputs[name] = out
            del lane_word_codes

        result = {
            "seq_len": T,
            "timing_ms": timings,
            "speed_ratio": {
                "tiled_real_over_uniform_random": float(
                    timings["uniform_random"] / timings["tiled_real"]
                ),
                "histogram_scrambled_over_uniform_random": float(
                    timings["uniform_random"] / timings["histogram_scrambled"]
                ),
                "tiled_real_over_histogram_scrambled": float(
                    timings["histogram_scrambled"] / timings["tiled_real"]
                ),
            },
            "code_distribution": distribution_stats,
            "output_shape": list(outputs["tiled_real"].shape),
        }
        print(json.dumps(result, indent=2))
        results.append(result)

        del residual_signs, residual_norms, lane_nibble_signs
        del real_codes, hist_codes, uniform_codes
        del outputs
        torch.cuda.empty_cache()

    payload = {
        "benchmark": "true_turboquant_real_code_distribution_speed",
        "method": (
            "factor_lut_combined_kernel_speed_under_uniform_vs_"
            "histogram_scrambled_vs_real_tiled_scalar_codes"
        ),
        "config": vars(args),
        "capture": {
            "model_id": args.model_id,
            "layer_idx": int(args.layer_idx),
            "capture_seq_len": int(args.capture_seq_len),
            "query_tokens": int(args.num_query_tokens),
            "heads_q": int(qk.num_attention_heads),
            "heads_kv": int(qk.num_key_value_heads),
            "key_heads_expanded": bool(qk.key_heads_expanded),
            "rope_applied": bool(qk.rope_applied),
            "rope_detail": str(qk.rope_detail),
            "centroids": [float(x) for x in centroids.detach().cpu()],
        },
        "source_code_distribution": source_stats,
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
