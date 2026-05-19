#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_eval_svdmodel import (
    apply_svd_ranks_inplace,
    estimate_params_from_ranks,
    load_rank_json,
)
from turboquant import dense_fp32_logits
from turboquant.real_qk_capture import capture_llama_style_qk


DEFAULT_UNIFORM_TARGET_LINEAR_REGEX = (
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

DEFAULT_EVAL_TEXTS = [
    (
        "Large language model compression reduces memory footprint and compute cost, "
        "but compression must preserve stable internal activations across layers. "
        "This sentence is repeated to create a deterministic loss probe."
    ),
    (
        "Attention logits depend on the scale of query and key vectors. "
        "A compressed model can remain functional only when those scales do not drift uncontrollably."
    ),
    (
        "Singular value decomposition replaces a dense linear layer with two lower-rank matrices. "
        "The approximation quality should be checked end to end, not only layer by layer."
    ),
]


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


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def jsonable_dataclass(x: Any) -> Any:
    if hasattr(x, "__dict__"):
        return {
            str(k): jsonable_dataclass(v)
            for k, v in vars(x).items()
        }
    if isinstance(x, dict):
        return {str(k): jsonable_dataclass(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable_dataclass(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _round_rank_down_to_multiple(rank: int, multiple: int, *, min_rank: int) -> int:
    if int(multiple) <= 1:
        return int(rank)
    rounded = (int(rank) // int(multiple)) * int(multiple)
    return max(int(min_rank), int(rounded))


def build_uniform_rank_dict_from_model(
    model: torch.nn.Module,
    *,
    linear_param_ratio: float,
    name_regex: str,
    min_rank: int,
    rank_multiple: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    ratio = float(linear_param_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(
            f"--uniform_linear_param_ratio must be in (0, 1], got {ratio}."
        )
    if int(min_rank) <= 0:
        raise ValueError("--uniform_min_rank must be positive.")
    if int(rank_multiple) <= 0:
        raise ValueError("--uniform_rank_multiple must be positive.")

    cre = re.compile(str(name_regex))
    rank_dict: dict[str, int] = {}
    details: list[dict[str, Any]] = []

    selected_weight_params = 0
    estimated_factorized_params = 0
    selected_layers = 0
    skipped_linear_layers_by_regex = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if cre.match(name) is None:
            skipped_linear_layers_by_regex += 1
            continue

        in_f = int(mod.in_features)
        out_f = int(mod.out_features)
        full_rank = int(min(in_f, out_f))
        weight_params = int(in_f * out_f)

        raw_rank = int((ratio * weight_params) // (in_f + out_f))
        rank = max(int(min_rank), int(raw_rank))
        rank = _round_rank_down_to_multiple(
            rank,
            int(rank_multiple),
            min_rank=int(min_rank),
        )
        rank = min(int(rank), int(full_rank))

        rank_dict[str(name)] = int(rank)
        factorized_params = int(rank * (in_f + out_f))
        selected_weight_params += weight_params
        estimated_factorized_params += factorized_params
        selected_layers += 1

        details.append(
            {
                "name": str(name),
                "in_features": in_f,
                "out_features": out_f,
                "full_rank": full_rank,
                "rank": int(rank),
                "weight_params": weight_params,
                "estimated_factorized_params": factorized_params,
                "linear_weight_param_ratio": (
                    float(factorized_params) / float(weight_params)
                    if weight_params
                    else 0.0
                ),
            }
        )

    if not rank_dict:
        raise RuntimeError(
            "Uniform rank generation selected zero Linear layers. "
            f"Check --uniform_name_regex={name_regex!r}."
        )

    summary = {
        "formula": "rank=floor(ratio*out*in/(out+in)), then optional rank-multiple rounding",
        "requested_linear_param_ratio": ratio,
        "target_name_regex": str(name_regex),
        "min_rank": int(min_rank),
        "rank_multiple": int(rank_multiple),
        "selected_layers": int(selected_layers),
        "selected_weight_params": int(selected_weight_params),
        "estimated_factorized_params": int(estimated_factorized_params),
        "achieved_selected_linear_weight_param_ratio": (
            float(estimated_factorized_params) / float(selected_weight_params)
            if selected_weight_params
            else 0.0
        ),
        "skipped_linear_layers_by_regex": int(skipped_linear_layers_by_regex),
        "rank_preview": details[:20],
        "rank_details": details,
    }
    return rank_dict, summary


def save_rank_dict_json(path: str, rank_dict: dict[str, int], summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "value": "rank_flat",
        "ranks": {str(k): int(v) for k, v in rank_dict.items()},
        "uniform_rank_generation": summary,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] Uniform rank JSON: {out}")


def resolve_svd_rebuild_device(requested: str) -> str:
    requested = str(requested).strip()
    if requested.lower() == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


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


def select_eval_texts(args: argparse.Namespace) -> list[str]:
    if args.eval_text_file:
        text = Path(args.eval_text_file).read_text(encoding="utf-8")
        texts = [x.strip() for x in text.split("\n\n") if x.strip()]
        if not texts:
            raise ValueError("--eval_text_file produced no non-empty text blocks.")
        return texts
    return list(DEFAULT_EVAL_TEXTS)


@torch.no_grad()
def loss_ppl_proxy(
    model: torch.nn.Module,
    tokenizer,
    *,
    texts: list[str],
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    losses: list[float] = []
    token_counts: list[int] = []

    for idx, text in enumerate(texts):
        input_ids = build_fixed_input_ids(
            tokenizer,
            seq_len=int(seq_len),
            text=str(text),
            device=device,
        )
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss = float(outputs.loss.detach().float().item())
        losses.append(loss)
        token_counts.append(int(input_ids.numel()))

        print(
            f"[LossProbe] text={idx} seq_len={seq_len} "
            f"loss={loss:.6f} tokens={int(input_ids.numel())}"
        )

        del input_ids, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    median_loss = float(statistics.median(losses))
    mean_loss = float(statistics.fmean(losses))
    return {
        "seq_len": int(seq_len),
        "num_texts": int(len(texts)),
        "losses": [float(x) for x in losses],
        "token_counts": token_counts,
        "mean_loss": mean_loss,
        "median_loss": median_loss,
        "ppl_proxy_from_mean_loss": float(math.exp(mean_loss)) if mean_loss < 80 else float("inf"),
        "ppl_proxy_from_median_loss": float(math.exp(median_loss)) if median_loss < 80 else float("inf"),
        "note": (
            "Deterministic fixed-text autoregressive loss probe. "
            "This is a stability comparison proxy, not a standardized benchmark PPL."
        ),
    }


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    if xf.numel() == 0:
        raise ValueError("Cannot compute stats for an empty tensor.")
    flat_last = xf.reshape(-1, xf.shape[-1]) if xf.ndim >= 1 else xf.reshape(1, 1)
    l2 = torch.linalg.vector_norm(flat_last, ord=2, dim=-1)
    return {
        "abs_max": float(xf.abs().max().item()),
        "abs_mean": float(xf.abs().mean().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "l2_mean_over_last_dim": float(l2.mean().item()),
        "l2_max_over_last_dim": float(l2.max().item()),
    }


def logits_stats(logits: torch.Tensor) -> dict[str, float]:
    lf = logits.detach().float()
    return {
        "abs_max": float(lf.abs().max().item()),
        "abs_mean": float(lf.abs().mean().item()),
        "min": float(lf.min().item()),
        "max": float(lf.max().item()),
        "mean": float(lf.mean().item()),
        "std": float(lf.std(unbiased=False).item()),
    }


@torch.no_grad()
def probe_qk_logits_by_layer(
    model: torch.nn.Module,
    tokenizer,
    *,
    layers: list[int],
    seq_lens: list[int],
    probe_text: str,
    device: torch.device,
    apply_rope: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for layer_idx in layers:
        for seq_len in seq_lens:
            T = int(seq_len)
            print(f"[QKProbe] layer={layer_idx}, T={T}")
            input_ids = build_fixed_input_ids(
                tokenizer,
                seq_len=T,
                text=probe_text,
                device=device,
            )
            qk = capture_llama_style_qk(
                model,
                input_ids,
                layer_idx=int(layer_idx),
                num_query_tokens=1,
                apply_rope=bool(apply_rope),
            )
            dense_logits = dense_fp32_logits(qk.queries, qk.keys)

            item = {
                "layer_idx": int(layer_idx),
                "seq_len": T,
                "qk_capture": {
                    "queries_shape": list(qk.queries.shape),
                    "keys_shape": list(qk.keys.shape),
                    "rope_applied": bool(qk.rope_applied),
                    "rope_detail": str(qk.rope_detail),
                    "num_attention_heads": int(qk.num_attention_heads),
                    "num_key_value_heads": int(qk.num_key_value_heads),
                    "key_heads_expanded": bool(qk.key_heads_expanded),
                },
                "query_stats": tensor_stats(qk.queries),
                "key_stats": tensor_stats(qk.keys),
                "dense_attention_logits_stats": logits_stats(dense_logits),
            }
            print(json.dumps(item, indent=2))
            results.append(item)

            del input_ids, qk, dense_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def index_probe_results(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        out[(int(row["layer_idx"]), int(row["seq_len"]))] = row
    return out


def safe_ratio(numer: float, denom: float) -> float | None:
    if float(denom) == 0.0:
        return None
    return float(numer) / float(denom)


def compare_probe_stats(
    full_rows: list[dict[str, Any]],
    svd_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_idx = index_probe_results(full_rows)
    svd_idx = index_probe_results(svd_rows)
    keys = sorted(set(full_idx).intersection(svd_idx))

    comparisons: list[dict[str, Any]] = []
    for key in keys:
        full = full_idx[key]
        svd = svd_idx[key]
        comp = {
            "layer_idx": int(key[0]),
            "seq_len": int(key[1]),
            "svd_over_full_ratio": {
                "query_abs_mean": safe_ratio(
                    svd["query_stats"]["abs_mean"],
                    full["query_stats"]["abs_mean"],
                ),
                "query_l2_mean": safe_ratio(
                    svd["query_stats"]["l2_mean_over_last_dim"],
                    full["query_stats"]["l2_mean_over_last_dim"],
                ),
                "key_abs_mean": safe_ratio(
                    svd["key_stats"]["abs_mean"],
                    full["key_stats"]["abs_mean"],
                ),
                "key_l2_mean": safe_ratio(
                    svd["key_stats"]["l2_mean_over_last_dim"],
                    full["key_stats"]["l2_mean_over_last_dim"],
                ),
                "logits_abs_mean": safe_ratio(
                    svd["dense_attention_logits_stats"]["abs_mean"],
                    full["dense_attention_logits_stats"]["abs_mean"],
                ),
                "logits_abs_max": safe_ratio(
                    svd["dense_attention_logits_stats"]["abs_max"],
                    full["dense_attention_logits_stats"]["abs_max"],
                ),
                "logits_std": safe_ratio(
                    svd["dense_attention_logits_stats"]["std"],
                    full["dense_attention_logits_stats"]["std"],
                ),
            },
            "full_rank": {
                "query_stats": full["query_stats"],
                "key_stats": full["key_stats"],
                "dense_attention_logits_stats": full["dense_attention_logits_stats"],
            },
            "svd_uniform_08": {
                "query_stats": svd["query_stats"],
                "key_stats": svd["key_stats"],
                "dense_attention_logits_stats": svd["dense_attention_logits_stats"],
            },
        }
        comparisons.append(comp)
    return comparisons


def extract_flag_rows(
    comparisons: list[dict[str, Any]],
    *,
    logits_abs_mean_ratio_threshold: float,
    logits_abs_max_ratio_threshold: float,
) -> list[dict[str, Any]]:
    flagged = []
    for row in comparisons:
        ratios = row["svd_over_full_ratio"]
        abs_mean_ratio = ratios["logits_abs_mean"]
        abs_max_ratio = ratios["logits_abs_max"]
        reasons = []
        if abs_mean_ratio is not None and abs_mean_ratio >= float(logits_abs_mean_ratio_threshold):
            reasons.append(
                f"logits_abs_mean_ratio={abs_mean_ratio:.3g} >= {float(logits_abs_mean_ratio_threshold):.3g}"
            )
        if abs_max_ratio is not None and abs_max_ratio >= float(logits_abs_max_ratio_threshold):
            reasons.append(
                f"logits_abs_max_ratio={abs_max_ratio:.3g} >= {float(logits_abs_max_ratio_threshold):.3g}"
            )
        if reasons:
            flagged.append(
                {
                    "layer_idx": int(row["layer_idx"]),
                    "seq_len": int(row["seq_len"]),
                    "reasons": reasons,
                    "ratios": ratios,
                }
            )
    return flagged



def save_rebuilt_svd_model(
    model: torch.nn.Module,
    tokenizer,
    *,
    save_dir: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a rebuilt SVD model in HuggingFace-compatible format."""
    out = Path(str(save_dir))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Save:SVDModel] Writing rebuilt SVD model to: {out}")
    model.save_pretrained(
        str(out),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(out))

    if metadata is not None:
        meta_path = out / "svd_rebuild_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[Save:SVDModel] Metadata: {meta_path}")

    total_bytes = 0
    file_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total_bytes += size
            file_rows.append({"path": str(p.relative_to(out)), "bytes": int(size)})

    summary = {
        "save_dir": str(out),
        "total_bytes": int(total_bytes),
        "total_gib": float(total_bytes / (1024 ** 3)),
        "files": file_rows,
    }
    print(
        "[Save:SVDModel] Complete | "
        f"total={summary['total_gib']:.2f} GiB | files={len(file_rows)}"
    )
    return summary


def load_saved_svd_model(
    *,
    model_dir: str,
    torch_dtype: torch.dtype | str,
    device: torch.device,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load a previously saved rebuilt SVD model and tokenizer."""
    model_dir = str(model_dir)
    print(f"[Load:SVDModel] {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=bool(trust_remote_code),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=bool(trust_remote_code),
        use_fast=True,
    )
    ensure_tokenizer_padding(tokenizer)
    model.to(device)
    model.eval()

    metadata_path = Path(model_dir) / "svd_rebuild_metadata.json"
    loaded_metadata = None
    if metadata_path.exists():
        try:
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            loaded_metadata = {"metadata_load_error": str(e)}

    metadata = {
        "rank_source": "saved_rebuilt_svd_model",
        "svd_model_dir": model_dir,
        "saved_metadata": loaded_metadata,
    }
    return model, tokenizer, metadata

def load_full_rank_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    dtype = parse_dtype(args.torch_dtype)
    model_id = str(args.base_model)
    print(f"[Load:full] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    ensure_tokenizer_padding(tokenizer)
    model.to(device)
    model.eval()
    metadata = {
        "model_id": model_id,
        "dtype": str(args.torch_dtype),
        "device": str(device),
    }
    return model, tokenizer, metadata


def load_uniform_svd_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    dtype = parse_dtype(args.torch_dtype)

    if args.svd_model_dir:
        if args.save_rebuilt_svd_model_dir:
            raise ValueError(
                "Use either --svd_model_dir or --save_rebuilt_svd_model_dir, not both."
            )
        return load_saved_svd_model(
            model_dir=str(args.svd_model_dir),
            torch_dtype=dtype,
            device=device,
            trust_remote_code=bool(args.trust_remote_code),
        )

    model_id = str(args.base_model)
    print(f"[Load:svd-base] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    ensure_tokenizer_padding(tokenizer)

    metadata: dict[str, Any] = {
        "model_id": model_id,
        "dtype": str(args.torch_dtype),
        "device": str(device),
        "rank_source": None,
    }

    if args.rank_json and args.uniform_linear_param_ratio is not None:
        raise ValueError("Use either --rank_json or --uniform_linear_param_ratio, not both.")

    if args.rank_json:
        rank_dict = load_rank_json(str(args.rank_json))
        metadata["rank_source"] = "rank_json"
        metadata["rank_json"] = str(args.rank_json)
    else:
        ratio = (
            float(args.uniform_linear_param_ratio)
            if args.uniform_linear_param_ratio is not None
            else 0.8
        )
        rank_dict, uniform_summary = build_uniform_rank_dict_from_model(
            model,
            linear_param_ratio=ratio,
            name_regex=str(args.uniform_name_regex),
            min_rank=int(args.uniform_min_rank),
            rank_multiple=int(args.uniform_rank_multiple),
        )
        metadata["rank_source"] = "uniform_linear_param_ratio"
        metadata["uniform_rank_generation"] = uniform_summary
        if args.save_uniform_rank_json:
            save_rank_dict_json(
                str(args.save_uniform_rank_json),
                rank_dict,
                uniform_summary,
            )

    acct = estimate_params_from_ranks(model, rank_dict, strict=bool(args.strict))
    metadata["rank_entries"] = int(len(rank_dict))
    metadata["rank_param_accounting"] = jsonable_dataclass(acct)

    svd_rebuild_device = resolve_svd_rebuild_device(str(args.svd_rebuild_device))
    metadata["svd_rebuild_device_requested"] = str(args.svd_rebuild_device)
    metadata["svd_rebuild_device_resolved"] = str(svd_rebuild_device)

    start = time.perf_counter()
    model = apply_svd_ranks_inplace(
        model,
        rank_dict,
        dtype=torch.float16 if dtype == "auto" else dtype,
        svd_device=svd_rebuild_device,
        progress=True,
    )
    metadata["svd_rebuild_elapsed_sec"] = float(time.perf_counter() - start)

    if args.save_rebuilt_svd_model_dir:
        metadata["saved_rebuilt_svd_model"] = save_rebuilt_svd_model(
            model,
            tokenizer,
            save_dir=str(args.save_rebuilt_svd_model_dir),
            metadata=metadata,
        )

    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def summarize_loss_comparison(
    full_loss: dict[str, Any],
    svd_loss: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seq_len": int(full_loss["seq_len"]),
        "full_rank_mean_loss": float(full_loss["mean_loss"]),
        "svd_uniform_08_mean_loss": float(svd_loss["mean_loss"]),
        "mean_loss_delta_svd_minus_full": float(svd_loss["mean_loss"] - full_loss["mean_loss"]),
        "full_rank_ppl_proxy": float(full_loss["ppl_proxy_from_mean_loss"]),
        "svd_uniform_08_ppl_proxy": float(svd_loss["ppl_proxy_from_mean_loss"]),
        "ppl_proxy_ratio_svd_over_full": safe_ratio(
            svd_loss["ppl_proxy_from_mean_loss"],
            full_loss["ppl_proxy_from_mean_loss"],
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate uniform SVD 80% model stability before full TurboQuant attention replacement. "
            "Compares full-rank Llama vs rebuilt SVD model on fixed-text loss probes and layerwise Q/K/logit scales."
        )
    )
    p.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch_dtype", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--rank_json", default=None)
    p.add_argument("--uniform_linear_param_ratio", type=float, default=0.8)
    p.add_argument("--uniform_name_regex", default=DEFAULT_UNIFORM_TARGET_LINEAR_REGEX)
    p.add_argument("--uniform_min_rank", type=int, default=1)
    p.add_argument("--uniform_rank_multiple", type=int, default=1)
    p.add_argument("--save_uniform_rank_json", default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--svd_rebuild_device", default="auto")
    p.add_argument(
        "--svd_model_dir",
        default=None,
        help=(
            "Load a previously saved rebuilt SVD model from this directory and "
            "skip the 224-layer SVD rebuild."
        ),
    )
    p.add_argument(
        "--save_rebuilt_svd_model_dir",
        default=None,
        help=(
            "After uniform-SVD rebuild, save the rebuilt model/tokenizer here "
            "with safe_serialization=True and 5GB shards."
        ),
    )

    p.add_argument("--loss_probe_seq_len", type=int, default=512)
    p.add_argument("--eval_text_file", default=None)

    p.add_argument("--probe_layers", type=int, nargs="+", default=[0, 4, 8, 12, 15, 20, 24, 28, 31])
    p.add_argument("--probe_seq_lens", type=int, nargs="+", default=[2048, 4096])
    p.add_argument("--probe_text", default=DEFAULT_EVAL_TEXTS[0])
    p.add_argument("--no_rope", action="store_true")

    p.add_argument("--logits_abs_mean_ratio_threshold", type=float, default=10.0)
    p.add_argument("--logits_abs_max_ratio_threshold", type=float, default=10.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")

    device = torch.device(str(args.device))
    texts = select_eval_texts(args)

    print("========== Uniform SVD 80% model stability benchmark ==========")
    print("[Purpose]")
    print("  Compare full-rank model vs uniform-SVD model before full TurboQuant attention replacement.")
    print("[Metrics]")
    print("  1) Fixed-text autoregressive loss / perplexity proxy")
    print("  2) Layerwise Q/K norm statistics")
    print("  3) Layerwise dense qK^T attention-logit scale statistics")
    print("  4) SVD/full-rank drift ratios and flagged layers")
    print(f"[Config] loss_probe_seq_len={args.loss_probe_seq_len}")
    print(f"[Config] probe_layers={list(args.probe_layers)}")
    print(f"[Config] probe_seq_lens={list(args.probe_seq_lens)}")

    # ------------------------------------------------------------------
    # Full-rank baseline
    # ------------------------------------------------------------------
    full_model, full_tokenizer, full_meta = load_full_rank_model(args, device=device)
    full_loss = loss_ppl_proxy(
        full_model,
        full_tokenizer,
        texts=texts,
        seq_len=int(args.loss_probe_seq_len),
        device=device,
    )
    full_qk_probe = probe_qk_logits_by_layer(
        full_model,
        full_tokenizer,
        layers=[int(x) for x in args.probe_layers],
        seq_lens=[int(x) for x in args.probe_seq_lens],
        probe_text=str(args.probe_text),
        device=device,
        apply_rope=not bool(args.no_rope),
    )

    # IMPORTANT:
    # Passing full_model into cleanup_model(...) alone does not delete the
    # caller-side reference. Explicitly delete it here before loading/rebuilding
    # the SVD model, otherwise both full-rank and SVD models coexist on GPU.
    cleanup_model(full_model)
    del full_model
    del full_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(device)
        print(
            "[GPU:after-full-cleanup] "
            f"free={free_b / (1024**3):.2f} GiB / total={total_b / (1024**3):.2f} GiB | "
            f"allocated={torch.cuda.memory_allocated(device) / (1024**3):.2f} GiB | "
            f"reserved={torch.cuda.memory_reserved(device) / (1024**3):.2f} GiB"
        )

    # ------------------------------------------------------------------
    # Uniform SVD model
    # ------------------------------------------------------------------
    svd_model, svd_tokenizer, svd_meta = load_uniform_svd_model(args, device=device)
    svd_loss = loss_ppl_proxy(
        svd_model,
        svd_tokenizer,
        texts=texts,
        seq_len=int(args.loss_probe_seq_len),
        device=device,
    )
    svd_qk_probe = probe_qk_logits_by_layer(
        svd_model,
        svd_tokenizer,
        layers=[int(x) for x in args.probe_layers],
        seq_lens=[int(x) for x in args.probe_seq_lens],
        probe_text=str(args.probe_text),
        device=device,
        apply_rope=not bool(args.no_rope),
    )

    comparisons = compare_probe_stats(full_qk_probe, svd_qk_probe)
    flagged = extract_flag_rows(
        comparisons,
        logits_abs_mean_ratio_threshold=float(args.logits_abs_mean_ratio_threshold),
        logits_abs_max_ratio_threshold=float(args.logits_abs_max_ratio_threshold),
    )
    loss_comparison = summarize_loss_comparison(full_loss, svd_loss)

    print("=" * 78)
    print("[Loss comparison]")
    print(json.dumps(loss_comparison, indent=2))
    print("=" * 78)
    print("[Flagged layer/sequence drift]")
    print(json.dumps(flagged, indent=2))

    payload = {
        "benchmark": "true_turboquant_svd_uniform_08_model_stability",
        "purpose": (
            "Sanity-check the uniformly SVD-compressed model itself before "
            "full TurboQuant attention replacement."
        ),
        "config": vars(args),
        "full_rank_model": {
            "metadata": full_meta,
            "loss_probe": full_loss,
            "qk_logit_probe": full_qk_probe,
        },
        "svd_uniform_08_model": {
            "metadata": svd_meta,
            "loss_probe": svd_loss,
            "qk_logit_probe": svd_qk_probe,
        },
        "comparisons": {
            "loss": loss_comparison,
            "qk_logit_scale_drift": comparisons,
            "flagged_layer_seq_rows": flagged,
        },
    }

    out = Path(str(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Save] {out}")

    cleanup_model(svd_model)
    del svd_tokenizer


if __name__ == "__main__":
    main()
