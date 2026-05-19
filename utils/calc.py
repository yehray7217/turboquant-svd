from __future__ import annotations

from typing import Any


def _linear_dims(module: Any) -> tuple[int, int]:
    try:
        in_features = int(module.in_features)
        out_features = int(module.out_features)
    except Exception as exc:
        raise TypeError(f"Expected a Linear-like module, got {type(module)!r}") from exc
    return in_features, out_features


def _rank_for_ratio(module: Any, param_ratio: float) -> int:
    in_features, out_features = _linear_dims(module)
    full_rank = min(in_features, out_features)
    rank = int(float(param_ratio) * in_features * out_features / (in_features + out_features))
    rank = max(1, min(full_rank, rank))
    return rank


def set_uniform_truncation_rank(
    module_dict: dict[str, Any],
    linear_info: dict[Any, dict[str, Any]],
    param_ratio_target: float,
) -> dict[str, int]:
    """Return layer_name -> uniform rank for llm_rs.py.

    The rank formula matches the low-rank parameter-count ratio convention:
        rank * (in + out) ~= ratio * in * out

    `compress_model_whiten(...)` already skips unsupported / unwhitened linears
    such as lm_head, so this function can safely emit every collected Linear.
    """
    result: dict[str, int] = {}
    for module, info in linear_info.items():
        full_name = str(info["full_name"])
        result[full_name] = _rank_for_ratio(module, float(param_ratio_target))
    return result


def rank_to_param_ratio(
    module_dict: dict[str, Any],
    sensitivity_dict: dict[str, dict[Any, float]],
    succinct: bool = False,
):
    """Convert rank-keyed sensitivity maps into parameter-ratio-keyed maps.

    This compatibility helper is primarily for existing greedy/binary search
    imports. The current project flow uses `search_method=uniform`, but keeping
    this function available preserves the older entry points.
    """
    converted: dict[str, dict[float, float]] = {}
    mapping: dict[str, dict[float, int]] = {}

    for layer_name, rank_to_score in sensitivity_dict.items():
        module = module_dict.get(layer_name)
        if module is None:
            continue
        in_features, out_features = _linear_dims(module)
        converted[layer_name] = {}
        mapping[layer_name] = {}
        for rank_raw, score in rank_to_score.items():
            rank = int(rank_raw)
            ratio = float(rank * (in_features + out_features) / (in_features * out_features))
            ratio = round(ratio, 12)
            converted[layer_name][ratio] = score
            mapping[layer_name][ratio] = rank
    return converted, mapping
