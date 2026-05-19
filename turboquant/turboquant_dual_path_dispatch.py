from __future__ import annotations

import torch

from .factor_lut import build_scalar_factor_lut_fp32
from .turboquant_combined_reduction_nonfactor_ablation_cuda import (
    turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)
from .turboquant_factor_lut_combined_reduction_full_cuda import (
    turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda,
)


DEFAULT_FACTOR_LUT_THRESHOLD_T = 65536


@torch.no_grad()
def turboquant_dual_path_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
    *,
    rotated_queries: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    centroids: torch.Tensor,
    factor_lut_threshold_t: int = DEFAULT_FACTOR_LUT_THRESHOLD_T,
) -> tuple[torch.Tensor, str]:
    """
    Dispatch the currently best validated true-TurboQuant logits path.

    Policy:
      - T < threshold:
          non-factor combined reduction
      - T >= threshold:
          build scalar factor LUT and use factor-LUT combined reduction

    The default threshold T=65536 follows the ablation where:
      - factor-LUT combined effective path is slower at 16K / 32K
      - factor-LUT combined effective path is faster at 64K / 128K

    Returns:
      (out, selected_path)

    selected_path:
      - "nonfactor_combined"
      - "factor_lut_combined"
    """
    if lane_word_scalar_codes.ndim != 4:
        raise ValueError(
            "lane_word_scalar_codes must be [1,H,T,64], "
            f"got {tuple(lane_word_scalar_codes.shape)}."
        )
    T = int(lane_word_scalar_codes.shape[2])

    if T >= int(factor_lut_threshold_t):
        scalar_factor_lut = build_scalar_factor_lut_fp32(rotated_queries, centroids)
        out = turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
            scalar_factor_lut=scalar_factor_lut,
            lane_word_scalar_codes=lane_word_scalar_codes,
            qjl_projected_queries=qjl_projected_queries,
            lane_nibble_qjl_signs=lane_nibble_qjl_signs,
            residual_norms=residual_norms,
        )
        return out, "factor_lut_combined"

    out = turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
        rotated_queries=rotated_queries,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
        centroids=centroids,
    )
    return out, "nonfactor_combined"
