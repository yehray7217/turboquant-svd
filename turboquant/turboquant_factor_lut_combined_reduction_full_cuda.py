from __future__ import annotations

import torch

from .factor_lut import build_scalar_factor_lut_fp32
from .turboquant_factor_lut_combined_reduction_ablation_cuda import (
    turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
)


@torch.no_grad()
def build_turboquant_scalar_factor_lut_fp32(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Query-conditioned scalar factor LUT used by the current combined-reduction
    factor-LUT full-kernel candidate.

    Input:
      rotated_queries: [1,H,1,128]
      centroids: [16]

    Output:
      scalar_factor_lut: [1,H,128,16]
    """
    return build_scalar_factor_lut_fp32(rotated_queries, centroids)


@torch.no_grad()
def turboquant_factor_lut_combined_reduction_4bit_qjl128_logits_b1q1_d128_cuda(
    *,
    scalar_factor_lut: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    """
    Current strongest long-context true-TurboQuant logits candidate:

      - scalar factor LUT: [1,H,128,16], FP32 query-side precompute
      - scalar codes: lane-word 4-bit layout
      - QJL signs: lane-nibble layout
      - scalar/QJL contributions merged before one warp reduction
    """
    return turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
        scalar_factor_lut=scalar_factor_lut,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
    )
