from __future__ import annotations

import torch

from .factor_lut import build_scalar_factor_lut_fp32
from .turboquant_scalar_factor_lut_ablation_cuda import (
    turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_logits_b1q1_d128_cuda,
)


@torch.no_grad()
def build_turboquant_scalar_factor_lut_fp32(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return build_scalar_factor_lut_fp32(rotated_queries, centroids)


@torch.no_grad()
def turboquant_factor_lut_4bit_qjl128_logits_b1q1_d128_cuda(
    *,
    scalar_factor_lut: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_logits_b1q1_d128_cuda(
        scalar_factor_lut=scalar_factor_lut,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
    )
