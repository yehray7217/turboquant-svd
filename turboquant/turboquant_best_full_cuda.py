from __future__ import annotations

import torch

from .turboquant_scalar_lane_word_ablation_cuda import (
    turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_cuda,
)


@torch.no_grad()
def turboquant_best_4bit_qjl128_logits_b1q1_d128_cuda(
    *,
    rotated_queries: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Current best exact-parity true TurboQuant full logits kernel.

    Layout:
      - scalar codes: 4-bit lane-word layout, 64 bytes/token/head
      - QJL signs: lane-nibble layout, 16 bytes/token/head
      - residual norm: FP32, 4 bytes/token/head

    Kernel:
      - true TurboQuant scalar contribution
      - QJL128 residual correction
    """
    return turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_cuda(
        rotated_queries=rotated_queries,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
        centroids=centroids,
    )
