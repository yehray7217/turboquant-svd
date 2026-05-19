from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        src = (
            Path(__file__).resolve().parent
            / "csrc"
            / "turboquant_qjl_sign_layout_ablation_cuda.cu"
        )
        _EXT = load(
            name="true_turboquant_qjl_sign_layout_ablation_cuda_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda(
    *,
    qjl_projected_queries: torch.Tensor,
    packed_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _load_ext().turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda(
        qjl_projected_queries.contiguous().to(torch.float32),
        packed_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
    )


@torch.no_grad()
def turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda(
    *,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _load_ext().turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda(
        qjl_projected_queries.contiguous().to(torch.float32),
        lane_nibble_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
    )


@torch.no_grad()
def turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda(
    *,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _load_ext().turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda(
        qjl_projected_queries.contiguous().to(torch.float32),
        lane_nibble_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
    )
