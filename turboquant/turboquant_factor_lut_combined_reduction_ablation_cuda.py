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
            / "turboquant_factor_lut_combined_reduction_ablation_cuda.cu"
        )
        _EXT = load(
            name="true_turboquant_factor_lut_combined_reduction_ablation_cuda_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
    *,
    scalar_factor_lut: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _load_ext().turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
        scalar_factor_lut.contiguous().to(torch.float32),
        lane_word_scalar_codes.contiguous(),
        qjl_projected_queries.contiguous().to(torch.float32),
        lane_nibble_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
    )
