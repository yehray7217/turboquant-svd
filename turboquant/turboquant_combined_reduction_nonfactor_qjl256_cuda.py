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
            / "turboquant_combined_reduction_nonfactor_qjl256_cuda.cu"
        )
        _EXT = load(
            name="true_turboquant_combined_reduction_nonfactor_ablation_cuda_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def turboquant_full_4bit_lane_word_lane_nibble_qjl256_combined_reduction_logits_b1q1_d128_cuda(
    *,
    rotated_queries: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    centroids: torch.Tensor,
    active_kv_len: int | None = None,
) -> torch.Tensor:
    active_kv_len_i = (
        int(lane_word_scalar_codes.shape[2])
        if active_kv_len is None
        else int(active_kv_len)
    )

    return _load_ext().turboquant_full_4bit_lane_word_lane_nibble_qjl256_combined_reduction_logits_b1q1_d128_cuda(
        rotated_queries.contiguous().to(torch.float32),
        lane_word_scalar_codes.contiguous(),
        qjl_projected_queries.contiguous().to(torch.float32),
        lane_nibble_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
        centroids.contiguous().to(torch.float32),
        active_kv_len_i,
    )
