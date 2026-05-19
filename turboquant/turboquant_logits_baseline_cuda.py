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
            / "turboquant_logits_baseline_cuda.cu"
        )
        _EXT = load(
            name="true_turboquant_logits_baseline_cuda_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def dense_fp32_qkt_b1q1_d128_cuda(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    return _load_ext().dense_fp32_qkt_b1q1_d128_cuda(
        queries.contiguous().to(torch.float32),
        keys.contiguous().to(torch.float32),
    )


@torch.no_grad()
def turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
    *,
    rotated_queries: torch.Tensor,
    packed_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    packed_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return _load_ext().turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
        rotated_queries.contiguous().to(torch.float32),
        packed_scalar_codes.contiguous(),
        qjl_projected_queries.contiguous().to(torch.float32),
        packed_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
        centroids.contiguous().to(torch.float32),
    )
