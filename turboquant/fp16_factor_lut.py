from __future__ import annotations

import torch

from .factor_lut import build_scalar_factor_lut_fp32


@torch.no_grad()
def build_scalar_factor_lut_fp16(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Build the current query-conditioned scalar factor LUT, then store it as FP16.

    Semantics:
      factor_lut_fp16[b,h,coord,code]
        = fp16(rotated_queries[b,h,0,coord] * centroids[code])

    Input:
      rotated_queries: [1,H,1,128]
      centroids: [16]

    Output:
      factor_lut_fp16: [1,H,128,16], float16 contiguous

    This deliberately matches the existing FP32 LUT builder followed by FP16
    storage, so the ablation isolates the storage/load precision change.
    """
    return build_scalar_factor_lut_fp32(
        rotated_queries,
        centroids,
    ).to(torch.float16).contiguous()
