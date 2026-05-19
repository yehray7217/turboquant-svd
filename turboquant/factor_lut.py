from __future__ import annotations

import torch


@torch.no_grad()
def build_scalar_factor_lut_fp32(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Build a query-conditioned scalar factor LUT:

        factor_lut[b,h,coord,code]
          = rotated_queries[b,h,0,coord] * centroids[code]

    Expected current benchmark/mainline shape:
      rotated_queries: [1,H,1,128], float32-compatible
      centroids:       [16], float32-compatible

    Output:
      factor_lut: [1,H,128,16], float32 contiguous

    This is a query-side precompute, analogous to the already precomputed
    QJL projected query used by the current logits kernels.
    """
    if rotated_queries.ndim != 4:
        raise ValueError(
            f"rotated_queries must be [1,H,1,128], got {tuple(rotated_queries.shape)}."
        )
    if tuple(rotated_queries.shape[::2]) != (1, 1):
        # shape[0] == 1 and shape[2] == 1
        raise ValueError(
            f"rotated_queries must be [1,H,1,128], got {tuple(rotated_queries.shape)}."
        )
    if rotated_queries.shape[-1] != 128:
        raise ValueError(
            f"rotated_queries last dim must be 128, got {rotated_queries.shape[-1]}."
        )
    if centroids.ndim != 1 or centroids.numel() != 16:
        raise ValueError(f"centroids must be [16], got {tuple(centroids.shape)}.")

    rq = rotated_queries.contiguous().to(torch.float32)[:, :, 0, :]  # [1,H,128]
    c = centroids.contiguous().to(torch.float32).view(1, 1, 1, 16)
    return (rq.unsqueeze(-1) * c).contiguous()
