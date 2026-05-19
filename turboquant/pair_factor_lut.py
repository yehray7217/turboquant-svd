from __future__ import annotations

import torch


@torch.no_grad()
def build_scalar_pair_factor_lut_fp32(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Build a two-stage scalar pair-factor LUT:

        pair_lut[b,h,pair,lane,pair_code]

    where:
      pair = 0 covers scalar stages (0,1)
      pair = 1 covers scalar stages (2,3)

      pair_code is one byte:
        low nibble  = first  4-bit scalar code in the pair
        high nibble = second 4-bit scalar code in the pair

    Input:
      rotated_queries: [1,H,1,128], float32-compatible
      centroids:       [16], float32-compatible

    Output:
      pair_lut: [1,H,2,32,256], float32 contiguous

    Semantics:
      pair_lut[h,0,lane,byte01]
        = rq[h,lane + 0*32] * centroid[code0]
        + rq[h,lane + 1*32] * centroid[code1]

      pair_lut[h,1,lane,byte23]
        = rq[h,lane + 2*32] * centroid[code2]
        + rq[h,lane + 3*32] * centroid[code3]

    This is a larger query-conditioned LUT than the current single-stage factor
    LUT, but it reduces runtime scalar scatter loads from 4 to 2 per lane.
    """
    if rotated_queries.ndim != 4:
        raise ValueError(
            f"rotated_queries must be [1,H,1,128], got {tuple(rotated_queries.shape)}."
        )
    if rotated_queries.shape[0] != 1 or rotated_queries.shape[2] != 1:
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
    rq_stage_lane = rq.reshape(rq.shape[0], rq.shape[1], 4, 32)  # [1,H,4,32]

    c = centroids.contiguous().to(torch.float32)
    pair_code = torch.arange(256, device=c.device, dtype=torch.int64)
    c_lo = c[(pair_code & 0x0F)]
    c_hi = c[(pair_code >> 4)]

    pair01 = (
        rq_stage_lane[:, :, 0, :, None] * c_lo[None, None, None, :]
        + rq_stage_lane[:, :, 1, :, None] * c_hi[None, None, None, :]
    )
    pair23 = (
        rq_stage_lane[:, :, 2, :, None] * c_lo[None, None, None, :]
        + rq_stage_lane[:, :, 3, :, None] * c_hi[None, None, None, :]
    )
    return torch.stack([pair01, pair23], dim=2).contiguous()
