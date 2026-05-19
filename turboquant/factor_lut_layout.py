from __future__ import annotations

import torch


@torch.no_grad()
def build_scalar_factor_lut_warp_code_major_fp32(
    rotated_queries: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Build a warp-code-major factor LUT:

        out[b,h,stage,code,lane]
          = rotated_queries[b,h,0,coord] * centroids[code]

        coord = lane + 32 * stage
        stage in {0,1,2,3}
        lane  in {0,...,31}

    Input:
      rotated_queries: [1,H,1,128], float32-compatible
      centroids:       [16], float32-compatible

    Output:
      factor_lut: [1,H,4,16,32], float32 contiguous

    Compared with the current [1,H,128,16] factor LUT, this physical layout
    groups the 32 lanes of a warp together inside each (stage, code) tile.
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
    # Coordinate order used by the CUDA kernel:
    # coord = lane + stage * 32
    rq_stage_lane = rq.reshape(rq.shape[0], rq.shape[1], 4, 32)  # [1,H,4,32]
    c = centroids.contiguous().to(torch.float32).view(1, 1, 1, 16, 1)
    return (rq_stage_lane.unsqueeze(-2) * c).contiguous()  # [1,H,4,16,32]
