from __future__ import annotations

import torch


@torch.no_grad()
def pack_qjl_signs_lane_nibble(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack QJL signs into lane-local nibble layout.

    Supports M = 128 * nblocks.

    For each 128-dim block, one warp lane maps to:
        lane, lane+32, lane+64, lane+96

    The four sign bits are stored in one nibble:
        bit 0 -> sign[block_base + lane]
        bit 1 -> sign[block_base + lane + 32]
        bit 2 -> sign[block_base + lane + 64]
        bit 3 -> sign[block_base + lane + 96]

    Two lane nibbles are packed into one byte:
        byte = low_lane_nibble | (high_lane_nibble << 4)

    Input:
      signs: [..., M], values may be {-1,+1}, bool, or numeric sign-like.

    Output:
      packed: [..., M // 8], uint8
    """
    if signs.ndim < 1:
        raise ValueError("Expected signs tensor with at least 1 dim.")

    M = int(signs.shape[-1])
    if M % 128 != 0:
        raise ValueError(f"Expected signs last dim divisible by 128, got {M}.")

    positive = (signs > 0).to(torch.uint8)

    nblocks = M // 128
    chunks = []

    lane_ids = torch.arange(32, device=signs.device, dtype=torch.long)
    coord_offsets = torch.tensor([0, 32, 64, 96], device=signs.device, dtype=torch.long)
    bit_shifts = torch.arange(4, device=signs.device, dtype=torch.uint8)

    for b in range(nblocks):
        base = b * 128
        gather_idx = base + lane_ids[:, None] + coord_offsets[None, :]  # [32,4]

        lane_bits = positive[..., gather_idx]  # [...,32,4]
        lane_nibbles = torch.sum(
            torch.bitwise_left_shift(lane_bits, bit_shifts),
            dim=-1,
            dtype=torch.int64,
        ).to(torch.uint8)  # [...,32]

        low = lane_nibbles[..., 0::2]
        high = lane_nibbles[..., 1::2]
        packed_block = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))  # [...,16]
        chunks.append(packed_block)

    return torch.cat(chunks, dim=-1).contiguous()


@torch.no_grad()
def unpack_qjl_signs_lane_nibble(
    packed: torch.Tensor,
    *,
    qjl_dim: int | None = None,
) -> torch.Tensor:
    """
    Inverse of pack_qjl_signs_lane_nibble.

    Input:
      packed: [..., M//8]
      qjl_dim: optional M. Defaults to packed.shape[-1] * 8.

    Output:
      signs: [..., M], int8 signs {-1,+1} in standard sketch-index order.
    """
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8.")
    if packed.ndim < 1:
        raise ValueError("Expected packed tensor with at least 1 dim.")

    if qjl_dim is None:
        M = int(packed.shape[-1]) * 8
    else:
        M = int(qjl_dim)

    if M % 128 != 0:
        raise ValueError(f"Expected qjl_dim divisible by 128, got {M}.")
    if int(packed.shape[-1]) != M // 8:
        raise ValueError(
            f"Expected packed last dim = {M // 8}, got {packed.shape[-1]}."
        )

    out = torch.empty(*packed.shape[:-1], M, device=packed.device, dtype=torch.int8)
    bit_shifts = torch.arange(4, device=packed.device, dtype=torch.uint8)

    nblocks = M // 128
    for b in range(nblocks):
        packed_block = packed[..., b * 16:(b + 1) * 16]

        low = torch.bitwise_and(packed_block, 0x0F)
        high = torch.bitwise_and(torch.bitwise_right_shift(packed_block, 4), 0x0F)
        lane_nibbles = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], 32)

        bits = torch.bitwise_and(
            torch.bitwise_right_shift(lane_nibbles.unsqueeze(-1), bit_shifts),
            torch.tensor(1, device=packed.device, dtype=torch.uint8),
        )  # [...,32,4]

        base = b * 128
        for i, offset in enumerate((0, 32, 64, 96)):
            out[..., base + offset:base + offset + 32] = torch.where(
                bits[..., :, i] > 0,
                torch.ones((), device=packed.device, dtype=torch.int8),
                -torch.ones((), device=packed.device, dtype=torch.int8),
            )

    return out.contiguous()
