from __future__ import annotations

import torch


@torch.no_grad()
def pack_qjl_signs_lane_nibble(signs: torch.Tensor) -> torch.Tensor:
    """
    Repack QJL128 signs into a lane-local nibble layout.

    The QJL-only CUDA kernel maps one warp lane `lane in [0,31]` to four
    sketch coordinates:
        lane, lane+32, lane+64, lane+96

    This layout stores those four sign bits in one nibble:
        bit 0 -> sign[lane]
        bit 1 -> sign[lane+32]
        bit 2 -> sign[lane+64]
        bit 3 -> sign[lane+96]

    Two lane nibbles are packed into one byte, so storage remains:
        32 lanes * 4 bits = 128 bits = 16 bytes / key.

    Input:
      signs: [..., 128], values may be {-1,+1}, bool, or numeric sign-like.

    Output:
      packed: [..., 16], uint8
    """
    if signs.shape[-1] != 128:
        raise ValueError(f"Expected signs last dim = 128, got {signs.shape[-1]}.")

    positive = (signs > 0).to(torch.uint8)
    base_shape = signs.shape[:-1]

    lane_ids = torch.arange(32, device=signs.device, dtype=torch.long)
    coord_offsets = torch.tensor([0, 32, 64, 96], device=signs.device, dtype=torch.long)
    gather_idx = lane_ids[:, None] + coord_offsets[None, :]  # [32,4]

    lane_bits = positive[..., gather_idx]  # [...,32,4]
    bit_shifts = torch.arange(4, device=signs.device, dtype=torch.uint8)
    lane_nibbles = torch.sum(
        torch.bitwise_left_shift(lane_bits, bit_shifts),
        dim=-1,
        dtype=torch.int64,
    ).to(torch.uint8)  # [...,32]

    low = lane_nibbles[..., 0::2]
    high = lane_nibbles[..., 1::2]
    packed = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))
    return packed.contiguous()


@torch.no_grad()
def unpack_qjl_signs_lane_nibble(packed: torch.Tensor) -> torch.Tensor:
    """
    Inverse of `pack_qjl_signs_lane_nibble`, returning int8 signs {-1,+1}
    in standard sketch-index order [...,128].
    """
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8.")
    if packed.shape[-1] != 16:
        raise ValueError(f"Expected packed last dim = 16, got {packed.shape[-1]}.")

    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    lane_nibbles = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], 32)

    bit_shifts = torch.arange(4, device=packed.device, dtype=torch.uint8)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(lane_nibbles.unsqueeze(-1), bit_shifts),
        torch.tensor(1, device=packed.device, dtype=torch.uint8),
    )  # [...,32,4]

    out = torch.empty(*packed.shape[:-1], 128, device=packed.device, dtype=torch.int8)
    for i, offset in enumerate((0, 32, 64, 96)):
        out[..., offset:offset + 32] = torch.where(
            bits[..., :, i] > 0,
            torch.ones((), device=packed.device, dtype=torch.int8),
            -torch.ones((), device=packed.device, dtype=torch.int8),
        )
    return out.contiguous()
