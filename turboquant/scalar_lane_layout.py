from __future__ import annotations

import torch


@torch.no_grad()
def pack_scalar_codes_lane_word_4bit(codes: torch.Tensor) -> torch.Tensor:
    """
    Repack 4-bit scalar codes into a lane-local 16-bit logical word, stored as
    two uint8 bytes per lane.

    Existing kernel lane mapping:
        one lane consumes coordinates:
          lane, lane+32, lane+64, lane+96

    New logical word layout per lane:
        bits  0.. 3: code[lane]
        bits  4.. 7: code[lane+32]
        bits  8..11: code[lane+64]
        bits 12..15: code[lane+96]

    Physical output remains uint8 and uses:
        32 lanes * 2 bytes = 64 bytes / token / head

    Input:
      codes: [..., 128] uint8 with values in [0,15]

    Output:
      packed: [..., 64] uint8
        packed[..., 2*lane + 0] = low byte
        packed[..., 2*lane + 1] = high byte
    """
    if codes.dtype != torch.uint8:
        raise ValueError("codes must be uint8.")
    if codes.shape[-1] != 128:
        raise ValueError(f"Expected last dim = 128, got {codes.shape[-1]}.")
    if torch.any(codes > 15):
        raise ValueError("4-bit scalar codes must be <= 15.")

    lane_ids = torch.arange(32, device=codes.device, dtype=torch.long)
    offsets = torch.tensor([0, 32, 64, 96], device=codes.device, dtype=torch.long)
    gather_idx = lane_ids[:, None] + offsets[None, :]  # [32,4]

    lane_codes = codes[..., gather_idx].to(torch.int32)  # [...,32,4]
    lane_words = (
        lane_codes[..., 0]
        | (lane_codes[..., 1] << 4)
        | (lane_codes[..., 2] << 8)
        | (lane_codes[..., 3] << 12)
    ).to(torch.int32)  # [...,32]

    low = torch.bitwise_and(lane_words, 0xFF).to(torch.uint8)
    high = torch.bitwise_and(torch.bitwise_right_shift(lane_words, 8), 0xFF).to(torch.uint8)
    return torch.stack([low, high], dim=-1).reshape(*codes.shape[:-1], 64).contiguous()


@torch.no_grad()
def unpack_scalar_codes_lane_word_4bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Inverse of `pack_scalar_codes_lane_word_4bit`, returning [...,128] uint8
    in standard coordinate order.
    """
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8.")
    if packed.shape[-1] != 64:
        raise ValueError(f"Expected last dim = 64, got {packed.shape[-1]}.")

    lane_bytes = packed.reshape(*packed.shape[:-1], 32, 2).to(torch.int32)
    lane_words = lane_bytes[..., 0] | (lane_bytes[..., 1] << 8)

    code0 = torch.bitwise_and(lane_words, 0x0F).to(torch.uint8)
    code1 = torch.bitwise_and(torch.bitwise_right_shift(lane_words, 4), 0x0F).to(torch.uint8)
    code2 = torch.bitwise_and(torch.bitwise_right_shift(lane_words, 8), 0x0F).to(torch.uint8)
    code3 = torch.bitwise_and(torch.bitwise_right_shift(lane_words, 12), 0x0F).to(torch.uint8)

    out = torch.empty(*packed.shape[:-1], 128, device=packed.device, dtype=torch.uint8)
    out[..., 0:32] = code0
    out[..., 32:64] = code1
    out[..., 64:96] = code2
    out[..., 96:128] = code3
    return out.contiguous()
