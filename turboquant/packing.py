from __future__ import annotations

import torch


@torch.no_grad()
def pack_scalar_codes_4bit(codes: torch.Tensor) -> torch.Tensor:
    """
    Pack scalar quantizer codes [0,15] pairwise into uint8 nibbles.

    Input:
      codes: [..., D] uint8, D even
    Output:
      packed: [..., D/2] uint8
    """
    if codes.dtype != torch.uint8:
        raise ValueError("codes must be uint8.")
    if codes.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be even for 4-bit packing.")
    if torch.any(codes > 15):
        raise ValueError("4-bit codes must be <= 15.")

    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return torch.bitwise_or(low, torch.bitwise_left_shift(high, 4)).contiguous()


@torch.no_grad()
def unpack_scalar_codes_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_scalar_codes_4bit`."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8.")
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@torch.no_grad()
def pack_qjl_signs_1bit(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack QJL signs {-1,+1} or boolean-like signs into bits.

    Bit convention:
      bit = 1 -> positive sign
      bit = 0 -> negative sign

    Input:
      signs: [..., M], M divisible by 8
    Output:
      packed: [..., M/8] uint8
    """
    if signs.shape[-1] % 8 != 0:
        raise ValueError("Last dimension must be divisible by 8 for bit packing.")

    positive = signs > 0
    positive_u8 = positive.to(torch.uint8)
    reshaped = positive_u8.reshape(*signs.shape[:-1], signs.shape[-1] // 8, 8)
    shifts = torch.arange(8, device=signs.device, dtype=torch.uint8)
    packed = torch.sum(
        torch.bitwise_left_shift(reshaped, shifts),
        dim=-1,
        dtype=torch.int64,
    ).to(torch.uint8)
    return packed.contiguous()


@torch.no_grad()
def unpack_qjl_signs_1bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack bits to int8 signs {-1,+1}.
    """
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8.")
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts),
        torch.tensor(1, device=packed.device, dtype=torch.uint8),
    )
    signs = torch.where(
        bits.reshape(*packed.shape[:-1], packed.shape[-1] * 8) > 0,
        torch.ones((), device=packed.device, dtype=torch.int8),
        -torch.ones((), device=packed.device, dtype=torch.int8),
    )
    return signs.contiguous()
