from __future__ import annotations

import torch


@torch.no_grad()
def make_random_orthogonal_rotation(
    dim: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Construct a reproducible dense orthogonal rotation R in R^{d x d}.

    Row-vector convention used throughout this patch:
      rotate(x)         = x @ R.T
      inverse_rotate(z) = z @ R

    This is a reference implementation. A later fast path can replace this
    with randomized Hadamard or another structured orthogonal transform.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)

    # Normalize column signs for deterministic QR output.
    diag_sign = torch.sign(torch.diagonal(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)

    return q.to(device=device, dtype=dtype).contiguous()


def rotate(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply z = x @ R.T."""
    if x.shape[-1] != rotation.shape[-1]:
        raise ValueError(
            f"Last dim mismatch: x={x.shape[-1]}, rotation={tuple(rotation.shape)}."
        )
    return torch.matmul(x.to(rotation.dtype), rotation.transpose(-2, -1))


def inverse_rotate(z: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply x = z @ R."""
    if z.shape[-1] != rotation.shape[-1]:
        raise ValueError(
            f"Last dim mismatch: z={z.shape[-1]}, rotation={tuple(rotation.shape)}."
        )
    return torch.matmul(z.to(rotation.dtype), rotation)
