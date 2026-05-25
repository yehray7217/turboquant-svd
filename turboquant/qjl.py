from __future__ import annotations

import math
import torch


@torch.no_grad()
def make_gaussian_sketch(
    dim: int,
    sketch_dim: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return S in R^{m x d} with iid N(0,1) entries.

    For this convention, the unbiased Monte Carlo inner-product residual
    estimator uses:
      sqrt(pi/2) / m * ||r|| * <S q, sign(S r)>.
    """
    if dim <= 0 or sketch_dim <= 0:
        raise ValueError(f"dim and sketch_dim must be positive, got {dim}, {sketch_dim}.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    s = torch.randn(sketch_dim, dim, generator=gen, dtype=torch.float32)
    return s.to(device=device, dtype=dtype).contiguous()


@torch.no_grad()
def qjl_encode_residual(
    residual: torch.Tensor,
    sketch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode residuals.

    Returns:
      signs: int8 tensor with values in {-1, +1}, shape [..., m]
      norms: float32 tensor, shape [...]
    """
    if residual.shape[-1] != sketch.shape[-1]:
        raise ValueError(
            f"Residual dim {residual.shape[-1]} != sketch dim {sketch.shape[-1]}."
        )

    residual_fp32 = residual.to(torch.float32)
    projected = torch.matmul(residual_fp32, sketch.to(residual.device).transpose(-2, -1))
    signs = torch.where(
        projected >= 0,
        torch.ones_like(projected, dtype=torch.int8),
        -torch.ones_like(projected, dtype=torch.int8),
    ).contiguous()
    norms = torch.linalg.vector_norm(residual_fp32, dim=-1).contiguous()
    return signs, norms


def qjl_project_query(
    query: torch.Tensor,
    sketch: torch.Tensor,
) -> torch.Tensor:
    """Compute S q for row-vector queries, shape [..., m]."""
    if query.shape[-1] != sketch.shape[-1]:
        raise ValueError(
            f"Query dim {query.shape[-1]} != sketch dim {sketch.shape[-1]}."
        )
    return torch.matmul(
        query.to(torch.float32),
        sketch.to(device=query.device, dtype=torch.float32).transpose(-2, -1),
    )


def qjl_residual_logits(
    query_projected: torch.Tensor,
    residual_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate query-residual inner products.

    Expected shapes:
      query_projected: [B,H,Q,M]
      residual_signs:  [B,H,T,M]
      residual_norms:  [B,H,T]

    Output:
      correction: [B,H,Q,T]
    """
    if query_projected.ndim != 4 or residual_signs.ndim != 4 or residual_norms.ndim != 3:
        raise ValueError("Expected [B,H,Q,M], [B,H,T,M], [B,H,T].")
    if query_projected.shape[-1] != residual_signs.shape[-1]:
        raise ValueError("QJL sketch dimension mismatch.")
    if query_projected.shape[:2] != residual_signs.shape[:2]:
        raise ValueError("Batch/head mismatch between query and residual signs.")
    if residual_signs.shape[:3] != residual_norms.shape:
        raise ValueError("Residual sign/norm shape mismatch.")

    sketch_dim = query_projected.shape[-1]
    scale = math.sqrt(math.pi / 2.0) / float(sketch_dim)
    signed_dot = torch.einsum(
        "bhqm,bhtm->bhqt",
        query_projected.to(torch.float32),
        residual_signs.to(torch.float32),
    )
    return scale * signed_dot * residual_norms.to(torch.float32).unsqueeze(-2)


def make_rademacher_sketch(
    dim: int,
    sketch_dim: int,
    *,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Rademacher sketch with entries ±1/sqrt(sketch_dim).

    Shape follows make_gaussian_sketch: [sketch_dim, dim].
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(sketch_dim), int(dim)),
        generator=gen,
        device=device,
        dtype=torch.int8,
    )
    sketch = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    sketch = sketch / (float(sketch_dim) ** 0.5)
    return sketch.contiguous()
