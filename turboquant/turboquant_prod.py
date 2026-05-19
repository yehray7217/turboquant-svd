from __future__ import annotations

from dataclasses import dataclass
import torch

from .rotation import rotate, inverse_rotate
from .scalar_quant import scalar_quantize, scalar_dequantize
from .qjl import qjl_encode_residual, qjl_project_query, qjl_residual_logits


@dataclass(frozen=True)
class TurboQuantProdEncoding:
    """
    Reference TurboQuant_prod-style key encoding.

    codes:
      scalar quantizer codes for rotated keys, [B,H,T,D], uint8
    residual_signs:
      QJL signs for original-domain residuals, [B,H,T,M], int8 {-1,+1}
    residual_norms:
      residual L2 norms, [B,H,T], float32
    """
    codes: torch.Tensor
    residual_signs: torch.Tensor
    residual_norms: torch.Tensor


def dense_fp32_logits(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Reference dense qK^T logits, shape [B,H,Q,T]."""
    if queries.ndim != 4 or keys.ndim != 4:
        raise ValueError("queries and keys must be [B,H,Q,D] and [B,H,T,D].")
    if queries.shape[:2] != keys.shape[:2] or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries/keys shape mismatch.")
    return torch.einsum("bhqd,bhtd->bhqt", queries.to(torch.float32), keys.to(torch.float32))


@torch.no_grad()
def encode_turboquant_prod_keys(
    keys: torch.Tensor,
    *,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> TurboQuantProdEncoding:
    """
    Reference encode:
      keys -> rotate -> scalar quantize -> reconstruct -> residual -> QJL.

    The residual is encoded in the original vector domain:
      residual = key - R^T scalar_dequant(R key)
    """
    if keys.ndim != 4:
        raise ValueError("keys must be [B,H,T,D].")

    rotated_keys = rotate(keys, rotation).to(torch.float32)
    codes = scalar_quantize(rotated_keys, centroids)
    reconstructed_rotated = scalar_dequantize(codes, centroids)
    reconstructed_keys = inverse_rotate(reconstructed_rotated, rotation).to(torch.float32)

    residual = keys.to(torch.float32) - reconstructed_keys
    residual_signs, residual_norms = qjl_encode_residual(residual, sketch)

    return TurboQuantProdEncoding(
        codes=codes.contiguous(),
        residual_signs=residual_signs.contiguous(),
        residual_norms=residual_norms.contiguous(),
    )


def turboquant_prod_reference_logits(
    queries: torch.Tensor,
    encoding: TurboQuantProdEncoding,
    *,
    rotation: torch.Tensor,
    centroids: torch.Tensor,
    sketch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TurboQuant_prod-style reference logits:
      scalar rotated-coordinate contribution
      + QJL residual correction.
    """
    if queries.ndim != 4:
        raise ValueError("queries must be [B,H,Q,D].")
    if encoding.codes.ndim != 4:
        raise ValueError("encoding.codes must be [B,H,T,D].")

    rotated_queries = rotate(queries, rotation).to(torch.float32)
    reconstructed_rotated_keys = scalar_dequantize(encoding.codes, centroids)

    scalar_logits = torch.einsum(
        "bhqd,bhtd->bhqt",
        rotated_queries,
        reconstructed_rotated_keys.to(torch.float32),
    )

    query_projected = qjl_project_query(queries, sketch)
    residual_logits = qjl_residual_logits(
        query_projected,
        encoding.residual_signs,
        encoding.residual_norms,
    )
    return scalar_logits + residual_logits
