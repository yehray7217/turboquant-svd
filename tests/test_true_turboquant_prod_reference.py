from __future__ import annotations

import torch

from turboquant import (
    make_random_orthogonal_rotation,
    fit_lloyd_scalar_codebook,
    make_gaussian_sketch,
    encode_turboquant_prod_keys,
    turboquant_prod_reference_logits,
    dense_fp32_logits,
)


def test_true_turboquant_prod_reference_shapes_and_finiteness() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B, H, Q, T, D, M = 1, 2, 1, 64, 16, 16
    q = torch.randn(B, H, Q, D, device=device)
    k = torch.randn(B, H, T, D, device=device)

    rotation = make_random_orthogonal_rotation(D, seed=7, device=device)
    sketch = make_gaussian_sketch(D, M, seed=11, device=device)

    calib_rotated = torch.matmul(
        torch.randn(4096, D, device=device),
        rotation.T,
    )
    centroids = fit_lloyd_scalar_codebook(
        calib_rotated,
        num_levels=8,
        max_iters=8,
        max_samples=100_000,
        seed=13,
    )

    enc = encode_turboquant_prod_keys(
        k,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )
    approx = turboquant_prod_reference_logits(
        q,
        enc,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )
    dense = dense_fp32_logits(q, k)

    assert approx.shape == (B, H, Q, T)
    assert dense.shape == (B, H, Q, T)
    assert enc.codes.shape == (B, H, T, D)
    assert enc.residual_signs.shape == (B, H, T, M)
    assert enc.residual_norms.shape == (B, H, T)
    assert torch.isfinite(approx).all()
    assert torch.isfinite(dense).all()
