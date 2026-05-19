from __future__ import annotations

import torch


@torch.no_grad()
def fit_lloyd_scalar_codebook(
    samples: torch.Tensor,
    *,
    num_levels: int,
    max_iters: int = 40,
    max_samples: int = 1_000_000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Fit a 1D scalar codebook with Lloyd-style centroid updates.

    TurboQuant's paper uses scalar quantizers after random rotation. This
    function provides a practical reference codebook fitter for the new repo.
    It is intentionally isolated so an analytic Beta-distribution codebook can
    replace it later without changing the encoder/logits API.
    """
    if num_levels <= 1:
        raise ValueError(f"num_levels must exceed 1, got {num_levels}.")
    if samples.numel() == 0:
        raise ValueError("samples must be non-empty.")

    x = samples.detach().to(torch.float32).reshape(-1)
    finite = torch.isfinite(x)
    x = x[finite]
    if x.numel() == 0:
        raise ValueError("samples contain no finite values.")

    if x.numel() > max_samples:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        idx = torch.randperm(x.numel(), generator=gen, device="cpu")[:max_samples]
        x = x.cpu()[idx].to(samples.device)
    else:
        x = x.to(samples.device)

    quantiles = torch.linspace(
        0.5 / num_levels,
        1.0 - 0.5 / num_levels,
        num_levels,
        device=x.device,
        dtype=torch.float32,
    )
    centroids = torch.quantile(x, quantiles).contiguous()

    for _ in range(int(max_iters)):
        distances = torch.abs(x.unsqueeze(-1) - centroids.view(1, -1))
        assignment = torch.argmin(distances, dim=-1)

        updated = centroids.clone()
        for k in range(num_levels):
            mask = assignment == k
            if torch.any(mask):
                updated[k] = x[mask].mean()

        updated, _ = torch.sort(updated)
        if torch.max(torch.abs(updated - centroids)).item() < 1e-7:
            centroids = updated
            break
        centroids = updated

    return centroids.contiguous()


@torch.no_grad()
def scalar_quantize(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Nearest-centroid scalar quantization, returning uint8 codes."""
    if centroids.ndim != 1:
        raise ValueError("centroids must be 1D.")
    if centroids.numel() > 256:
        raise ValueError("reference uint8 codes support at most 256 centroids.")

    distances = torch.abs(
        x.to(torch.float32).unsqueeze(-1)
        - centroids.to(device=x.device, dtype=torch.float32).view(
            *([1] * x.ndim), -1
        )
    )
    return torch.argmin(distances, dim=-1).to(torch.uint8).contiguous()


def scalar_dequantize(
    codes: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Decode uint8 scalar codes back to centroid values."""
    if centroids.ndim != 1:
        raise ValueError("centroids must be 1D.")
    return centroids.to(device=codes.device, dtype=torch.float32)[codes.long()]
