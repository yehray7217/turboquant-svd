from __future__ import annotations

"""
Scalar-codebook utilities for the local TurboQuant benchmark stack.

API restored:
  - fit_lloyd_scalar_codebook(...)
  - scalar_quantize(...)
  - scalar_dequantize(...)

Important project convention:
  The production TurboQuant factor-LUT path uses ONE shared scalar centroid
  codebook with shape [K], e.g. [16] for 4-bit scalar codes.

This file therefore returns [K] from fit_lloyd_scalar_codebook(), while
scalar_quantize() also tolerates legacy per-dimension [D, K] centroids for
debugging compatibility.

The quantization path is chunked to avoid allocating a single giant
[N, D, K] distance tensor at long sequence lengths.
"""

import os
from typing import Any, Optional

import torch


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError("expected a torch.Tensor")
    if not x.is_floating_point():
        x = x.float()
    return x


def _resolve_num_levels(
    *,
    num_levels: Optional[int] = None,
    n_levels: Optional[int] = None,
    levels: Optional[int] = None,
    bits: Optional[int] = None,
    kwargs: Optional[dict[str, Any]] = None,
) -> int:
    if num_levels is not None:
        k = int(num_levels)
    elif n_levels is not None:
        k = int(n_levels)
    elif levels is not None:
        k = int(levels)
    elif bits is not None:
        k = 1 << int(bits)
    elif kwargs and kwargs.get("num_centroids") is not None:
        k = int(kwargs["num_centroids"])
    elif kwargs and kwargs.get("k") is not None:
        k = int(kwargs["k"])
    else:
        k = 16
    if k <= 0:
        raise ValueError(f"num_levels must be positive, got {k}")
    return k


def _resolve_iters(
    *,
    iters: Optional[int] = None,
    num_iters: Optional[int] = None,
    lloyd_iters: Optional[int] = None,
    kwargs: Optional[dict[str, Any]] = None,
) -> int:
    if lloyd_iters is not None:
        return max(0, int(lloyd_iters))
    if num_iters is not None:
        return max(0, int(num_iters))
    if iters is not None:
        return max(0, int(iters))
    if kwargs:
        for name in ("n_iters", "iterations", "max_iters"):
            if kwargs.get(name) is not None:
                return max(0, int(kwargs[name]))
    return 10


def _subsample_flat(
    flat: torch.Tensor,
    max_samples: Optional[int],
    seed: Optional[int],
) -> torch.Tensor:
    if max_samples is None:
        return flat
    max_samples = int(max_samples)
    if max_samples <= 0 or flat.numel() <= max_samples:
        return flat

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0 if seed is None else int(seed))
    idx = torch.randperm(int(flat.numel()), generator=gen, device="cpu")[:max_samples]
    return flat.index_select(0, idx.to(flat.device))


@torch.no_grad()
def fit_lloyd_scalar_codebook(
    samples: torch.Tensor,
    num_levels: Optional[int] = None,
    *,
    n_levels: Optional[int] = None,
    levels: Optional[int] = None,
    bits: Optional[int] = None,
    iters: Optional[int] = None,
    num_iters: Optional[int] = None,
    lloyd_iters: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: Optional[int] = 0,
    eps: float = 1e-12,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Fit a shared 1D Lloyd scalar codebook.

    Returns:
      centroids: [K]

    All values from `samples` are flattened into one scalar sample pool. This
    matches the factor-LUT path, which expects centroids with shape [16].
    """
    samples = _as_float_tensor(samples)
    flat = samples.reshape(-1).float().contiguous()
    flat = _subsample_flat(flat, max_samples=max_samples, seed=seed)

    if flat.numel() == 0:
        raise ValueError("fit_lloyd_scalar_codebook: empty sample set")

    k = _resolve_num_levels(
        num_levels=num_levels,
        n_levels=n_levels,
        levels=levels,
        bits=bits,
        kwargs=kwargs,
    )
    n_iter = _resolve_iters(
        iters=iters,
        num_iters=num_iters,
        lloyd_iters=lloyd_iters,
        kwargs=kwargs,
    )

    q = (torch.arange(k, device=flat.device, dtype=torch.float32) + 0.5) / float(k)
    centroids = torch.quantile(flat, q).contiguous()  # [K]
    centroids, _ = torch.sort(centroids)

    fit_chunk = max(1, int(os.environ.get("TURBOQUANT_LLOYD_CHUNK", "1048576")))

    for _ in range(n_iter):
        sums = torch.zeros((k,), device=flat.device, dtype=torch.float32)
        counts = torch.zeros((k,), device=flat.device, dtype=torch.float32)

        for begin in range(0, int(flat.numel()), fit_chunk):
            x = flat[begin : begin + fit_chunk]                 # [C]
            dist = torch.abs(x.unsqueeze(-1) - centroids)       # [C, K]
            codes = torch.argmin(dist, dim=-1)                  # [C]
            sums.scatter_add_(0, codes, x)
            counts.scatter_add_(0, codes, torch.ones_like(x))

        updated = sums / counts.clamp_min(float(eps))
        centroids = torch.where(counts > 0, updated, centroids)
        centroids, _ = torch.sort(centroids)

    return centroids.contiguous()


fit_scalar_codebook = fit_lloyd_scalar_codebook
lloyd_scalar_codebook = fit_lloyd_scalar_codebook


def _centroid_layout(values: torch.Tensor, centroids: torch.Tensor) -> tuple[str, torch.Tensor]:
    """
    Return:
      ("shared", [1, 1, K]) for centroids [K]
      ("per_dim", [1, D, K]) for centroids [D, K] or [K, D]
    """
    values = _as_float_tensor(values)
    centroids = _as_float_tensor(centroids).to(values.device)
    d = int(values.shape[-1])

    if centroids.dim() == 1:
        return "shared", centroids.view(1, 1, -1)

    if centroids.dim() == 2:
        if int(centroids.shape[0]) == d:
            return "per_dim", centroids.unsqueeze(0)
        if int(centroids.shape[1]) == d:
            return "per_dim", centroids.t().contiguous().unsqueeze(0)

    raise ValueError(
        f"unsupported centroids shape={tuple(centroids.shape)} for values last dim D={d}"
    )


@torch.no_grad()
def scalar_quantize(
    values: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Nearest-centroid scalar quantization.

    Shared-codebook case [K]:
      codes[..., d] = argmin_k |values[..., d] - centroids[k]|

    Per-dim debug case [D, K]:
      codes[..., d] = argmin_k |values[..., d] - centroids[d, k]|

    Returns uint8 codes with the same shape as `values`.
    """
    values = _as_float_tensor(values)
    d = int(values.shape[-1])
    flat = values.reshape(-1, d)
    _, cent = _centroid_layout(values, centroids)

    if chunk_size is None:
        chunk_size = int(os.environ.get("TURBOQUANT_SCALAR_QUANT_CHUNK", "4096"))
    chunk_size = max(1, int(chunk_size))

    chunks = []
    for begin in range(0, int(flat.shape[0]), chunk_size):
        x = flat[begin : begin + chunk_size]                     # [C, D]
        dist = torch.abs(x.unsqueeze(-1) - cent)                 # [C, D, K]
        codes = torch.argmin(dist, dim=-1).to(torch.uint8)       # [C, D]
        chunks.append(codes)

    return torch.cat(chunks, dim=0).reshape(values.shape).contiguous()


@torch.no_grad()
def scalar_dequantize(codes: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Dequantize scalar codes.

    Supports:
      - shared centroids [K]
      - per-dim centroids [D, K] / [K, D]
    """
    if not torch.is_tensor(codes):
        raise TypeError("codes must be a torch.Tensor")

    d = int(codes.shape[-1])
    dummy = torch.empty((*codes.shape[:-1], d), device=codes.device, dtype=torch.float32)
    layout, cent = _centroid_layout(dummy, centroids)

    flat_codes = codes.reshape(-1, d).to(torch.long)

    if layout == "shared":
        table = cent.reshape(-1)  # [K]
        out = table[flat_codes]
    else:
        table = cent.squeeze(0)   # [D, K]
        dim_idx = torch.arange(d, device=codes.device, dtype=torch.long).view(1, d).expand_as(flat_codes)
        out = table[dim_idx, flat_codes]

    return out.reshape(codes.shape).contiguous()


dequantize_scalar_codes = scalar_dequantize


__all__ = [
    "fit_lloyd_scalar_codebook",
    "fit_scalar_codebook",
    "lloyd_scalar_codebook",
    "scalar_quantize",
    "scalar_dequantize",
    "dequantize_scalar_codes",
]
