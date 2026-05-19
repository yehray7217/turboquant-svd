from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        src = (
            Path(__file__).resolve().parent
            / "csrc"
            / "turboquant_factor_lut_block_geometry_ablation_cuda.cu"
        )
        _EXT = load(
            name="true_turboquant_factor_lut_block_geometry_ablation_cuda_ext",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def _call(name: str, **kwargs) -> torch.Tensor:
    ext = _load_ext()
    fn = getattr(ext, name)
    return fn(
        kwargs["scalar_factor_lut"].contiguous().to(torch.float32),
        kwargs["lane_word_scalar_codes"].contiguous(),
        kwargs["qjl_projected_queries"].contiguous().to(torch.float32),
        kwargs["lane_nibble_qjl_signs"].contiguous(),
        kwargs["residual_norms"].contiguous().to(torch.float32),
    )


@torch.no_grad()
def turboquant_factor_lut_combined_geometry_wpb4_cuda(
    *,
    scalar_factor_lut: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _call(
        "turboquant_factor_lut_combined_geometry_wpb4_cuda",
        scalar_factor_lut=scalar_factor_lut,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
    )


@torch.no_grad()
def turboquant_factor_lut_combined_geometry_wpb8_cuda(
    *,
    scalar_factor_lut: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _call(
        "turboquant_factor_lut_combined_geometry_wpb8_cuda",
        scalar_factor_lut=scalar_factor_lut,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
    )


@torch.no_grad()
def turboquant_factor_lut_combined_geometry_wpb16_cuda(
    *,
    scalar_factor_lut: torch.Tensor,
    lane_word_scalar_codes: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    lane_nibble_qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
) -> torch.Tensor:
    return _call(
        "turboquant_factor_lut_combined_geometry_wpb16_cuda",
        scalar_factor_lut=scalar_factor_lut,
        lane_word_scalar_codes=lane_word_scalar_codes,
        qjl_projected_queries=qjl_projected_queries,
        lane_nibble_qjl_signs=lane_nibble_qjl_signs,
        residual_norms=residual_norms,
    )
