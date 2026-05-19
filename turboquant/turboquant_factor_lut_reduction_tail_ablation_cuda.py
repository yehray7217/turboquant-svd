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
            / "turboquant_factor_lut_reduction_tail_ablation_cuda.cu"
        )
        _EXT = load(
            name="true_turboquant_factor_lut_reduction_tail_ablation_cuda_ext",
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
def turboquant_factor_lut_combined_reduction_tail_down_cuda(**kwargs) -> torch.Tensor:
    return _call("turboquant_factor_lut_combined_reduction_tail_down_cuda", **kwargs)

@torch.no_grad()
def turboquant_factor_lut_combined_reduction_tail_xor_cuda(**kwargs) -> torch.Tensor:
    return _call("turboquant_factor_lut_combined_reduction_tail_xor_cuda", **kwargs)

@torch.no_grad()
def turboquant_factor_lut_combined_reduction_tail_halfwidth_cuda(**kwargs) -> torch.Tensor:
    return _call("turboquant_factor_lut_combined_reduction_tail_halfwidth_cuda", **kwargs)
