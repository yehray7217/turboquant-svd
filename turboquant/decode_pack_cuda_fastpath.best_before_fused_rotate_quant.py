from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

import torch
from torch.utils.cpp_extension import load_inline


_CPP_SRC = r'''
#include <torch/extension.h>
#include <vector>

torch::Tensor pack_scalar_codes_4bit_cuda(torch::Tensor codes, int64_t mode);
torch::Tensor pack_qjl_signs_1bit_cuda(torch::Tensor signs, int64_t mode);
torch::Tensor scalar_quantize_16_cuda(torch::Tensor values, torch::Tensor centroids);

std::vector<torch::Tensor> fused_dequant_residual_qjl_cuda(
    torch::Tensor new_keys,
    torch::Tensor codes,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_scalar_codes_4bit_cuda", &pack_scalar_codes_4bit_cuda,
        "Pack scalar 4-bit codes [*,128] -> [*,64] (CUDA)");
  m.def("pack_qjl_signs_1bit_cuda", &pack_qjl_signs_1bit_cuda,
        "Pack QJL sign bits [*,128] -> [*,16] (CUDA)");
  m.def("scalar_quantize_16_cuda", &scalar_quantize_16_cuda,
        "Scalar quantize shared 16-centroid codebook [*,128] -> uint8 [*,128] (CUDA)");
  m.def("fused_dequant_residual_qjl_cuda", &fused_dequant_residual_qjl_cuda,
        "Fused scalar dequant + inverse rotate + residual + QJL encode [*,128] (CUDA)");
}
'''

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename scalar_t>
__global__ void pack_scalar_codes_4bit_kernel(
    const scalar_t* __restrict__ codes,
    uint8_t* __restrict__ out,
    int64_t outer,
    int64_t mode) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = outer * 64;
  if (idx >= total) return;

  const int64_t row = idx / 64;
  const int64_t j = idx % 64;

  // Match turboquant.scalar_lane_layout.pack_scalar_codes_lane_word_4bit:
  //
  // j=0  -> (0,16)
  // j=1  -> (64,80)
  // j=2  -> (1,17)
  // j=3  -> (65,81)
  // ...
  // j=32 -> (32,48)
  // j=33 -> (96,112)
  const int64_t group = j / 32;          // 0 or 1
  const int64_t within = j % 32;
  const int64_t i = within / 2;          // 0..15
  const int64_t half = within % 2;       // 0 or 1

  const int64_t lo_local = group * 32 + i + half * 64;
  const int64_t hi_local = lo_local + 16;

  const int64_t lo_idx = row * 128 + lo_local;
  const int64_t hi_idx = row * 128 + hi_local;

  const uint8_t a = static_cast<uint8_t>(codes[lo_idx] & static_cast<scalar_t>(0xF));
  const uint8_t b = static_cast<uint8_t>(codes[hi_idx] & static_cast<scalar_t>(0xF));

  if (mode == 0) {
    out[idx] = static_cast<uint8_t>(a | (b << 4));
  } else {
    out[idx] = static_cast<uint8_t>((a << 4) | b);
  }
}

template <typename scalar_t>
__device__ __forceinline__ uint8_t sign_to_bit(scalar_t v, int64_t sign_mode) {
  // sign_mode:
  // 0: nonzero -> 1
  // 1: value > 0 -> 1
  // 2: value < 0 -> 1
  if (sign_mode == 0) {
    return static_cast<uint8_t>(v != static_cast<scalar_t>(0));
  } else if (sign_mode == 1) {
    return static_cast<uint8_t>(v > static_cast<scalar_t>(0));
  } else {
    return static_cast<uint8_t>(v < static_cast<scalar_t>(0));
  }
}

template <typename scalar_t>
__global__ void pack_qjl_signs_1bit_kernel(
    const scalar_t* __restrict__ signs,
    uint8_t* __restrict__ out,
    int64_t outer,
    int64_t mode) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = outer * 16;
  if (idx >= total) return;

  const int64_t row = idx / 16;
  const int64_t j = idx % 16;

  // Match turboquant.qjl_sign_layout.pack_qjl_signs_lane_nibble:
  //
  // byte j packs:
  // bit0 <- signs[  2j]
  // bit4 <- signs[  2j+1]
  // bit1 <- signs[ 32+2j]
  // bit5 <- signs[ 32+2j+1]
  // bit2 <- signs[ 64+2j]
  // bit6 <- signs[ 64+2j+1]
  // bit3 <- signs[ 96+2j]
  // bit7 <- signs[ 96+2j+1]
  //
  // The reference packer accepts bool signs, so sign_mode=0
  // (nonzero -> 1) is the matching convention.
  const int64_t base = row * 128;
  const int64_t even = 2 * j;
  const int64_t odd  = 2 * j + 1;

  uint8_t packed = 0;

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + even], 0) << 0);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + odd], 0) << 4);

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 32 + even], 0) << 1);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 32 + odd], 0) << 5);

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 64 + even], 0) << 2);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 64 + odd], 0) << 6);

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 96 + even], 0) << 3);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 96 + odd], 0) << 7);

  out[idx] = packed;
}



template <typename value_t>
__global__ void scalar_quantize_16_kernel(
    const value_t* __restrict__ values,      // [outer,128]
    const float* __restrict__ centroids,     // [16]
    uint8_t* __restrict__ codes,             // [outer,128]
    int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const float x = static_cast<float>(values[idx]);

  float best_dist = fabsf(x - centroids[0]);
  int best = 0;

  #pragma unroll
  for (int k = 1; k < 16; ++k) {
    const float dist = fabsf(x - centroids[k]);
    if (dist < best_dist) {
      best_dist = dist;
      best = k;
    }
  }

  codes[idx] = static_cast<uint8_t>(best);
}

torch::Tensor scalar_quantize_16_cuda(torch::Tensor values, torch::Tensor centroids) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(values.size(-1) == 128, "values last dim must be 128");
  TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids must be float32");
  TORCH_CHECK(centroids.numel() == 16, "centroids must have 16 entries");

  auto out = torch::empty(values.sizes(), values.options().dtype(torch::kUInt8));

  const int64_t total = values.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "scalar_quantize_16_cuda", [&] {
    scalar_quantize_16_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        values.data_ptr<scalar_t>(),
        centroids.data_ptr<float>(),
        out.data_ptr<uint8_t>(),
        total);
  });

  return out;
}


template <typename key_t>
__global__ void fused_dequant_residual_qjl_kernel(
    const key_t* __restrict__ new_keys,      // [outer,128]
    const uint8_t* __restrict__ codes,       // [outer,128]
    const float* __restrict__ centroids,     // [16]
    const float* __restrict__ rotation,      // [128,128], row-major, inverse_rotate: z @ R
    const float* __restrict__ sketch,        // [128,128], row-major, projected[m] = sum_d residual[d] * sketch[m,d]
    int8_t* __restrict__ signs,              // [outer,128]
    float* __restrict__ norms,               // [outer]
    int64_t outer) {
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

  if (row >= outer || tid >= 128) return;

  __shared__ float residual[128];
  __shared__ float reduce_buf[128];

  // reconstructed_key[d] = sum_j centroids[codes[j]] * rotation[j,d]
  float recon_d = 0.0f;
  #pragma unroll
  for (int j = 0; j < 128; ++j) {
    const uint8_t c = static_cast<uint8_t>(codes[row * 128 + j] & 0x0F);
    const float z_j = centroids[static_cast<int>(c)];
    recon_d += z_j * rotation[j * 128 + tid];
  }

  const float x_d = static_cast<float>(new_keys[row * 128 + tid]);
  const float r_d = x_d - recon_d;

  residual[tid] = r_d;
  reduce_buf[tid] = r_d * r_d;
  __syncthreads();

  // Reduce residual norm.
  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) {
      reduce_buf[tid] += reduce_buf[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    norms[row] = sqrtf(reduce_buf[0]);
  }

  // Each thread computes one QJL projection m=tid.
  float dot = 0.0f;
  #pragma unroll
  for (int d = 0; d < 128; ++d) {
    dot += residual[d] * sketch[tid * 128 + d];
  }

  signs[row * 128 + tid] = (dot >= 0.0f) ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);
}

std::vector<torch::Tensor> fused_dequant_residual_qjl_cuda(
    torch::Tensor new_keys,
    torch::Tensor codes,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch) {
  TORCH_CHECK(new_keys.is_cuda(), "new_keys must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
  TORCH_CHECK(rotation.is_cuda(), "rotation must be CUDA");
  TORCH_CHECK(sketch.is_cuda(), "sketch must be CUDA");

  TORCH_CHECK(new_keys.is_contiguous(), "new_keys must be contiguous");
  TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(rotation.is_contiguous(), "rotation must be contiguous");
  TORCH_CHECK(sketch.is_contiguous(), "sketch must be contiguous");

  TORCH_CHECK(new_keys.size(-1) == 128, "new_keys last dim must be 128");
  TORCH_CHECK(codes.size(-1) == 128, "codes last dim must be 128");
  TORCH_CHECK(codes.scalar_type() == at::kByte, "codes must be torch.uint8");
  TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids must be float32");
  TORCH_CHECK(rotation.scalar_type() == at::kFloat, "rotation must be float32");
  TORCH_CHECK(sketch.scalar_type() == at::kFloat, "sketch must be float32");
  TORCH_CHECK(centroids.numel() == 16, "centroids must have 16 entries");
  TORCH_CHECK(rotation.numel() == 128 * 128, "rotation must be [128,128]");
  TORCH_CHECK(sketch.numel() == 128 * 128, "sketch must be [128,128]");

  auto sign_sizes = new_keys.sizes().vec();
  auto norm_sizes = new_keys.sizes().vec();
  norm_sizes.pop_back();

  auto signs = torch::empty(sign_sizes, new_keys.options().dtype(torch::kInt8));
  auto norms = torch::empty(norm_sizes, new_keys.options().dtype(torch::kFloat32));

  const int64_t outer = new_keys.numel() / 128;
  const dim3 blocks(static_cast<unsigned int>(outer));
  const dim3 threads(128);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(new_keys.scalar_type(), "fused_dequant_residual_qjl_cuda", [&] {
    fused_dequant_residual_qjl_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        new_keys.data_ptr<scalar_t>(),
        codes.data_ptr<uint8_t>(),
        centroids.data_ptr<float>(),
        rotation.data_ptr<float>(),
        sketch.data_ptr<float>(),
        signs.data_ptr<int8_t>(),
        norms.data_ptr<float>(),
        outer);
  });

  return {signs, norms};
}


torch::Tensor pack_scalar_codes_4bit_cuda(torch::Tensor codes, int64_t mode) {
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
  TORCH_CHECK(codes.size(-1) == 128, "codes last dim must be 128");
  TORCH_CHECK(mode == 0 || mode == 1, "scalar pack mode must be 0 or 1");

  auto out_sizes = codes.sizes().vec();
  out_sizes.back() = 64;
  auto out = torch::empty(out_sizes, codes.options().dtype(torch::kUInt8));

  const int64_t outer = codes.numel() / 128;
  const int64_t total = outer * 64;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_INTEGRAL_TYPES(codes.scalar_type(), "pack_scalar_codes_4bit_cuda", [&] {
    pack_scalar_codes_4bit_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        codes.data_ptr<scalar_t>(),
        out.data_ptr<uint8_t>(),
        outer,
        mode);
  });

  return out;
}

torch::Tensor pack_qjl_signs_1bit_cuda(torch::Tensor signs, int64_t mode) {
  TORCH_CHECK(signs.is_cuda(), "signs must be CUDA");
  TORCH_CHECK(signs.is_contiguous(), "signs must be contiguous");
  TORCH_CHECK(signs.size(-1) == 128, "signs last dim must be 128");
  TORCH_CHECK(mode >= 0 && mode <= 5, "qjl pack mode must be in [0,5]");

  auto out_sizes = signs.sizes().vec();
  out_sizes.back() = 16;
  auto out = torch::empty(out_sizes, signs.options().dtype(torch::kUInt8));

  const int64_t outer = signs.numel() / 128;
  const int64_t total = outer * 16;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool,
                             signs.scalar_type(), "pack_qjl_signs_1bit_cuda", [&] {
    pack_qjl_signs_1bit_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        signs.data_ptr<scalar_t>(),
        out.data_ptr<uint8_t>(),
        outer,
        mode);
  });

  return out;
}
'''


@lru_cache(maxsize=1)
def _ext():
    return load_inline(
        name="turboquant_decode_pack_cuda_fastpath_ext_fused_qjl_quant_v1",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("TURBOQUANT_PACK_CUDA_VERBOSE", "0"))),
    )


@torch.no_grad()
def scalar_quantize_16_cuda(
    values: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA fastpath for scalar_quantize(values, centroids) when:
      values last dim = 128
      shared centroids shape = [16]
    Returns uint8 codes with same shape as values.
    """
    return _ext().scalar_quantize_16_cuda(
        values.contiguous(),
        centroids.contiguous(),
    )


@torch.no_grad()
def fused_dequant_residual_qjl_cuda(
    new_keys: torch.Tensor,
    codes: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sketch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused replacement for:
      scalar_dequantize(codes, centroids)
      inverse_rotate(..., rotation)
      residual = new_keys - reconstructed
      qjl_encode_residual(residual, sketch)

    Returns:
      residual_signs_raw: int8 [*,128] with {-1,+1}
      residual_norms_raw: float32 [*]
    """
    return tuple(
        _ext().fused_dequant_residual_qjl_cuda(
            new_keys.contiguous(),
            codes.contiguous(),
            centroids.contiguous(),
            rotation.contiguous(),
            sketch.contiguous(),
        )
    )


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(a, b)


@torch.no_grad()
def _infer_scalar_mode(
    ref_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[int, torch.dtype]:
    codes = (torch.arange(128, device=device, dtype=torch.uint8) % 16).view(1, 1, 1, 128).contiguous()
    ref = ref_fn(codes)
    ext = _ext()

    for mode in (0, 1):
        out = ext.pack_scalar_codes_4bit_cuda(codes.contiguous(), mode)
        if out.dtype != ref.dtype:
            out = out.to(ref.dtype)
        if _same_tensor(out, ref):
            return mode, ref.dtype

    raise RuntimeError(
        "Could not infer scalar packing mode from the repo reference packer. "
        f"reference shape={tuple(ref.shape)} dtype={ref.dtype}"
    )


@torch.no_grad()
def _infer_qjl_mode(
    ref_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[int, torch.dtype, torch.dtype]:
    patterns = [
        torch.tensor([False, True, False, True, True, False, True, False] * 16,
                     device=device, dtype=torch.bool).view(1, 1, 1, 128),
        torch.tensor([0, 1, 0, 1, 1, 0, 1, 0] * 16,
                     device=device, dtype=torch.uint8).view(1, 1, 1, 128),
        torch.tensor([-1, 1, -1, 1, 1, -1, 1, -1] * 16,
                     device=device, dtype=torch.int8).view(1, 1, 1, 128),
        torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0] * 16,
                     device=device, dtype=torch.float32).view(1, 1, 1, 128),
    ]

    ext = _ext()
    errors: list[str] = []

    for signs in patterns:
        try:
            ref = ref_fn(signs)
        except Exception as exc:
            errors.append(f"ref rejected dtype={signs.dtype}: {type(exc).__name__}: {exc}")
            continue

        for mode in range(6):
            try:
                out = ext.pack_qjl_signs_1bit_cuda(signs.contiguous(), mode)
                if out.dtype != ref.dtype:
                    out = out.to(ref.dtype)
                if _same_tensor(out, ref):
                    return mode, ref.dtype, signs.dtype
            except Exception as exc:
                errors.append(
                    f"candidate failed dtype={signs.dtype} mode={mode}: "
                    f"{type(exc).__name__}: {exc}"
                )

    raise RuntimeError(
        "Could not infer QJL packing mode from the repo reference packer. "
        + " | ".join(errors[:12])
    )


class DecodePackCudaFastPath:
    def __init__(
        self,
        *,
        scalar_ref_fn: Callable[[torch.Tensor], torch.Tensor],
        qjl_ref_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("DecodePackCudaFastPath requires a CUDA device.")

        self.scalar_mode, self.scalar_out_dtype = _infer_scalar_mode(scalar_ref_fn, device)
        self.qjl_mode, self.qjl_out_dtype, self.qjl_validation_input_dtype = _infer_qjl_mode(qjl_ref_fn, device)
        self.extension = _ext()

    @torch.no_grad()
    def pack_scalar(self, codes: torch.Tensor) -> torch.Tensor:
        out = self.extension.pack_scalar_codes_4bit_cuda(codes.contiguous(), int(self.scalar_mode))
        if out.dtype != self.scalar_out_dtype:
            out = out.to(self.scalar_out_dtype)
        return out.contiguous()

    @torch.no_grad()
    def pack_qjl(self, signs: torch.Tensor) -> torch.Tensor:
        out = self.extension.pack_qjl_signs_1bit_cuda(signs.contiguous(), int(self.qjl_mode))
        if out.dtype != self.qjl_out_dtype:
            out = out.to(self.qjl_out_dtype)
        return out.contiguous()

    def summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "scalar_mode": int(self.scalar_mode),
            "scalar_out_dtype": str(self.scalar_out_dtype),
            "qjl_mode": int(self.qjl_mode),
            "qjl_out_dtype": str(self.qjl_out_dtype),
            "qjl_validation_input_dtype": str(self.qjl_validation_input_dtype),
        }
