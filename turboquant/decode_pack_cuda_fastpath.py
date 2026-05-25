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

std::vector<torch::Tensor> fused_scalar_quant_pack_4bit_cuda(
    torch::Tensor values,
    torch::Tensor centroids);

void fused_append_compressed_k_cuda(
    torch::Tensor new_keys,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch,
    torch::Tensor scalar_storage,
    torch::Tensor qjl_storage,
    torch::Tensor norm_storage,
    int64_t start);

std::vector<torch::Tensor> fused_rotate_quant_residual_qjl_cuda(
    torch::Tensor new_keys,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch);

std::vector<torch::Tensor> fused_dequant_residual_qjl_cuda(
    torch::Tensor new_keys,
    torch::Tensor codes,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch);

std::vector<torch::Tensor> fused_residual_qjl256_pack_cuda(
    torch::Tensor residual,
    torch::Tensor sketch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_scalar_codes_4bit_cuda", &pack_scalar_codes_4bit_cuda,
        "Pack scalar 4-bit codes [*,128] -> [*,64] (CUDA)");
  m.def("pack_qjl_signs_1bit_cuda", &pack_qjl_signs_1bit_cuda,
        "Pack QJL sign bits [*,128] -> [*,16] (CUDA)");
  m.def("scalar_quantize_16_cuda", &scalar_quantize_16_cuda,
        "Scalar quantize shared 16-centroid codebook [*,128] -> uint8 [*,128] (CUDA)");
  m.def("fused_scalar_quant_pack_4bit_cuda", &fused_scalar_quant_pack_4bit_cuda,
        "Fused scalar quant + current lane-word 4bit pack [*,128] -> codes [*,128], packed [*,64] (CUDA)");
  m.def("fused_append_compressed_k_cuda", &fused_append_compressed_k_cuda,
        "Fused rotate+quant+residual-QJL+pack+direct compressed-cache append (CUDA)");
  m.def("fused_rotate_quant_residual_qjl_cuda", &fused_rotate_quant_residual_qjl_cuda,
        "Fused rotate + scalar quant + residual + QJL encode [*,128] (CUDA)");
  m.def("fused_dequant_residual_qjl_cuda", &fused_dequant_residual_qjl_cuda,
        "Fused scalar dequant + inverse rotate + residual + QJL encode [*,128] (CUDA)");
  m.def("fused_residual_qjl256_pack_cuda", &fused_residual_qjl256_pack_cuda,
        "Fused residual norm + QJL256 sign encode + lane-nibble pack (CUDA)");
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
    int64_t M,
    int64_t mode) {
  const int64_t bytes_per_row = M / 8;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = outer * bytes_per_row;
  if (idx >= total) return;

  const int64_t row = idx / bytes_per_row;
  const int64_t j_global = idx % bytes_per_row;

  // Generic lane-nibble layout for M = 128 * nblocks.
  //
  // For each 128-dim block:
  // byte j packs:
  // bit0 <- signs[block +  2j]
  // bit4 <- signs[block +  2j+1]
  // bit1 <- signs[block + 32+2j]
  // bit5 <- signs[block + 32+2j+1]
  // bit2 <- signs[block + 64+2j]
  // bit6 <- signs[block + 64+2j+1]
  // bit3 <- signs[block + 96+2j]
  // bit7 <- signs[block + 96+2j+1]
  const int64_t block_id = j_global / 16;
  const int64_t j = j_global % 16;

  const int64_t base = row * M + block_id * 128;
  const int64_t even = 2 * j;
  const int64_t odd  = 2 * j + 1;

  uint8_t packed = 0;

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + even], mode) << 0);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + odd], mode) << 4);

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 32 + even], mode) << 1);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 32 + odd], mode) << 5);

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 64 + even], mode) << 2);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 64 + odd], mode) << 6);

  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 96 + even], mode) << 3);
  packed |= static_cast<uint8_t>(
      sign_to_bit<scalar_t>(signs[base + 96 + odd], mode) << 7);

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


template <typename scalar_t>
__global__ void fused_residual_qjl256_pack_kernel(
    const scalar_t* __restrict__ residual,  // [outer,128]
    const float* __restrict__ sketch,       // [256,128]
    uint8_t* __restrict__ packed,           // [outer,32]
    float* __restrict__ norms,              // [outer]
    int64_t outer) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;  // 0..255

  if (row >= outer) return;

  __shared__ float r[128];
  __shared__ float norm_sq_shared[256];

  if (tid < 128) {
    const float v = static_cast<float>(residual[row * 128 + tid]);
    r[tid] = v;
    norm_sq_shared[tid] = v * v;
  } else {
    norm_sq_shared[tid] = 0.0f;
  }
  __syncthreads();

  // Reduce norm over 256 shared entries.
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      norm_sq_shared[tid] += norm_sq_shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    norms[row] = sqrtf(norm_sq_shared[0]);
  }

  // Each tid 0..255 computes one QJL sign dot.
  float dot = 0.0f;
  #pragma unroll
  for (int d = 0; d < 128; ++d) {
    dot += sketch[tid * 128 + d] * r[d];
  }

  const uint8_t bit = (dot > 0.0f) ? 1 : 0;

  // Pack according to lane-nibble layout:
  // block 0: dims 0..127 -> bytes 0..15
  // block 1: dims 128..255 -> bytes 16..31
  const int block_id = tid / 128;
  const int local = tid - block_id * 128;
  const int lane = local % 32;
  const int offset_group = local / 32;  // 0,1,2,3
  const int byte_j = block_id * 16 + lane / 2;
  const int nibble_shift = (lane & 1) ? 4 : 0;
  const int bit_shift = nibble_shift + offset_group;

  if (bit) {
    atomicOr(reinterpret_cast<unsigned int*>(&packed[row * 32 + (byte_j & ~3)]),
             static_cast<unsigned int>(1u << ((byte_j & 3) * 8 + bit_shift)));
  }
}

std::vector<torch::Tensor> fused_residual_qjl256_pack_cuda(
    torch::Tensor residual,
    torch::Tensor sketch) {
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(sketch.is_cuda(), "sketch must be CUDA");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(sketch.is_contiguous(), "sketch must be contiguous");
  TORCH_CHECK(residual.size(-1) == 128, "residual last dim must be 128");
  TORCH_CHECK(sketch.numel() == 256 * 128, "sketch must be [256,128]");

  const int64_t outer = residual.numel() / 128;
  auto packed_sizes = residual.sizes().vec();
  packed_sizes.back() = 32;
  auto norm_sizes = residual.sizes().vec();
  norm_sizes.pop_back();

  auto packed = torch::zeros(packed_sizes, residual.options().dtype(torch::kUInt8));
  auto norms = torch::empty(norm_sizes, residual.options().dtype(torch::kFloat32));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(residual.scalar_type(), "fused_residual_qjl256_pack_cuda", [&] {
    fused_residual_qjl256_pack_kernel<scalar_t><<<outer, 256, 0, stream>>>(
        residual.data_ptr<scalar_t>(),
        sketch.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        norms.data_ptr<float>(),
        outer);
  });

  return {packed, norms};
}



__global__ void fused_scalar_quant_pack_4bit_kernel(
    const float* __restrict__ values,     // [outer,128]
    const float* __restrict__ centroids,  // [16]
    uint8_t* __restrict__ codes,          // [outer,128]
    uint8_t* __restrict__ packed,         // [outer,64]
    int64_t outer) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;  // 0..127
  if (row >= outer) return;

  __shared__ uint8_t c[128];

  const float x = values[row * 128 + tid];

  float best_dist = fabsf(x - centroids[0]);
  uint8_t best = 0;
  #pragma unroll
  for (int k = 1; k < 16; ++k) {
    const float d = fabsf(x - centroids[k]);
    if (d < best_dist) {
      best_dist = d;
      best = static_cast<uint8_t>(k);
    }
  }

  c[tid] = best;
  codes[row * 128 + tid] = best;
  __syncthreads();

  // Match turboquant.scalar_lane_layout.pack_scalar_codes_lane_word_4bit:
  // for lane in [0,31]:
  //   packed[2*lane+0] = code[lane]    | code[lane+32] << 4
  //   packed[2*lane+1] = code[lane+64] | code[lane+96] << 4
  if (tid < 32) {
    const uint8_t c0 = c[tid] & 0x0F;
    const uint8_t c1 = c[tid + 32] & 0x0F;
    const uint8_t c2 = c[tid + 64] & 0x0F;
    const uint8_t c3 = c[tid + 96] & 0x0F;

    packed[row * 64 + 2 * tid + 0] = static_cast<uint8_t>(c0 | (c1 << 4));
    packed[row * 64 + 2 * tid + 1] = static_cast<uint8_t>(c2 | (c3 << 4));
  }
}

std::vector<torch::Tensor> fused_scalar_quant_pack_4bit_cuda(
    torch::Tensor values,
    torch::Tensor centroids) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must be float32");
  TORCH_CHECK(centroids.scalar_type() == torch::kFloat32,
              "centroids must be float32");
  TORCH_CHECK(values.size(-1) == 128, "values last dim must be 128");
  TORCH_CHECK(centroids.numel() == 16, "centroids must have 16 values");

  const int64_t outer = values.numel() / 128;

  auto codes_sizes = values.sizes().vec();
  auto packed_sizes = values.sizes().vec();
  packed_sizes.back() = 64;

  auto codes = torch::empty(codes_sizes, values.options().dtype(torch::kUInt8));
  auto packed = torch::empty(packed_sizes, values.options().dtype(torch::kUInt8));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
  fused_scalar_quant_pack_4bit_kernel<<<outer, 128, 0, stream>>>(
      values.data_ptr<float>(),
      centroids.data_ptr<float>(),
      codes.data_ptr<uint8_t>(),
      packed.data_ptr<uint8_t>(),
      outer);

  return {codes, packed};
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
__global__ void fused_append_compressed_k_kernel(
    const key_t* __restrict__ new_keys,       // [B,H,1,128], contiguous, B=1
    const float* __restrict__ centroids,      // [16]
    const float* __restrict__ rotation,       // [128,128]
    const float* __restrict__ sketch,         // [128,128]
    uint8_t* __restrict__ scalar_storage,     // [1,H,capacity,64]
    uint8_t* __restrict__ qjl_storage,        // [1,H,capacity,16]
    float* __restrict__ norm_storage,         // [1,H,capacity]
    int64_t H,
    int64_t capacity,
    int64_t start) {
  const int h = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

  if (h >= H || tid >= 128) return;

  __shared__ float recon_rot[128];
  __shared__ float residual[128];
  __shared__ float reduce_buf[128];
  __shared__ uint8_t codes_s[128];
  __shared__ int8_t signs_s[128];

  // new_keys row offset for B=1,T=1: [H,128]
  const int64_t key_base = static_cast<int64_t>(h) * 128;

  // rotate(x) = x @ R.T
  float z_d = 0.0f;
  #pragma unroll
  for (int j = 0; j < 128; ++j) {
    const float x_j = static_cast<float>(new_keys[key_base + j]);
    z_d += x_j * rotation[tid * 128 + j];
  }

  // scalar quantize 16 centroids
  float best_dist = fabsf(z_d - centroids[0]);
  int best = 0;
  #pragma unroll
  for (int k = 1; k < 16; ++k) {
    const float dist = fabsf(z_d - centroids[k]);
    if (dist < best_dist) {
      best_dist = dist;
      best = k;
    }
  }

  codes_s[tid] = static_cast<uint8_t>(best);
  recon_rot[tid] = centroids[best];

  __syncthreads();

  // inverse rotate reconstructed key: recon = recon_rot @ R
  float recon_d = 0.0f;
  #pragma unroll
  for (int j = 0; j < 128; ++j) {
    recon_d += recon_rot[j] * rotation[j * 128 + tid];
  }

  const float x_d = static_cast<float>(new_keys[key_base + tid]);
  const float r_d = x_d - recon_d;
  residual[tid] = r_d;
  reduce_buf[tid] = r_d * r_d;

  __syncthreads();

  // norm reduce
  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) {
      reduce_buf[tid] += reduce_buf[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    norm_storage[static_cast<int64_t>(h) * capacity + start] = sqrtf(reduce_buf[0]);
  }

  // QJL sign projection
  float dot = 0.0f;
  #pragma unroll
  for (int d = 0; d < 128; ++d) {
    dot += residual[d] * sketch[tid * 128 + d];
  }
  signs_s[tid] = (dot >= 0.0f) ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);

  __syncthreads();

  // Pack scalar codes using the same lane layout as
  // pack_scalar_codes_4bit_kernel(mode=0):
  //
  // byte j packs codes[lo_local] in low nibble and codes[hi_local] in high nibble
  // j=0  -> (0,16)
  // j=1  -> (64,80)
  // j=2  -> (1,17)
  // j=3  -> (65,81)
  // ...
  if (tid < 64) {
    const int j = tid;
    const int group = j / 32;       // 0 or 1
    const int within = j % 32;
    const int i = within / 2;       // 0..15
    const int half = within % 2;    // 0 or 1

    const int lo_local = group * 32 + i + half * 64;
    const int hi_local = lo_local + 16;

    const uint8_t a = static_cast<uint8_t>(codes_s[lo_local] & 0x0F);
    const uint8_t b = static_cast<uint8_t>(codes_s[hi_local] & 0x0F);
    const uint8_t packed = static_cast<uint8_t>(a | (b << 4));

    scalar_storage[
      (static_cast<int64_t>(h) * capacity + start) * 64 + tid
    ] = packed;
  }

  // Pack QJL signs using the same lane-nibble layout as
  // pack_qjl_signs_1bit_kernel(mode=0).
  //
  // Current fastpath mode=0 means nonzero -> 1. Since signs_s is {-1,+1},
  // all sign values are nonzero and therefore map to bit 1, matching the
  // existing pack_qjl fastpath behavior.
  if (tid < 16) {
    const int j = tid;
    const int even = 2 * j;
    const int odd = 2 * j + 1;

    uint8_t packed = 0;

    packed |= static_cast<uint8_t>((signs_s[even] != 0) << 0);
    packed |= static_cast<uint8_t>((signs_s[odd]  != 0) << 4);

    packed |= static_cast<uint8_t>((signs_s[32 + even] != 0) << 1);
    packed |= static_cast<uint8_t>((signs_s[32 + odd]  != 0) << 5);

    packed |= static_cast<uint8_t>((signs_s[64 + even] != 0) << 2);
    packed |= static_cast<uint8_t>((signs_s[64 + odd]  != 0) << 6);

    packed |= static_cast<uint8_t>((signs_s[96 + even] != 0) << 3);
    packed |= static_cast<uint8_t>((signs_s[96 + odd]  != 0) << 7);

    qjl_storage[
      (static_cast<int64_t>(h) * capacity + start) * 16 + tid
    ] = packed;
  }
}

void fused_append_compressed_k_cuda(
    torch::Tensor new_keys,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch,
    torch::Tensor scalar_storage,
    torch::Tensor qjl_storage,
    torch::Tensor norm_storage,
    int64_t start) {
  TORCH_CHECK(new_keys.is_cuda(), "new_keys must be CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
  TORCH_CHECK(rotation.is_cuda(), "rotation must be CUDA");
  TORCH_CHECK(sketch.is_cuda(), "sketch must be CUDA");
  TORCH_CHECK(scalar_storage.is_cuda(), "scalar_storage must be CUDA");
  TORCH_CHECK(qjl_storage.is_cuda(), "qjl_storage must be CUDA");
  TORCH_CHECK(norm_storage.is_cuda(), "norm_storage must be CUDA");

  TORCH_CHECK(new_keys.is_contiguous(), "new_keys must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(rotation.is_contiguous(), "rotation must be contiguous");
  TORCH_CHECK(sketch.is_contiguous(), "sketch must be contiguous");
  TORCH_CHECK(scalar_storage.is_contiguous(), "scalar_storage must be contiguous");
  TORCH_CHECK(qjl_storage.is_contiguous(), "qjl_storage must be contiguous");
  TORCH_CHECK(norm_storage.is_contiguous(), "norm_storage must be contiguous");

  TORCH_CHECK(new_keys.dim() == 4, "new_keys must be [B,H,1,128]");
  TORCH_CHECK(new_keys.size(0) == 1, "only B=1 supported");
  TORCH_CHECK(new_keys.size(2) == 1, "only T=1 append supported");
  TORCH_CHECK(new_keys.size(3) == 128, "new_keys last dim must be 128");

  TORCH_CHECK(scalar_storage.dim() == 4, "scalar_storage must be [1,H,capacity,64]");
  TORCH_CHECK(qjl_storage.dim() == 4, "qjl_storage must be [1,H,capacity,16]");
  TORCH_CHECK(norm_storage.dim() == 3, "norm_storage must be [1,H,capacity]");

  const int64_t H = new_keys.size(1);
  const int64_t capacity = scalar_storage.size(2);

  TORCH_CHECK(scalar_storage.size(0) == 1 && scalar_storage.size(1) == H && scalar_storage.size(3) == 64,
              "bad scalar_storage shape");
  TORCH_CHECK(qjl_storage.size(0) == 1 && qjl_storage.size(1) == H && qjl_storage.size(2) == capacity && qjl_storage.size(3) == 16,
              "bad qjl_storage shape");
  TORCH_CHECK(norm_storage.size(0) == 1 && norm_storage.size(1) == H && norm_storage.size(2) == capacity,
              "bad norm_storage shape");
  TORCH_CHECK(start >= 0 && start < capacity, "start out of range");

  TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids must be float32");
  TORCH_CHECK(rotation.scalar_type() == at::kFloat, "rotation must be float32");
  TORCH_CHECK(sketch.scalar_type() == at::kFloat, "sketch must be float32");
  TORCH_CHECK(scalar_storage.scalar_type() == at::kByte, "scalar_storage must be uint8");
  TORCH_CHECK(qjl_storage.scalar_type() == at::kByte, "qjl_storage must be uint8");
  TORCH_CHECK(norm_storage.scalar_type() == at::kFloat, "norm_storage must be float32");

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(new_keys.scalar_type(), "fused_append_compressed_k_cuda", [&] {
    fused_append_compressed_k_kernel<scalar_t><<<static_cast<unsigned int>(H), 128, 0, stream>>>(
        new_keys.data_ptr<scalar_t>(),
        centroids.data_ptr<float>(),
        rotation.data_ptr<float>(),
        sketch.data_ptr<float>(),
        scalar_storage.data_ptr<uint8_t>(),
        qjl_storage.data_ptr<uint8_t>(),
        norm_storage.data_ptr<float>(),
        H,
        capacity,
        start);
  });
}


template <typename key_t>
__global__ void fused_rotate_quant_residual_qjl_kernel(
    const key_t* __restrict__ new_keys,      // [outer,128]
    const float* __restrict__ centroids,     // [16]
    const float* __restrict__ rotation,      // [128,128], row-major
    const float* __restrict__ sketch,        // [128,128], row-major
    uint8_t* __restrict__ codes,             // [outer,128]
    int8_t* __restrict__ signs,              // [outer,128]
    float* __restrict__ norms,               // [outer]
    int64_t outer) {
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

  if (row >= outer || tid >= 128) return;

  __shared__ float rotated[128];
  __shared__ float recon_rot[128];
  __shared__ float residual[128];
  __shared__ float reduce_buf[128];

  // rotate(x) = x @ R.T
  // rotated[d] = sum_j x[j] * R[d,j]
  float z_d = 0.0f;
  #pragma unroll
  for (int j = 0; j < 128; ++j) {
    const float x_j = static_cast<float>(new_keys[row * 128 + j]);
    z_d += x_j * rotation[tid * 128 + j];
  }
  rotated[tid] = z_d;

  // Shared 16-level scalar quantize.
  float best_dist = fabsf(z_d - centroids[0]);
  int best = 0;
  #pragma unroll
  for (int k = 1; k < 16; ++k) {
    const float dist = fabsf(z_d - centroids[k]);
    if (dist < best_dist) {
      best_dist = dist;
      best = k;
    }
  }

  codes[row * 128 + tid] = static_cast<uint8_t>(best);
  recon_rot[tid] = centroids[best];

  __syncthreads();

  // inverse_rotate(z_hat) = z_hat @ R
  // reconstructed_key[d] = sum_j recon_rot[j] * R[j,d]
  float recon_d = 0.0f;
  #pragma unroll
  for (int j = 0; j < 128; ++j) {
    recon_d += recon_rot[j] * rotation[j * 128 + tid];
  }

  const float x_d = static_cast<float>(new_keys[row * 128 + tid]);
  const float r_d = x_d - recon_d;

  residual[tid] = r_d;
  reduce_buf[tid] = r_d * r_d;

  __syncthreads();

  // Norm reduction.
  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) {
      reduce_buf[tid] += reduce_buf[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    norms[row] = sqrtf(reduce_buf[0]);
  }

  // QJL sign: projected[m] = sum_d residual[d] * sketch[m,d]
  float dot = 0.0f;
  #pragma unroll
  for (int d = 0; d < 128; ++d) {
    dot += residual[d] * sketch[tid * 128 + d];
  }

  signs[row * 128 + tid] = (dot >= 0.0f) ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);
}

std::vector<torch::Tensor> fused_rotate_quant_residual_qjl_cuda(
    torch::Tensor new_keys,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch) {
  TORCH_CHECK(new_keys.is_cuda(), "new_keys must be CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
  TORCH_CHECK(rotation.is_cuda(), "rotation must be CUDA");
  TORCH_CHECK(sketch.is_cuda(), "sketch must be CUDA");

  TORCH_CHECK(new_keys.is_contiguous(), "new_keys must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(rotation.is_contiguous(), "rotation must be contiguous");
  TORCH_CHECK(sketch.is_contiguous(), "sketch must be contiguous");

  TORCH_CHECK(new_keys.size(-1) == 128, "new_keys last dim must be 128");
  TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids must be float32");
  TORCH_CHECK(rotation.scalar_type() == at::kFloat, "rotation must be float32");
  TORCH_CHECK(sketch.scalar_type() == at::kFloat, "sketch must be float32");
  TORCH_CHECK(centroids.numel() == 16, "centroids must have 16 entries");
  TORCH_CHECK(rotation.numel() == 128 * 128, "rotation must be [128,128]");
  TORCH_CHECK(sketch.numel() == 128 * 128, "sketch must be [128,128]");

  auto code_sizes = new_keys.sizes().vec();
  auto norm_sizes = new_keys.sizes().vec();
  norm_sizes.pop_back();

  auto codes = torch::empty(code_sizes, new_keys.options().dtype(torch::kUInt8));
  auto signs = torch::empty(code_sizes, new_keys.options().dtype(torch::kInt8));
  auto norms = torch::empty(norm_sizes, new_keys.options().dtype(torch::kFloat32));

  const int64_t outer = new_keys.numel() / 128;
  const dim3 blocks(static_cast<unsigned int>(outer));
  const dim3 threads(128);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(new_keys.scalar_type(), "fused_rotate_quant_residual_qjl_cuda", [&] {
    fused_rotate_quant_residual_qjl_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        new_keys.data_ptr<scalar_t>(),
        centroids.data_ptr<float>(),
        rotation.data_ptr<float>(),
        sketch.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        signs.data_ptr<int8_t>(),
        norms.data_ptr<float>(),
        outer);
  });

  return {codes, signs, norms};
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
  TORCH_CHECK(signs.scalar_type() == torch::kInt8,
              "pack_qjl_signs_1bit_cuda currently expects int8 signs {-1,+1}");

  const int64_t M = signs.size(-1);
  TORCH_CHECK(M == 128 || M == 256 || M == 512,
              "signs last dim must be one of {128,256,512}");
  TORCH_CHECK(M % 128 == 0, "signs last dim must be divisible by 128");

  auto out_sizes = signs.sizes().vec();
  out_sizes.back() = M / 8;
  auto out = torch::empty(out_sizes, signs.options().dtype(torch::kUInt8));

  const int64_t outer = signs.numel() / M;
  const int64_t total = outer * (M / 8);
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  pack_qjl_signs_1bit_kernel<int8_t><<<blocks, threads, 0, stream>>>(
      signs.data_ptr<int8_t>(),
      out.data_ptr<uint8_t>(),
      outer,
      M,
      mode);

  return out;
}
'''


@lru_cache(maxsize=1)
def _ext():
    return load_inline(
        name="turboquant_decode_pack_cuda_fastpath_ext_fused_append_k_v2",
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
def fused_append_compressed_k_cuda(
    new_keys: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sketch: torch.Tensor,
    scalar_storage: torch.Tensor,
    qjl_storage: torch.Tensor,
    norm_storage: torch.Tensor,
    start: int,
) -> None:
    """
    Direct steady-state append:
      new_keys -> rotate + scalar quant + residual-QJL
      -> pack scalar/qjl -> write to preallocated compressed cache storage.
    Supports B=1, T=1, D=128.
    """
    _ext().fused_append_compressed_k_cuda(
        new_keys.contiguous(),
        centroids.contiguous(),
        rotation.contiguous(),
        sketch.contiguous(),
        scalar_storage,
        qjl_storage,
        norm_storage,
        int(start),
    )
    return None


@torch.no_grad()
def fused_rotate_quant_residual_qjl_cuda(
    new_keys: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sketch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused replacement for:
      rotate(new_keys, rotation)
      scalar_quantize(rotated_keys, centroids)
      scalar_dequantize(codes, centroids)
      inverse_rotate(..., rotation)
      residual = new_keys - reconstructed
      qjl_encode_residual(residual, sketch)

    Returns:
      codes_raw: uint8 [*,128]
      residual_signs_raw: int8 [*,128]
      residual_norms_raw: float32 [*]
    """
    return tuple(
        _ext().fused_rotate_quant_residual_qjl_cuda(
            new_keys.contiguous(),
            centroids.contiguous(),
            rotation.contiguous(),
            sketch.contiguous(),
        )
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

@torch.no_grad()
def pack_qjl_signs_1bit_cuda(signs: torch.Tensor, mode: int = 1) -> torch.Tensor:
    """
    CUDA lane-nibble QJL sign pack.
    Supports signs[..., M] where M should be 128/256/512 after generic kernel patch.
    Returns uint8 packed[..., M//8].
    """
    if not signs.is_cuda:
        raise ValueError("signs must be CUDA")
    return _ext().pack_qjl_signs_1bit_cuda(signs.contiguous(), int(mode))

@torch.no_grad()
def fused_residual_qjl256_pack_cuda(
    residual: torch.Tensor,
    sketch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    residual: [...,128], sketch: [256,128]
    returns:
      packed_qjl: [...,32] uint8
      norms: [...] fp32
    """
    if not residual.is_cuda or not sketch.is_cuda:
        raise ValueError("residual and sketch must be CUDA")
    return tuple(_ext().fused_residual_qjl256_pack_cuda(
        residual.contiguous(),
        sketch.contiguous(),
    ))

@torch.no_grad()
def fused_scalar_quant_pack_4bit_cuda(
    values: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    values: [...,128] float32
    centroids: [16] float32
    returns:
      codes: [...,128] uint8
      packed: [...,64] uint8, matching scalar_lane_layout.py
    """
    if not values.is_cuda or not centroids.is_cuda:
        raise ValueError("values and centroids must be CUDA")
    return tuple(_ext().fused_scalar_quant_pack_4bit_cuda(
        values.contiguous(),
        centroids.contiguous(),
    ))

