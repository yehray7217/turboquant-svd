import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    this_dir = Path(__file__).resolve().parent
    build_dir = Path(os.environ.get("TURBOQUANT_EXT_BUILD_DIR", Path.home() / ".cache" / "torch_extensions"))
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__device__ __forceinline__ float load_prob(const scalar_t* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float load_prob<at::Half>(const at::Half* p) {
    return __half2float(reinterpret_cast<const __half*>(p)[0]);
}

template <typename scalar_t>
__device__ __forceinline__ void store_out(scalar_t* p, float x) {
    *p = static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ void store_out<at::Half>(at::Half* p, float x) {
    reinterpret_cast<__half*>(p)[0] = __float2half_rn(x);
}

// partial[h, chunk, d] = sum_t probs[h,t] * scale[h,t] * q[h,t,d]
template <typename prob_t>
__global__ void quant_v_partial_kernel(
    const prob_t* __restrict__ probs,       // [1,H,1,T]
    const int8_t* __restrict__ vq,          // [1,H,T,D]
    const at::Half* __restrict__ vscale,    // [1,H,T,1]
    float* __restrict__ partial,            // [H,C,D]
    int H,
    int T,
    int D,
    int chunks,
    int chunk_size
) {
    int h = blockIdx.x;
    int c = blockIdx.y;
    int d = threadIdx.x;

    if (h >= H || c >= chunks || d >= D) return;

    int start = c * chunk_size;
    int end = min(T, start + chunk_size);

    float acc = 0.0f;

    // probs offset: ((0*H + h)*1 + 0)*T + t
    const prob_t* p_base = probs + h * T;

    // vq offset: ((0*H + h)*T + t)*D + d
    const int8_t* q_base = vq + h * T * D + d;

    // vscale offset: ((0*H + h)*T + t)
    const at::Half* s_base = vscale + h * T;

    for (int t = start; t < end; ++t) {
        float p = load_prob<prob_t>(p_base + t);
        float s = __half2float(reinterpret_cast<const __half*>(s_base + t)[0]);
        float q = static_cast<float>(q_base[t * D]);
        acc += p * s * q;
    }

    partial[(h * chunks + c) * D + d] = acc;
}

template <typename out_t>
__global__ void quant_v_reduce_kernel(
    const float* __restrict__ partial,  // [H,C,D]
    out_t* __restrict__ out,            // [1,H,1,D]
    int H,
    int D,
    int chunks
) {
    int h = blockIdx.x;
    int d = threadIdx.x;

    if (h >= H || d >= D) return;

    float acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        acc += partial[(h * chunks + c) * D + d];
    }

    out[h * D + d] = static_cast<out_t>(acc);
}

template <>
__global__ void quant_v_reduce_kernel<at::Half>(
    const float* __restrict__ partial,
    at::Half* __restrict__ out,
    int H,
    int D,
    int chunks
) {
    int h = blockIdx.x;
    int d = threadIdx.x;

    if (h >= H || d >= D) return;

    float acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        acc += partial[(h * chunks + c) * D + d];
    }

    reinterpret_cast<__half*>(out)[h * D + d] = __float2half_rn(acc);
}

torch::Tensor quant_v_matmul_forward(
    torch::Tensor probs,
    torch::Tensor vq,
    torch::Tensor vscale,
    int64_t active_len,
    int64_t chunk_size
) {
    CHECK_INPUT(probs);
    CHECK_INPUT(vq);
    CHECK_INPUT(vscale);

    TORCH_CHECK(vq.scalar_type() == torch::kInt8, "vq must be int8");
    TORCH_CHECK(vscale.scalar_type() == torch::kFloat16, "vscale must be float16");
    TORCH_CHECK(probs.dim() == 4, "probs must be [1,H,1,T]");
    TORCH_CHECK(vq.dim() == 4, "vq must be [1,H,T,D]");
    TORCH_CHECK(vscale.dim() == 4, "vscale must be [1,H,T,1]");

    int B = probs.size(0);
    int H = probs.size(1);
    int Q = probs.size(2);
    int T = static_cast<int>(active_len);
    int D = vq.size(3);

    TORCH_CHECK(B == 1, "only B=1 supported");
    TORCH_CHECK(Q == 1, "only Q=1 supported");
    TORCH_CHECK(vq.size(0) == 1, "vq B must be 1");
    TORCH_CHECK(vq.size(1) == H, "vq H mismatch");
    TORCH_CHECK(vq.size(2) >= T, "vq T shorter than active_len");
    TORCH_CHECK(vscale.size(0) == 1, "vscale B must be 1");
    TORCH_CHECK(vscale.size(1) == H, "vscale H mismatch");
    TORCH_CHECK(vscale.size(2) >= T, "vscale T shorter than active_len");
    TORCH_CHECK(vscale.size(3) == 1, "vscale last dim must be 1");
    TORCH_CHECK(D == 128, "only D=128 supported");

    int cs = static_cast<int>(chunk_size);
    if (cs <= 0) cs = 256;
    int chunks = (T + cs - 1) / cs;

    auto partial = torch::empty({H, chunks, D}, probs.options().dtype(torch::kFloat32));
    auto out = torch::empty({1, H, 1, D}, probs.options());

    dim3 grid1(H, chunks);
    dim3 block1(D);

    if (probs.scalar_type() == torch::kFloat16) {
        quant_v_partial_kernel<at::Half><<<grid1, block1>>>(
            probs.data_ptr<at::Half>(),
            vq.data_ptr<int8_t>(),
            vscale.data_ptr<at::Half>(),
            partial.data_ptr<float>(),
            H,
            T,
            D,
            chunks,
            cs
        );
    } else if (probs.scalar_type() == torch::kFloat32) {
        quant_v_partial_kernel<float><<<grid1, block1>>>(
            probs.data_ptr<float>(),
            vq.data_ptr<int8_t>(),
            vscale.data_ptr<at::Half>(),
            partial.data_ptr<float>(),
            H,
            T,
            D,
            chunks,
            cs
        );
    } else {
        TORCH_CHECK(false, "probs dtype must be fp16 or fp32");
    }

    dim3 grid2(H);
    dim3 block2(D);

    if (out.scalar_type() == torch::kFloat16) {
        quant_v_reduce_kernel<at::Half><<<grid2, block2>>>(
            partial.data_ptr<float>(),
            out.data_ptr<at::Half>(),
            H,
            D,
            chunks
        );
    } else if (out.scalar_type() == torch::kFloat32) {
        quant_v_reduce_kernel<float><<<grid2, block2>>>(
            partial.data_ptr<float>(),
            out.data_ptr<float>(),
            H,
            D,
            chunks
        );
    } else {
        TORCH_CHECK(false, "out dtype must be fp16 or fp32");
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_v_matmul_forward", &quant_v_matmul_forward, "fused quantized V matmul forward");
}
'''

    src_path = build_dir / "turboquant_quant_v_matmul_cuda.cu"
    src_path.write_text(cuda_src)

    _EXT = load(
        name="turboquant_quant_v_matmul_cuda_ext_v1",
        sources=[str(src_path)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("TURBOQUANT_QUANT_V_VERBOSE", "0"))),
    )
    return _EXT


def quant_v_matmul_cuda(
    probs: torch.Tensor,
    vq: torch.Tensor,
    vscale: torch.Tensor,
    active_len: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    ext = _load_ext()
    return ext.quant_v_matmul_forward(
        probs.contiguous(),
        vq.contiguous(),
        vscale.contiguous(),
        int(active_len),
        int(chunk_size),
    )
