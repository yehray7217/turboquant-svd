#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARP_SIZE_ = 32;
constexpr int D = 128;
constexpr int M = 128;
constexpr int SCALAR_LEVELS = 16;
constexpr int PACKED_CODE_BYTES = D / 2;
constexpr int PACKED_SIGN_BYTES = M / 8;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

__global__ void dense_fp32_qkt_b1q1_d128_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float* __restrict__ out,
    int H,
    int T
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

    if (h >= H || t >= T) {
        return;
    }

    const float* q_ptr = queries + static_cast<int64_t>(h) * D;
    const float* k_ptr = keys + (static_cast<int64_t>(h) * T + t) * D;

    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = lane + i * 32;
        acc += q_ptr[idx] * k_ptr[idx];
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        out[static_cast<int64_t>(h) * T + t] = acc;
    }
}

__global__ void turboquant_4bit_qjl128_logits_b1q1_d128_kernel(
    const float* __restrict__ rotated_queries,
    const uint8_t* __restrict__ packed_scalar_codes,
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ packed_qjl_signs,
    const float* __restrict__ residual_norms,
    const float* __restrict__ centroids,
    float* __restrict__ out,
    int H,
    int T
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

    if (h >= H || t >= T) {
        return;
    }

    const float* rq_ptr = rotated_queries + static_cast<int64_t>(h) * D;
    const uint8_t* code_ptr =
        packed_scalar_codes + (static_cast<int64_t>(h) * T + t) * PACKED_CODE_BYTES;

    const float* qp_ptr = qjl_projected_queries + static_cast<int64_t>(h) * M;
    const uint8_t* sign_ptr =
        packed_qjl_signs + (static_cast<int64_t>(h) * T + t) * PACKED_SIGN_BYTES;

    float scalar_acc = 0.0f;
    float qjl_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int coord = lane + i * 32;
        const int code_byte_idx = coord >> 1;
        const uint8_t packed_code = code_ptr[code_byte_idx];
        const uint8_t code =
            (coord & 1)
                ? static_cast<uint8_t>((packed_code >> 4) & 0x0Fu)
                : static_cast<uint8_t>(packed_code & 0x0Fu);
        scalar_acc += rq_ptr[coord] * centroids[code];

        const int sketch_idx = coord;  // M == D == 128 in this baseline.
        const uint8_t packed_sign = sign_ptr[sketch_idx >> 3];
        const uint8_t bit =
            static_cast<uint8_t>((packed_sign >> (sketch_idx & 7)) & 0x01u);
        const float sign = bit ? 1.0f : -1.0f;
        qjl_acc += qp_ptr[sketch_idx] * sign;
    }

    scalar_acc = warp_reduce_sum(scalar_acc);
    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float norm = residual_norms[static_cast<int64_t>(h) * T + t];
        constexpr float QJL_SCALE =
            1.2533141373155001f / static_cast<float>(M); // sqrt(pi/2)/M
        out[static_cast<int64_t>(h) * T + t] =
            scalar_acc + QJL_SCALE * norm * qjl_acc;
    }
}

void validate_dense_inputs(torch::Tensor queries, torch::Tensor keys) {
    TORCH_CHECK(queries.is_cuda() && keys.is_cuda(), "queries/keys must be CUDA");
    TORCH_CHECK(queries.dtype() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(keys.dtype() == torch::kFloat32, "keys must be float32");
    TORCH_CHECK(
        queries.dim() == 4 &&
        queries.size(0) == 1 &&
        queries.size(2) == 1 &&
        queries.size(3) == D,
        "queries must be [1,H,1,128]"
    );
    TORCH_CHECK(
        keys.dim() == 4 &&
        keys.size(0) == 1 &&
        keys.size(3) == D,
        "keys must be [1,H,T,128]"
    );
    TORCH_CHECK(queries.size(1) == keys.size(1), "head count mismatch");
}

void validate_turbo_inputs(
    torch::Tensor rotated_queries,
    torch::Tensor packed_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor packed_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
    TORCH_CHECK(rotated_queries.is_cuda(), "rotated_queries must be CUDA");
    TORCH_CHECK(packed_scalar_codes.is_cuda(), "packed_scalar_codes must be CUDA");
    TORCH_CHECK(qjl_projected_queries.is_cuda(), "qjl_projected_queries must be CUDA");
    TORCH_CHECK(packed_qjl_signs.is_cuda(), "packed_qjl_signs must be CUDA");
    TORCH_CHECK(residual_norms.is_cuda(), "residual_norms must be CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");

    TORCH_CHECK(rotated_queries.dtype() == torch::kFloat32, "rotated_queries must be float32");
    TORCH_CHECK(qjl_projected_queries.dtype() == torch::kFloat32, "qjl_projected_queries must be float32");
    TORCH_CHECK(residual_norms.dtype() == torch::kFloat32, "residual_norms must be float32");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be float32");
    TORCH_CHECK(packed_scalar_codes.dtype() == torch::kUInt8, "packed_scalar_codes must be uint8");
    TORCH_CHECK(packed_qjl_signs.dtype() == torch::kUInt8, "packed_qjl_signs must be uint8");

    TORCH_CHECK(
        rotated_queries.dim() == 4 &&
        rotated_queries.size(0) == 1 &&
        rotated_queries.size(2) == 1 &&
        rotated_queries.size(3) == D,
        "rotated_queries must be [1,H,1,128]"
    );
    TORCH_CHECK(
        qjl_projected_queries.dim() == 4 &&
        qjl_projected_queries.size(0) == 1 &&
        qjl_projected_queries.size(2) == 1 &&
        qjl_projected_queries.size(3) == M,
        "qjl_projected_queries must be [1,H,1,128]"
    );
    TORCH_CHECK(
        packed_scalar_codes.dim() == 4 &&
        packed_scalar_codes.size(0) == 1 &&
        packed_scalar_codes.size(3) == PACKED_CODE_BYTES,
        "packed_scalar_codes must be [1,H,T,64]"
    );
    TORCH_CHECK(
        packed_qjl_signs.dim() == 4 &&
        packed_qjl_signs.size(0) == 1 &&
        packed_qjl_signs.size(3) == PACKED_SIGN_BYTES,
        "packed_qjl_signs must be [1,H,T,16]"
    );
    TORCH_CHECK(
        residual_norms.dim() == 3 &&
        residual_norms.size(0) == 1,
        "residual_norms must be [1,H,T]"
    );
    TORCH_CHECK(centroids.dim() == 1 && centroids.size(0) == SCALAR_LEVELS, "centroids must be [16]");

    TORCH_CHECK(rotated_queries.size(1) == packed_scalar_codes.size(1), "head mismatch");
    TORCH_CHECK(rotated_queries.size(1) == qjl_projected_queries.size(1), "head mismatch");
    TORCH_CHECK(rotated_queries.size(1) == packed_qjl_signs.size(1), "head mismatch");
    TORCH_CHECK(rotated_queries.size(1) == residual_norms.size(1), "head mismatch");
    TORCH_CHECK(packed_scalar_codes.size(2) == packed_qjl_signs.size(2), "T mismatch");
    TORCH_CHECK(packed_scalar_codes.size(2) == residual_norms.size(2), "T mismatch");
}

} // namespace

torch::Tensor dense_fp32_qkt_b1q1_d128_cuda(
    torch::Tensor queries,
    torch::Tensor keys
) {
    validate_dense_inputs(queries, keys);
    const int H = static_cast<int>(queries.size(1));
    const int T = static_cast<int>(keys.size(2));

    auto out = torch::empty({1, H, 1, T}, queries.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    dense_fp32_qkt_b1q1_d128_kernel<<<grid, block>>>(
        queries.contiguous().data_ptr<float>(),
        keys.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor turboquant_4bit_qjl128_logits_b1q1_d128_cuda(
    torch::Tensor rotated_queries,
    torch::Tensor packed_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor packed_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
    validate_turbo_inputs(
        rotated_queries,
        packed_scalar_codes,
        qjl_projected_queries,
        packed_qjl_signs,
        residual_norms,
        centroids
    );

    const int H = static_cast<int>(rotated_queries.size(1));
    const int T = static_cast<int>(packed_scalar_codes.size(2));

    auto out = torch::empty({1, H, 1, T}, rotated_queries.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    turboquant_4bit_qjl128_logits_b1q1_d128_kernel<<<grid, block>>>(
        rotated_queries.contiguous().data_ptr<float>(),
        packed_scalar_codes.contiguous().data_ptr<uint8_t>(),
        qjl_projected_queries.contiguous().data_ptr<float>(),
        packed_qjl_signs.contiguous().data_ptr<uint8_t>(),
        residual_norms.contiguous().data_ptr<float>(),
        centroids.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dense_fp32_qkt_b1q1_d128_cuda",
        &dense_fp32_qkt_b1q1_d128_cuda,
        "Dense FP32 qK^T attention logits CUDA baseline"
    );
    m.def(
        "turboquant_4bit_qjl128_logits_b1q1_d128_cuda",
        &turboquant_4bit_qjl128_logits_b1q1_d128_cuda,
        "True TurboQuant 4-bit scalar packed + QJL128 packed-sign logits CUDA baseline"
    );
}
