#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int M = 128;
constexpr int PACKED_STANDARD_SIGN_BYTES = M / 8;   // 16
constexpr int PACKED_LANE_NIBBLE_BYTES = 16;        // 32 lanes * 4 bits / 8
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

__device__ __forceinline__ float sign_from_bit(uint8_t bit) {
    return bit ? 1.0f : -1.0f;
}

// Candidate 1: current standard packed sign layout, but residual norm is hoisted
// before the QJL load/decode/accumulate work.
__global__ void turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_kernel(
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ packed_qjl_signs,
    const float* __restrict__ residual_norms,
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

    const float norm = residual_norms[static_cast<int64_t>(h) * T + t];

    const float* qp_ptr = qjl_projected_queries + static_cast<int64_t>(h) * M;
    const uint8_t* sign_ptr =
        packed_qjl_signs + (static_cast<int64_t>(h) * T + t) * PACKED_STANDARD_SIGN_BYTES;

    float qjl_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sketch_idx = lane + i * 32;
        const uint8_t packed_sign = sign_ptr[sketch_idx >> 3];
        const uint8_t bit =
            static_cast<uint8_t>((packed_sign >> (sketch_idx & 7)) & 0x01u);
        qjl_acc += qp_ptr[sketch_idx] * sign_from_bit(bit);
    }

    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        constexpr float QJL_SCALE =
            1.2533141373155001f / static_cast<float>(M); // sqrt(pi/2)/M
        out[static_cast<int64_t>(h) * T + t] =
            QJL_SCALE * norm * qjl_acc;
    }
}

// Candidate 2: lane-nibble sign layout. Each lane loads one byte (containing its
// own nibble plus the neighboring lane's nibble) instead of four sign bytes.
__global__ void turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_kernel(
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
    const float* __restrict__ residual_norms,
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

    const float* qp_ptr = qjl_projected_queries + static_cast<int64_t>(h) * M;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + (static_cast<int64_t>(h) * T + t) * PACKED_LANE_NIBBLE_BYTES;

    const uint8_t pair_byte = sign_ptr[lane >> 1];
    const uint8_t lane_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((pair_byte >> 4) & 0x0Fu)
            : static_cast<uint8_t>(pair_byte & 0x0Fu);

    float qjl_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sketch_idx = lane + i * 32;
        const uint8_t bit =
            static_cast<uint8_t>((lane_nibble >> i) & 0x01u);
        qjl_acc += qp_ptr[sketch_idx] * sign_from_bit(bit);
    }

    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float norm = residual_norms[static_cast<int64_t>(h) * T + t];
        constexpr float QJL_SCALE =
            1.2533141373155001f / static_cast<float>(M);
        out[static_cast<int64_t>(h) * T + t] =
            QJL_SCALE * norm * qjl_acc;
    }
}

// Candidate 3: lane-nibble layout + early residual norm load.
__global__ void turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_kernel(
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
    const float* __restrict__ residual_norms,
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

    const float norm = residual_norms[static_cast<int64_t>(h) * T + t];

    const float* qp_ptr = qjl_projected_queries + static_cast<int64_t>(h) * M;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + (static_cast<int64_t>(h) * T + t) * PACKED_LANE_NIBBLE_BYTES;

    const uint8_t pair_byte = sign_ptr[lane >> 1];
    const uint8_t lane_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((pair_byte >> 4) & 0x0Fu)
            : static_cast<uint8_t>(pair_byte & 0x0Fu);

    float qjl_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sketch_idx = lane + i * 32;
        const uint8_t bit =
            static_cast<uint8_t>((lane_nibble >> i) & 0x01u);
        qjl_acc += qp_ptr[sketch_idx] * sign_from_bit(bit);
    }

    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        constexpr float QJL_SCALE =
            1.2533141373155001f / static_cast<float>(M);
        out[static_cast<int64_t>(h) * T + t] =
            QJL_SCALE * norm * qjl_acc;
    }
}

void validate_query(torch::Tensor qjl_projected_queries) {
    TORCH_CHECK(qjl_projected_queries.is_cuda(), "qjl_projected_queries must be CUDA");
    TORCH_CHECK(qjl_projected_queries.dtype() == torch::kFloat32, "qjl_projected_queries must be float32");
    TORCH_CHECK(
        qjl_projected_queries.dim() == 4 &&
        qjl_projected_queries.size(0) == 1 &&
        qjl_projected_queries.size(2) == 1 &&
        qjl_projected_queries.size(3) == M,
        "qjl_projected_queries must be [1,H,1,128]"
    );
}

void validate_norms(torch::Tensor residual_norms) {
    TORCH_CHECK(residual_norms.is_cuda(), "residual_norms must be CUDA");
    TORCH_CHECK(residual_norms.dtype() == torch::kFloat32, "residual_norms must be float32");
    TORCH_CHECK(
        residual_norms.dim() == 3 &&
        residual_norms.size(0) == 1,
        "residual_norms must be [1,H,T]"
    );
}

void validate_standard_signs(torch::Tensor packed_qjl_signs) {
    TORCH_CHECK(packed_qjl_signs.is_cuda(), "packed_qjl_signs must be CUDA");
    TORCH_CHECK(packed_qjl_signs.dtype() == torch::kUInt8, "packed_qjl_signs must be uint8");
    TORCH_CHECK(
        packed_qjl_signs.dim() == 4 &&
        packed_qjl_signs.size(0) == 1 &&
        packed_qjl_signs.size(3) == PACKED_STANDARD_SIGN_BYTES,
        "packed_qjl_signs must be [1,H,T,16]"
    );
}

void validate_lane_nibble_signs(torch::Tensor lane_nibble_qjl_signs) {
    TORCH_CHECK(lane_nibble_qjl_signs.is_cuda(), "lane_nibble_qjl_signs must be CUDA");
    TORCH_CHECK(lane_nibble_qjl_signs.dtype() == torch::kUInt8, "lane_nibble_qjl_signs must be uint8");
    TORCH_CHECK(
        lane_nibble_qjl_signs.dim() == 4 &&
        lane_nibble_qjl_signs.size(0) == 1 &&
        lane_nibble_qjl_signs.size(3) == PACKED_LANE_NIBBLE_BYTES,
        "lane_nibble_qjl_signs must be [1,H,T,16]"
    );
}

void validate_matching_shapes(
    torch::Tensor q,
    torch::Tensor signs,
    torch::Tensor norms
) {
    TORCH_CHECK(q.size(1) == signs.size(1), "head mismatch between q and signs");
    TORCH_CHECK(q.size(1) == norms.size(1), "head mismatch between q and norms");
    TORCH_CHECK(signs.size(2) == norms.size(2), "T mismatch between signs and norms");
}

torch::Tensor launch_qjl_variant(
    torch::Tensor qjl_projected_queries,
    torch::Tensor signs,
    torch::Tensor residual_norms,
    int variant
) {
    const int H = static_cast<int>(qjl_projected_queries.size(1));
    const int T = static_cast<int>(signs.size(2));

    auto out = torch::empty({1, H, 1, T}, qjl_projected_queries.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    if (variant == 0) {
        turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_kernel<<<grid, block>>>(
            qjl_projected_queries.contiguous().data_ptr<float>(),
            signs.contiguous().data_ptr<uint8_t>(),
            residual_norms.contiguous().data_ptr<float>(),
            out.data_ptr<float>(),
            H,
            T
        );
    } else if (variant == 1) {
        turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_kernel<<<grid, block>>>(
            qjl_projected_queries.contiguous().data_ptr<float>(),
            signs.contiguous().data_ptr<uint8_t>(),
            residual_norms.contiguous().data_ptr<float>(),
            out.data_ptr<float>(),
            H,
            T
        );
    } else {
        turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_kernel<<<grid, block>>>(
            qjl_projected_queries.contiguous().data_ptr<float>(),
            signs.contiguous().data_ptr<uint8_t>(),
            residual_norms.contiguous().data_ptr<float>(),
            out.data_ptr<float>(),
            H,
            T
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace

torch::Tensor turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda(
    torch::Tensor qjl_projected_queries,
    torch::Tensor packed_qjl_signs,
    torch::Tensor residual_norms
) {
    validate_query(qjl_projected_queries);
    validate_standard_signs(packed_qjl_signs);
    validate_norms(residual_norms);
    validate_matching_shapes(qjl_projected_queries, packed_qjl_signs, residual_norms);
    return launch_qjl_variant(qjl_projected_queries, packed_qjl_signs, residual_norms, 0);
}

torch::Tensor turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda(
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    validate_query(qjl_projected_queries);
    validate_lane_nibble_signs(lane_nibble_qjl_signs);
    validate_norms(residual_norms);
    validate_matching_shapes(qjl_projected_queries, lane_nibble_qjl_signs, residual_norms);
    return launch_qjl_variant(qjl_projected_queries, lane_nibble_qjl_signs, residual_norms, 1);
}

torch::Tensor turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda(
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    validate_query(qjl_projected_queries);
    validate_lane_nibble_signs(lane_nibble_qjl_signs);
    validate_norms(residual_norms);
    validate_matching_shapes(qjl_projected_queries, lane_nibble_qjl_signs, residual_norms);
    return launch_qjl_variant(qjl_projected_queries, lane_nibble_qjl_signs, residual_norms, 2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda",
        &turboquant_qjl_only_qjl128_early_norm_logits_b1q1_d128_cuda,
        "QJL-only QJL128 with early residual norm load"
    );
    m.def(
        "turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda",
        &turboquant_qjl_only_qjl128_lane_nibble_logits_b1q1_d128_cuda,
        "QJL-only QJL128 with lane-nibble sign layout"
    );
    m.def(
        "turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda",
        &turboquant_qjl_only_qjl128_lane_nibble_early_norm_logits_b1q1_d128_cuda,
        "QJL-only QJL128 with lane-nibble signs and early residual norm load"
    );
}
