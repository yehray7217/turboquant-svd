#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int D = 128;
constexpr int M = 128;
constexpr int SCALAR_LEVELS = 16;
constexpr int LANE_WORD_PACKED_CODE_BYTES = D / 2;  // 64 bytes/token/head
constexpr int LANE_NIBBLE_SIGN_BYTES = M / 8;       // 16 bytes/token/head
constexpr int SHARED_FLOATS = D + M + SCALAR_LEVELS;
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

__device__ __forceinline__ uint16_t load_lane_code_word(
    const uint8_t* __restrict__ lane_word_scalar_codes,
    int lane
) {
    const uint16_t* words =
        reinterpret_cast<const uint16_t*>(lane_word_scalar_codes);
    return words[lane];
}

__device__ __forceinline__ float finalize_qjl_scale(
    float scalar_acc,
    float qjl_acc,
    float norm
) {
    constexpr float QJL_SCALE =
        1.2533141373155001f / static_cast<float>(M);  // sqrt(pi/2)/M
    return scalar_acc + QJL_SCALE * norm * qjl_acc;
}

/*
Variant A: stage rotated query + QJL projected query into shared memory.
Centroids remain global/indexed loads.

Per block:
  - 128 rotated-query floats
  - 128 qjl-query floats
Shared bytes:
  - 256 * 4 = 1024 bytes
*/
__global__ void turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_kernel(
    const float* __restrict__ rotated_queries,
    const uint8_t* __restrict__ lane_word_scalar_codes,
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
    const float* __restrict__ residual_norms,
    const float* __restrict__ centroids,
    float* __restrict__ out,
    int H,
    int T
) {
    __shared__ float shared_q[D + M];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

    if (h >= H) {
        return;
    }

    if (tid < D) {
        shared_q[tid] =
            rotated_queries[static_cast<int64_t>(h) * D + tid];
    }
    if (tid < M) {
        shared_q[D + tid] =
            qjl_projected_queries[static_cast<int64_t>(h) * M + tid];
    }
    __syncthreads();

    if (t >= T) {
        return;
    }

    const int64_t token_idx = static_cast<int64_t>(h) * T + t;
    const uint8_t* code_ptr =
        lane_word_scalar_codes + token_idx * LANE_WORD_PACKED_CODE_BYTES;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + token_idx * LANE_NIBBLE_SIGN_BYTES;

    const uint16_t lane_word = load_lane_code_word(code_ptr, lane);

    const uint8_t sign_pair_byte = sign_ptr[lane >> 1];
    const uint8_t lane_sign_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((sign_pair_byte >> 4) & 0x0Fu)
            : static_cast<uint8_t>(sign_pair_byte & 0x0Fu);

    float scalar_acc = 0.0f;
    float qjl_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int coord = lane + i * 32;

        const uint8_t scalar_code =
            static_cast<uint8_t>((lane_word >> (i * 4)) & 0x0Fu);
        scalar_acc += shared_q[coord] * centroids[scalar_code];

        const uint8_t sign_bit =
            static_cast<uint8_t>((lane_sign_nibble >> i) & 0x01u);
        qjl_acc += shared_q[D + coord] * sign_from_bit(sign_bit);
    }

    scalar_acc = warp_reduce_sum(scalar_acc);
    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float norm = residual_norms[token_idx];
        out[token_idx] = finalize_qjl_scale(scalar_acc, qjl_acc, norm);
    }
}

/*
Variant B: stage rotated query + QJL projected query + all 16 centroids into
shared memory.

Per block:
  - 128 rotated-query floats
  - 128 qjl-query floats
  -  16 centroid floats
Shared bytes:
  - 272 * 4 = 1088 bytes
*/
__global__ void turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_kernel(
    const float* __restrict__ rotated_queries,
    const uint8_t* __restrict__ lane_word_scalar_codes,
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
    const float* __restrict__ residual_norms,
    const float* __restrict__ centroids,
    float* __restrict__ out,
    int H,
    int T
) {
    __shared__ float shared_values[SHARED_FLOATS];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

    if (h >= H) {
        return;
    }

    if (tid < D) {
        shared_values[tid] =
            rotated_queries[static_cast<int64_t>(h) * D + tid];
    }
    if (tid < M) {
        shared_values[D + tid] =
            qjl_projected_queries[static_cast<int64_t>(h) * M + tid];
    }
    if (tid < SCALAR_LEVELS) {
        shared_values[D + M + tid] = centroids[tid];
    }
    __syncthreads();

    if (t >= T) {
        return;
    }

    const int64_t token_idx = static_cast<int64_t>(h) * T + t;
    const uint8_t* code_ptr =
        lane_word_scalar_codes + token_idx * LANE_WORD_PACKED_CODE_BYTES;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + token_idx * LANE_NIBBLE_SIGN_BYTES;

    const uint16_t lane_word = load_lane_code_word(code_ptr, lane);

    const uint8_t sign_pair_byte = sign_ptr[lane >> 1];
    const uint8_t lane_sign_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((sign_pair_byte >> 4) & 0x0Fu)
            : static_cast<uint8_t>(sign_pair_byte & 0x0Fu);

    float scalar_acc = 0.0f;
    float qjl_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int coord = lane + i * 32;

        const uint8_t scalar_code =
            static_cast<uint8_t>((lane_word >> (i * 4)) & 0x0Fu);
        scalar_acc +=
            shared_values[coord] * shared_values[D + M + scalar_code];

        const uint8_t sign_bit =
            static_cast<uint8_t>((lane_sign_nibble >> i) & 0x01u);
        qjl_acc += shared_values[D + coord] * sign_from_bit(sign_bit);
    }

    scalar_acc = warp_reduce_sum(scalar_acc);
    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float norm = residual_norms[token_idx];
        out[token_idx] = finalize_qjl_scale(scalar_acc, qjl_acc, norm);
    }
}

void validate_float_query(torch::Tensor q, const char* name, int last_dim) {
    TORCH_CHECK(q.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(
        q.dim() == 4 &&
        q.size(0) == 1 &&
        q.size(2) == 1 &&
        q.size(3) == last_dim,
        name, " must be [1,H,1,", last_dim, "]"
    );
}

void validate_lane_word_codes(torch::Tensor codes) {
    TORCH_CHECK(codes.is_cuda(), "lane_word_scalar_codes must be CUDA");
    TORCH_CHECK(codes.dtype() == torch::kUInt8, "lane_word_scalar_codes must be uint8");
    TORCH_CHECK(
        codes.dim() == 4 &&
        codes.size(0) == 1 &&
        codes.size(3) == LANE_WORD_PACKED_CODE_BYTES,
        "lane_word_scalar_codes must be [1,H,T,64]"
    );
}

void validate_lane_nibble_signs(torch::Tensor signs) {
    TORCH_CHECK(signs.is_cuda(), "lane_nibble_qjl_signs must be CUDA");
    TORCH_CHECK(signs.dtype() == torch::kUInt8, "lane_nibble_qjl_signs must be uint8");
    TORCH_CHECK(
        signs.dim() == 4 &&
        signs.size(0) == 1 &&
        signs.size(3) == LANE_NIBBLE_SIGN_BYTES,
        "lane_nibble_qjl_signs must be [1,H,T,16]"
    );
}

void validate_norms(torch::Tensor norms) {
    TORCH_CHECK(norms.is_cuda(), "residual_norms must be CUDA");
    TORCH_CHECK(norms.dtype() == torch::kFloat32, "residual_norms must be float32");
    TORCH_CHECK(
        norms.dim() == 3 &&
        norms.size(0) == 1,
        "residual_norms must be [1,H,T]"
    );
}

void validate_centroids(torch::Tensor centroids) {
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be float32");
    TORCH_CHECK(
        centroids.dim() == 1 && centroids.size(0) == SCALAR_LEVELS,
        "centroids must be [16]"
    );
}

void validate_full_inputs(
    torch::Tensor rotated_queries,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
    validate_float_query(rotated_queries, "rotated_queries", D);
    validate_lane_word_codes(lane_word_scalar_codes);
    validate_float_query(qjl_projected_queries, "qjl_projected_queries", M);
    validate_lane_nibble_signs(lane_nibble_qjl_signs);
    validate_norms(residual_norms);
    validate_centroids(centroids);

    TORCH_CHECK(rotated_queries.size(1) == lane_word_scalar_codes.size(1), "head mismatch");
    TORCH_CHECK(rotated_queries.size(1) == qjl_projected_queries.size(1), "head mismatch");
    TORCH_CHECK(rotated_queries.size(1) == lane_nibble_qjl_signs.size(1), "head mismatch");
    TORCH_CHECK(rotated_queries.size(1) == residual_norms.size(1), "head mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == lane_nibble_qjl_signs.size(2), "T mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == residual_norms.size(2), "T mismatch");
}

template <typename Kernel>
torch::Tensor launch_full_kernel(
    Kernel kernel,
    torch::Tensor rotated_queries,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
    validate_full_inputs(
        rotated_queries,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms,
        centroids
    );

    const int H = static_cast<int>(rotated_queries.size(1));
    const int T = static_cast<int>(lane_word_scalar_codes.size(2));

    auto out =
        torch::empty({1, H, 1, T}, rotated_queries.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    kernel<<<grid, block>>>(
        rotated_queries.contiguous().data_ptr<float>(),
        lane_word_scalar_codes.contiguous().data_ptr<uint8_t>(),
        qjl_projected_queries.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
        residual_norms.contiguous().data_ptr<float>(),
        centroids.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace

torch::Tensor turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_cuda(
    torch::Tensor rotated_queries,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
    return launch_full_kernel(
        turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_kernel,
        rotated_queries,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms,
        centroids
    );
}

torch::Tensor turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_cuda(
    torch::Tensor rotated_queries,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
    return launch_full_kernel(
        turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_kernel,
        rotated_queries,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms,
        centroids
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_cuda",
        &turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_logits_b1q1_d128_cuda,
        "Best full TurboQuant with block-shared query staging"
    );
    m.def(
        "turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_cuda",
        &turboquant_full_4bit_lane_word_lane_nibble_qjl128_shared_queries_centroids_logits_b1q1_d128_cuda,
        "Best full TurboQuant with block-shared query and centroid staging"
    );
}
