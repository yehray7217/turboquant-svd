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
constexpr int STANDARD_PACKED_CODE_BYTES = D / 2;     // 64
constexpr int LANE_WORD_PACKED_CODE_BYTES = D / 2;    // 64, 32 lanes * 2 bytes
constexpr int LANE_NIBBLE_SIGN_BYTES = M / 8;         // 16
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
    // The token/head base pointer is 64-byte aligned in normal tensor storage,
    // and each lane uses a 2-byte stride. Use a single logical 16-bit load.
    const uint16_t* words = reinterpret_cast<const uint16_t*>(lane_word_scalar_codes);
    return words[lane];
}

/*
Scalar-only ablation:
  - standard scalar kernel loads four packed U8 code bytes per lane.
  - this variant loads one 16-bit lane-local code word per lane.
*/
__global__ void turboquant_scalar_only_4bit_lane_word_logits_b1q1_d128_kernel(
    const float* __restrict__ rotated_queries,
    const uint8_t* __restrict__ lane_word_scalar_codes,
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
        lane_word_scalar_codes + (static_cast<int64_t>(h) * T + t) * LANE_WORD_PACKED_CODE_BYTES;

    const uint16_t lane_word = load_lane_code_word(code_ptr, lane);

    float scalar_acc = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int coord = lane + i * 32;
        const uint8_t code = static_cast<uint8_t>((lane_word >> (i * 4)) & 0x0Fu);
        scalar_acc += rq_ptr[coord] * centroids[code];
    }

    scalar_acc = warp_reduce_sum(scalar_acc);

    if (lane == 0) {
        out[static_cast<int64_t>(h) * T + t] = scalar_acc;
    }
}

/*
Full optimized variant:
  - scalar codes use lane-local 16-bit code words.
  - QJL signs use the already-validated lane-nibble layout.
  - residual norm follows the prior full optimized kernel's "early" C++ source
    placement, although compiler scheduling should be verified separately.
*/
__global__ void turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_kernel(
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
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

    if (h >= H || t >= T) {
        return;
    }

    const float norm = residual_norms[static_cast<int64_t>(h) * T + t];

    const float* rq_ptr = rotated_queries + static_cast<int64_t>(h) * D;
    const uint8_t* code_ptr =
        lane_word_scalar_codes + (static_cast<int64_t>(h) * T + t) * LANE_WORD_PACKED_CODE_BYTES;

    const float* qp_ptr = qjl_projected_queries + static_cast<int64_t>(h) * M;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + (static_cast<int64_t>(h) * T + t) * LANE_NIBBLE_SIGN_BYTES;

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
        scalar_acc += rq_ptr[coord] * centroids[scalar_code];

        const uint8_t sign_bit =
            static_cast<uint8_t>((lane_sign_nibble >> i) & 0x01u);
        qjl_acc += qp_ptr[coord] * sign_from_bit(sign_bit);
    }

    scalar_acc = warp_reduce_sum(scalar_acc);
    qjl_acc = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        constexpr float QJL_SCALE =
            1.2533141373155001f / static_cast<float>(M);
        out[static_cast<int64_t>(h) * T + t] =
            scalar_acc + QJL_SCALE * norm * qjl_acc;
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

} // namespace

torch::Tensor turboquant_scalar_only_4bit_lane_word_logits_b1q1_d128_cuda(
    torch::Tensor rotated_queries,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor centroids
) {
    validate_float_query(rotated_queries, "rotated_queries", D);
    validate_lane_word_codes(lane_word_scalar_codes);
    validate_centroids(centroids);
    TORCH_CHECK(rotated_queries.size(1) == lane_word_scalar_codes.size(1), "head mismatch");

    const int H = static_cast<int>(rotated_queries.size(1));
    const int T = static_cast<int>(lane_word_scalar_codes.size(2));

    auto out = torch::empty({1, H, 1, T}, rotated_queries.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    turboquant_scalar_only_4bit_lane_word_logits_b1q1_d128_kernel<<<grid, block>>>(
        rotated_queries.contiguous().data_ptr<float>(),
        lane_word_scalar_codes.contiguous().data_ptr<uint8_t>(),
        centroids.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_cuda(
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

    const int H = static_cast<int>(rotated_queries.size(1));
    const int T = static_cast<int>(lane_word_scalar_codes.size(2));

    auto out = torch::empty({1, H, 1, T}, rotated_queries.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_kernel<<<grid, block>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_scalar_only_4bit_lane_word_logits_b1q1_d128_cuda",
        &turboquant_scalar_only_4bit_lane_word_logits_b1q1_d128_cuda,
        "Scalar-only 4-bit lane-word code layout logits CUDA"
    );
    m.def(
        "turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_cuda",
        &turboquant_full_4bit_lane_word_scalar_lane_nibble_qjl128_logits_b1q1_d128_cuda,
        "Full TurboQuant: 4-bit scalar lane-word codes + lane-nibble QJL128 signs"
    );
}
