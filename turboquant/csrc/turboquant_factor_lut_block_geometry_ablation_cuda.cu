#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int M = 128;
constexpr int SCALAR_LEVELS = 16;
constexpr int LANE_WORD_PACKED_CODE_BYTES = D / 2;  // 64 bytes/token/head
constexpr int LANE_NIBBLE_SIGN_BYTES = M / 8;       // 16 bytes/token/head
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

/*
Block-geometry ablation for the current long-context best kernel:

  FP32 scalar factor LUT
  + lane-word 4-bit scalar codes
  + lane-nibble QJL signs
  + combined scalar/QJL reduction

Only launch/block geometry changes.

WARPS_PER_BLOCK:
  4  -> 128 threads/block
  8  -> 256 threads/block  (current production)
  16 -> 512 threads/block

Each warp still processes exactly one token/head logit.
*/
template <int WARPS_PER_BLOCK>
__global__ void turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_combined_geometry_logits_b1q1_d128_kernel(
    const float* __restrict__ scalar_factor_lut,
    const uint8_t* __restrict__ lane_word_scalar_codes,
    const float* __restrict__ qjl_projected_queries,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
    const float* __restrict__ residual_norms,
    float* __restrict__ out,
    int H,
    int T
) {
    static_assert(
        WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8 || WARPS_PER_BLOCK == 16,
        "WARPS_PER_BLOCK must be 4, 8, or 16."
    );

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

    if (h >= H || t >= T) {
        return;
    }

    const int64_t token_idx = static_cast<int64_t>(h) * T + t;
    const uint8_t* code_ptr =
        lane_word_scalar_codes + token_idx * LANE_WORD_PACKED_CODE_BYTES;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + token_idx * LANE_NIBBLE_SIGN_BYTES;
    const float* qp_ptr =
        qjl_projected_queries + static_cast<int64_t>(h) * M;

    const uint16_t lane_word = load_lane_code_word(code_ptr, lane);

    const uint8_t sign_pair_byte = sign_ptr[lane >> 1];
    const uint8_t lane_sign_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((sign_pair_byte >> 4) & 0x0Fu)
            : static_cast<uint8_t>(sign_pair_byte & 0x0Fu);

    float scalar_acc = 0.0f;
    float qjl_acc = 0.0f;

    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        const int coord = lane + stage * 32;

        const uint8_t scalar_code =
            static_cast<uint8_t>((lane_word >> (stage * 4)) & 0x0Fu);
        const int64_t lut_idx =
            (static_cast<int64_t>(h) * D + coord) * SCALAR_LEVELS
            + scalar_code;
        scalar_acc += scalar_factor_lut[lut_idx];

        const uint8_t sign_bit =
            static_cast<uint8_t>((lane_sign_nibble >> stage) & 0x01u);
        qjl_acc += qp_ptr[coord] * sign_from_bit(sign_bit);
    }

    float norm_lane0 = 0.0f;
    if (lane == 0) {
        norm_lane0 = residual_norms[token_idx];
    }
    const float norm = __shfl_sync(FULL_MASK, norm_lane0, 0);

    constexpr float QJL_SCALE =
        1.2533141373155001f / static_cast<float>(M);

    const float combined_acc =
        scalar_acc + QJL_SCALE * norm * qjl_acc;

    const float combined_sum = warp_reduce_sum(combined_acc);

    if (lane == 0) {
        out[token_idx] = combined_sum;
    }
}

void validate_factor_lut(torch::Tensor factor_lut) {
    TORCH_CHECK(factor_lut.is_cuda(), "scalar_factor_lut must be CUDA");
    TORCH_CHECK(factor_lut.dtype() == torch::kFloat32, "scalar_factor_lut must be float32");
    TORCH_CHECK(
        factor_lut.dim() == 4 &&
        factor_lut.size(0) == 1 &&
        factor_lut.size(2) == D &&
        factor_lut.size(3) == SCALAR_LEVELS,
        "scalar_factor_lut must be [1,H,128,16]"
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

void validate_qjl_query(torch::Tensor q) {
    TORCH_CHECK(q.is_cuda(), "qjl_projected_queries must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "qjl_projected_queries must be float32");
    TORCH_CHECK(
        q.dim() == 4 &&
        q.size(0) == 1 &&
        q.size(2) == 1 &&
        q.size(3) == M,
        "qjl_projected_queries must be [1,H,1,128]"
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

void validate_all(
    torch::Tensor scalar_factor_lut,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    validate_factor_lut(scalar_factor_lut);
    validate_lane_word_codes(lane_word_scalar_codes);
    validate_qjl_query(qjl_projected_queries);
    validate_lane_nibble_signs(lane_nibble_qjl_signs);
    validate_norms(residual_norms);

    TORCH_CHECK(scalar_factor_lut.size(1) == lane_word_scalar_codes.size(1), "head mismatch");
    TORCH_CHECK(scalar_factor_lut.size(1) == qjl_projected_queries.size(1), "head mismatch");
    TORCH_CHECK(scalar_factor_lut.size(1) == lane_nibble_qjl_signs.size(1), "head mismatch");
    TORCH_CHECK(scalar_factor_lut.size(1) == residual_norms.size(1), "head mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == lane_nibble_qjl_signs.size(2), "T mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == residual_norms.size(2), "T mismatch");
}

template <int WARPS_PER_BLOCK>
torch::Tensor launch_geometry(
    torch::Tensor scalar_factor_lut,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    validate_all(
        scalar_factor_lut,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms
    );

    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    const int H = static_cast<int>(scalar_factor_lut.size(1));
    const int T = static_cast<int>(lane_word_scalar_codes.size(2));

    auto out =
        torch::empty({1, H, 1, T}, scalar_factor_lut.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    turboquant_full_4bit_lane_word_factor_lut_lane_nibble_qjl128_combined_geometry_logits_b1q1_d128_kernel<WARPS_PER_BLOCK><<<grid, block>>>(
        scalar_factor_lut.contiguous().data_ptr<float>(),
        lane_word_scalar_codes.contiguous().data_ptr<uint8_t>(),
        qjl_projected_queries.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
        residual_norms.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace

torch::Tensor turboquant_factor_lut_combined_geometry_wpb4_cuda(
    torch::Tensor scalar_factor_lut,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    return launch_geometry<4>(
        scalar_factor_lut,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms
    );
}

torch::Tensor turboquant_factor_lut_combined_geometry_wpb8_cuda(
    torch::Tensor scalar_factor_lut,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    return launch_geometry<8>(
        scalar_factor_lut,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms
    );
}

torch::Tensor turboquant_factor_lut_combined_geometry_wpb16_cuda(
    torch::Tensor scalar_factor_lut,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    return launch_geometry<16>(
        scalar_factor_lut,
        lane_word_scalar_codes,
        qjl_projected_queries,
        lane_nibble_qjl_signs,
        residual_norms
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_factor_lut_combined_geometry_wpb4_cuda",
        &turboquant_factor_lut_combined_geometry_wpb4_cuda,
        "Factor-LUT combined kernel with 4 warps/block"
    );
    m.def(
        "turboquant_factor_lut_combined_geometry_wpb8_cuda",
        &turboquant_factor_lut_combined_geometry_wpb8_cuda,
        "Factor-LUT combined kernel with 8 warps/block"
    );
    m.def(
        "turboquant_factor_lut_combined_geometry_wpb16_cuda",
        &turboquant_factor_lut_combined_geometry_wpb16_cuda,
        "Factor-LUT combined kernel with 16 warps/block"
    );
}
