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
constexpr int PAIRS = 2;
constexpr int LANES = 32;
constexpr int PAIR_CODES = 256;
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
Pair-factor-LUT combined-reduction kernel.

Current factor-LUT combined scalar path:
  scalar_acc =
      factor_lut[coord0, code0]
    + factor_lut[coord1, code1]
    + factor_lut[coord2, code2]
    + factor_lut[coord3, code3]

Pair-factor-LUT scalar path:
  byte01 = [code1 | code0]
  byte23 = [code3 | code2]

  scalar_acc =
      pair_lut[pair01, lane, byte01]
    + pair_lut[pair23, lane, byte23]

The objective is to reduce four highly uncoalesced scalar factor-LUT global
loads to two larger pair-LUT global loads while preserving the current
warp-per-token / combined-reduction structure.
*/
__global__ void turboquant_full_4bit_lane_word_pair_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_kernel(
    const float* __restrict__ pair_factor_lut,
    const uint8_t* __restrict__ lane_word_scalar_codes,
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

    const int64_t token_idx = static_cast<int64_t>(h) * T + t;
    const uint8_t* code_ptr =
        lane_word_scalar_codes + token_idx * LANE_WORD_PACKED_CODE_BYTES;
    const uint8_t* sign_ptr =
        lane_nibble_qjl_signs + token_idx * LANE_NIBBLE_SIGN_BYTES;
    const float* qp_ptr =
        qjl_projected_queries + static_cast<int64_t>(h) * M;

    const uint16_t lane_word = load_lane_code_word(code_ptr, lane);
    const uint8_t pair_code_01 = static_cast<uint8_t>(lane_word & 0x00FFu);
    const uint8_t pair_code_23 = static_cast<uint8_t>((lane_word >> 8) & 0x00FFu);

    const int64_t pair01_idx =
        (((static_cast<int64_t>(h) * PAIRS + 0) * LANES + lane) * PAIR_CODES)
        + pair_code_01;
    const int64_t pair23_idx =
        (((static_cast<int64_t>(h) * PAIRS + 1) * LANES + lane) * PAIR_CODES)
        + pair_code_23;

    float scalar_acc =
        pair_factor_lut[pair01_idx]
        + pair_factor_lut[pair23_idx];

    const uint8_t sign_pair_byte = sign_ptr[lane >> 1];
    const uint8_t lane_sign_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((sign_pair_byte >> 4) & 0x0Fu)
            : static_cast<uint8_t>(sign_pair_byte & 0x0Fu);

    float qjl_acc = 0.0f;
    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        const int coord = lane + stage * 32;
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

void validate_pair_factor_lut(torch::Tensor pair_lut) {
    TORCH_CHECK(pair_lut.is_cuda(), "pair_factor_lut must be CUDA");
    TORCH_CHECK(pair_lut.dtype() == torch::kFloat32, "pair_factor_lut must be float32");
    TORCH_CHECK(
        pair_lut.dim() == 5 &&
        pair_lut.size(0) == 1 &&
        pair_lut.size(2) == PAIRS &&
        pair_lut.size(3) == LANES &&
        pair_lut.size(4) == PAIR_CODES,
        "pair_factor_lut must be [1,H,2,32,256]"
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

} // namespace

torch::Tensor turboquant_full_4bit_lane_word_pair_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
    torch::Tensor pair_factor_lut,
    torch::Tensor lane_word_scalar_codes,
    torch::Tensor qjl_projected_queries,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms
) {
    validate_pair_factor_lut(pair_factor_lut);
    validate_lane_word_codes(lane_word_scalar_codes);
    validate_qjl_query(qjl_projected_queries);
    validate_lane_nibble_signs(lane_nibble_qjl_signs);
    validate_norms(residual_norms);

    TORCH_CHECK(pair_factor_lut.size(1) == lane_word_scalar_codes.size(1), "head mismatch");
    TORCH_CHECK(pair_factor_lut.size(1) == qjl_projected_queries.size(1), "head mismatch");
    TORCH_CHECK(pair_factor_lut.size(1) == lane_nibble_qjl_signs.size(1), "head mismatch");
    TORCH_CHECK(pair_factor_lut.size(1) == residual_norms.size(1), "head mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == lane_nibble_qjl_signs.size(2), "T mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == residual_norms.size(2), "T mismatch");

    const int H = static_cast<int>(pair_factor_lut.size(1));
    const int T = static_cast<int>(lane_word_scalar_codes.size(2));

    auto out =
        torch::empty({1, H, 1, T}, pair_factor_lut.options().dtype(torch::kFloat32));
    const dim3 block(THREADS, 1, 1);
    const dim3 grid((T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, 1);

    turboquant_full_4bit_lane_word_pair_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_kernel<<<grid, block>>>(
        pair_factor_lut.contiguous().data_ptr<float>(),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_full_4bit_lane_word_pair_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda",
        &turboquant_full_4bit_lane_word_pair_factor_lut_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda,
        "Full TurboQuant combined-reduction kernel with 2-stage scalar pair-factor LUT"
    );
}
