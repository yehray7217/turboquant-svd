#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re


WRAPPER = Path("turboquant/turboquant_combined_reduction_nonfactor_ablation_cuda.py")
CUDA = Path("turboquant/csrc/turboquant_combined_reduction_nonfactor_ablation_cuda.cu")
BENCH = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def patch_wrapper() -> None:
    p = WRAPPER
    s = p.read_text(encoding="utf-8")

    if "active_kv_len: int | None = None" not in s:
        old = '''    residual_norms: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
'''
        new = '''    residual_norms: torch.Tensor,
    centroids: torch.Tensor,
    active_kv_len: int | None = None,
) -> torch.Tensor:
'''
        if old not in s:
            fail("Could not find wrapper signature tail.")
        s = s.replace(old, new, 1)

    if "active_kv_len_i = " not in s:
        old = '''    return _load_ext().turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
        rotated_queries.contiguous().to(torch.float32),
        lane_word_scalar_codes.contiguous(),
        qjl_projected_queries.contiguous().to(torch.float32),
        lane_nibble_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
        centroids.contiguous().to(torch.float32),
    )
'''
        new = '''    active_kv_len_i = (
        int(lane_word_scalar_codes.shape[2])
        if active_kv_len is None
        else int(active_kv_len)
    )

    return _load_ext().turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda(
        rotated_queries.contiguous().to(torch.float32),
        lane_word_scalar_codes.contiguous(),
        qjl_projected_queries.contiguous().to(torch.float32),
        lane_nibble_qjl_signs.contiguous(),
        residual_norms.contiguous().to(torch.float32),
        centroids.contiguous().to(torch.float32),
        active_kv_len_i,
    )
'''
        if old not in s:
            fail("Could not find wrapper extension call block.")
        s = s.replace(old, new, 1)

    p.write_text(s, encoding="utf-8")
    print("[OK] Patched Python wrapper with optional active_kv_len.")


def patch_cuda() -> None:
    p = CUDA
    s = p.read_text(encoding="utf-8")

    if "int64_t active_kv_len" not in s:
        old = '''    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids
) {
'''
        new = '''    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor residual_norms,
    torch::Tensor centroids,
    int64_t active_kv_len
) {
'''
        if old not in s:
            fail("Could not find CUDA extension function signature tail.")
        s = s.replace(old, new, 1)

    old = '''    TORCH_CHECK(lane_word_scalar_codes.size(2) == lane_nibble_qjl_signs.size(2), "T mismatch");
    TORCH_CHECK(lane_word_scalar_codes.size(2) == residual_norms.size(2), "T mismatch");

    const int H = static_cast<int>(rotated_queries.size(1));
    const int T = static_cast<int>(lane_word_scalar_codes.size(2));
'''
    new = '''    TORCH_CHECK(
        lane_word_scalar_codes.size(2) == lane_nibble_qjl_signs.size(2),
        "storage T mismatch: lane_word_scalar_codes vs lane_nibble_qjl_signs"
    );
    TORCH_CHECK(
        lane_word_scalar_codes.size(2) == residual_norms.size(2),
        "storage T mismatch: lane_word_scalar_codes vs residual_norms"
    );
    TORCH_CHECK(
        active_kv_len >= 0 &&
        active_kv_len <= lane_word_scalar_codes.size(2),
        "active_kv_len must satisfy 0 <= active_kv_len <= storage T; got ",
        active_kv_len,
        " with storage T=",
        lane_word_scalar_codes.size(2)
    );

    const int H = static_cast<int>(rotated_queries.size(1));
    const int T = static_cast<int>(active_kv_len);
'''
    if old in s:
        s = s.replace(old, new, 1)
    elif "active_kv_len <= lane_word_scalar_codes.size(2)" not in s:
        fail("Could not patch CUDA T checks and T assignment.")

    p.write_text(s, encoding="utf-8")
    print("[OK] Patched CUDA extension to use active_kv_len as logical T.")


def insert_active_storage_helper(s: str) -> str:
    if "_active_compressed_cache_inputs" in s:
        return s

    marker = "    def _make_forward(\n"
    helper = '''    def _active_compressed_cache_inputs(
        self,
        state: PackedLayerState,
    ):
        """
        Return full contiguous storage plus logical active KV length when
        preallocation is active. Fall back to active tensors otherwise.
        """
        if (
            state.scalar_lane_words_storage is not None
            and state.qjl_lane_nibbles_storage is not None
            and state.residual_norms_storage is not None
            and int(state.compressed_cache_len) > 0
        ):
            return (
                state.scalar_lane_words_storage,
                state.qjl_lane_nibbles_storage,
                state.residual_norms_storage,
                int(state.compressed_cache_len),
            )

        assert state.scalar_lane_words is not None
        assert state.qjl_lane_nibbles is not None
        assert state.residual_norms is not None
        return (
            state.scalar_lane_words,
            state.qjl_lane_nibbles,
            state.residual_norms,
            int(state.scalar_lane_words.shape[-2]),
        )

'''
    if marker not in s:
        fail("Could not find _make_forward marker for helper insertion.")
    return s.replace(marker, helper + marker, 1)


def locate_kernel_call_line(s: str) -> int:
    needles = [
        "lane_word_scalar_codes=state.scalar_lane_words",
        "lane_word_scalar_codes = state.scalar_lane_words",
    ]
    positions = [s.find(n) for n in needles if s.find(n) >= 0]
    if not positions:
        fail("Could not find kernel call kwarg `lane_word_scalar_codes=state.scalar_lane_words`.")
    return min(positions)


def patch_benchmark_callsite(s: str) -> str:
    pos = locate_kernel_call_line(s)
    line_start = s.rfind("\n", 0, pos) + 1
    line = s[line_start:s.find("\n", line_start)]
    indent = re.match(r"\s*", line).group(0)

    # Insert locals once near the callsite.
    nearby = s[max(0, line_start - 600): line_start + 600]
    if "active_scalar_lane_words" not in nearby:
        active_block = (
            f"{indent}(\n"
            f"{indent}    active_scalar_lane_words,\n"
            f"{indent}    active_qjl_lane_nibbles,\n"
            f"{indent}    active_residual_norms,\n"
            f"{indent}    active_kv_len,\n"
            f"{indent}) = patcher._active_compressed_cache_inputs(state)\n\n"
        )
        s = s[:line_start] + active_block + s[line_start:]

    s = s.replace(
        "lane_word_scalar_codes=state.scalar_lane_words",
        "lane_word_scalar_codes=active_scalar_lane_words",
    )
    s = s.replace(
        "lane_nibble_qjl_signs=state.qjl_lane_nibbles",
        "lane_nibble_qjl_signs=active_qjl_lane_nibbles",
    )
    s = s.replace(
        "residual_norms=state.residual_norms",
        "residual_norms=active_residual_norms",
    )

    # Insert active_kv_len kwarg directly after centroids=state.centroids
    pattern = re.compile(r"(?P<indent>\s*)centroids=state\.centroids,\n")
    matches = list(pattern.finditer(s))
    inserted = 0
    rebuilt = []
    last = 0
    for m in matches:
        rebuilt.append(s[last:m.end()])
        window = s[max(0, m.start() - 500): min(len(s), m.end() + 300)]
        if (
            "lane_word_scalar_codes=active_scalar_lane_words" in window
            and "active_kv_len=active_kv_len" not in window
        ):
            rebuilt.append(f"{m.group('indent')}active_kv_len=active_kv_len,\n")
            inserted += 1
        last = m.end()
    rebuilt.append(s[last:])
    s = "".join(rebuilt)

    if inserted < 1 and "active_kv_len=active_kv_len" not in s:
        fail("Could not insert active_kv_len kwarg at kernel callsite.")

    return s


def patch_benchmark_adapter_attempts(s: str) -> str:
    # The benchmark has adapter/signature-heuristic machinery in some revisions.
    # If explicit accepted keyword lists exist, add active_kv_len after centroids.
    if '"active_kv_len"' not in s and "'active_kv_len'" not in s:
        s = s.replace(
            '"centroids",\n',
            '"centroids",\n            "active_kv_len",\n',
        )
        s = s.replace(
            "'centroids',\n",
            "'centroids',\n            'active_kv_len',\n",
        )
    return s


def patch_benchmark() -> None:
    p = BENCH
    s = p.read_text(encoding="utf-8")
    s = insert_active_storage_helper(s)
    s = patch_benchmark_callsite(s)
    s = patch_benchmark_adapter_attempts(s)
    p.write_text(s, encoding="utf-8")
    print("[OK] Patched benchmark to pass full contiguous storage + active_kv_len.")


def main() -> None:
    patch_wrapper()
    patch_cuda()
    patch_benchmark()
    print("[OK] active_kv_len full-storage patch completed.")


if __name__ == "__main__":
    main()
