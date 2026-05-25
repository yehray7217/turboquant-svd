#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1) Ensure the lower-level encode helpers are imported.
#    This mirrors turboquant.turboquant_prod.encode_turboquant_prod_keys:
#      rotate -> scalar quantize -> scalar dequantize -> inverse rotate
#      -> residual -> qjl_encode_residual
# ---------------------------------------------------------------------
if "from turboquant.qjl import qjl_project_query, qjl_encode_residual" not in s:
    old = "from turboquant.qjl import qjl_project_query\n"
    new = "from turboquant.qjl import qjl_project_query, qjl_encode_residual\n"
    if old not in s:
        raise SystemExit("[FAIL] Could not find qjl_project_query import line.")
    s = s.replace(old, new, 1)

if "from turboquant.rotation import rotate, inverse_rotate" not in s:
    old = "from turboquant.rotation import rotate\n"
    new = "from turboquant.rotation import rotate, inverse_rotate\n"
    if old not in s:
        raise SystemExit("[FAIL] Could not find rotate import line.")
    s = s.replace(old, new, 1)

if "from turboquant.scalar_quant import scalar_quantize, scalar_dequantize" not in s:
    anchor = "from turboquant.rotation import rotate, inverse_rotate\n"
    if anchor not in s:
        raise SystemExit("[FAIL] Could not find rotation import anchor for scalar_quant import.")
    s = s.replace(
        anchor,
        anchor + "from turboquant.scalar_quant import scalar_quantize, scalar_dequantize\n",
        1,
    )

# ---------------------------------------------------------------------
# 2) Replace _append_new_k() with encode-internal breakdown.
#    This assumes the v2 append breakdown patch is already applied.
# ---------------------------------------------------------------------
start = s.find("    def _append_new_k(")
if start < 0:
    raise SystemExit("[FAIL] Could not find _append_new_k().")

end = s.find("\n    def _make_forward(", start)
if end < 0:
    raise SystemExit("[FAIL] Could not find end marker after _append_new_k().")

new_append = r'''    def _append_new_k(
        self,
        *,
        state: PackedLayerState,
        new_keys: torch.Tensor,
    ) -> None:
        assert state.rotation is not None
        assert state.sketch is not None
        assert state.centroids is not None
        assert state.scalar_lane_words is not None
        assert state.qjl_lane_nibbles is not None
        assert state.residual_norms is not None

        # -----------------------------------------------------------------
        # Encode new K, split using the same sequence as:
        # turboquant.turboquant_prod.encode_turboquant_prod_keys()
        # -----------------------------------------------------------------
        rotated_keys, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: rotate(new_keys, state.rotation).to(torch.float32),
        )
        state.add_component_ms("encode_rotate_new_k", ms)

        codes_raw, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: scalar_quantize(rotated_keys, state.centroids),
        )
        state.add_component_ms("encode_scalar_quantize_new_k", ms)

        reconstructed_rotated, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: scalar_dequantize(codes_raw, state.centroids),
        )
        state.add_component_ms("encode_scalar_dequantize_new_k", ms)

        reconstructed_keys, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: inverse_rotate(reconstructed_rotated, state.rotation).to(torch.float32),
        )
        state.add_component_ms("encode_inverse_rotate_reconstruct_new_k", ms)

        residual, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: new_keys.to(torch.float32) - reconstructed_keys,
        )
        state.add_component_ms("encode_residual_subtract_new_k", ms)

        (residual_signs_raw, residual_norms_raw), ms = _profile_cuda_ms(
            self.profile_components,
            lambda: qjl_encode_residual(residual, state.sketch),
        )
        state.add_component_ms("encode_qjl_encode_residual_new_k", ms)

        codes, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: codes_raw.contiguous(),
        )
        state.add_component_ms("encode_contiguous_codes_new_k", ms)

        residual_signs, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: residual_signs_raw.contiguous(),
        )
        state.add_component_ms("encode_contiguous_residual_signs_new_k", ms)

        norms_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: residual_norms_raw.contiguous(),
        )
        state.add_component_ms("encode_contiguous_residual_norms_new_k", ms)

        # -----------------------------------------------------------------
        # Pack + append compressed state.
        # These names are kept compatible with the prior v2 breakdown.
        # -----------------------------------------------------------------
        scalar_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: pack_scalar_codes_lane_word_4bit(codes).contiguous(),
        )
        state.add_component_ms("pack_scalar_codes_new_k", ms)

        qjl_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: pack_qjl_signs_lane_nibble(residual_signs).contiguous(),
        )
        state.add_component_ms("pack_qjl_signs_new_k", ms)

        def _cat_scalar():
            return torch.cat([state.scalar_lane_words, scalar_new], dim=-2).contiguous()

        scalar_concat, ms = _profile_cuda_ms(
            self.profile_components,
            _cat_scalar,
        )
        state.add_component_ms("cat_append_scalar_lane_words", ms)
        state.scalar_lane_words = scalar_concat

        def _cat_qjl():
            return torch.cat([state.qjl_lane_nibbles, qjl_new], dim=-2).contiguous()

        qjl_concat, ms = _profile_cuda_ms(
            self.profile_components,
            _cat_qjl,
        )
        state.add_component_ms("cat_append_qjl_lane_nibbles", ms)
        state.qjl_lane_nibbles = qjl_concat

        def _cat_norms():
            return torch.cat([state.residual_norms, norms_new], dim=-1).contiguous()

        norms_concat, ms = _profile_cuda_ms(
            self.profile_components,
            _cat_norms,
        )
        state.add_component_ms("cat_append_residual_norms", ms)
        state.residual_norms = norms_concat

        state.append_calls += 1
        state.last_kv_len = int(state.scalar_lane_words.shape[-2])
'''

s = s[:start] + new_append + s[end:]
p.write_text(s, encoding="utf-8")
print("[OK] Split new-K encode into rotate / scalar quantize / dequantize / inverse rotate / residual / QJL internals.")
