#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1) Split rope_and_cache_update into rope_apply + cache_update_only
# ---------------------------------------------------------------------
old = r'''            def _rope_and_cache():
                query_states_local, key_states_new_local = _safe_apply_rope(
                    self_module,
                    query_states,
                    key_states_new,
                    value_states_new,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                full_keys_kv_local, full_values_kv_local, present_local = _cache_update(
                    module=self_module,
                    past=past,
                    key_states=key_states_new_local,
                    value_states=value_states_new,
                    kwargs=kwargs,
                )
                return (
                    query_states_local,
                    key_states_new_local,
                    full_keys_kv_local,
                    full_values_kv_local,
                    present_local,
                )

            (
                query_states,
                key_states_new,
                full_keys_kv,
                full_values_kv,
                present,
            ), ms = _profile_cuda_ms(
                patcher.profile_components,
                _rope_and_cache,
            )
            state.add_component_ms("rope_and_cache_update", ms)
'''

new = r'''            def _rope_apply_only():
                return _safe_apply_rope(
                    self_module,
                    query_states,
                    key_states_new,
                    value_states_new,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

            (query_states, key_states_new), ms = _profile_cuda_ms(
                patcher.profile_components,
                _rope_apply_only,
            )
            state.add_component_ms("rope_apply", ms)

            def _cache_update_only():
                return _cache_update(
                    module=self_module,
                    past=past,
                    key_states=key_states_new,
                    value_states=value_states_new,
                    kwargs=kwargs,
                )

            (full_keys_kv, full_values_kv, present), ms = _profile_cuda_ms(
                patcher.profile_components,
                _cache_update_only,
            )
            state.add_component_ms("cache_update_only", ms)
'''

if old not in s:
    raise SystemExit("[FAIL] Could not find rope_and_cache_update profiling block. Make sure v1 component profiling patch is already applied.")
s = s.replace(old, new, 1)

# ---------------------------------------------------------------------
# 2) Replace _append_new_k() with fine-grained profiling
# ---------------------------------------------------------------------
start = s.find("    def _append_new_k(")
if start < 0:
    raise SystemExit("[FAIL] Could not find _append_new_k().")

end = s.find("\n    def _make_forward(", start)
if end < 0:
    raise SystemExit("[FAIL] Could not find end of _append_new_k().")

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

        def _encode_new_k_only():
            return encode_turboquant_prod_keys(
                new_keys,
                rotation=state.rotation,
                centroids=state.centroids,
                sketch=state.sketch,
            )

        encoding, ms = _profile_cuda_ms(
            self.profile_components,
            _encode_new_k_only,
        )
        state.add_component_ms("encode_new_k", ms)

        scalar_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: pack_scalar_codes_lane_word_4bit(encoding.codes).contiguous(),
        )
        state.add_component_ms("pack_scalar_codes_new_k", ms)

        qjl_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: pack_qjl_signs_lane_nibble(encoding.residual_signs).contiguous(),
        )
        state.add_component_ms("pack_qjl_signs_new_k", ms)

        norms_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: encoding.residual_norms.contiguous(),
        )
        state.add_component_ms("materialize_residual_norms_new_k", ms)

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
print("[OK] Added v2 profiling split: rope/cache and append encode/pack/cat breakdown.")
