#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

old = '''            rotated_queries = rotate(query_states, state.rotation).to(torch.float32).contiguous()
            qjl_projected_queries = qjl_project_query(query_states, state.sketch).to(torch.float32).contiguous()

            tq_logits, adapter = _try_call_nonfactor_kernel(
'''
new = '''            rotated_queries = rotate(query_states, state.rotation).to(torch.float32).contiguous()
            qjl_projected_queries = qjl_project_query(query_states, state.sketch).to(torch.float32).contiguous()

            # Integration shape diagnostic. This is intentionally printed before
            # the CUDA wrapper so the next failure identifies whether the decode
            # query/cache states match the already-validated microbenchmark layout.
            print(
                json.dumps(
                    {
                        "tq_decode_pre_kernel_shapes": True,
                        "layer_idx": int(state.layer_idx),
                        "hidden_states_shape": list(hidden_states.shape),
                        "query_states_shape": list(query_states.shape),
                        "key_states_new_shape": list(key_states_new.shape),
                        "value_states_new_shape": list(value_states_new.shape),
                        "full_keys_kv_shape": list(full_keys_kv.shape),
                        "full_values_kv_shape": list(full_values_kv.shape),
                        "full_keys_expanded_shape": list(full_keys.shape),
                        "full_values_expanded_shape": list(full_values.shape),
                        "rotated_queries_shape": list(rotated_queries.shape),
                        "qjl_projected_queries_shape": list(qjl_projected_queries.shape),
                        "lane_word_scalar_codes_shape": list(state.scalar_lane_words.shape),
                        "lane_nibble_qjl_signs_shape": list(state.qjl_lane_nibbles.shape),
                        "residual_norms_shape": list(state.residual_norms.shape),
                        "centroids_shape": list(state.centroids.shape),
                    }
                ),
                flush=True,
            )

            tq_logits, adapter = _try_call_nonfactor_kernel(
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find pre-kernel query block.")

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("[OK] Added decode pre-kernel shape diagnostic.")
