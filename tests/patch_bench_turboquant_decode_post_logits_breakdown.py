#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

old = '''            def _post_logits_attention():
                attn_logits_local = attn_logits
                mask = _normalize_mask(
                    kwargs.get("attention_mask", None),
                    q_len=q_len,
                    kv_len=int(full_keys.shape[-2]),
                    dtype=attn_logits_local.dtype,
                    device=attn_logits_local.device,
                )
                if mask is not None:
                    attn_logits_local = attn_logits_local + mask

                attn_probs = torch.softmax(attn_logits_local, dim=-1, dtype=torch.float32).to(full_values.dtype)
                attn_output_local = torch.matmul(attn_probs, full_values.to(attn_probs.dtype))
                attn_output_local = attn_output_local.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_size)
                attn_output_local = self_module.o_proj(attn_output_local.to(hidden_states.dtype))
                return attn_output_local

            attn_output, ms = _profile_cuda_ms(
                patcher.profile_components,
                _post_logits_attention,
            )
            state.add_component_ms("mask_softmax_matmul_v_o_proj", ms)
'''

new = '''            def _mask_build_only():
                return _normalize_mask(
                    kwargs.get("attention_mask", None),
                    q_len=q_len,
                    kv_len=int(full_keys.shape[-2]),
                    dtype=attn_logits.dtype,
                    device=attn_logits.device,
                )

            mask, ms = _profile_cuda_ms(
                patcher.profile_components,
                _mask_build_only,
            )
            state.add_component_ms("post_logits_mask_build", ms)

            def _mask_add_only():
                if mask is None:
                    return attn_logits
                return attn_logits + mask

            attn_logits_masked, ms = _profile_cuda_ms(
                patcher.profile_components,
                _mask_add_only,
            )
            state.add_component_ms("post_logits_mask_add", ms)

            def _softmax_only():
                return torch.softmax(
                    attn_logits_masked,
                    dim=-1,
                    dtype=torch.float32,
                )

            attn_probs_fp32, ms = _profile_cuda_ms(
                patcher.profile_components,
                _softmax_only,
            )
            state.add_component_ms("post_logits_softmax_fp32", ms)

            def _probs_cast_only():
                return attn_probs_fp32.to(full_values.dtype)

            attn_probs, ms = _profile_cuda_ms(
                patcher.profile_components,
                _probs_cast_only,
            )
            state.add_component_ms("post_logits_probs_cast_to_v_dtype", ms)

            def _values_cast_only():
                return full_values.to(attn_probs.dtype)

            full_values_for_matmul, ms = _profile_cuda_ms(
                patcher.profile_components,
                _values_cast_only,
            )
            state.add_component_ms("post_logits_values_cast_to_probs_dtype", ms)

            def _matmul_probs_v_only():
                return torch.matmul(attn_probs, full_values_for_matmul)

            attn_output_heads, ms = _profile_cuda_ms(
                patcher.profile_components,
                _matmul_probs_v_only,
            )
            state.add_component_ms("post_logits_matmul_probs_v", ms)

            def _reshape_only():
                return (
                    attn_output_heads
                    .transpose(1, 2)
                    .contiguous()
                    .reshape(bsz, q_len, hidden_size)
                )

            attn_output_reshaped, ms = _profile_cuda_ms(
                patcher.profile_components,
                _reshape_only,
            )
            state.add_component_ms("post_logits_transpose_contiguous_reshape", ms)

            def _o_proj_input_cast_only():
                return attn_output_reshaped.to(hidden_states.dtype)

            attn_output_for_o_proj, ms = _profile_cuda_ms(
                patcher.profile_components,
                _o_proj_input_cast_only,
            )
            state.add_component_ms("post_logits_o_proj_input_cast", ms)

            def _o_proj_only():
                return self_module.o_proj(attn_output_for_o_proj)

            attn_output, ms = _profile_cuda_ms(
                patcher.profile_components,
                _o_proj_only,
            )
            state.add_component_ms("post_logits_o_proj", ms)
'''

if old not in s:
    raise SystemExit(
        "[FAIL] Could not find existing mask_softmax_matmul_v_o_proj block. "
        "Apply after the component profiling patches currently in the repo."
    )

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("[OK] Split post-logits attention path into mask / softmax / casts / matmul / reshape / o_proj.")
