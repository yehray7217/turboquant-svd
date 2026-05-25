#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

old = "from dataclasses import dataclass\n"
new = "from dataclasses import dataclass, field\n"
if old not in s:
    raise SystemExit("[FAIL] Could not find dataclasses import.")
s = s.replace(old, new, 1)

marker = "\n\n@dataclass\nclass PackedLayerState:"
helper = r'''

def _profile_cuda_ms(enabled: bool, fn):
    # Diagnostic-only synchronous CUDA event timing.
    # This intentionally synchronizes after each component so attribution is clear.
    # Do not use profiling-mode total runtime as a production latency number.
    if not enabled:
        return fn(), None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))
'''
if marker not in s:
    raise SystemExit("[FAIL] Could not find PackedLayerState marker.")
s = s.replace(marker, helper + marker, 1)

old = '''    kernel_adapter: Optional[str] = None
    last_kv_len: int = 0
'''
new = '''    kernel_adapter: Optional[str] = None
    last_kv_len: int = 0
    component_ms_total: dict[str, float] = field(default_factory=dict)
    component_ms_count: dict[str, int] = field(default_factory=dict)

    def add_component_ms(self, name: str, ms: Optional[float]) -> None:
        if ms is None:
            return
        self.component_ms_total[name] = float(self.component_ms_total.get(name, 0.0) + float(ms))
        self.component_ms_count[name] = int(self.component_ms_count.get(name, 0) + 1)
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find PackedLayerState field block.")
s = s.replace(old, new, 1)

old = '''        codebook_seed: int,
        verbose_build: bool,
    ) -> None:
'''
new = '''        codebook_seed: int,
        verbose_build: bool,
        profile_components: bool,
    ) -> None:
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find patcher __init__ signature block.")
s = s.replace(old, new, 1)

old = '''        self.codebook_seed = int(codebook_seed)
        self.verbose_build = bool(verbose_build)
        self.states: dict[int, PackedLayerState] = {}
'''
new = '''        self.codebook_seed = int(codebook_seed)
        self.verbose_build = bool(verbose_build)
        self.profile_components = bool(profile_components)
        self.states: dict[int, PackedLayerState] = {}
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find patcher __init__ body block.")
s = s.replace(old, new, 1)

start_anchor = '''            query_states = _reshape_projected(
                self_module.q_proj(hidden_states),
                bsz=bsz,
                seqlen=q_len,
                heads=num_heads,
                head_dim=head_dim,
            )
'''
start = s.find(start_anchor)
if start < 0:
    raise SystemExit("[FAIL] Could not find decode replacement q_proj anchor.")

end_anchor = '''            # User's current stack accepts this LLaMA attention return structure.
            return attn_output, None, present
'''
end = s.find(end_anchor, start)
if end < 0:
    raise SystemExit("[FAIL] Could not find decode replacement return anchor.")
end += len(end_anchor)

replacement = r'''            def _project_qkv():
                query_states_local = _reshape_projected(
                    self_module.q_proj(hidden_states),
                    bsz=bsz,
                    seqlen=q_len,
                    heads=num_heads,
                    head_dim=head_dim,
                )
                key_states_new_local = _reshape_projected(
                    self_module.k_proj(hidden_states),
                    bsz=bsz,
                    seqlen=q_len,
                    heads=num_kv_heads,
                    head_dim=head_dim,
                )
                value_states_new_local = _reshape_projected(
                    self_module.v_proj(hidden_states),
                    bsz=bsz,
                    seqlen=q_len,
                    heads=num_kv_heads,
                    head_dim=head_dim,
                )
                return query_states_local, key_states_new_local, value_states_new_local

            (query_states, key_states_new, value_states_new), ms = _profile_cuda_ms(
                patcher.profile_components,
                _project_qkv,
            )
            state.add_component_ms("qkv_projection", ms)

            position_ids = kwargs.get("position_ids", None)
            position_embeddings = kwargs.get("position_embeddings", None)

            def _rope_and_cache():
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

            def _repeat_kv_states():
                full_keys_local = _repeat_kv(full_keys_kv, num_kv_groups).contiguous()
                full_values_local = _repeat_kv(full_values_kv, num_kv_groups).contiguous()
                new_keys_expanded_local = _repeat_kv(key_states_new, num_kv_groups).contiguous()
                return full_keys_local, full_values_local, new_keys_expanded_local

            (full_keys, full_values, new_keys_expanded), ms = _profile_cuda_ms(
                patcher.profile_components,
                _repeat_kv_states,
            )
            state.add_component_ms("repeat_kv_materialize", ms)

            if not state.ready():
                _, ms = _profile_cuda_ms(
                    patcher.profile_components,
                    lambda: patcher._fit_state_from_full_k(
                        state=state,
                        full_keys=full_keys,
                        head_dim=head_dim,
                    ),
                )
                state.add_component_ms("prime_build_full_prefix_state", ms)
            else:
                _, ms = _profile_cuda_ms(
                    patcher.profile_components,
                    lambda: patcher._append_new_k(
                        state=state,
                        new_keys=new_keys_expanded,
                    ),
                )
                state.add_component_ms("append_encode_pack_cat_new_k", ms)

            assert state.rotation is not None
            assert state.sketch is not None
            assert state.centroids is not None
            assert state.scalar_lane_words is not None
            assert state.qjl_lane_nibbles is not None
            assert state.residual_norms is not None

            def _prepare_query_factors():
                rotated_queries_local = rotate(query_states, state.rotation).to(torch.float32).contiguous()
                qjl_projected_queries_local = qjl_project_query(query_states, state.sketch).to(torch.float32).contiguous()
                return rotated_queries_local, qjl_projected_queries_local

            (rotated_queries, qjl_projected_queries), ms = _profile_cuda_ms(
                patcher.profile_components,
                _prepare_query_factors,
            )
            state.add_component_ms("rotate_q_and_qjl_project_q", ms)

            def _kernel_call():
                return _try_call_nonfactor_kernel(
                    rotated_queries=rotated_queries,
                    qjl_projected_queries=qjl_projected_queries,
                    centroids=state.centroids,
                    scalar_lane_words=state.scalar_lane_words,
                    qjl_lane_nibbles=state.qjl_lane_nibbles,
                    residual_norms=state.residual_norms,
                )

            (tq_logits, adapter), ms = _profile_cuda_ms(
                patcher.profile_components,
                _kernel_call,
            )
            state.add_component_ms("turboquant_cuda_logits_kernel_wrapper", ms)
            state.kernel_adapter = adapter

            scale = float(getattr(self_module, "scaling", 1.0 / math.sqrt(float(head_dim))))
            attn_logits = tq_logits.to(torch.float32) * float(scale)

            def _post_logits_attention():
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

            state.decode_calls += 1
            state.last_kv_len = int(full_keys.shape[-2])

            # User's current stack accepts this LLaMA attention return structure.
            return attn_output, None, present
'''
s = s[:start] + replacement + s[end:]

old = '''                "residual_norm_shape": list(v.residual_norms.shape) if v.residual_norms is not None else None,
            }
'''
new = '''                "residual_norm_shape": list(v.residual_norms.shape) if v.residual_norms is not None else None,
                "component_ms_total": dict(v.component_ms_total),
                "component_ms_count": dict(v.component_ms_count),
                "component_ms_mean": {
                    name: (
                        float(v.component_ms_total[name] / v.component_ms_count[name])
                        if int(v.component_ms_count.get(name, 0)) > 0 else None
                    )
                    for name in v.component_ms_total
                },
            }
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find summary shape block.")
s = s.replace(old, new, 1)

old = '''    p.add_argument("--quiet_build", action="store_true")
'''
new = '''    p.add_argument("--quiet_build", action="store_true")
    p.add_argument(
        "--profile_components",
        action="store_true",
        help="Synchronous CUDA-event profiling of replacement subcomponents. Diagnostic only; do not use total runtime as a performance result.",
    )
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find --quiet_build argparse line.")
s = s.replace(old, new, 1)

old = '''        codebook_seed=int(args.codebook_seed),
        verbose_build=not bool(args.quiet_build),
    )
'''
new = '''        codebook_seed=int(args.codebook_seed),
        verbose_build=not bool(args.quiet_build),
        profile_components=bool(args.profile_components),
    )
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find TurboQuantDecodeAttentionPatcher constructor call.")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("[OK] Added synchronous component profiling for decode replacement path.")
