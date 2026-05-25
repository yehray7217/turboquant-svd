#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

old = """    component_ms_total: dict[str, float] = field(default_factory=dict)
    component_ms_count: dict[str, int] = field(default_factory=dict)

    def add_component_ms(self, name: str, ms: Optional[float]) -> None:
"""
new = """    component_ms_total: dict[str, float] = field(default_factory=dict)
    component_ms_count: dict[str, int] = field(default_factory=dict)
    cache_update_path_counts: dict[str, int] = field(default_factory=dict)
    cache_update_past_type_counts: dict[str, int] = field(default_factory=dict)
    cache_position_present_count: int = 0
    cache_position_absent_count: int = 0

    def add_component_ms(self, name: str, ms: Optional[float]) -> None:
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find PackedLayerState profiling fields. Apply v1/v2 profiling patches first.")
s = s.replace(old, new, 1)

old = """        self.component_ms_total[name] = float(self.component_ms_total.get(name, 0.0) + float(ms))
        self.component_ms_count[name] = int(self.component_ms_count.get(name, 0) + 1)
"""
new = """        self.component_ms_total[name] = float(self.component_ms_total.get(name, 0.0) + float(ms))
        self.component_ms_count[name] = int(self.component_ms_count.get(name, 0) + 1)

    def add_cache_update_path(self, name: str) -> None:
        self.cache_update_path_counts[name] = int(self.cache_update_path_counts.get(name, 0) + 1)

    def add_cache_past_type(self, name: str) -> None:
        self.cache_update_past_type_counts[name] = int(self.cache_update_past_type_counts.get(name, 0) + 1)
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find add_component_ms body.")
s = s.replace(old, new, 1)

start = s.find("def _cache_update(")
if start < 0:
    raise SystemExit("[FAIL] Could not find _cache_update().")

end = s.find("\n\ndef _normalize_mask(", start)
if end < 0:
    raise SystemExit("[FAIL] Could not find end marker after _cache_update().")

new_cache_update = """def _cache_update(
    *,
    module: torch.nn.Module,
    past: Any,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    kwargs: dict[str, Any],
    cos: Any = None,
    sin: Any = None,
    profile_components: bool = False,
    state: Optional["PackedLayerState"] = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    # Cache-update root-cause diagnostics:
    # - past cache type
    # - whether cache_position is present
    # - which update branch is used
    # - inner CUDA time of the actual update/cat branch
    if state is not None:
        state.add_cache_past_type(type(past).__name__ if past is not None else "None")
        if kwargs.get("cache_position", None) is None:
            state.cache_position_absent_count += 1
        else:
            state.cache_position_present_count += 1

    if past is None:
        if state is not None:
            state.add_cache_update_path("past_none_return")
        return key_states, value_states, None

    if hasattr(past, "update"):
        if state is not None:
            state.add_cache_update_path("cache_object_update")

        layer_idx = int(getattr(module, "layer_idx", 0))
        cache_kwargs = {}
        if cos is not None:
            cache_kwargs["cos"] = cos
        if sin is not None:
            cache_kwargs["sin"] = sin
        cache_position = kwargs.get("cache_position", None)
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position

        def _do_cache_object_update():
            try:
                return past.update(key_states, value_states, layer_idx, cache_kwargs)
            except TypeError:
                if state is not None:
                    state.add_cache_update_path("cache_object_update_typeerror_fallback")
                return past.update(key_states, value_states, layer_idx)

        (key_states, value_states), ms = _profile_cuda_ms(
            bool(profile_components),
            _do_cache_object_update,
        )
        if state is not None:
            state.add_component_ms("cache_object_past_update_inner", ms)
        return key_states, value_states, past

    if isinstance(past, (tuple, list)) and len(past) >= 2:
        if state is not None:
            state.add_cache_update_path("tuple_or_list_cat")

        def _tuple_cat_update():
            key_states_local = torch.cat([past[0], key_states], dim=-2)
            value_states_local = torch.cat([past[1], value_states], dim=-2)
            return key_states_local, value_states_local

        (key_states, value_states), ms = _profile_cuda_ms(
            bool(profile_components),
            _tuple_cat_update,
        )
        if state is not None:
            state.add_component_ms("tuple_cache_cat_inner", ms)
        return key_states, value_states, (key_states, value_states)

    if state is not None:
        state.add_cache_update_path("unsupported_cache_type")
    raise RuntimeError(f"Unsupported past_key_value type: {type(past)!r}")
"""
s = s[:start] + new_cache_update + s[end:]

old = """                return _cache_update(
                    module=self_module,
                    past=past,
                    key_states=key_states_new,
                    value_states=value_states_new,
                    kwargs=kwargs,
                )
"""
new = """                return _cache_update(
                    module=self_module,
                    past=past,
                    key_states=key_states_new,
                    value_states=value_states_new,
                    kwargs=kwargs,
                    profile_components=patcher.profile_components,
                    state=state,
                )
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find _cache_update_only() call block.")
s = s.replace(old, new, 1)

old = """                "component_ms_mean": {
                    name: (
                        float(v.component_ms_total[name] / v.component_ms_count[name])
                        if int(v.component_ms_count.get(name, 0)) > 0 else None
                    )
                    for name in v.component_ms_total
                },
            }
"""
new = """                "component_ms_mean": {
                    name: (
                        float(v.component_ms_total[name] / v.component_ms_count[name])
                        if int(v.component_ms_count.get(name, 0)) > 0 else None
                    )
                    for name in v.component_ms_total
                },
                "cache_update_path_counts": dict(v.cache_update_path_counts),
                "cache_update_past_type_counts": dict(v.cache_update_past_type_counts),
                "cache_position_present_count": int(v.cache_position_present_count),
                "cache_position_absent_count": int(v.cache_position_absent_count),
            }
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find summary component_ms_mean block.")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("[OK] Added cache-update root-cause profiling and branch diagnostics.")
