#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1) Wrap TQ replacement forward with whole-module timing.
# ---------------------------------------------------------------------
old = """        return types.MethodType(tq_decode_forward, module)
"""
new = """        @torch.no_grad()
        def profiled_tq_decode_forward(self_module: torch.nn.Module, *f_args: Any, **f_kwargs: Any):
            if not bool(patcher.profile_components):
                return tq_decode_forward(self_module, *f_args, **f_kwargs)

            try:
                hidden_states_probe, kwargs_probe = _extract_hidden_and_kwargs(f_args, f_kwargs)
                past_probe = _extract_past(kwargs_probe)
                is_replacement_decode = int(hidden_states_probe.shape[1]) == 1 and past_probe is not None
            except Exception:
                is_replacement_decode = False

            if not is_replacement_decode:
                return tq_decode_forward(self_module, *f_args, **f_kwargs)

            total_key = (
                "replacement_attention_forward_total_steady"
                if state.ready()
                else "replacement_attention_forward_total_prime_build"
            )
            out, ms = _profile_cuda_ms(
                True,
                lambda: tq_decode_forward(self_module, *f_args, **f_kwargs),
            )
            state.add_component_ms(total_key, ms)
            return out

        return types.MethodType(profiled_tq_decode_forward, module)
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find TQ forward return line. Apply this after the current decode timing patches.")
s = s.replace(old, new, 1)

# ---------------------------------------------------------------------
# 2) Add baseline original-attention decode profiler.
# ---------------------------------------------------------------------
marker = "\n\ndef _parse_replace_layers(raw: str) -> Optional[set[int]]:"
baseline_profiler = """

@dataclass
class BaselineAttentionLayerProfile:
    name: str
    layer_idx: int
    total_ms: float = 0.0
    count: int = 0

    def add_ms(self, ms: Optional[float]) -> None:
        if ms is None:
            return
        self.total_ms += float(ms)
        self.count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_idx": int(self.layer_idx),
            "baseline_attention_forward_total_ms": float(self.total_ms),
            "baseline_attention_forward_total_calls": int(self.count),
            "baseline_attention_forward_total_mean_ms": (
                float(self.total_ms / self.count) if self.count > 0 else None
            ),
        }


class BaselineDecodeAttentionProfiler:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.states: dict[int, BaselineAttentionLayerProfile] = {}
        self.original_forwards: list[tuple[torch.nn.Module, Any]] = []

    def _make_forward(
        self,
        module: torch.nn.Module,
        state: BaselineAttentionLayerProfile,
        original_forward: Any,
    ):
        profiler = self

        @torch.no_grad()
        def profiled_baseline_forward(self_module: torch.nn.Module, *f_args: Any, **f_kwargs: Any):
            if not profiler.enabled:
                return original_forward(*f_args, **f_kwargs)

            try:
                hidden_states_probe, kwargs_probe = _extract_hidden_and_kwargs(f_args, f_kwargs)
                past_probe = _extract_past(kwargs_probe)
                is_decode_attention = int(hidden_states_probe.shape[1]) == 1 and past_probe is not None
            except Exception:
                is_decode_attention = False

            if not is_decode_attention:
                return original_forward(*f_args, **f_kwargs)

            out, ms = _profile_cuda_ms(
                True,
                lambda: original_forward(*f_args, **f_kwargs),
            )
            state.add_ms(ms)
            return out

        return types.MethodType(profiled_baseline_forward, module)

    def install(self, model: torch.nn.Module, profile_layers: Optional[set[int]]) -> list[dict[str, Any]]:
        installed = []
        layer_re = re.compile(r"(?:^|\\.)(?:layers|h)\\.(\\d+)\\.")
        fallback_counter = 0

        for name, module in model.named_modules():
            has_attn_proj = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))
            if not has_attn_proj:
                continue

            m = layer_re.search(name + ".")
            if m:
                layer_idx = int(m.group(1))
            else:
                layer_idx = fallback_counter
                fallback_counter += 1

            if profile_layers is not None and layer_idx not in profile_layers:
                continue

            state = BaselineAttentionLayerProfile(name=name, layer_idx=layer_idx)
            self.states[layer_idx] = state
            original_forward = module.forward
            self.original_forwards.append((module, original_forward))
            module.forward = self._make_forward(module, state, original_forward)

            installed.append(
                {
                    "layer_idx": int(layer_idx),
                    "name": name,
                    "module_type": type(module).__name__,
                }
            )

        return installed

    def restore(self) -> None:
        for module, original_forward in self.original_forwards:
            module.forward = original_forward
        self.original_forwards.clear()

    def summary(self) -> dict[str, Any]:
        return {
            str(k): v.summary()
            for k, v in sorted(self.states.items(), key=lambda kv: kv[0])
        }
"""
if marker not in s:
    raise SystemExit("[FAIL] Could not find _parse_replace_layers marker.")
s = s.replace(marker, baseline_profiler + marker, 1)

# ---------------------------------------------------------------------
# 3) Wrap the baseline run with profiler install/restore.
# ---------------------------------------------------------------------
old = """    baseline = None
    if not bool(args.skip_baseline):
        baseline = _manual_decode_run(
            model,
            prompt_ids=prompt_ids,
            timed_decode_tokens=int(args.timed_decode_tokens),
            prime_decode_tokens=int(args.prime_decode_tokens),
            label="baseline_original_attention_decode",
        )
        print(json.dumps({"baseline_decode": baseline}, indent=2))
    else:
        print(json.dumps({"baseline_decode": "skipped"}, indent=2))
"""
new = """    baseline = None
    baseline_attention_profiler = BaselineDecodeAttentionProfiler(
        enabled=bool(args.profile_components) and not bool(args.skip_baseline)
    )
    baseline_attention_profile_installed = []

    if not bool(args.skip_baseline):
        if bool(args.profile_components):
            baseline_attention_profile_installed = baseline_attention_profiler.install(
                model,
                profile_layers=_parse_replace_layers(args.replace_layers),
            )

        baseline = _manual_decode_run(
            model,
            prompt_ids=prompt_ids,
            timed_decode_tokens=int(args.timed_decode_tokens),
            prime_decode_tokens=int(args.prime_decode_tokens),
            label="baseline_original_attention_decode",
        )

        if bool(args.profile_components):
            baseline_attention_profiler.restore()

        print(json.dumps({"baseline_decode": baseline}, indent=2))
    else:
        print(json.dumps({"baseline_decode": "skipped"}, indent=2))
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find baseline run block. Make sure the skip-baseline patch is applied.")
s = s.replace(old, new, 1)

# ---------------------------------------------------------------------
# 4) Add baseline profiler outputs into the JSON payload.
# ---------------------------------------------------------------------
old = """        "baseline": baseline,
        "replacement": replacement,
        "installed": installed,
        "patcher_summary": patcher_summary,
        "comparisons": comparisons,
"""
new = """        "baseline": baseline,
        "replacement": replacement,
        "installed": installed,
        "baseline_attention_profile_installed": baseline_attention_profile_installed,
        "baseline_attention_profile_summary": baseline_attention_profiler.summary(),
        "patcher_summary": patcher_summary,
        "comparisons": comparisons,
"""
if old not in s:
    raise SystemExit("[FAIL] Could not find payload block.")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("[OK] Added baseline-vs-TQ decode attention module-total delta profiling.")
