#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

old = '''        state.rotation = make_random_orthogonal_rotation(
            int(head_dim),
            seed=self.rotation_seed + int(state.layer_idx),
            device=device,
        )
        state.sketch = make_gaussian_sketch(
            int(head_dim),
            int(self.qjl_dim),
            seed=self.sketch_seed + int(state.layer_idx),
            device=device,
        )
'''
new = '''        # Build random matrices on CPU first, then move to CUDA.
        # torch.linalg.qr on CUDA can obscure earlier async CUDA failures by
        # reporting device-side assert at the rotation construction line.
        state.rotation = make_random_orthogonal_rotation(
            int(head_dim),
            seed=self.rotation_seed + int(state.layer_idx),
            device=torch.device("cpu"),
        ).to(device=device, dtype=torch.float32).contiguous()
        state.sketch = make_gaussian_sketch(
            int(head_dim),
            int(self.qjl_dim),
            seed=self.sketch_seed + int(state.layer_idx),
            device=torch.device("cpu"),
        ).to(device=device, dtype=torch.float32).contiguous()
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find rotation/sketch construction block.")
s = s.replace(old, new, 1)

old = '''    p.add_argument("--quiet_build", action="store_true")
    p.add_argument(
        "--text",
'''
new = '''    p.add_argument("--quiet_build", action="store_true")
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Run replacement only. Useful after a baseline run, and avoids carrying any prior CUDA error state into replacement.",
    )
    p.add_argument(
        "--text",
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find argparse insertion point.")
s = s.replace(old, new, 1)

old = '''    baseline = _manual_decode_run(
        model,
        prompt_ids=prompt_ids,
        timed_decode_tokens=int(args.timed_decode_tokens),
        prime_decode_tokens=int(args.prime_decode_tokens),
        label="baseline_original_attention_decode",
    )
    print(json.dumps({"baseline_decode": baseline}, indent=2))
'''
new = '''    baseline = None
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
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find baseline run block.")
s = s.replace(old, new, 1)

old = '''    b_mean = float(baseline["decode_ms_per_token"]["mean_ms"])
    r_mean = float(replacement["decode_ms_per_token"]["mean_ms"])
    baseline_tokens = baseline["generated_token_ids"]
    replacement_tokens = replacement["generated_token_ids"]
    aligned = min(len(baseline_tokens), len(replacement_tokens))
    exact_matches = sum(int(baseline_tokens[i] == replacement_tokens[i]) for i in range(aligned))

    comparisons = {
        "steady_state_decode_latency": {
            "baseline_mean_ms_per_token": b_mean,
            "replacement_mean_ms_per_token": r_mean,
            "replacement_over_baseline": float(r_mean / b_mean) if b_mean > 0 else None,
            "baseline_over_replacement_speedup": float(b_mean / r_mean) if r_mean > 0 else None,
            "baseline_tokens_per_sec": baseline["tokens_per_sec"],
            "replacement_tokens_per_sec": replacement["tokens_per_sec"],
        },
        "generated_token_agreement": {
            "aligned_tokens": int(aligned),
            "exact_match_count": int(exact_matches),
            "exact_match_ratio": float(exact_matches / aligned) if aligned > 0 else None,
        },
        "scope": {
            "prefill_attention": "original_attention",
            "timed_decode_attention": "turboquant_cuda_nonfactor_combined_logits",
            "prime_step_excluded_from_steady_state": True,
            "compressed_cache_append": "Python/PyTorch glue with packed tensor concatenation",
        },
    }
'''
new = '''    r_mean = float(replacement["decode_ms_per_token"]["mean_ms"])
    if baseline is not None:
        b_mean = float(baseline["decode_ms_per_token"]["mean_ms"])
        baseline_tokens = baseline["generated_token_ids"]
        replacement_tokens = replacement["generated_token_ids"]
        aligned = min(len(baseline_tokens), len(replacement_tokens))
        exact_matches = sum(int(baseline_tokens[i] == replacement_tokens[i]) for i in range(aligned))
        latency_cmp = {
            "baseline_mean_ms_per_token": b_mean,
            "replacement_mean_ms_per_token": r_mean,
            "replacement_over_baseline": float(r_mean / b_mean) if b_mean > 0 else None,
            "baseline_over_replacement_speedup": float(b_mean / r_mean) if r_mean > 0 else None,
            "baseline_tokens_per_sec": baseline["tokens_per_sec"],
            "replacement_tokens_per_sec": replacement["tokens_per_sec"],
        }
        token_cmp = {
            "aligned_tokens": int(aligned),
            "exact_match_count": int(exact_matches),
            "exact_match_ratio": float(exact_matches / aligned) if aligned > 0 else None,
        }
    else:
        latency_cmp = {
            "baseline_mean_ms_per_token": None,
            "replacement_mean_ms_per_token": r_mean,
            "replacement_over_baseline": None,
            "baseline_over_replacement_speedup": None,
            "baseline_tokens_per_sec": None,
            "replacement_tokens_per_sec": replacement["tokens_per_sec"],
        }
        token_cmp = {
            "aligned_tokens": None,
            "exact_match_count": None,
            "exact_match_ratio": None,
        }

    comparisons = {
        "steady_state_decode_latency": latency_cmp,
        "generated_token_agreement": token_cmp,
        "scope": {
            "prefill_attention": "original_attention",
            "timed_decode_attention": "turboquant_cuda_nonfactor_combined_logits",
            "prime_step_excluded_from_steady_state": True,
            "compressed_cache_append": "Python/PyTorch glue with packed tensor concatenation",
            "baseline_skipped": bool(args.skip_baseline),
        },
    }
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find comparisons block.")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("[OK] Patched decode timing script: CPU rotation/sketch build + --skip_baseline.")
