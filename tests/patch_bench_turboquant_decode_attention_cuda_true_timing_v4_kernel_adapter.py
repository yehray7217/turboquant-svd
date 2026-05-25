#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

start = s.find("def _try_call_nonfactor_kernel(")
if start < 0:
    raise SystemExit("[FAIL] Could not find _try_call_nonfactor_kernel().")

next_marker = "\n\n@dataclass\nclass PackedLayerState:"
end = s.find(next_marker, start)
if end < 0:
    raise SystemExit("[FAIL] Could not find end marker after _try_call_nonfactor_kernel().")

new_func = '''def _try_call_nonfactor_kernel(
    *,
    rotated_queries: torch.Tensor,
    qjl_projected_queries: torch.Tensor,
    centroids: torch.Tensor,
    scalar_lane_words: torch.Tensor,
    qjl_lane_nibbles: torch.Tensor,
    residual_norms: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """
    Call the project-local nonfactor CUDA wrapper.

    The CUDA wrapper is keyword-only in the current repo state. The earlier
    decode integration only had a narrow name mapping; once those aliases missed,
    it fell back to positional calls and failed with:

      takes 0 positional arguments but 6 were given

    This adapter:
      1) introspects the Python signature when possible,
      2) maps parameter names to the six tensors using robust heuristics,
      3) tries a small set of explicit keyword layouts used by project variants,
      4) emits the discovered signature and all failures if none match.
    """
    fn = turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda

    try:
        sig = inspect.signature(fn)
        sig_text = str(sig)
        params = list(sig.parameters.values())
    except Exception as e:
        sig = None
        sig_text = f"<inspect.signature failed: {type(e).__name__}: {e}>"
        params = []

    def choose_tensor(param_name: str) -> torch.Tensor | None:
        n = param_name.lower()

        if ("rot" in n or "rotated" in n) and ("quer" in n or n.startswith("q_")):
            return rotated_queries

        if ("project" in n or "proj" in n) and ("quer" in n or "qjl" in n):
            return qjl_projected_queries

        if "centroid" in n or "codebook" in n:
            return centroids

        if "norm" in n:
            return residual_norms

        if (
            ("scalar" in n or "code" in n)
            and ("lane" in n or "word" in n or "packed" in n)
            and "qjl" not in n
        ):
            return scalar_lane_words

        if (
            "qjl" in n
            and ("lane" in n or "nibble" in n or "packed" in n or "sign" in n)
        ):
            return qjl_lane_nibbles

        if "sign" in n and "scalar" not in n:
            return qjl_lane_nibbles

        return None

    errors: list[str] = []

    # First: signature-driven heuristic mapping.
    if params:
        heuristic_kwargs: dict[str, torch.Tensor] = {}
        unresolved: list[str] = []
        keyword_compatible = True

        for p in params:
            if p.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                keyword_compatible = False
                unresolved.append(f"{p.name}:{p.kind}")
                continue

            t = choose_tensor(p.name)
            if t is None:
                unresolved.append(p.name)
            else:
                heuristic_kwargs[p.name] = t

        if keyword_compatible and not unresolved and heuristic_kwargs:
            try:
                return fn(**heuristic_kwargs), "signature_heuristic_kwargs:" + ",".join(heuristic_kwargs.keys())
            except Exception as e:
                errors.append(
                    "signature_heuristic_kwargs "
                    + repr(list(heuristic_kwargs.keys()))
                    + f": {type(e).__name__}: {e}"
                )
        else:
            errors.append(
                "signature_heuristic_unresolved "
                + f"signature={sig_text} unresolved={unresolved} mapped={list(heuristic_kwargs.keys())}"
            )

    # Second: explicit known/plausible keyword layouts.
    keyword_candidates: list[tuple[str, dict[str, torch.Tensor]]] = [
        (
            "kw_rot_qjl_centroids_packed_scalar_packed_qjl_norms",
            {
                "rotated_queries": rotated_queries,
                "qjl_projected_queries": qjl_projected_queries,
                "centroids": centroids,
                "packed_scalar_codes_lane_word": scalar_lane_words,
                "packed_qjl_signs_lane_nibble": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_rot_qjl_centroids_scalar_lane_qjl_lane_norms",
            {
                "rotated_queries": rotated_queries,
                "qjl_projected_queries": qjl_projected_queries,
                "centroids": centroids,
                "scalar_lane_words": scalar_lane_words,
                "qjl_lane_nibbles": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_rot_qjl_scalar_centroids_packed_scalar_packed_qjl_norms",
            {
                "rotated_queries": rotated_queries,
                "qjl_projected_queries": qjl_projected_queries,
                "scalar_centroids": centroids,
                "packed_scalar_codes_lane_word": scalar_lane_words,
                "packed_qjl_signs_lane_nibble": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_rot_proj_centroids_packed_codes_packed_signs_norms",
            {
                "rotated_queries": rotated_queries,
                "projected_queries": qjl_projected_queries,
                "centroids": centroids,
                "packed_codes_lane_word": scalar_lane_words,
                "packed_signs_lane_nibble": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
        (
            "kw_qrot_qproj_centroids_packed_codes_packed_signs_norms",
            {
                "q_rot": rotated_queries,
                "qjl_projected_query": qjl_projected_queries,
                "centroids": centroids,
                "packed_scalar_codes": scalar_lane_words,
                "packed_qjl_signs": qjl_lane_nibbles,
                "residual_norms": residual_norms,
            },
        ),
    ]

    for name, kwargs in keyword_candidates:
        try:
            return fn(**kwargs), name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not call TurboQuant nonfactor combined CUDA kernel with keyword adapter. "
        f"Discovered signature: {sig_text}. Adapter attempts:\n"
        + "\n".join(errors)
    )
'''

s = s[:start] + new_func + s[end:]
p.write_text(s, encoding="utf-8")
print("[OK] Patched _try_call_nonfactor_kernel() with keyword-only signature-aware adapter.")
