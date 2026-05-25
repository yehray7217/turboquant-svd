#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any


TARGET_MODULE = "turboquant.turboquant_combined_reduction_nonfactor_ablation_cuda"
TARGET_FN = "turboquant_full_4bit_lane_word_lane_nibble_qjl128_combined_reduction_logits_b1q1_d128_cuda"


def _safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as e:
        return f"<signature unavailable: {type(e).__name__}: {e}>"


def _read_window(path: Path, line_no_1based: int, radius: int = 60) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"<could not read {path}: {type(e).__name__}: {e}>"
    start = max(1, line_no_1based - radius)
    end = min(len(lines), line_no_1based + radius)
    out = []
    for i in range(start, end + 1):
        out.append(f"{i:5d}: {lines[i-1]}")
    return "\n".join(out)


def _grep_nearby(root: Path) -> list[dict[str, Any]]:
    patterns = [
        r"active_kv_len",
        r"kv_len",
        r"valid_kv",
        r"lane_word",
        r"lane_nibble",
        r"combined_reduction",
        r"logits_b1q1_d128",
    ]
    wanted_ext = {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}
    hits: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in wanted_ext:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in patterns:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append({
                    "path": str(path),
                    "pattern": pat,
                    "line": line,
                })
                break
    return hits[:200]


def main() -> None:
    repo_root = Path.cwd()
    module = importlib.import_module(TARGET_MODULE)
    fn = getattr(module, TARGET_FN)

    module_file = Path(inspect.getsourcefile(module) or getattr(module, "__file__", "<unknown>"))
    fn_file = Path(inspect.getsourcefile(fn) or module_file)
    try:
        _, fn_line = inspect.getsourcelines(fn)
    except Exception:
        fn_line = 1

    report = {
        "target_module": TARGET_MODULE,
        "target_fn": TARGET_FN,
        "repo_root": str(repo_root),
        "module_file": str(module_file),
        "fn_file": str(fn_file),
        "module_signature": _safe_signature(module),
        "fn_signature": _safe_signature(fn),
        "fn_source_line": int(fn_line),
        "fn_module": getattr(fn, "__module__", None),
        "fn_name": getattr(fn, "__name__", None),
        "module_dir": str(module_file.parent),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nearby_source_hits": _grep_nearby(module_file.parent),
    }

    report_path = Path("runs/svd_uniform_08/eval/debug_active_kv_len_kernel_surface.json")
    text_path = Path("runs/svd_uniform_08/eval/debug_active_kv_len_kernel_surface.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fn_window = _read_window(fn_file, fn_line, radius=90)
    module_window = _read_window(module_file, fn_line, radius=90)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text = []
    text.append("=== TurboQuant active_kv_len kernel surface diagnostic ===")
    text.append("")
    text.append(f"target_module = {TARGET_MODULE}")
    text.append(f"target_fn     = {TARGET_FN}")
    text.append(f"module_file   = {module_file}")
    text.append(f"fn_file       = {fn_file}")
    text.append(f"fn_line       = {fn_line}")
    text.append(f"fn_signature  = {report['fn_signature']}")
    text.append("")
    text.append("=== Function source window ===")
    text.append(fn_window)
    text.append("")
    text.append("=== Nearby module source window ===")
    text.append(module_window)
    text.append("")
    text.append("=== Nearby source hits ===")
    for hit in report["nearby_source_hits"]:
        text.append(f"{hit['path']}:{hit['line']}  pattern={hit['pattern']}")
    text.append("")
    text_path.write_text("\n".join(text), encoding="utf-8")

    print("[OK] Wrote:")
    print(f"  {report_path}")
    print(f"  {text_path}")
    print("")
    print("=== Key surface ===")
    print(f"module_file  = {module_file}")
    print(f"fn_signature = {report['fn_signature']}")
    print("")
    print("Paste back the text report or at least:")
    print("  - fn_signature")
    print("  - Function source window")
    print("  - any .cu/.cpp hits that define the extension launch")


if __name__ == "__main__":
    main()
