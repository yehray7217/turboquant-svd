#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/eval_turboquant_attention_replacement_ppl.py")
s = p.read_text(encoding="utf-8")

old = '''        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], original_output: Any):
            if not args:
                return original_output
            hidden_states = args[0]
            if not torch.is_tensor(hidden_states):
                return original_output
            if hidden_states.ndim != 3:
                return original_output
'''

new = '''        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], original_output: Any):
            # Llama attention may receive hidden_states positionally or as a keyword,
            # depending on the model wrapper / Transformers version.
            if args:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.get("hidden_states", None)

            if not torch.is_tensor(hidden_states):
                return original_output
            if hidden_states.ndim != 3:
                return original_output
'''

if old not in s:
    raise SystemExit("[FAIL] Target hook block not found; source drifted.")

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("[OK] Patched hook to accept kwargs['hidden_states'] as well as positional args.")
