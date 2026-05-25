#!/usr/bin/env python3
from pathlib import Path
import re

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text()

# ============================================================
# 1. Add argparse flag after --use_pack_cuda_fastpath block
# ============================================================
if "--compressed_cache_reserve_tokens" not in s:
    lines = s.splitlines(keepends=True)
    anchor_idx = None

    for i, line in enumerate(lines):
        if "--use_pack_cuda_fastpath" in line:
            anchor_idx = i
            break

    if anchor_idx is None:
        raise SystemExit("[FAIL] Could not find --use_pack_cuda_fastpath argparse anchor.")

    block_end = None
    for j in range(anchor_idx, min(anchor_idx + 20, len(lines))):
        if lines[j].strip() == ")":
            block_end = j
            break

    if block_end is None:
        raise SystemExit("[FAIL] Could not find end of --use_pack_cuda_fastpath add_argument block.")

    arg_block = [
        '    p.add_argument(\n',
        '        "--compressed_cache_reserve_tokens",\n',
        '        type=int,\n',
        '        default=0,\n',
        '        help=(\n',
        '            "Reserve compressed KV append capacity per layer. "\n',
        '            "0 means auto = prime_decode_tokens + timed_decode_tokens + 8."\n',
        '        ),\n',
        '    )\n',
    ]

    lines[block_end + 1:block_end + 1] = arg_block
    s = "".join(lines)

# ============================================================
# 2. Extend PackedLayerState
# ============================================================
field_anchor = "    residual_norms: Optional[torch.Tensor] = None\n"

extra_fields = (
    "    scalar_lane_words_storage: Optional[torch.Tensor] = None\n"
    "    qjl_lane_nibbles_storage: Optional[torch.Tensor] = None\n"
    "    residual_norms_storage: Optional[torch.Tensor] = None\n"
    "    compressed_cache_len: int = 0\n"
    "    compressed_cache_capacity: int = 0\n"
)

if "scalar_lane_words_storage" not in s:
    if field_anchor not in s:
        raise SystemExit("[FAIL] Could not find PackedLayerState residual_norms field.")
    s = s.replace(field_anchor, field_anchor + extra_fields, 1)

# ============================================================
# 3. Patch patcher constructor signature/body
# ============================================================
old = (
    "        use_pack_cuda_fastpath: bool,\n"
    "    ) -> None:\n"
)
new = (
    "        use_pack_cuda_fastpath: bool,\n"
    "        compressed_cache_reserve_tokens: int,\n"
    "    ) -> None:\n"
)

if old in s:
    s = s.replace(old, new, 1)
elif "compressed_cache_reserve_tokens: int" not in s:
    raise SystemExit("[FAIL] Could not patch patcher __init__ signature.")

old = (
    "        self.use_pack_cuda_fastpath = bool(use_pack_cuda_fastpath)\n"
    "        self.pack_cuda_fastpath = None\n"
)
new = (
    "        self.use_pack_cuda_fastpath = bool(use_pack_cuda_fastpath)\n"
    "        self.pack_cuda_fastpath = None\n"
    "        self.compressed_cache_reserve_tokens = max(0, int(compressed_cache_reserve_tokens))\n"
)

if old in s:
    s = s.replace(old, new, 1)
elif "self.compressed_cache_reserve_tokens" not in s:
    raise SystemExit("[FAIL] Could not patch patcher __init__ body.")

# ============================================================
# 4. Pass reserve arg in main constructor call
# ============================================================
old = (
    "        use_pack_cuda_fastpath=bool(args.use_pack_cuda_fastpath),\n"
    "    )\n"
)
new = (
    "        use_pack_cuda_fastpath=bool(args.use_pack_cuda_fastpath),\n"
    "        compressed_cache_reserve_tokens=(\n"
    "            int(args.compressed_cache_reserve_tokens)\n"
    "            if int(args.compressed_cache_reserve_tokens) > 0\n"
    "            else int(args.prime_decode_tokens) + int(args.timed_decode_tokens) + 8\n"
    "        ),\n"
    "    )\n"
)

if old in s:
    s = s.replace(old, new, 1)
elif "compressed_cache_reserve_tokens=(" not in s:
    raise SystemExit("[FAIL] Could not patch patcher constructor call.")

# ============================================================
# 5. Insert preallocated-cache helper methods
# ============================================================
marker = "    def _fit_state_from_full_k(\n"

helpers = '''    def _install_preallocated_compressed_cache(
        self,
        *,
        state: PackedLayerState,
        scalar_lane_words: torch.Tensor,
        qjl_lane_nibbles: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> None:
        prefix_len = int(scalar_lane_words.shape[-2])
        capacity = prefix_len + max(0, int(self.compressed_cache_reserve_tokens))

        scalar_shape = list(scalar_lane_words.shape)
        qjl_shape = list(qjl_lane_nibbles.shape)
        norm_shape = list(residual_norms.shape)

        scalar_shape[-2] = capacity
        qjl_shape[-2] = capacity
        norm_shape[-1] = capacity

        scalar_storage = torch.empty(
            scalar_shape,
            dtype=scalar_lane_words.dtype,
            device=scalar_lane_words.device,
        )
        qjl_storage = torch.empty(
            qjl_shape,
            dtype=qjl_lane_nibbles.dtype,
            device=qjl_lane_nibbles.device,
        )
        norm_storage = torch.empty(
            norm_shape,
            dtype=residual_norms.dtype,
            device=residual_norms.device,
        )

        scalar_storage[..., :prefix_len, :].copy_(scalar_lane_words)
        qjl_storage[..., :prefix_len, :].copy_(qjl_lane_nibbles)
        norm_storage[..., :prefix_len].copy_(residual_norms)

        state.scalar_lane_words_storage = scalar_storage
        state.qjl_lane_nibbles_storage = qjl_storage
        state.residual_norms_storage = norm_storage

        state.compressed_cache_len = prefix_len
        state.compressed_cache_capacity = capacity

        state.scalar_lane_words = scalar_storage[..., :prefix_len, :]
        state.qjl_lane_nibbles = qjl_storage[..., :prefix_len, :]
        state.residual_norms = norm_storage[..., :prefix_len]

    def _append_preallocated_compressed_cache(
        self,
        *,
        state: PackedLayerState,
        scalar_new: torch.Tensor,
        qjl_new: torch.Tensor,
        norms_new: torch.Tensor,
    ) -> None:
        assert state.scalar_lane_words_storage is not None
        assert state.qjl_lane_nibbles_storage is not None
        assert state.residual_norms_storage is not None

        start = int(state.compressed_cache_len)
        add = int(scalar_new.shape[-2])
        end = start + add

        if end > int(state.compressed_cache_capacity):
            raise RuntimeError(
                "Preallocated compressed cache capacity exceeded: "
                f"need end={end}, capacity={state.compressed_cache_capacity}. "
                "Increase --compressed_cache_reserve_tokens."
            )

        state.scalar_lane_words_storage[..., start:end, :].copy_(scalar_new)
        state.qjl_lane_nibbles_storage[..., start:end, :].copy_(qjl_new)
        state.residual_norms_storage[..., start:end].copy_(norms_new)

        state.compressed_cache_len = end
        state.scalar_lane_words = state.scalar_lane_words_storage[..., :end, :]
        state.qjl_lane_nibbles = state.qjl_lane_nibbles_storage[..., :end, :]
        state.residual_norms = state.residual_norms_storage[..., :end]

'''

if "_install_preallocated_compressed_cache" not in s:
    if marker not in s:
        raise SystemExit("[FAIL] Could not find _fit_state_from_full_k marker.")
    s = s.replace(marker, helpers + marker, 1)

# ============================================================
# 6. Replace prefix compressed-cache assignment trio
# ============================================================
pattern = re.compile(
    r'(?P<indent>\s*)'
    r'state\.scalar_lane_words\s*=\s*(?P<scalar>[^\n]+)\n'
    r'(?P=indent)state\.qjl_lane_nibbles\s*=\s*(?P<qjl>[^\n]+)\n'
    r'(?P=indent)state\.residual_norms\s*=\s*(?P<norm>[^\n]+)\n',
    re.M,
)

matches = list(pattern.finditer(s))
append_pos = s.find("    def _append_new_k(")

candidate = None
for m in matches:
    if append_pos < 0 or m.start() < append_pos:
        candidate = m
        break

if candidate is None:
    raise SystemExit("[FAIL] Could not find prefix compressed-cache assignment trio.")

indent = candidate.group("indent")
scalar_expr = candidate.group("scalar").strip()
qjl_expr = candidate.group("qjl").strip()
norm_expr = candidate.group("norm").strip()

replacement = (
    f"{indent}scalar_lane_words_built = {scalar_expr}\n"
    f"{indent}qjl_lane_nibbles_built = {qjl_expr}\n"
    f"{indent}residual_norms_built = {norm_expr}\n"
    f"{indent}self._install_preallocated_compressed_cache(\n"
    f"{indent}    state=state,\n"
    f"{indent}    scalar_lane_words=scalar_lane_words_built,\n"
    f"{indent}    qjl_lane_nibbles=qjl_lane_nibbles_built,\n"
    f"{indent}    residual_norms=residual_norms_built,\n"
    f"{indent})\n"
)

s = s[:candidate.start()] + replacement + s[candidate.end():]

# ============================================================
# 7. Replace profiled torch.cat append block with slice-write
# ============================================================
start = s.find("        def _cat_scalar():")
end_marker = "        state.append_calls += 1\n"
end = s.find(end_marker, start)

if start < 0 or end < 0:
    raise SystemExit("[FAIL] Could not find profiled cat-append block.")

new_block = '''        def _slice_write_preallocated_cache():
            self._append_preallocated_compressed_cache(
                state=state,
                scalar_new=scalar_new,
                qjl_new=qjl_new,
                norms_new=norms_new,
            )
            return None

        _, ms = _profile_cuda_ms(
            self.profile_components,
            _slice_write_preallocated_cache,
        )
        state.add_component_ms("preallocated_compressed_cache_slice_write", ms)

'''

s = s[:start] + new_block + s[end:]

p.write_text(s)
print("[OK] Added preallocated compressed KV cache append path and removed per-token torch.cat append.")
