#!/usr/bin/env python3
from pathlib import Path

p = Path("tests/bench_turboquant_decode_attention_cuda_true_timing.py")
s = p.read_text(encoding="utf-8")

# 1) Import fast path.
anchor = "from turboquant.qjl_sign_layout import pack_qjl_signs_lane_nibble\n"
insert = anchor + "from turboquant.decode_pack_cuda_fastpath import DecodePackCudaFastPath\n"
if "DecodePackCudaFastPath" not in s:
    if anchor not in s:
        raise SystemExit("[FAIL] Could not find qjl pack import anchor.")
    s = s.replace(anchor, insert, 1)

# 2) Add argparse flag.
quiet_line = '    p.add_argument("--quiet_build", action="store_true")\n'
arg_block = '''    p.add_argument(
        "--use_pack_cuda_fastpath",
        action="store_true",
        help="Use validated CUDA fast path for scalar/QJL new-K packing.",
    )
'''
if "--use_pack_cuda_fastpath" not in s:
    if quiet_line not in s:
        raise SystemExit("[FAIL] Could not find quiet_build argparse line.")
    s = s.replace(quiet_line, quiet_line + arg_block, 1)

# 3) Add constructor arg.
old = '''        verbose_build: bool,
        profile_components: bool,
    ) -> None:
'''
new = '''        verbose_build: bool,
        profile_components: bool,
        use_pack_cuda_fastpath: bool,
    ) -> None:
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find TurboQuantDecodeAttentionPatcher __init__ signature.")
s = s.replace(old, new, 1)

old = '''        self.verbose_build = bool(verbose_build)
        self.profile_components = bool(profile_components)
        self.states: dict[int, PackedLayerState] = {}
'''
new = '''        self.verbose_build = bool(verbose_build)
        self.profile_components = bool(profile_components)
        self.use_pack_cuda_fastpath = bool(use_pack_cuda_fastpath)
        self.pack_cuda_fastpath = None
        self.states: dict[int, PackedLayerState] = {}
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find patcher __init__ body.")
s = s.replace(old, new, 1)

# 4) Add lazy initializer inside patcher class.
marker = "    def _fit_state_from_full_k(\n"
helper = '''    def _ensure_pack_cuda_fastpath(self, device: torch.device):
        if not self.use_pack_cuda_fastpath:
            return None
        if self.pack_cuda_fastpath is None:
            self.pack_cuda_fastpath = DecodePackCudaFastPath(
                scalar_ref_fn=pack_scalar_codes_lane_word_4bit,
                qjl_ref_fn=pack_qjl_signs_lane_nibble,
                device=device,
            )
            print(
                json.dumps({"pack_cuda_fastpath": self.pack_cuda_fastpath.summary()}),
                flush=True,
            )
        return self.pack_cuda_fastpath

'''
if "def _ensure_pack_cuda_fastpath" not in s:
    if marker not in s:
        raise SystemExit("[FAIL] Could not find _fit_state_from_full_k marker.")
    s = s.replace(marker, helper + marker, 1)

# 5) Pass constructor arg in main.
old = '''        verbose_build=not bool(args.quiet_build),
        profile_components=bool(args.profile_components),
    )
'''
new = '''        verbose_build=not bool(args.quiet_build),
        profile_components=bool(args.profile_components),
        use_pack_cuda_fastpath=bool(args.use_pack_cuda_fastpath),
    )
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find patcher constructor call in main.")
s = s.replace(old, new, 1)

# 6) Replace new-K pack block from the new-K encode breakdown patch.
old = '''        scalar_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: pack_scalar_codes_lane_word_4bit(codes).contiguous(),
        )
        state.add_component_ms("pack_scalar_codes_new_k", ms)

        qjl_new, ms = _profile_cuda_ms(
            self.profile_components,
            lambda: pack_qjl_signs_lane_nibble(residual_signs).contiguous(),
        )
        state.add_component_ms("pack_qjl_signs_new_k", ms)
'''
new = '''        pack_fastpath = self._ensure_pack_cuda_fastpath(codes.device)

        scalar_new, ms = _profile_cuda_ms(
            self.profile_components,
            (
                (lambda: pack_fastpath.pack_scalar(codes))
                if pack_fastpath is not None
                else (lambda: pack_scalar_codes_lane_word_4bit(codes).contiguous())
            ),
        )
        state.add_component_ms(
            "pack_scalar_codes_new_k_cuda_fastpath"
            if pack_fastpath is not None
            else "pack_scalar_codes_new_k",
            ms,
        )

        qjl_new, ms = _profile_cuda_ms(
            self.profile_components,
            (
                (lambda: pack_fastpath.pack_qjl(residual_signs))
                if pack_fastpath is not None
                else (lambda: pack_qjl_signs_lane_nibble(residual_signs).contiguous())
            ),
        )
        state.add_component_ms(
            "pack_qjl_signs_new_k_cuda_fastpath"
            if pack_fastpath is not None
            else "pack_qjl_signs_new_k",
            ms,
        )
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find new-K pack block. Apply after the new-K encode breakdown patch.")
s = s.replace(old, new, 1)

# 7) Add summary to payload.
old = '''        "patcher_summary": patcher_summary,
        "comparisons": comparisons,
'''
new = '''        "patcher_summary": patcher_summary,
        "pack_cuda_fastpath_summary": (
            patcher.pack_cuda_fastpath.summary()
            if patcher.pack_cuda_fastpath is not None
            else {"enabled": False}
        ),
        "comparisons": comparisons,
'''
if old not in s:
    raise SystemExit("[FAIL] Could not find payload patcher_summary block.")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("[OK] Added optional validated CUDA new-K packing fast path.")
