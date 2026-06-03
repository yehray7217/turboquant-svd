#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from modules.svd_hf_registry import register_svdllama_auto_classes
    register_svdllama_auto_classes()
except Exception:
    pass

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
except Exception as e:
    raise RuntimeError("This prototype currently expects LLaMA-style attention.") from e


def _event_time_ms(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


def _quantize_symmetric_token(x, bits: int):
    # x: [B, H, T, D]
    if bits <= 0:
        return x.detach().contiguous(), None
    qmax = max((1 << (bits - 1)) - 1, 1)
    xf = x.detach().float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / float(qmax)
    q = torch.round(xf / scale).clamp(-qmax, qmax).to(torch.int8).contiguous()
    return q, scale.to(torch.float16).contiguous()


def _dequantize_symmetric_token(q, scale, dtype):
    if scale is None:
        return q.to(dtype)
    return (q.float() * scale.float()).to(dtype)


def _get_layer_cache(past, layer_idx: int):
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return past.key_cache[layer_idx], past.value_cache[layer_idx]
    if isinstance(past, (tuple, list)):
        item = past[layer_idx]
        return item[0], item[1]
    raise RuntimeError(f"Unsupported past type: {type(past)!r}")


class DirectIntKVRuntimePatcher:
    def __init__(self, model, *, k_mode: str, v_mode: str):
        self.model = model
        self.k_mode = k_mode
        self.v_mode = v_mode
        self.orig = {}
        self.cache = {}

    def _bits(self, mode):
        mode = (mode or "dense").lower()
        if mode in ("dense", "none", "fp16"):
            return 0
        if mode in ("int8", "int8_token"):
            return 8
        if mode in ("int4", "int4_token"):
            return 4
        raise ValueError(f"unsupported mode={mode}")

    def install(self):
        for _, module in self.model.named_modules():
            if module.__class__.__name__ in ("LlamaAttention", "LlamaSdpaAttention"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is None:
                    continue
                self.orig[int(layer_idx)] = module.forward
                module.forward = MethodType(self._make_forward(int(layer_idx), module), module)

    def import_from_past(self, past):
        kb = self._bits(self.k_mode)
        vb = self._bits(self.v_mode)

        for layer_idx in self.orig.keys():
            k, v = _get_layer_cache(past, int(layer_idx))
            kq, ks = _quantize_symmetric_token(k, kb)
            vq, vs = _quantize_symmetric_token(v, vb)
            self.cache[int(layer_idx)] = {
                "kq": kq,
                "ks": ks,
                "vq": vq,
                "vs": vs,
            }

    def _append_cache(self, layer_idx, k_new, v_new):
        kb = self._bits(self.k_mode)
        vb = self._bits(self.v_mode)

        kq, ks = _quantize_symmetric_token(k_new, kb)
        vq, vs = _quantize_symmetric_token(v_new, vb)

        st = self.cache.get(layer_idx)
        if st is None:
            st = {"kq": kq, "ks": ks, "vq": vq, "vs": vs}
        else:
            st["kq"] = torch.cat([st["kq"], kq], dim=2)
            if ks is not None:
                st["ks"] = torch.cat([st["ks"], ks], dim=2)
            st["vq"] = torch.cat([st["vq"], vq], dim=2)
            if vs is not None:
                st["vs"] = torch.cat([st["vs"], vs], dim=2)

        self.cache[layer_idx] = st
        return st

    def _make_forward(self, layer_idx, attn_module):
        patcher = self

        def forward(self_module, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=None, position_embeddings=None, **kwargs):
            bsz, q_len, _ = hidden_states.size()

            # Safety: patched path is decode-only. Never run full prefill here.
            if int(q_len) != 1:
                return patcher.orig[layer_idx](
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            query_states = self_module.q_proj(hidden_states)
            key_states = self_module.k_proj(hidden_states)
            value_states = self_module.v_proj(hidden_states)

            num_heads = self_module.num_heads
            num_kv_heads = self_module.num_key_value_heads
            head_dim = self_module.head_dim

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Decode-only RoPE:
            # Do not reuse position_embeddings from the outer model here. In some
            # HF versions it may be length-1 while position_ids is absolute
            # e.g. 16384, causing CUDA index out-of-bounds inside
            # apply_rotary_pos_emb(). Always construct cos/sin with enough
            # seq_len for the absolute position id.
            rotary = getattr(self_module, "rotary_emb", None)
            if rotary is not None:
                if position_ids is None:
                    prev_len = 0
                    st_prev = patcher.cache.get(layer_idx)
                    if st_prev is not None:
                        prev_len = int(st_prev["kq"].shape[-2])
                    position_ids = torch.tensor(
                        [[prev_len]],
                        device=query_states.device,
                        dtype=torch.long,
                    )

                seq_len = int(position_ids.detach().max().item()) + 1

                try:
                    cos, sin = rotary(value_states, seq_len=seq_len)
                except TypeError:
                    # Newer HF rotary API may accept position_ids directly and
                    # return already-positioned cos/sin. If so, call
                    # apply_rotary_pos_emb without re-indexing by position_ids.
                    cos, sin = rotary(value_states, position_ids)
                    position_ids_for_apply = None
                else:
                    position_ids_for_apply = position_ids

                try:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                        position_ids_for_apply,
                    )
                except TypeError:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                    )

            st = patcher._append_cache(layer_idx, key_states, value_states)

            key_full = _dequantize_symmetric_token(st["kq"], st["ks"], query_states.dtype)
            value_full = _dequantize_symmetric_token(st["vq"], st["vs"], query_states.dtype)

            key_full = repeat_kv(key_full, self_module.num_key_value_groups)
            value_full = repeat_kv(value_full, self_module.num_key_value_groups)

            attn_weights = torch.matmul(query_states.float(), key_full.transpose(2, 3).float())
            attn_weights = attn_weights / math.sqrt(head_dim)

            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_probs, value_full)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self_module.hidden_size)
            attn_output = self_module.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, None

        return forward


def _load_prompt(path, prompt_len, tokenizer, device):
    if path and Path(path).exists():
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ("input_ids", "prompt_ids", "ids"):
                if k in obj:
                    input_ids = obj[k]
                    break
            else:
                raise RuntimeError(f"Unsupported prompt dict keys: {sorted(obj.keys())}")
        else:
            input_ids = obj

        if not torch.is_tensor(input_ids):
            raise RuntimeError(f"Loaded prompt is not a tensor: {type(input_ids)!r}")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids[:, :prompt_len].to(device)

    text = "Hello world. " * (prompt_len + 16)
    return tokenizer(text, return_tensors="pt").input_ids[:, :prompt_len].to(device)


@torch.no_grad()
def dense_prefill_chunked(model, input_ids, *, chunk_size: int):
    """
    Dense prefill with original HF attention, but chunked to avoid full 16K
    activation/logit peak memory. Returns final outputs and accumulated past.
    """
    chunk_size = max(1, int(chunk_size))
    past = None
    outputs = None
    total = int(input_ids.shape[1])

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        chunk = input_ids[:, start:end]

        def _call():
            if past is None:
                return model(input_ids=chunk, use_cache=True)
            return model(input_ids=chunk, past_key_values=past, use_cache=True)

        outputs, _ = _event_time_ms(_call)
        past = outputs.past_key_values

        # Keep only current outputs/past references. This avoids retaining
        # large per-chunk logits longer than necessary.
        torch.cuda.empty_cache()

    return outputs, past


@torch.no_grad()
def manual_decode_run(model, patcher, input_ids, *, prime_decode_tokens, timed_decode_tokens):
    # 1. Dense prefill, no patched attention yet.
    outputs, prefill_ms = _event_time_ms(lambda: model(input_ids=input_ids, use_cache=True))
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1:].argmax(dim=-1)

    # 2. Import dense prefill KV into quantized internal cache.
    patcher.import_from_past(past)

    cur_pos = int(input_ids.shape[1])

    prime_times = []
    for _ in range(prime_decode_tokens):
        pos = torch.tensor([[cur_pos]], device=input_ids.device, dtype=torch.long)
        outputs, dt = _event_time_ms(
            lambda: model(input_ids=next_token, position_ids=pos, use_cache=False)
        )
        prime_times.append(dt)
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        cur_pos += 1

    times = []
    generated = []
    for _ in range(timed_decode_tokens):
        pos = torch.tensor([[cur_pos]], device=input_ids.device, dtype=torch.long)
        outputs, dt = _event_time_ms(
            lambda: model(input_ids=next_token, position_ids=pos, use_cache=False)
        )
        times.append(dt)
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated.append(int(next_token.item()))
        cur_pos += 1

    mean = sum(times) / len(times)
    med = sorted(times)[len(times) // 2]

    return {
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": {
            "count": len(times),
            "mean_ms": mean,
            "median_ms": med,
            "min_ms": min(times),
            "max_ms": max(times),
            "times_ms": times,
        },
        "tokens_per_sec": 1000.0 / mean,
        "generated_token_ids": generated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt_len", type=int, default=16384)
    ap.add_argument("--timed_decode_tokens", type=int, default=128)
    ap.add_argument("--prime_decode_tokens", type=int, default=1)
    ap.add_argument("--k_mode", default="int8_token")
    ap.add_argument("--v_mode", default="int8_token")
    ap.add_argument("--prompt_ids_pt", default=None)
    ap.add_argument("--prefill_chunk_size", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.torch_dtype]
    device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    ).to(device)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    input_ids = _load_prompt(args.prompt_ids_pt, args.prompt_len, tok, device)

    patcher = DirectIntKVRuntimePatcher(model, k_mode=args.k_mode, v_mode=args.v_mode)

    print("========== Direct INT KV runtime prototype ==========")
    print("model_name =", args.model_name)
    print("prompt_len =", args.prompt_len)
    print("k_mode =", args.k_mode)
    print("v_mode =", args.v_mode)

    # Install only after prefill is done inside manual_decode_run.
    # The manual function does dense prefill first, then imports past, then installs.
    (outputs, past), prefill_ms = _event_time_ms(
        lambda: dense_prefill_chunked(
            model,
            input_ids,
            chunk_size=int(args.prefill_chunk_size),
        )
    )
    next_token = outputs.logits[:, -1:].argmax(dim=-1)

    patcher.import_from_past(past)

    # Important: after importing dense prefill KV into quantized internal cache,
    # release HF dense cache references so runtime reflects the prototype cache.
    del past
    del outputs
    torch.cuda.empty_cache()

    patcher.install()

    cur_pos = int(input_ids.shape[1])

    # prime
    for _ in range(args.prime_decode_tokens):
        pos = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
        outputs, _ = _event_time_ms(lambda: model(input_ids=next_token, position_ids=pos, use_cache=False))
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        cur_pos += 1

    times = []
    generated = []
    for _ in range(args.timed_decode_tokens):
        pos = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
        outputs, dt = _event_time_ms(lambda: model(input_ids=next_token, position_ids=pos, use_cache=False))
        times.append(dt)
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated.append(int(next_token.item()))
        cur_pos += 1

    mean = sum(times) / len(times)
    med = sorted(times)[len(times) // 2]

    replacement = {
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": {
            "count": len(times),
            "mean_ms": mean,
            "median_ms": med,
            "min_ms": min(times),
            "max_ms": max(times),
            "times_ms": times,
        },
        "tokens_per_sec": 1000.0 / mean,
        "generated_token_ids": generated,
    }

    out = {
        "kind": "direct_int_kv_runtime_prototype_decode_only",
        "model_name": args.model_name,
        "prompt_len": args.prompt_len,
        "k_mode": args.k_mode,
        "v_mode": args.v_mode,
        "replacement": replacement,
        "note": "Dense prefill, then import prefill KV into quantized internal cache; decode-only prototype, not fused optimized kernel.",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("[Save]", args.out)


if __name__ == "__main__":
    main()
