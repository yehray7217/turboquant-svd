# Layer-sharded whiten cache patch

This patch replaces the monolithic in-RAM whiten scaling-matrix accumulation
with a layer-sharded disk cache.

## Why

The first memory hotfix moved whitening scale matrices to CPU FP16 and removed
an unused inverse, but the job still accumulated all 32 layers in RAM and was
killed near `29/32`.

## New behavior

During whitening profile generation:

```text
cache/whiten/
  meta-llama_Llama-2-7b-hf_w2_scaling_matrices_fp16_layers/
    manifest.json
    layer_00.pt
    layer_01.pt
    ...
    layer_31.pt
```

Each layer shard is:
1. computed,
2. saved immediately,
3. removed from RAM.

During compression:
- `compress_model_whiten(...)` lazily loads only one layer shard at a time;
- attaches that shard's scaling matrices just before compressing the layer;
- releases it after use.

## Apply in `turboquant-svd`

```bash
cd ~/turboquant-svd
unzip -o turboquant_svd_whiten_layer_sharded_cache_patch.zip -d .
```

## Cleanup previous incomplete caches

```bash
rm -f cache/whiten/meta-llama_Llama-2-7b-hf_w2_scaling_matrices_fp16.pt
rm -rf cache/whiten/meta-llama_Llama-2-7b-hf_w2_scaling_matrices_fp16_layers
```

## Rerun

```bash
bash scripts/run_whiten_uniform_08.sh
```

## Resume behavior

If the process stops after some `layer_XX.pt` files are already written, rerun
the same command. Existing layer shards are reused while activations are still
propagated through the corresponding layers.
