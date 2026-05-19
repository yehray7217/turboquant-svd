# Whiten uniform-0.8 compression pipeline inside `turboquant-svd`

This patch moves the old `llm_rs.py` whitening compression workflow into the
new repo, so compression and TurboQuant evaluation can live in one project.

## Why this patch

The previous run of:

```bash
python llm_rs.py --method whiten --search_method uniform ...
```

was killed during whiten scale-matrix generation at `23/32` layers. The patch
includes the memory hotfix:

- generated whitening scaling matrices cached in CPU FP16,
- unused profiling-time inverse removed,
- large per-module temporaries deleted,
- cache-file path overwrite fixed.

## Files added

```text
llm_rs.py
whiten_utils.py
datautils.py
evaluate_utils.py
act_aware_utils.py
sensitivity.py
binary_search.py
greedy.py
linear_prog.py
spectrum_greedy.py
config_translate.py
huggingface_utils.py

modules/
  __init__.py
  svd_linear.py

utils/
  __init__.py
  calc.py
  svd_logger.py

scripts/
  run_whiten_uniform_08.sh
```

## Apply

```bash
cd ~/turboquant-svd
unzip -o turboquant_svd_whiten_uniform_pipeline_patch.zip -d .
```

## Recommended run

Remove any incomplete whiten cache first:

```bash
rm -f cache/whiten/meta-llama_Llama-2-7b-hf_w2_scaling_matrices_fp16.pt
```

Then run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python llm_rs.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --method whiten \
  --search_method uniform \
  --param_ratio_target 0.8 \
  --calib_dataset wikitext2 \
  --n_calib_samples 256 \
  --step_type param_ratio \
  --dump_config \
  --dump_huggingface_model \
  --config_root configs_uniform_whiten_08 \
  --save_folder svd_models_uniform_whiten_08 \
  --record_file uniform_whiten_08_result.txt \
  --device cuda:0
```

Equivalent convenience command:

```bash
bash scripts/run_whiten_uniform_08.sh
```

For a lower-memory fallback:

```bash
N_CALIB_SAMPLES=128 bash scripts/run_whiten_uniform_08.sh
```

## Expected outputs

```text
cache/whiten/
  meta-llama_Llama-2-7b-hf_w2_scaling_matrices_fp16.pt

configs_uniform_whiten_08/
  ...

svd_models_uniform_whiten_08/
  ...

output/uniform_whiten_08_result.txt
config_dump.json
```

## Notes

- This patch is aimed at the `whiten + uniform 0.8` production path.
- The support functions in `utils/calc.py` are compatible with the uniform path
  and preserve import compatibility for the older search helpers.
- Once the whiten model is dumped, the next step is to load that saved model in
  the existing `turboquant-svd` E2E / TurboQuant-readiness benchmarks.
