#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python llm_rs.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --method whiten \
  --search_method uniform \
  --param_ratio_target 0.8 \
  --calib_dataset wikitext2 \
  --n_calib_samples "${N_CALIB_SAMPLES:-256}" \
  --step_type param_ratio \
  --dump_config \
  --dump_huggingface_model \
  --config_root configs_uniform_whiten_08 \
  --save_folder svd_models_uniform_whiten_08 \
  --record_file uniform_whiten_08_result.txt \
  --device cuda:0
