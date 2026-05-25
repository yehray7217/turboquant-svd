# TurboQuant-SVD SVDLlama loader patch v2

This version fixes the compatibility mismatch where older test snippets import:

```python
from modules.svd_hf_registry import register_svdllama_auto_classes
```

It provides both:

- `utils/svdllama_loader.py`
- `modules/svd_hf_registry.py`

## Apply from repo root

```bash
cd ~/turboquant-svd
unzip -o turboquant_svd_svdllama_loader_patch_v2.zip -d .
bash turboquant_svd_svdllama_loader_patch_v2_build/apply_patch.sh .
```

## Minimal verification

```bash
PYTHONPATH=$PWD python - <<'PY'
from modules.svd_hf_registry import register_svdllama_auto_classes
register_svdllama_auto_classes()

from transformers import AutoConfig
p = "svd_models_uniform_whiten_08/whiten_w2_uniform_param_ratio_0.1_80_8.38"
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
print("model_type =", cfg.model_type)
print("architectures =", cfg.architectures)
print("num_svd_linears =", len(cfg.truncation_ranks))
PY
```

Alternative new helper:

```python
from utils.svdllama_loader import ensure_svdllama_registered
ensure_svdllama_registered(model_path)
```
