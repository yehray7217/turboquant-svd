"""Small helper for loading saved SVDLlama checkpoints through Transformers Auto* APIs.

Why this exists
---------------
This project saves compressed checkpoints with config.json fields like:
    "model_type": "svdllama"
    "architectures": ["ASVDLlamaForCausalLM"]

Transformers can reload those checkpoints only after the local ``svd_llama``
package has registered its custom AutoConfig / AutoModel classes. Scripts that
call ``AutoModelForCausalLM.from_pretrained(...)`` directly can otherwise fail
with an unknown model type / architecture error.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


_SVD_MODEL_TYPES = {"svdllama", "succinctllama"}


def _local_model_type(model_id_or_path: str | Path) -> Optional[str]:
    """Return config.model_type for a local checkpoint directory, if available."""
    path = Path(model_id_or_path).expanduser()
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type")
        return str(model_type).lower() if model_type is not None else None
    except Exception:
        # Do not mask the original Transformers loading error.
        return None


def ensure_svdllama_registered(model_id_or_path: str | Path) -> bool:
    """Import ``svd_llama`` when it is likely needed.

    Returns
    -------
    bool
        True if ``svd_llama`` was imported successfully, False if it was not
        required or unavailable for a non-local/non-SVD checkpoint.

    Raises
    ------
    ImportError
        If a *local* checkpoint clearly declares an SVDLlama-family model type
        but the ``svd_llama`` package cannot be imported.
    """
    local_type = _local_model_type(model_id_or_path)

    # For local compressed checkpoints, fail early with a targeted message.
    if local_type in _SVD_MODEL_TYPES:
        try:
            import svd_llama  # noqa: F401  # registers Auto* classes
            return True
        except ImportError as e:
            raise ImportError(
                "This checkpoint declares model_type='{}', but Python could not import "
                "the local 'svd_llama' package. Run from the turboquant-svd repo root "
                "or add the repo to PYTHONPATH before loading the checkpoint.".format(local_type)
            ) from e

    # For remote IDs or ordinary local checkpoints, opportunistically register
    # SVDLlama if the package is present; otherwise leave normal models alone.
    try:
        import svd_llama  # noqa: F401
        return True
    except ImportError:
        return False


def load_causal_lm_svdllama_aware(model_id_or_path: str | Path, **kwargs: Any):
    """Load a causal LM after ensuring SVDLlama Auto* registration is attempted."""
    from transformers import AutoModelForCausalLM

    ensure_svdllama_registered(model_id_or_path)
    return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)


def load_tokenizer_svdllama_aware(model_id_or_path: str | Path, **kwargs: Any):
    """Tokenizer wrapper kept for call-site symmetry."""
    from transformers import AutoTokenizer

    ensure_svdllama_registered(model_id_or_path)
    return AutoTokenizer.from_pretrained(model_id_or_path, **kwargs)
