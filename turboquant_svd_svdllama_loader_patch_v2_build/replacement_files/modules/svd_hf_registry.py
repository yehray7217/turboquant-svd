"""Compatibility registry for saved SVDLlama checkpoints.

Older notes/snippets may use:
    from modules.svd_hf_registry import register_svdllama_auto_classes
    register_svdllama_auto_classes()

The real work is to import the local svd_llama package, whose import side effect
registers the SVDLlama config/model classes with Hugging Face Transformers.
"""
from __future__ import annotations


def register_svdllama_auto_classes() -> bool:
    """Register local SVDLlama AutoConfig/AutoModel classes.

    Returns True when registration import succeeds. Raises ImportError with a
    targeted message when the local svd_llama package is not importable.
    """
    try:
        import svd_llama  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "Could not import local 'svd_llama'. Run this from the turboquant-svd "
            "repo root, or set PYTHONPATH=$PWD before running Transformers Auto* loaders."
        ) from e


# Alias for scripts that prefer an ensure_* name.
ensure_svdllama_registered = register_svdllama_auto_classes
