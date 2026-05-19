from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class SVDLogger:
    """Small JSONL logger compatible with the old SVD hooks.

    Logging is optional. The compression workflow works with no SVDLOG_PATH.
    """

    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = None if path is None else Path(path)

    def _write(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def run_header(self, metadata: dict[str, Any]) -> None:
        self._write({"type": "header", "metadata": metadata})

    def log(self, **kwargs: Any) -> None:
        payload = {"type": "svd", **kwargs}
        # Avoid serializing giant tensors into the optional JSONL log.
        for key in ("S", "U", "V"):
            if key in payload:
                value = payload[key]
                try:
                    payload[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                    }
                except Exception:
                    payload[key] = str(type(value))
        self._write(payload)


def get_logger_from_env() -> SVDLogger | None:
    path = os.environ.get("SVDLOG_PATH")
    if not path:
        return None
    return SVDLogger(path)
