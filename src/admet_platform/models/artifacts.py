"""Artifact helpers for local ADMET baseline training."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(float(value))
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    return value


def _safe_float(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return value
