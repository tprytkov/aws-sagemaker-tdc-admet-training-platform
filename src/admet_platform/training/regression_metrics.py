"""Original-unit and normalized validation metrics for regression."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(
    target_original: np.ndarray,
    prediction_original: np.ndarray,
    target_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
) -> dict[str, Any]:
    original = _paired(target_original, prediction_original, "original")
    normalized = _paired(target_normalized, prediction_normalized, "normalized")
    if len(original[0]) != len(normalized[0]):
        raise ValueError("Original and normalized regression arrays must have equal length.")
    y_true, y_pred = original
    z_true, z_pred = normalized
    residual = y_pred - y_true
    normalized_residual = z_pred - z_true
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": _r2(y_true, y_pred),
        "spearman": _correlation(_ranks(y_true), _ranks(y_pred)),
        "pearson": _correlation(y_true, y_pred),
        "normalized_rmse": float(
            np.sqrt(np.mean(np.square(normalized_residual)))
        ),
        "normalized_mae": float(np.mean(np.abs(normalized_residual))),
        "row_count": int(len(y_true)),
    }


def _paired(
    first: np.ndarray, second: np.ndarray, scale: str
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if not len(left) or len(left) != len(right):
        raise ValueError(f"{scale} regression arrays must be non-empty and equal length.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{scale} regression arrays must be finite.")
    return left, right


def _r2(target: np.ndarray, prediction: np.ndarray) -> float | None:
    denominator = float(np.sum(np.square(target - np.mean(target))))
    if denominator == 0:
        return None
    return float(1.0 - np.sum(np.square(target - prediction)) / denominator)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


__all__ = ["regression_metrics"]
