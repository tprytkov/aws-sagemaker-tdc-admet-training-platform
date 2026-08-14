"""Endpoint-complete metrics for Chemprop and matched baselines."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Regression arrays must be finite, non-empty, and aligned.")
    spearman = spearmanr(y, p).statistic if len(np.unique(p)) > 1 else None
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "spearman": float(spearman) if spearman is not None and np.isfinite(spearman) else None,
        "median_absolute_error": float(np.median(np.abs(y - p))),
    }


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, ece_bins: int = 10,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p) or set(np.unique(y)) - {0, 1}:
        raise ValueError("Classification arrays must be aligned with binary labels.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1].")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": expected_calibration_error(y, p, ece_bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if bins < 2:
        raise ValueError("ECE requires at least two bins.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return None, None
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs").fit(logits, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])
