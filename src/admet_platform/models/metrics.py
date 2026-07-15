"""Metric helpers for local ADMET baseline models."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    """Calculate binary classification metrics without crashing on tiny splits."""

    metric_warnings: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metrics: dict[str, Any] = {
            "roc_auc": None,
            "pr_auc": None,
            "accuracy": _json_float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": _json_float(balanced_accuracy_score(y_true, y_pred)),
            "precision": _json_float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": _json_float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": _json_float(f1_score(y_true, y_pred, zero_division=0)),
            "matthews_correlation_coefficient": _json_float(matthews_corrcoef(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        }

    if len(np.unique(y_true)) < 2:
        metric_warnings.append("ROC AUC and PR AUC are unavailable because y_true has one class.")
        return metrics, metric_warnings

    metrics["roc_auc"] = _json_float(roc_auc_score(y_true, y_probability))
    metrics["pr_auc"] = _json_float(average_precision_score(y_true, y_probability))
    return metrics, metric_warnings


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    """Calculate regression metrics with nulls and warnings for unavailable values."""

    metric_warnings: list[str] = []
    metrics: dict[str, Any] = {
        "rmse": _json_float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": _json_float(mean_absolute_error(y_true, y_pred)),
        "r2": None,
        "pearson_correlation": None,
        "spearman_correlation": None,
    }

    if len(y_true) >= 2:
        metrics["r2"] = _json_float(r2_score(y_true, y_pred))
    else:
        metric_warnings.append("R2 is unavailable because fewer than two rows were provided.")

    if len(y_true) >= 2 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConstantInputWarning)
            try:
                metrics["pearson_correlation"] = _json_float(pearsonr(y_true, y_pred).statistic)
            except (ConstantInputWarning, ValueError):
                metric_warnings.append("Pearson correlation is unavailable for constant values.")
            try:
                metrics["spearman_correlation"] = _json_float(spearmanr(y_true, y_pred).statistic)
            except (ConstantInputWarning, ValueError):
                metric_warnings.append("Spearman correlation is unavailable for constant values.")
    else:
        metric_warnings.append("Pearson and Spearman correlations are unavailable for this split.")

    return metrics, metric_warnings


def _json_float(value: float | np.floating) -> float | None:
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return as_float
