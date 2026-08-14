"""Validation-only BBB calibration and operating-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float
    fit_split: str = "validation"

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))
        calibrated_logits = self.coefficient * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-calibrated_logits))


def fit_platt_calibrator(y_validation: np.ndarray, validation_probabilities: np.ndarray) -> PlattCalibrator:
    y = np.asarray(y_validation, dtype=int).reshape(-1)
    p = np.asarray(validation_probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or len(np.unique(y)) < 2:
        raise ValueError("Platt calibration requires aligned validation predictions with both classes.")
    logits = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", random_state=42).fit(logits, y)
    return PlattCalibrator(float(model.coef_[0, 0]), float(model.intercept_[0]))


def select_maximum_mcc_threshold(y_validation: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_validation, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Threshold selection requires aligned validation arrays.")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    scored = [(float(matthews_corrcoef(y, p >= threshold)), float(threshold)) for threshold in candidates]
    best_score = max(score for score, _ in scored)
    return min(threshold for score, threshold in scored if score == best_score)
