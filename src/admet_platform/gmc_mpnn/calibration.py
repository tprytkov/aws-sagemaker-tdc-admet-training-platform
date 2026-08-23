"""Frozen TRAIN-OOF Platt calibration for GMC-MPNN BBB probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from sklearn.linear_model import LogisticRegression

from admet_platform.chemprop.calibration import select_maximum_mcc_threshold
from admet_platform.chemprop.metrics import classification_metrics
from admet_platform.training.multitask_calibration import calibration_metrics


GMC_CALIBRATION_VERSION: Final = "gmc-mpnn-bbb-train-oof-platt-v1"
CALIBRATION_METHOD: Final = "platt_scaling"
FIT_SPLIT: Final = "train_oof"
PROBABILITY_CLIP_EPSILON: Final = 1e-7
LOGISTIC_REGRESSION_C: Final = 1_000_000.0
LOGISTIC_REGRESSION_SOLVER: Final = "lbfgs"
LOGISTIC_REGRESSION_RANDOM_STATE: Final = 42
THRESHOLD_SELECTION_METHOD: Final = "maximum_mcc"
THRESHOLD_CANDIDATE_POLICY: Final = (
    "sorted unique calibrated TRAIN-OOF probabilities plus 0.0, 0.5, and 1.0"
)
THRESHOLD_TIE_BREAK_POLICY: Final = "smallest threshold with exactly maximum MCC"


class GMCCalibrationError(ValueError):
    """A GMC calibration input or frozen-artifact contract violation."""


@dataclass(frozen=True)
class FrozenGMCPlattCalibrator:
    coefficient: float
    intercept: float
    selected_threshold: float
    calibration_version: str = GMC_CALIBRATION_VERSION
    method: str = CALIBRATION_METHOD
    fit_split: str = FIT_SPLIT
    probability_clip_epsilon: float = PROBABILITY_CLIP_EPSILON

    def __post_init__(self) -> None:
        if self.calibration_version != GMC_CALIBRATION_VERSION:
            raise GMCCalibrationError("Frozen GMC calibration version is incompatible.")
        if self.method != CALIBRATION_METHOD or self.fit_split != FIT_SPLIT:
            raise GMCCalibrationError("Frozen GMC calibrator has incompatible provenance.")
        if not np.isfinite([self.coefficient, self.intercept]).all():
            raise GMCCalibrationError("Frozen GMC calibration parameters must be finite.")
        if self.probability_clip_epsilon != PROBABILITY_CLIP_EPSILON:
            raise GMCCalibrationError("Frozen GMC probability clipping epsilon is incompatible.")
        if not np.isfinite(self.selected_threshold) or not 0 <= self.selected_threshold <= 1:
            raise GMCCalibrationError("Frozen GMC operating threshold must be within [0, 1].")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _probabilities(probabilities)
        clipped = np.clip(
            values,
            self.probability_clip_epsilon,
            1.0 - self.probability_clip_epsilon,
        )
        logits = np.log(clipped / (1.0 - clipped))
        calibrated_logits = self.coefficient * logits + self.intercept
        calibrated = _sigmoid(calibrated_logits)
        if not np.isfinite(calibrated).all():
            raise GMCCalibrationError("Calibrated probabilities contain NaN or infinity.")
        return calibrated.astype(np.float64, copy=False)


def fit_train_oof_platt_calibrator(
    labels: np.ndarray,
    ensemble_probabilities: np.ndarray,
) -> FrozenGMCPlattCalibrator:
    """Fit the exact Chemprop BBB Platt convention using TRAIN OOF only."""

    y = _labels(labels)
    probabilities = _probabilities(ensemble_probabilities)
    if len(y) != len(probabilities) or len(y) == 0:
        raise GMCCalibrationError("TRAIN OOF labels and probabilities must be aligned.")
    clipped = np.clip(
        probabilities,
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        C=LOGISTIC_REGRESSION_C,
        solver=LOGISTIC_REGRESSION_SOLVER,
        random_state=LOGISTIC_REGRESSION_RANDOM_STATE,
    )
    model.fit(logits, y)
    coefficient = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    provisional = FrozenGMCPlattCalibrator(
        coefficient=coefficient,
        intercept=intercept,
        selected_threshold=0.5,
    )
    calibrated = provisional.transform(probabilities)
    threshold = select_maximum_mcc_threshold(y, calibrated)
    return FrozenGMCPlattCalibrator(
        coefficient=coefficient,
        intercept=intercept,
        selected_threshold=threshold,
    )


def complete_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Combine existing repository classification and binary-log-loss definitions."""

    y = _labels(labels)
    p = _probabilities(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise GMCCalibrationError("Metric labels and probabilities must be aligned.")
    classified = classification_metrics(y, p, threshold=threshold, ece_bins=10)
    calibrated = calibration_metrics(y, p)
    classified["binary_log_loss"] = calibrated["binary_log_loss"]
    if not np.isclose(classified["auroc"], calibrated["roc_auc"], rtol=0.0, atol=1e-15):
        raise GMCCalibrationError("Repository AUROC definitions disagree.")
    if not np.isclose(classified["auprc"], calibrated["average_precision"], rtol=0.0, atol=1e-15):
        raise GMCCalibrationError("Repository AUPRC definitions disagree.")
    if not np.isclose(classified["brier_score"], calibrated["brier_score"], rtol=0.0, atol=1e-15):
        raise GMCCalibrationError("Repository Brier-score definitions disagree.")
    if not np.isclose(
        classified["expected_calibration_error"],
        calibrated["expected_calibration_error_10_bins"],
        rtol=0.0,
        atol=1e-15,
    ):
        raise GMCCalibrationError("Repository ECE definitions disagree.")
    return classified


def _labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values).reshape(-1)
    try:
        numeric = labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise GMCCalibrationError("Labels must be finite binary values.") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise GMCCalibrationError("Labels must be finite binary values.")
    result = numeric.astype(np.int64)
    if set(result.tolist()) != {0, 1}:
        raise GMCCalibrationError("Calibration requires both binary classes.")
    return result


def _probabilities(values: np.ndarray) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise GMCCalibrationError("Probabilities must be numeric.") from exc
    if not np.isfinite(probabilities).all():
        raise GMCCalibrationError("Probabilities must be finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise GMCCalibrationError("Probabilities must be within [0, 1].")
    return probabilities


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


__all__ = [
    "CALIBRATION_METHOD",
    "FIT_SPLIT",
    "GMC_CALIBRATION_VERSION",
    "GMCCalibrationError",
    "FrozenGMCPlattCalibrator",
    "LOGISTIC_REGRESSION_C",
    "LOGISTIC_REGRESSION_RANDOM_STATE",
    "LOGISTIC_REGRESSION_SOLVER",
    "PROBABILITY_CLIP_EPSILON",
    "THRESHOLD_CANDIDATE_POLICY",
    "THRESHOLD_SELECTION_METHOD",
    "THRESHOLD_TIE_BREAK_POLICY",
    "complete_binary_metrics",
    "fit_train_oof_platt_calibrator",
    "select_maximum_mcc_threshold",
]
