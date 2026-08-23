from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from admet_platform.chemprop.calibration import (
    fit_platt_calibrator,
    select_maximum_mcc_threshold,
)
from admet_platform.chemprop.metrics import classification_metrics
from admet_platform.gmc_mpnn.calibration import (
    FIT_SPLIT,
    PROBABILITY_CLIP_EPSILON,
    FrozenGMCPlattCalibrator,
    complete_binary_metrics,
    fit_train_oof_platt_calibrator,
)
from admet_platform.training.multitask_calibration import calibration_metrics


def test_train_oof_platt_matches_existing_chemprop_mathematics() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.05, 0.2, 0.4, 0.6, 0.8, 0.95], dtype=np.float64)

    frozen = fit_train_oof_platt_calibrator(labels, probabilities)
    existing = fit_platt_calibrator(labels, probabilities)

    assert frozen.fit_split == FIT_SPLIT == "train_oof"
    assert frozen.coefficient == pytest.approx(existing.coefficient, rel=0.0, abs=1e-12)
    assert frozen.intercept == pytest.approx(existing.intercept, rel=0.0, abs=1e-12)
    expected = 1.0 / (
        1.0
        + np.exp(
            -(
                frozen.coefficient
                * np.log(
                    np.clip(probabilities, 1e-7, 1 - 1e-7)
                    / (1 - np.clip(probabilities, 1e-7, 1 - 1e-7))
                )
                + frozen.intercept
            )
        )
    )
    np.testing.assert_allclose(frozen.transform(probabilities), expected, rtol=0.0, atol=1e-15)


def test_zero_and_one_probabilities_are_safely_clipped() -> None:
    calibrator = FrozenGMCPlattCalibrator(
        coefficient=1.0,
        intercept=0.0,
        selected_threshold=0.5,
    )
    transformed = calibrator.transform(np.asarray([0.0, 1.0]))

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(
        transformed,
        [PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON],
        rtol=0.0,
        atol=1e-15,
    )


def test_threshold_selection_matches_existing_exact_tie_policy() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    probabilities = np.asarray([0.2, 0.8], dtype=np.float64)
    selected = select_maximum_mcc_threshold(labels, probabilities)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [matthews_corrcoef(labels, probabilities >= threshold) for threshold in candidates]
    expected = min(
        threshold
        for threshold, score in zip(candidates, scores, strict=True)
        if score == max(scores)
    )
    assert scores[list(candidates).index(0.5)] == scores[list(candidates).index(0.8)] == max(scores)
    assert selected == expected == 0.5


def test_complete_metrics_reuse_repository_definitions() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    observed = complete_binary_metrics(labels, probabilities, threshold=0.5)
    classification = classification_metrics(labels, probabilities, threshold=0.5, ece_bins=10)
    calibration = calibration_metrics(labels, probabilities)

    for key, value in classification.items():
        assert observed[key] == value
    assert observed["binary_log_loss"] == calibration["binary_log_loss"]
