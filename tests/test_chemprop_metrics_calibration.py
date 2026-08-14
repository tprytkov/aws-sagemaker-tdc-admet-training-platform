import numpy as np
import pytest

from admet_platform.chemprop.calibration import fit_platt_calibrator, select_maximum_mcc_threshold
from admet_platform.chemprop.metrics import classification_metrics, regression_metrics


def test_regression_metrics_include_required_values() -> None:
    result = regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.5, 1.5]))
    assert set(result) == {"mae", "rmse", "r2", "spearman", "median_absolute_error"}
    assert result["mae"] == pytest.approx(1 / 3)


def test_validation_only_platt_and_maximum_mcc_threshold() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.55, 0.7, 0.9])
    calibrator = fit_platt_calibrator(labels, probabilities)
    assert calibrator.fit_split == "validation"
    calibrated = calibrator.transform(probabilities)
    threshold = select_maximum_mcc_threshold(labels, calibrated)
    assert 0 <= threshold <= 1
    result = classification_metrics(labels, calibrated, threshold)
    assert result["mcc"] == pytest.approx(1.0)
    assert "expected_calibration_error" in result
