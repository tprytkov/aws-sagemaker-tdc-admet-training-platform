from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_platform.training.multitask_regression_control import (
    PREDICTION_COLUMNS,
    initial_regression_control_state,
    update_regression_checkpoint_selection,
)
from admet_platform.training.regression_metrics import regression_metrics


TASKS = ("caco2_wang", "solubility_aqsoldb")


def _evaluation(step: int, nrmse: float, spearman: float) -> dict:
    return {
        "global_step": step,
        "split": "validation",
        "test_data_used": False,
        "mean_normalized_rmse": nrmse,
        "mean_spearman": spearman,
        "endpoints": {},
    }


def test_original_and_normalized_regression_metrics() -> None:
    metrics = regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 3.0, 2.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.0, 1.0, 0.0]),
    )

    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["r2"] == pytest.approx(0.0)
    assert metrics["pearson"] == pytest.approx(0.5)
    assert metrics["spearman"] == pytest.approx(0.5)
    assert metrics["normalized_rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["normalized_mae"] == pytest.approx(2 / 3)
    assert metrics["row_count"] == 3


def test_checkpoint_selection_minimizes_normalized_rmse_then_maximizes_spearman() -> None:
    state = initial_regression_control_state(TASKS)

    first = update_regression_checkpoint_selection(
        state, _evaluation(100, 0.8, 0.4)
    )
    worse = update_regression_checkpoint_selection(
        state, _evaluation(200, 0.9, 0.9)
    )
    tied = update_regression_checkpoint_selection(
        state, _evaluation(300, 0.8, 0.6)
    )

    assert first["selections"][0]["reason"] == (
        "lower_mean_validation_normalized_rmse"
    )
    assert not worse["composite_improved"]
    assert tied["selections"][0]["reason"] == (
        "higher_mean_validation_spearman_tiebreaker"
    )
    assert state["best_composite_step"] == 300
    assert state["shared_checkpoint_only"] is True


def test_checkpoint_selection_rejects_test_metrics() -> None:
    evaluation = _evaluation(1, 0.5, 0.5)
    evaluation["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        update_regression_checkpoint_selection(
            initial_regression_control_state(TASKS), evaluation
        )


def test_prediction_schema_is_stable(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "canonical_smiles": "CCO",
                "target_original": 1.0,
                "prediction_original": 1.2,
                "residual_original": 0.2,
                "target_normalized": 0.0,
                "prediction_normalized": 0.1,
            }
        ],
        columns=PREDICTION_COLUMNS,
    )
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    assert tuple(pd.read_csv(path).columns) == PREDICTION_COLUMNS
