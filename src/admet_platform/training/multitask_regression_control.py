"""Validation-only evaluation and shared checkpoint selection for regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from admet_platform.data.regression_transforms import FittedRegressionTransform
from admet_platform.training.regression_metrics import regression_metrics


PREDICTION_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "target_original",
    "prediction_original",
    "residual_original",
    "target_normalized",
    "prediction_normalized",
)


def evaluate_regression_validation(
    trainer: Any,
    validation_loaders: Mapping[str, Any],
    transforms: Mapping[str, FittedRegressionTransform],
    output_dir: str | Path,
    global_step: int,
) -> dict[str, Any]:
    if set(validation_loaders) != set(trainer.model.task_names):
        raise ValueError("Validation loaders must exactly match regression tasks.")
    if set(transforms) != set(trainer.model.task_names):
        raise ValueError("Target transforms must exactly match regression tasks.")
    destination = Path(output_dir) / "validation" / f"step_{global_step:08d}"
    destination.mkdir(parents=True, exist_ok=True)
    endpoints: dict[str, Any] = {}
    for task in trainer.model.task_names:
        rows: list[dict[str, Any]] = []
        weighted_raw_loss = 0.0
        example_count = 0
        for batch in validation_loaders[task]:
            record = trainer.evaluation_step(task, batch)
            prediction_normalized = record["logits"].numpy().astype(np.float64)
            target_normalized = batch["labels"].numpy().astype(np.float64)
            prediction_original = transforms[task].inverse_values(
                prediction_normalized
            )
            target_original = batch["target_original"].numpy().astype(np.float64)
            count = int(record["example_count"])
            raw_loss = float(record["raw_losses"][task])
            weighted_raw_loss += raw_loss * count
            example_count += count
            for molecule_id, smiles, raw_target, raw_prediction, target_z, prediction_z in zip(
                batch["molecule_id"],
                batch["canonical_smiles"],
                target_original,
                prediction_original,
                target_normalized,
                prediction_normalized,
            ):
                rows.append(
                    {
                        "molecule_id": str(molecule_id),
                        "canonical_smiles": str(smiles),
                        "target_original": float(raw_target),
                        "prediction_original": float(raw_prediction),
                        "residual_original": float(raw_prediction - raw_target),
                        "target_normalized": float(target_z),
                        "prediction_normalized": float(prediction_z),
                    }
                )
        predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        prediction_name = f"validation_predictions_{task}.csv"
        predictions.to_csv(destination / prediction_name, index=False)
        predictions.to_csv(Path(output_dir) / prediction_name, index=False)
        metrics = regression_metrics(
            predictions["target_original"].to_numpy(),
            predictions["prediction_original"].to_numpy(),
            predictions["target_normalized"].to_numpy(),
            predictions["prediction_normalized"].to_numpy(),
        )
        metrics["validation_loss"] = weighted_raw_loss / example_count
        metrics["prediction_file"] = str(
            Path("validation") / destination.name / prediction_name
        )
        endpoints[task] = metrics
    normalized_rmse = [endpoints[task]["normalized_rmse"] for task in trainer.model.task_names]
    spearman = [endpoints[task]["spearman"] for task in trainer.model.task_names]
    return {
        "global_step": int(global_step),
        "split": "validation",
        "endpoints": endpoints,
        "mean_normalized_rmse": float(np.mean(normalized_rmse)),
        "mean_spearman": (
            float(np.mean(spearman)) if all(value is not None for value in spearman) else None
        ),
        "test_data_used": False,
    }


def initial_regression_control_state(tasks: tuple[str, ...]) -> dict[str, Any]:
    return {
        "best_mean_normalized_rmse": None,
        "best_mean_spearman": None,
        "best_composite_step": None,
        "evaluations_without_improvement": 0,
        "evaluation_count": 0,
        "stopped_early": False,
        "stop_reason": None,
        "selection_events": [],
        "validation_history": [],
        "shared_checkpoint_only": True,
        "endpoint_names": list(tasks),
    }


def update_regression_checkpoint_selection(
    state: dict[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    if evaluation.get("split") != "validation" or evaluation.get("test_data_used") is not False:
        raise ValueError("Regression checkpoint selection accepts validation data only.")
    score = float(evaluation["mean_normalized_rmse"])
    tie = evaluation.get("mean_spearman")
    if not np.isfinite(score):
        raise ValueError("mean_normalized_rmse must be finite.")
    if tie is not None:
        tie = float(tie)
        if not np.isfinite(tie):
            raise ValueError("mean_spearman must be finite when provided.")
    state["evaluation_count"] += 1
    best = state["best_mean_normalized_rmse"]
    best_tie = state["best_mean_spearman"]
    improved = best is None or score < best
    reason = "lower_mean_validation_normalized_rmse"
    if not improved and score == best and tie is not None:
        improved = best_tie is None or tie > best_tie
        reason = "higher_mean_validation_spearman_tiebreaker"
    selections = []
    if improved:
        state["best_mean_normalized_rmse"] = score
        state["best_mean_spearman"] = tie
        state["best_composite_step"] = int(evaluation["global_step"])
        selections.append({"kind": "shared_composite", "reason": reason})
    state["evaluations_without_improvement"] = (
        0 if improved else state["evaluations_without_improvement"] + 1
    )
    event = {
        "global_step": int(evaluation["global_step"]),
        "source_split": "validation",
        "mean_normalized_rmse": score,
        "mean_spearman": tie,
        "composite_improved": improved,
        "selections": selections,
    }
    state["selection_events"].append(event)
    return event


def should_stop_regression_early(
    state: Mapping[str, Any],
    *,
    global_step: int,
    patience_evaluations: int,
    minimum_training_steps: int,
) -> bool:
    return bool(
        patience_evaluations > 0
        and global_step >= minimum_training_steps
        and state["evaluations_without_improvement"] >= patience_evaluations
    )


__all__ = [
    "PREDICTION_COLUMNS",
    "evaluate_regression_validation",
    "initial_regression_control_state",
    "should_stop_regression_early",
    "update_regression_checkpoint_selection",
]
