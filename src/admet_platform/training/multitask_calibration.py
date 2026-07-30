"""Validation-only Platt calibration for multi-task binary classifiers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

from admet_platform.data.multitask import (
    EndpointDatasetSplits,
    MultiTaskConfig,
    MultiTaskTrainingConfig,
    _load_prepared_split,
    build_task_dataloaders,
    load_multitask_config,
)
from admet_platform.models.multitask_chemberta import (
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.training.multitask_control import evaluate_split
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss
from admet_platform.training.multitask_trainer import MultiTaskTrainer


CALIBRATION_SCHEMA_VERSION = "1.0.0"
CALIBRATION_METHOD = "platt_scaling"
DEFAULT_MINIMUM_CLASS_SUPPORT = 20
DEFAULT_CALIBRATION_SEED = 42
RAW_PREDICTION_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "target",
    "raw_logit",
    "probability",
    "prediction",
)


def sigmoid(values: np.ndarray | list[float]) -> np.ndarray:
    """Calculate a numerically stable sigmoid."""

    logits = np.asarray(values, dtype=np.float64)
    result = np.empty_like(logits)
    nonnegative = logits >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def fit_platt_calibrator(
    raw_logits: np.ndarray | list[float],
    targets: np.ndarray | list[int],
    *,
    minimum_class_support: int = DEFAULT_MINIMUM_CLASS_SUPPORT,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> dict[str, Any]:
    """Fit deterministic logistic regression to one endpoint's validation logits."""

    logits, labels = _validated_binary_inputs(raw_logits, targets)
    if minimum_class_support <= 0:
        raise ValueError("minimum_class_support must be positive.")
    class_counts = _class_counts(labels)
    parameters: dict[str, Any] = {
        "calibration_method": CALIBRATION_METHOD,
        "coefficient_a": None,
        "intercept_b": None,
        "fit_status": "insufficient_class_support",
        "class_counts": class_counts,
        "minimum_class_support_per_class": int(minimum_class_support),
        "source_split": "validation",
    }
    if min(class_counts["class_0"], class_counts["class_1"]) < minimum_class_support:
        return parameters

    estimator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=seed,
    )
    estimator.fit(logits.reshape(-1, 1), labels)
    parameters.update(
        {
            "coefficient_a": float(estimator.coef_[0, 0]),
            "intercept_b": float(estimator.intercept_[0]),
            "fit_status": "fitted",
        }
    )
    return parameters


def apply_platt_calibration(
    raw_logits: np.ndarray | list[float],
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Apply fitted parameters, or retain uncalibrated probabilities if not fitted."""

    logits = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError("raw_logits must contain only finite values.")
    if parameters.get("fit_status") != "fitted":
        return sigmoid(logits)
    coefficient = parameters.get("coefficient_a")
    intercept = parameters.get("intercept_b")
    if not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient):
        raise ValueError("Fitted calibration coefficient_a must be finite.")
    if not isinstance(intercept, (int, float)) or not math.isfinite(intercept):
        raise ValueError("Fitted calibration intercept_b must be finite.")
    return sigmoid(float(coefficient) * logits + float(intercept))


def expected_calibration_error(
    targets: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    *,
    bins: int = 10,
) -> float:
    """Calculate ECE with fixed equal-width probability bins."""

    labels, scores = _validated_metric_inputs(targets, probabilities)
    if bins <= 0:
        raise ValueError("bins must be positive.")
    bin_indices = np.minimum((scores * bins).astype(int), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(scores[mask].mean()) - float(labels[mask].mean())
        )
    return float(error)


def calibration_metrics(
    targets: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
) -> dict[str, Any]:
    """Calculate endpoint discrimination, calibration, and class-support metrics."""

    labels, scores = _validated_metric_inputs(targets, probabilities)
    both_classes = len(np.unique(labels)) == 2
    return {
        "brier_score": float(brier_score_loss(labels, scores)),
        "binary_log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "expected_calibration_error_10_bins": expected_calibration_error(
            labels, scores, bins=10
        ),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, scores)) if both_classes else None
        ),
        "class_counts": _class_counts(labels),
    }


def calibrate_endpoint_predictions(
    predictions: pd.DataFrame,
    *,
    minimum_class_support: int = DEFAULT_MINIMUM_CLASS_SUPPORT,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit and apply one validation calibrator while retaining the raw columns."""

    missing = [column for column in RAW_PREDICTION_COLUMNS if column not in predictions]
    if missing:
        raise ValueError(
            "Validation predictions are missing required column(s): " + ", ".join(missing)
        )
    frame = predictions.loc[:, RAW_PREDICTION_COLUMNS].copy()
    logits = frame["raw_logit"].to_numpy(dtype=np.float64)
    targets = frame["target"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    expected = sigmoid(logits)
    if not np.allclose(expected, probabilities, rtol=1e-6, atol=1e-7):
        raise ValueError("Validation probability does not equal sigmoid(raw_logit).")

    parameters = fit_platt_calibrator(
        logits,
        targets,
        minimum_class_support=minimum_class_support,
        seed=seed,
    )
    calibrated = (
        apply_platt_calibration(logits, parameters)
        if parameters["fit_status"] == "fitted"
        else probabilities.copy()
    )
    if np.any((calibrated < 0.0) | (calibrated > 1.0)):
        raise RuntimeError("Calibrated probabilities fell outside [0, 1].")
    frame["calibrated_probability"] = calibrated
    frame["calibrated_prediction"] = (calibrated >= 0.5).astype(int)
    metrics = {
        "fit_status": parameters["fit_status"],
        "uncalibrated": calibration_metrics(targets, probabilities),
        "calibrated": calibration_metrics(targets, calibrated),
    }
    return frame, parameters, metrics


def run_multitask_calibration(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    prepared_root: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    source_split: str = "validation",
    minimum_class_support: int = DEFAULT_MINIMUM_CLASS_SUPPORT,
) -> dict[str, Any]:
    """Generate validation predictions and fit endpoint-specific calibrators."""

    if source_split != "validation":
        raise ValueError("Calibration source_split must be 'validation'.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config_source = Path(config_path).resolve()
    checkpoint_source = Path(checkpoint_path).resolve()
    prepared_source = Path(prepared_root).resolve()
    output = Path(output_dir).resolve()
    if not checkpoint_source.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint_source}")
    config = load_multitask_config(config_source)
    checkpoint = MultiTaskTrainer.read_checkpoint(checkpoint_source, "cpu")
    model_config = MultiTaskChemBERTaConfig.from_dict(checkpoint["model_config"])
    if tuple(config.tasks) != model_config.tasks:
        raise ValueError("Calibration config endpoint order does not match the checkpoint.")
    training_config = MultiTaskTrainingConfig(**checkpoint["training_config"])

    run_root = checkpoint_source.parents[1]
    encoder_config_path = run_root / "model" / "encoder_config"
    tokenizer_path = run_root / "tokenizer"
    if not encoder_config_path.is_dir() or not tokenizer_path.is_dir():
        raise FileNotFoundError(
            "Selected run is missing its local encoder configuration or tokenizer."
        )
    encoder_config = AutoConfig.from_pretrained(
        encoder_config_path, local_files_only=True
    )
    encoder = AutoModel.from_config(encoder_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = MultiTaskChemBERTa(model_config, encoder=encoder)
    loss_metadata = checkpoint["loss_metadata"]
    loss_module = MultiTaskBinaryLoss(
        loss_metadata["positive_class_weights"],
        loss_metadata["task_loss_weights"],
    )
    trainer = MultiTaskTrainer(
        model,
        None,
        loss_module,
        training_config,
        device=device,
        evaluation_only=True,
    )
    selection_state_before = json.dumps(trainer.control_state, sort_keys=True)
    trainer.load_checkpoint_for_evaluation(checkpoint_source)
    if trainer.optimizer is not None:
        raise RuntimeError("Calibration constructed an optimizer.")

    datasets = _load_validation_datasets(
        config,
        prepared_source,
    )
    loaders = build_task_dataloaders(
        datasets,
        tokenizer,
        seed=training_config.random_seed,
        train_batch_size=training_config.train_batch_size,
        evaluation_batch_size=training_config.evaluation_batch_size,
        max_length=training_config.max_sequence_length,
        splits=("validation",),
    )
    if any(set(task_loaders) != {"validation"} for task_loaders in loaders.values()):
        raise RuntimeError("Calibration constructed a non-validation DataLoader.")

    output.mkdir(parents=True, exist_ok=True)
    evaluate_split(
        trainer,
        {
            task: task_loaders["validation"]
            for task, task_loaders in loaders.items()
        },
        output,
        trainer.global_step,
        split="validation",
    )
    if json.dumps(trainer.control_state, sort_keys=True) != selection_state_before:
        raise RuntimeError("Calibration altered checkpoint-selection state.")

    checkpoint_identity = _file_identity(checkpoint_source)
    config_identity = _file_identity(config_source)
    split_manifest = _split_manifest_identity(run_root)
    git_commit = _git_commit()
    endpoint_parameters: dict[str, Any] = {}
    endpoint_metrics: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for endpoint in model_config.tasks:
        generated_path = output / f"validation_predictions_{endpoint}.csv"
        raw_path = output / f"validation_predictions_raw_{endpoint}.csv"
        raw_frame = pd.read_csv(generated_path)
        raw_frame = raw_frame.loc[:, RAW_PREDICTION_COLUMNS]
        raw_frame.to_csv(raw_path, index=False)
        calibrated_frame, parameters, metrics = calibrate_endpoint_predictions(
            raw_frame,
            minimum_class_support=minimum_class_support,
            seed=training_config.random_seed,
        )
        calibrated_frame.to_csv(
            output / f"validation_predictions_calibrated_{endpoint}.csv",
            index=False,
        )
        parameter_record = {
            "endpoint": endpoint,
            **parameters,
            "checkpoint": checkpoint_identity,
            "config": config_identity,
            "coordinated_split_manifest_identifier": split_manifest["identifier"],
            "git_commit": git_commit,
        }
        endpoint_parameters[endpoint] = parameter_record
        endpoint_metrics[endpoint] = metrics
        summary_rows.append(_metrics_summary_row(endpoint, metrics))

    parameters_payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_split": "validation",
        "endpoint_order": list(model_config.tasks),
        "minimum_class_support_per_class": minimum_class_support,
        "checkpoint": checkpoint_identity,
        "config": config_identity,
        "coordinated_split_manifest": split_manifest,
        "git_commit": git_commit,
        "endpoints": endpoint_parameters,
    }
    metrics_payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_split": "validation",
        "endpoint_order": list(model_config.tasks),
        "endpoints": endpoint_metrics,
    }
    manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_split": "validation",
        "test_data_accessed": False,
        "checkpoint_selection_performed": False,
        "training_performed": False,
        "threshold_optimization_performed": False,
        "descriptive_prediction_threshold": 0.5,
        "endpoint_order": list(model_config.tasks),
        "checkpoint": checkpoint_identity,
        "config": config_identity,
        "prepared_root": str(prepared_source),
        "coordinated_split_manifest": split_manifest,
        "git_commit": git_commit,
        "artifacts": {
            "calibration_manifest": "calibration_manifest.json",
            "calibration_parameters": "calibration_parameters.json",
            "calibration_metrics_by_endpoint": "calibration_metrics_by_endpoint.json",
            "calibration_metrics_summary": "calibration_metrics_summary.csv",
            "validation_predictions_raw": {
                endpoint: f"validation_predictions_raw_{endpoint}.csv"
                for endpoint in model_config.tasks
            },
            "validation_predictions_calibrated": {
                endpoint: f"validation_predictions_calibrated_{endpoint}.csv"
                for endpoint in model_config.tasks
            },
        },
    }
    _write_json(output / "calibration_parameters.json", parameters_payload)
    _write_json(output / "calibration_metrics_by_endpoint.json", metrics_payload)
    pd.DataFrame(summary_rows).to_csv(
        output / "calibration_metrics_summary.csv", index=False
    )
    _write_json(output / "calibration_manifest.json", manifest)
    return manifest


def _load_validation_datasets(
    config: MultiTaskConfig,
    prepared_root: Path,
) -> dict[str, EndpointDatasetSplits]:
    """Load validation frames without constructing paths for any other split."""

    datasets: dict[str, EndpointDatasetSplits] = {}
    validation_filename = config.split_files["validation"]
    for task_name, endpoint in config.tasks.items():
        validation_path = (
            prepared_root / endpoint.endpoint_id / validation_filename
        )
        validation = _load_prepared_split(
            validation_path,
            endpoint,
            "validation",
            config.training.allow_smiles_fallback,
        )
        datasets[task_name] = EndpointDatasetSplits(
            endpoint=endpoint,
            train=None,
            validation=validation,
            test=None,
            paths={"validation": validation_path},
        )
    return datasets


def _validated_binary_inputs(
    raw_logits: np.ndarray | list[float],
    targets: np.ndarray | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(targets).reshape(-1)
    if logits.size == 0 or logits.size != labels.size:
        raise ValueError("raw_logits and targets must have the same non-zero length.")
    if not np.isfinite(logits).all():
        raise ValueError("raw_logits must contain only finite values.")
    numeric_labels = labels.astype(np.float64)
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, [0, 1]).all():
        raise ValueError("targets must contain only binary 0/1 values.")
    return logits, numeric_labels.astype(int)


def _validated_metric_inputs(
    targets: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    _, labels = _validated_binary_inputs(np.zeros_like(scores), targets)
    if scores.size != labels.size:
        raise ValueError("targets and probabilities must have the same length.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite and constrained to [0, 1].")
    return labels, scores


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "class_0": int((labels == 0).sum()),
        "class_1": int((labels == 1).sum()),
        "total": int(labels.size),
    }


def _metrics_summary_row(endpoint: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "endpoint": endpoint,
        "fit_status": metrics["fit_status"],
    }
    for stage in ("uncalibrated", "calibrated"):
        for metric in (
            "brier_score",
            "binary_log_loss",
            "expected_calibration_error_10_bins",
            "roc_auc",
            "average_precision",
        ):
            row[f"{stage}_{metric}"] = metrics[stage][metric]
    counts = metrics["uncalibrated"]["class_counts"]
    row.update(
        {
            "class_0_count": counts["class_0"],
            "class_1_count": counts["class_1"],
            "row_count": counts["total"],
        }
    )
    return row


def _file_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _split_manifest_identity(run_root: Path) -> dict[str, Any]:
    run_manifest_path = run_root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"Selected run is missing run_manifest.json: {run_manifest_path}"
        )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    prepared_manifest = run_manifest.get("prepared_split_manifest")
    identifier = (
        prepared_manifest.get("split_manifest_id")
        if isinstance(prepared_manifest, dict)
        else None
    )
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(
            "Selected run manifest does not record a coordinated split manifest identifier."
        )
    return {
        "identifier": identifier,
        "training_run_manifest": str(run_manifest_path),
        "training_run_manifest_sha256": hashlib.sha256(
            run_manifest_path.read_bytes()
        ).hexdigest(),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CALIBRATION_METHOD",
    "DEFAULT_MINIMUM_CLASS_SUPPORT",
    "RAW_PREDICTION_COLUMNS",
    "apply_platt_calibration",
    "calibrate_endpoint_predictions",
    "calibration_metrics",
    "expected_calibration_error",
    "fit_platt_calibrator",
    "run_multitask_calibration",
    "sigmoid",
]
