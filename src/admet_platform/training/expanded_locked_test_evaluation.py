"""Frozen, evaluation-only workflow for the expanded ten-endpoint classifier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

from admet_platform.config import _load_yaml_mapping
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
from admet_platform.training.multitask_calibration import (
    RAW_PREDICTION_COLUMNS,
    calibration_metrics,
    sigmoid,
)
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss
from admet_platform.training.multitask_trainer import MultiTaskTrainer


SCHEMA_VERSION = "1.0.0"
EXPECTED_ENDPOINT_ORDER = (
    "hia_hou",
    "pgp_broccatelli",
    "bbb_martins",
    "cyp1a2_veith",
    "cyp2c19_veith",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "herg_karim",
    "ames",
)
EXPECTED_OUTPUT_ARTIFACTS = (
    "test_predictions_raw_<endpoint>.csv",
    "test_predictions_calibrated_<endpoint>.csv",
    "test_metrics_by_endpoint.json",
    "test_metrics_summary.csv",
    "test_bootstrap_confidence_intervals.csv",
    "test_evaluation_manifest.json",
    "manuscript_test_results_table.csv",
)
CI_METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "sensitivity",
    "specificity",
)


@dataclass(frozen=True)
class ExpandedEvaluationConfig:
    source_path: Path
    project_root: Path
    endpoint_order: tuple[str, ...]
    training_config: Path
    training_config_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_global_step: int
    checkpoint_random_seed: int
    checkpoint_task_sampling: str
    checkpoint_task_sampling_alpha: float
    prepared_root: Path
    coordinated_manifest: Path
    coordinated_manifest_id: str
    training_run_manifest: Path
    calibration_root: Path
    calibration_hashes: Mapping[str, str]
    preserve_uncalibrated_endpoints: tuple[str, ...]
    threshold: float
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    output_artifacts: tuple[str, ...]


def load_expanded_evaluation_config(path: str | Path) -> ExpandedEvaluationConfig:
    """Load and validate the immutable expanded-evaluation contract."""

    source = Path(path).resolve()
    root = source.parent.parent
    raw = _load_yaml_mapping(source.read_text(encoding="utf-8"), source=str(source))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expanded evaluation schema_version must be '{SCHEMA_VERSION}'.")
    if raw.get("source_split") != "test" or raw.get("evaluation_only") is not True:
        raise ValueError("Expanded evaluation must be evaluation-only on source_split=test.")
    forbidden_multi_model_fields = {"checkpoints", "experiments", "models"} & set(raw)
    if forbidden_multi_model_fields:
        raise ValueError(
            "Expanded evaluation accepts exactly one checkpoint; unsupported field(s): "
            + ", ".join(sorted(forbidden_multi_model_fields))
        )
    endpoint_order = tuple(raw.get("endpoint_order", ()))
    if endpoint_order != EXPECTED_ENDPOINT_ORDER:
        raise ValueError("Expanded evaluation endpoint order does not match the frozen order.")

    training = _mapping(raw, "training_config")
    checkpoint = _mapping(raw, "checkpoint")
    coordinated = _mapping(raw, "coordinated_manifest")
    calibration = _mapping(raw, "calibration")
    bootstrap = _mapping(raw, "bootstrap")
    artifact_hashes = _mapping(calibration, "artifact_hashes")
    required_calibration = {
        "calibration_parameters.json",
        "calibration_manifest.json",
        "calibration_artifact_hashes.txt",
    }
    if set(artifact_hashes) != required_calibration:
        raise ValueError("Calibration artifact hash set is incomplete or unexpected.")
    for name, digest in artifact_hashes.items():
        _validate_sha256(digest, f"calibration.artifact_hashes.{name}")
    if calibration.get("method") != "platt_scaling":
        raise ValueError("Expanded evaluation requires frozen Platt scaling.")
    threshold = calibration.get("descriptive_threshold")
    if threshold != 0.5:
        raise ValueError("Expanded evaluation threshold must remain frozen at 0.5.")
    preserved = tuple(calibration.get("preserve_uncalibrated_endpoints", ()))
    if preserved != ("hia_hou",):
        raise ValueError("Only hia_hou may preserve uncalibrated probability.")

    output_artifacts = tuple(raw.get("output_artifacts", ()))
    if output_artifacts != EXPECTED_OUTPUT_ARTIFACTS:
        raise ValueError("Expanded evaluation output artifact plan does not match the contract.")
    if bootstrap.get("strategy") != "endpoint_stratified":
        raise ValueError("Bootstrap strategy must be endpoint_stratified.")
    replicates = bootstrap.get("replicates")
    seed = bootstrap.get("seed")
    confidence = bootstrap.get("confidence_level")
    if replicates != 2000 or not isinstance(seed, int) or confidence != 0.95:
        raise ValueError("Bootstrap must use 2000 replicates, an integer seed, and 95% CIs.")

    training_path = _project_path(root, training.get("path"), "training_config.path")
    checkpoint_path = _project_path(root, checkpoint.get("path"), "checkpoint.path")
    training_sha = training.get("sha256")
    checkpoint_sha = checkpoint.get("sha256")
    _validate_sha256(training_sha, "training_config.sha256")
    _validate_sha256(checkpoint_sha, "checkpoint.sha256")
    return ExpandedEvaluationConfig(
        source_path=source,
        project_root=root,
        endpoint_order=endpoint_order,
        training_config=training_path,
        training_config_sha256=training_sha,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_global_step=int(checkpoint.get("global_step")),
        checkpoint_random_seed=int(checkpoint.get("random_seed")),
        checkpoint_task_sampling=str(checkpoint.get("task_sampling")),
        checkpoint_task_sampling_alpha=float(checkpoint.get("task_sampling_alpha")),
        prepared_root=_project_path(root, raw.get("prepared_root"), "prepared_root"),
        coordinated_manifest=_project_path(root, coordinated.get("path"), "coordinated_manifest.path"),
        coordinated_manifest_id=str(coordinated.get("split_manifest_id")),
        training_run_manifest=_project_path(root, raw.get("training_run_manifest"), "training_run_manifest"),
        calibration_root=_project_path(root, calibration.get("root"), "calibration.root"),
        calibration_hashes=dict(artifact_hashes),
        preserve_uncalibrated_endpoints=preserved,
        threshold=float(threshold),
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        confidence_level=float(confidence),
        output_artifacts=output_artifacts,
    )


def validate_expanded_evaluation_dry_run(
    config: ExpandedEvaluationConfig,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate frozen inputs without constructing, opening, or hashing a test path."""

    verified = {
        "training_config": _verify_file_hash(
            config.training_config, config.training_config_sha256, "Training config"
        ),
        "checkpoint": _verify_file_hash(
            config.checkpoint, config.checkpoint_sha256, "Checkpoint"
        ),
    }
    parsed_training = load_multitask_config(config.training_config)
    if tuple(parsed_training.tasks) != config.endpoint_order:
        raise ValueError("Training config endpoint order does not match the frozen order.")
    _validate_checkpoint_metadata(config)

    coordinated = _read_json(config.coordinated_manifest, "Coordinated manifest")
    if coordinated.get("split_manifest_id") != config.coordinated_manifest_id:
        raise ValueError("Coordinated split manifest identifier mismatch.")
    calibration_hashes = {
        name: _verify_file_hash(config.calibration_root / name, digest, name)
        for name, digest in config.calibration_hashes.items()
    }
    parameters = _read_json(
        config.calibration_root / "calibration_parameters.json",
        "Calibration parameters",
    )
    calibration_manifest = _read_json(
        config.calibration_root / "calibration_manifest.json",
        "Calibration manifest",
    )
    endpoint_parameters = _validate_calibration_contract(
        config, parameters, calibration_manifest
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run",
        "dry_run": True,
        "source_split": "test",
        "test_data_accessed": False,
        "evaluation_executed": False,
        "evaluation_only": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_fitting_performed": False,
        "threshold_optimization_performed": False,
        "model_comparison_performed": False,
        "number_of_model_checkpoints_evaluated": 1,
        "endpoint_order": list(config.endpoint_order),
        "verified_hashes": {**verified, "calibration": calibration_hashes},
        "calibration_fit_status": {
            endpoint: endpoint_parameters[endpoint]["fit_status"]
            for endpoint in config.endpoint_order
        },
        "output_dir": str(Path(output_dir).resolve()),
        "expected_output_plan": _expanded_output_plan(config.endpoint_order),
    }


def run_expanded_locked_test_evaluation(
    *,
    evaluation_config: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate frozen inputs or evaluate exactly one checkpoint once."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_expanded_evaluation_config(evaluation_config)
    output = Path(output_dir).resolve()
    if not dry_run:
        _require_empty_output_directory(output)
    validation = validate_expanded_evaluation_dry_run(config, output_dir=output_dir)
    if dry_run:
        return validation

    expected_test_hashes = _load_and_validate_test_manifest_contracts(config)
    verified_test_hashes = _hash_locked_test_files(config, expected_test_hashes)
    parameters = _read_json(
        config.calibration_root / "calibration_parameters.json",
        "Calibration parameters",
    )["endpoints"]
    staging = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    raw_predictions, global_step = _generate_test_predictions(config, device=device)

    metrics_by_endpoint: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(config.endpoint_order):
        raw = raw_predictions[endpoint].loc[:, RAW_PREDICTION_COLUMNS].copy()
        raw.to_csv(staging / f"test_predictions_raw_{endpoint}.csv", index=False)
        calibrated = apply_frozen_calibration(raw, parameters[endpoint], endpoint=endpoint)
        calibrated.to_csv(
            staging / f"test_predictions_calibrated_{endpoint}.csv", index=False
        )
        labels = raw["target"].to_numpy(dtype=int)
        uncalibrated_scores = raw["probability"].to_numpy(dtype=float)
        calibrated_scores = calibrated["calibrated_probability"].to_numpy(dtype=float)
        endpoint_metrics = {
            "calibration_status": parameters[endpoint]["fit_status"],
            "uncalibrated": expanded_binary_metrics(labels, uncalibrated_scores),
            "calibrated": expanded_binary_metrics(labels, calibrated_scores),
        }
        metrics_by_endpoint[endpoint] = endpoint_metrics
        summary_rows.extend(
            _summary_rows(
                endpoint,
                {
                    "uncalibrated": endpoint_metrics["uncalibrated"],
                    "calibrated": endpoint_metrics["calibrated"],
                },
                calibration_status=parameters[endpoint]["fit_status"],
            )
        )
        endpoint_seed = config.bootstrap_seed + endpoint_index
        for probability_type, scores in (
            ("uncalibrated", uncalibrated_scores),
            ("calibrated", calibrated_scores),
        ):
            bootstrap_rows.extend(
                stratified_bootstrap_confidence_intervals(
                    labels,
                    scores,
                    endpoint=endpoint,
                    probability_type=probability_type,
                    replicates=config.bootstrap_replicates,
                    seed=endpoint_seed,
                    confidence_level=config.confidence_level,
                )
            )

    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_split": "test",
        "endpoint_order": list(config.endpoint_order),
        "endpoints": metrics_by_endpoint,
    }
    _write_json(staging / "test_metrics_by_endpoint.json", metrics_payload)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(staging / "test_metrics_summary.csv", index=False)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(
        staging / "test_bootstrap_confidence_intervals.csv", index=False
    )
    manuscript = _build_manuscript_table(
        config.endpoint_order,
        metrics_by_endpoint,
        bootstrap_frame,
    )
    manuscript.to_csv(staging / "manuscript_test_results_table.csv", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_split": "test",
        "test_data_accessed": True,
        "evaluation_only": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_fitting_performed": False,
        "threshold_optimization_performed": False,
        "model_comparison_performed": False,
        "number_of_model_checkpoints_evaluated": 1,
        "checkpoint": validation["verified_hashes"]["checkpoint"],
        "training_config": validation["verified_hashes"]["training_config"],
        "calibration_artifacts": validation["verified_hashes"]["calibration"],
        "calibration_freeze_record": validation["verified_hashes"]["calibration"][
            "calibration_artifact_hashes.txt"
        ],
        "coordinated_split_manifest_id": config.coordinated_manifest_id,
        "verified_test_hashes": verified_test_hashes,
        "git_commit": _git_commit(),
        "endpoint_order": list(config.endpoint_order),
        "calibration_status_by_endpoint": {
            endpoint: parameters[endpoint]["fit_status"]
            for endpoint in config.endpoint_order
        },
        "global_step": global_step,
        "threshold": config.threshold,
        "bootstrap": {
            "replicates": config.bootstrap_replicates,
            "seed": config.bootstrap_seed,
            "confidence_level": config.confidence_level,
            "strategy": "endpoint_stratified",
            "class_support_preserved_by_stratification": True,
            "class_missing_replicates_expected_when_both_classes_observed": False,
        },
        "artifacts": _expanded_output_plan(config.endpoint_order),
        "completion_status": "complete",
    }
    _write_json(staging / "test_evaluation_manifest.json", manifest)
    _promote_completed_output(staging, output)
    return manifest


def apply_frozen_calibration(
    predictions: pd.DataFrame,
    parameters: Mapping[str, Any],
    *,
    endpoint: str,
) -> pd.DataFrame:
    """Apply immutable validation-fit parameters to aligned raw test predictions."""

    missing = [column for column in RAW_PREDICTION_COLUMNS if column not in predictions]
    if missing:
        raise ValueError("Raw predictions are missing: " + ", ".join(missing))
    frame = predictions.loc[:, RAW_PREDICTION_COLUMNS].copy()
    logits = frame["raw_logit"].to_numpy(dtype=float)
    probabilities = frame["probability"].to_numpy(dtype=float)
    if not np.allclose(sigmoid(logits), probabilities, rtol=1e-6, atol=1e-7):
        raise ValueError("Uncalibrated probability does not equal sigmoid(raw_logit).")
    if endpoint == "hia_hou":
        if parameters.get("fit_status") != "insufficient_class_support":
            raise ValueError("hia_hou must retain insufficient-support calibration status.")
        calibrated = probabilities.copy()
    else:
        if parameters.get("fit_status") != "fitted":
            raise ValueError(f"Endpoint '{endpoint}' does not have fitted calibration parameters.")
        coefficient = _finite_number(parameters.get("coefficient_a"), "coefficient_a")
        intercept = _finite_number(parameters.get("intercept_b"), "intercept_b")
        calibrated = sigmoid(coefficient * logits + intercept)
    frame["calibrated_probability"] = calibrated
    frame["calibrated_prediction"] = (calibrated >= 0.5).astype(int)
    return frame


def expanded_binary_metrics(
    labels: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
) -> dict[str, Any]:
    """Calculate frozen-threshold discrimination, calibration, and confusion metrics."""

    y = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    calibration = calibration_metrics(y, scores)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y, predictions, labels=[0, 1]).ravel()]
    both = len(np.unique(y)) == 2
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    return {
        **calibration,
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)) if both else None,
        "mcc": float(matthews_corrcoef(y, predictions)) if both else None,
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "precision_ppv": float(ppv) if ppv is not None else None,
        "npv": float(npv) if npv is not None else None,
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "class_support": calibration["class_counts"],
    }


def stratified_bootstrap_confidence_intervals(
    labels: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    *,
    endpoint: str,
    probability_type: str,
    replicates: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    """Return deterministic endpoint-stratified percentile intervals."""

    y = np.asarray(labels, dtype=int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.size != scores.size or y.size == 0:
        raise ValueError("Bootstrap labels and probabilities must be aligned and nonempty.")
    if replicates <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap settings are invalid.")
    class_indices = [np.flatnonzero(y == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in CI_METRICS}
    rejected: dict[str, int] = {metric: 0 for metric in CI_METRICS}
    for _ in range(replicates):
        sampled_parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
            if len(indices)
        ]
        sampled = np.concatenate(sampled_parts)
        replicate_metrics = expanded_binary_metrics(y[sampled], scores[sampled])
        for metric in CI_METRICS:
            value = replicate_metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                rejected[metric] += 1
            else:
                values[metric].append(float(value))
    point = expanded_binary_metrics(y, scores)
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    for metric in CI_METRICS:
        successful = len(values[metric])
        lower = float(np.quantile(values[metric], alpha)) if successful else None
        upper = float(np.quantile(values[metric], 1.0 - alpha)) if successful else None
        rows.append(
            {
                "endpoint": endpoint,
                "probability_type": probability_type,
                "metric": metric,
                "point_estimate": point.get(metric),
                "confidence_level": confidence_level,
                "ci_lower": lower,
                "ci_upper": upper,
                "requested_replicates": replicates,
                "successful_replicates": successful,
                "rejected_replicates": rejected[metric],
                "bootstrap_seed": seed,
                "resampling": "endpoint_stratified",
                "source_sample_size": int(y.size),
                "both_classes_present_in_source": bool(all(len(indices) for indices in class_indices)),
                "class_support_preserved_by_stratification": True,
            }
        )
    return rows


def verify_locked_test_hashes(config: ExpandedEvaluationConfig) -> dict[str, Any]:
    """Hash test files only in real mode and match both frozen manifests."""

    expected = _load_and_validate_test_manifest_contracts(config)
    return _hash_locked_test_files(config, expected)


def _load_and_validate_test_manifest_contracts(
    config: ExpandedEvaluationConfig,
) -> dict[str, dict[str, str]]:
    """Validate both frozen manifests and collect hashes before test-path access."""

    coordinated = _read_json(config.coordinated_manifest, "Coordinated manifest")
    run_manifest = _read_json(config.training_run_manifest, "Training-run manifest")
    if coordinated.get("split_manifest_id") != config.coordinated_manifest_id:
        raise ValueError("Coordinated split manifest identifier mismatch.")
    if tuple(run_manifest.get("endpoint_order", ())) != config.endpoint_order:
        raise ValueError("Training-run manifest endpoint order mismatch.")
    if run_manifest.get("seed") != config.checkpoint_random_seed:
        raise ValueError("Training-run manifest random seed mismatch.")
    sampling = run_manifest.get("task_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("Training-run manifest task-sampling metadata is missing.")
    if sampling.get("strategy") != config.checkpoint_task_sampling:
        raise ValueError("Training-run manifest task-sampling strategy mismatch.")
    if sampling.get("alpha") != config.checkpoint_task_sampling_alpha:
        raise ValueError("Training-run manifest task-sampling alpha mismatch.")
    prepared = run_manifest.get("prepared_split_manifest")
    if not isinstance(prepared, dict) or prepared.get("split_manifest_id") != config.coordinated_manifest_id:
        raise ValueError("Training-run manifest coordinated split identifier mismatch.")
    output_checkpoint = run_manifest.get("output_checkpoint")
    if not isinstance(output_checkpoint, dict) or output_checkpoint.get("sha256") != config.checkpoint_sha256:
        raise ValueError("Training-run manifest checkpoint identity mismatch.")

    expected: dict[str, dict[str, str]] = {}
    for endpoint in config.endpoint_order:
        key = f"{endpoint}/test"
        coordinated_expected = _manifest_hash(coordinated, key)
        run_expected = _manifest_hash(run_manifest, key)
        if coordinated_expected is None:
            raise ValueError(f"Coordinated manifest is missing a frozen hash for {key}.")
        if run_expected is None:
            raise ValueError(f"Training-run manifest is missing a frozen hash for {key}.")
        expected[endpoint] = {
            "coordinated": coordinated_expected,
            "training_run": run_expected,
        }
    return expected


def _hash_locked_test_files(
    config: ExpandedEvaluationConfig,
    expected_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """First test-data access: hash each locked CSV against both validated manifests."""

    parsed = load_multitask_config(config.training_config)
    verified: dict[str, Any] = {}
    for endpoint in config.endpoint_order:
        key = f"{endpoint}/test"
        filename = parsed.split_files["test"]
        test_path = config.prepared_root / parsed.tasks[endpoint].endpoint_id / filename
        if test_path.name != "test.csv":
            raise ValueError(f"Locked split for '{endpoint}' must be named test.csv.")
        actual = _sha256(test_path)
        coordinated_expected = expected_hashes[endpoint]["coordinated"]
        run_expected = expected_hashes[endpoint]["training_run"]
        if actual != coordinated_expected:
            raise ValueError(f"Coordinated manifest hash mismatch for {key}.")
        if actual != run_expected:
            raise ValueError(f"Selected training-run manifest hash mismatch for {key}.")
        verified[endpoint] = {"sha256": actual}
    return verified


def _generate_test_predictions(
    config: ExpandedEvaluationConfig,
    *,
    device: str,
) -> tuple[dict[str, pd.DataFrame], int]:
    run_root = config.checkpoint.parents[1]
    tokenizer = AutoTokenizer.from_pretrained(
        run_root / "tokenizer", local_files_only=True
    )
    parsed = load_multitask_config(config.training_config)
    datasets = _load_test_datasets(parsed, config.prepared_root)
    loaders = build_task_dataloaders(
        datasets,
        tokenizer,
        seed=parsed.training.random_seed,
        train_batch_size=parsed.training.train_batch_size,
        evaluation_batch_size=parsed.training.evaluation_batch_size,
        max_length=parsed.training.max_sequence_length,
        splits=("test",),
    )
    if any(set(task_loaders) != {"test"} for task_loaders in loaders.values()):
        raise RuntimeError("Expanded evaluation constructed a non-test DataLoader.")

    checkpoint = MultiTaskTrainer.read_checkpoint(config.checkpoint, "cpu")
    model_config = MultiTaskChemBERTaConfig.from_dict(checkpoint["model_config"])
    training_config = MultiTaskTrainingConfig(**checkpoint["training_config"])
    encoder_config = AutoConfig.from_pretrained(
        run_root / "model" / "encoder_config", local_files_only=True
    )
    model = MultiTaskChemBERTa(model_config, encoder=AutoModel.from_config(encoder_config))
    loss_metadata = checkpoint["loss_metadata"]
    trainer = MultiTaskTrainer(
        model,
        None,
        MultiTaskBinaryLoss(
            loss_metadata["positive_class_weights"],
            loss_metadata["task_loss_weights"],
        ),
        training_config,
        device=device,
        evaluation_only=True,
    )
    selection_before = json.dumps(trainer.control_state, sort_keys=True)
    trainer.load_checkpoint_for_evaluation(config.checkpoint)
    if trainer.optimizer is not None or trainer.scheduler is not None:
        raise RuntimeError("Expanded evaluation constructed training state.")
    predictions = {
        endpoint: _predict_endpoint(trainer, endpoint, loaders[endpoint]["test"])
        for endpoint in config.endpoint_order
    }
    if json.dumps(trainer.control_state, sort_keys=True) != selection_before:
        raise RuntimeError("Expanded evaluation altered checkpoint-selection state.")
    return predictions, trainer.global_step


def _predict_endpoint(
    trainer: MultiTaskTrainer,
    endpoint: str,
    loader: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in loader:
        record = trainer.evaluation_step(endpoint, batch)
        logits = record["logits"].numpy().reshape(-1)
        probabilities = sigmoid(logits)
        labels = batch["labels"].numpy().reshape(-1)
        count = int(record["example_count"])
        lengths = (len(batch["molecule_id"]), len(batch["canonical_smiles"]), len(labels), len(logits))
        if any(length != count for length in lengths):
            raise RuntimeError(f"Prediction alignment failed for endpoint '{endpoint}'.")
        for index in range(count):
            rows.append(
                {
                    "molecule_id": batch["molecule_id"][index],
                    "canonical_smiles": batch["canonical_smiles"][index],
                    "target": int(labels[index]),
                    "raw_logit": float(logits[index]),
                    "probability": float(probabilities[index]),
                    "prediction": int(probabilities[index] >= 0.5),
                }
            )
    return pd.DataFrame(rows, columns=RAW_PREDICTION_COLUMNS)


def _load_test_datasets(
    config: MultiTaskConfig,
    prepared_root: Path,
) -> dict[str, EndpointDatasetSplits]:
    datasets: dict[str, EndpointDatasetSplits] = {}
    filename = config.split_files["test"]
    for name, endpoint in config.tasks.items():
        path = prepared_root / endpoint.endpoint_id / filename
        test = _load_prepared_split(
            path, endpoint, "test", config.training.allow_smiles_fallback
        )
        datasets[name] = EndpointDatasetSplits(
            endpoint=endpoint,
            train=None,
            validation=None,
            test=test,
            paths={"test": path},
        )
    return datasets


def _validate_checkpoint_metadata(config: ExpandedEvaluationConfig) -> None:
    checkpoint = MultiTaskTrainer.read_checkpoint(config.checkpoint, "cpu")
    model_config = checkpoint.get("model_config", {})
    if tuple(model_config.get("tasks", ())) != config.endpoint_order:
        raise ValueError("Checkpoint endpoint order does not match the frozen order.")
    if int(checkpoint.get("global_step", -1)) != config.checkpoint_global_step:
        raise ValueError("Checkpoint global step does not match the frozen selection.")
    training = checkpoint.get("training_config", {})
    expected = {
        "random_seed": config.checkpoint_random_seed,
        "task_sampling": config.checkpoint_task_sampling,
        "task_sampling_alpha": config.checkpoint_task_sampling_alpha,
    }
    for field, value in expected.items():
        if training.get(field) != value:
            raise ValueError(f"Checkpoint training metadata mismatch for '{field}'.")


def _validate_calibration_contract(
    config: ExpandedEvaluationConfig,
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    if parameters.get("source_split") != "validation":
        raise ValueError("Calibration parameters must originate from validation.")
    if tuple(parameters.get("endpoint_order", ())) != config.endpoint_order:
        raise ValueError("Calibration parameter endpoint order mismatch.")
    if tuple(manifest.get("endpoint_order", ())) != config.endpoint_order:
        raise ValueError("Calibration manifest endpoint order mismatch.")
    if manifest.get("source_split") != "validation" or manifest.get("test_data_accessed") is not False:
        raise ValueError("Calibration manifest does not prove validation-only calibration.")
    frozen_false_fields = (
        "checkpoint_selection_performed",
        "training_performed",
        "threshold_optimization_performed",
    )
    if any(manifest.get(field) is not False for field in frozen_false_fields):
        raise ValueError("Calibration manifest does not preserve the frozen calibration policy.")
    for payload_name, payload in (("parameters", parameters), ("manifest", manifest)):
        checkpoint_identity = payload.get("checkpoint")
        config_identity = payload.get("config")
        split_identity = payload.get("coordinated_split_manifest")
        if not isinstance(checkpoint_identity, dict) or checkpoint_identity.get("sha256") != config.checkpoint_sha256:
            raise ValueError(f"Calibration {payload_name} checkpoint identity mismatch.")
        if not isinstance(config_identity, dict) or config_identity.get("sha256") != config.training_config_sha256:
            raise ValueError(f"Calibration {payload_name} config identity mismatch.")
        if not isinstance(split_identity, dict) or split_identity.get("identifier") != config.coordinated_manifest_id:
            raise ValueError(f"Calibration {payload_name} split-manifest identity mismatch.")
    endpoints = parameters.get("endpoints")
    if not isinstance(endpoints, dict) or tuple(endpoints) != config.endpoint_order:
        raise ValueError("Calibration endpoint mapping order mismatch.")
    for endpoint in config.endpoint_order:
        record = endpoints[endpoint]
        if not isinstance(record, dict) or record.get("endpoint") != endpoint:
            raise ValueError(f"Calibration schema mismatch for endpoint '{endpoint}'.")
        if record.get("calibration_method") != "platt_scaling":
            raise ValueError(f"Calibration method mismatch for endpoint '{endpoint}'.")
        if endpoint == "hia_hou":
            if record.get("fit_status") != "insufficient_class_support":
                raise ValueError("hia_hou must remain uncalibrated.")
        else:
            if record.get("fit_status") != "fitted":
                raise ValueError(f"Endpoint '{endpoint}' must have fitted calibration.")
            _finite_number(record.get("coefficient_a"), "coefficient_a")
            _finite_number(record.get("intercept_b"), "intercept_b")
    return endpoints


def _summary_rows(
    endpoint: str,
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    calibration_status: str,
) -> list[dict[str, Any]]:
    rows = []
    for probability_type, values in metrics.items():
        row = {
            "endpoint": endpoint,
            "probability_type": probability_type,
            "calibration_status": calibration_status,
        }
        for key, value in values.items():
            row[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        rows.append(row)
    return rows


def _build_manuscript_table(
    endpoint_order: tuple[str, ...],
    metrics_by_endpoint: Mapping[str, Mapping[str, Any]],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Format final metric objects and their matching CIs without recomputation."""

    ci_lookup = {
        (row.endpoint, row.metric): row
        for row in bootstrap.loc[bootstrap["probability_type"] == "calibrated"].itertuples()
    }
    rows: list[dict[str, Any]] = []
    for endpoint in endpoint_order:
        endpoint_result = metrics_by_endpoint[endpoint]
        metrics = endpoint_result["calibrated"]
        support = metrics["class_support"]
        row: dict[str, Any] = {
            "endpoint": endpoint,
            "n": support["total"],
            "class_0": support["class_0"],
            "class_1": support["class_1"],
            "calibration_status": endpoint_result["calibration_status"],
        }
        for metric in CI_METRICS:
            ci = ci_lookup[(endpoint, metric)]
            row[metric] = metrics[metric]
            row[f"{metric}_ci_lower"] = ci.ci_lower
            row[f"{metric}_ci_upper"] = ci.ci_upper
        rows.append(row)
    return pd.DataFrame(rows)


def _expanded_output_plan(endpoint_order: tuple[str, ...]) -> list[str]:
    return [
        *(f"test_predictions_raw_{endpoint}.csv" for endpoint in endpoint_order),
        *(f"test_predictions_calibrated_{endpoint}.csv" for endpoint in endpoint_order),
        *EXPECTED_OUTPUT_ARTIFACTS[2:],
    ]


def _require_empty_output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {path}")


def _promote_completed_output(staging: Path, output: Path) -> None:
    """Atomically expose a completed run; partial work remains visibly incomplete."""

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
        output.rmdir()
    staging.replace(output)


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
    return result.stdout.strip() or None


def _manifest_hash(manifest: Mapping[str, Any], key: str) -> str | None:
    for field in ("output_file_sha256", "input_hashes", "test_file_sha256"):
        values = manifest.get(field)
        if isinstance(values, dict) and isinstance(values.get(key), str):
            return values[key]
    for field in ("dataset_manifest", "prepared_split_manifest"):
        nested = manifest.get(field)
        if isinstance(nested, dict):
            value = _manifest_hash(nested, key)
            if value is not None:
                return value
    endpoint, separator, split = key.partition("/")
    endpoints = manifest.get("endpoints")
    if separator and isinstance(endpoints, dict):
        endpoint_record = endpoints.get(endpoint)
        if isinstance(endpoint_record, dict):
            splits = endpoint_record.get("splits")
            split_record = splits.get(split) if isinstance(splits, dict) else None
            if isinstance(split_record, dict):
                value = split_record.get("output_csv_sha256")
                if isinstance(value, str):
                    return value
    return None


def _verify_file_hash(path: Path, expected: str, label: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch.")
    return {"path": str(path), "sha256": actual}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _mapping(parent: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = parent.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"Expanded evaluation field '{field}' must be a mapping.")
    return value


def _project_path(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Expanded evaluation field '{field}' must be a path.")
    return (root / raw).resolve()


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Expanded evaluation field '{field}' must be a lowercase SHA-256.")


def _finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Calibration field '{field}' must be finite.")
    return float(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EXPECTED_ENDPOINT_ORDER",
    "EXPECTED_OUTPUT_ARTIFACTS",
    "ExpandedEvaluationConfig",
    "apply_frozen_calibration",
    "expanded_binary_metrics",
    "load_expanded_evaluation_config",
    "run_expanded_locked_test_evaluation",
    "stratified_bootstrap_confidence_intervals",
    "validate_expanded_evaluation_dry_run",
    "verify_locked_test_hashes",
]
