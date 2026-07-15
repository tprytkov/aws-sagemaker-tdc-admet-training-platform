"""Load and validate existing local model-run artifacts."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from admet_platform.evaluation.schemas import ModelRun, SUPPORTED_MODEL_FAMILIES


REQUIRED_FILES = ("metrics.json", "predictions_validation.csv", "predictions_test.csv", "training_metadata.json")
CLASSIFICATION_PREDICTION_COLUMNS = {"observed_target", "predicted_class", "predicted_probability"}
REGRESSION_PREDICTION_COLUMNS = {"observed_target", "predicted_value"}


def load_model_runs(run_dirs: list[str | Path]) -> list[ModelRun]:
    """Load a list of run directories and reject duplicate run IDs."""

    runs = [load_model_run(run_dir) for run_dir in run_dirs]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for run in runs:
        if run.run_id in seen:
            duplicates.add(run.run_id)
        seen.add(run.run_id)
    if duplicates:
        raise ValueError(f"Duplicate run_id value(s) found: {', '.join(sorted(duplicates))}.")
    return runs


def discover_run_dirs(parent_dir: str | Path) -> list[Path]:
    """Discover direct or nested directories that look like model runs."""

    parent = Path(parent_dir)
    if not parent.exists():
        raise ValueError(f"Discovery parent directory does not exist: {parent}.")
    candidates = sorted(path.parent for path in parent.rglob("metrics.json"))
    if not candidates:
        raise ValueError(f"No model-run directories found under: {parent}.")
    malformed: list[str] = []
    valid: list[Path] = []
    for candidate in candidates:
        try:
            _validate_required_files(candidate)
            valid.append(candidate)
        except ValueError:
            malformed.append(str(candidate))
    if malformed:
        raise ValueError(f"Malformed model-run directories discovered: {', '.join(malformed)}.")
    return valid


def load_model_run(run_dir: str | Path) -> ModelRun:
    run_path = Path(run_dir)
    _validate_required_files(run_path)
    metrics = _read_json(run_path / "metrics.json")
    training_metadata = _read_json(run_path / "training_metadata.json")
    feature_metadata = _read_optional_json(run_path / "feature_metadata.json")
    model_config = _read_optional_json(run_path / "model_config.json")
    run_manifest = _read_optional_json(run_path / "run_manifest.json")

    endpoint_id = _required_matching_value("endpoint_id", metrics, training_metadata, run_manifest)
    task_type = _required_matching_value("task_type", metrics, training_metadata, run_manifest)
    if task_type not in {"binary_classification", "regression"}:
        raise ValueError(f"Unsupported task_type '{task_type}' in {run_path}.")
    validation_metrics = _required_metrics(metrics, "validation", run_path)
    test_metrics = metrics.get("test") or {}
    if not isinstance(test_metrics, dict):
        raise ValueError(f"metrics.json test field must be an object in {run_path}.")

    _validate_prediction_schema(run_path / "predictions_validation.csv", task_type)
    _validate_prediction_schema(run_path / "predictions_test.csv", task_type)

    feature_type = metrics.get("feature_type") or training_metadata.get("feature_type")
    pretrained = training_metadata.get("pretrained_model_name") or model_config.get("model_name")
    model_family = "chemberta" if pretrained or (run_path / "model").exists() else "classical"
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family '{model_family}' in {run_path}.")
    if model_family == "classical" and feature_type not in {"descriptors", "morgan"}:
        raise ValueError(f"Unsupported classical feature_type '{feature_type}' in {run_path}.")

    model_type = metrics.get("model_type") or training_metadata.get("model_type") or model_family
    source_dataset = (
        metrics.get("source_dataset")
        or training_metadata.get("source_dataset")
        or run_manifest.get("source_dataset")
        or "unavailable"
    )
    run_id = str(training_metadata.get("run_id") or run_manifest.get("run_id") or _stable_run_id(run_path))
    development_mode = bool(
        training_metadata.get("development_row_limit") is not None
        or model_config.get("development_row_limit") is not None
        or run_manifest.get("development_mode")
        or any("development" in str(warning).lower() or "smoke" in str(warning).lower() for warning in metrics.get("warnings", []))
    )
    package_versions = training_metadata.get("package_versions") or run_manifest.get("package_versions") or {}
    split_provenance = {
        "source_dataset": source_dataset,
        "split_strategy": training_metadata.get("split_strategy") or run_manifest.get("split_strategy") or "unavailable",
        "train_rows": _optional_int(training_metadata.get("training_row_count") or run_manifest.get("train_count")),
        "validation_rows": _optional_int(training_metadata.get("validation_row_count") or run_manifest.get("validation_count")),
        "test_rows": _optional_int(training_metadata.get("test_row_count") or run_manifest.get("test_count")),
    }

    return ModelRun(
        run_id=run_id,
        run_dir=run_path,
        endpoint_id=endpoint_id,
        task_type=task_type,
        source_dataset=source_dataset,
        model_family=model_family,
        model_type=str(model_type),
        feature_type=feature_type,
        pretrained_checkpoint=pretrained,
        development_mode=development_mode,
        train_rows=split_provenance["train_rows"],
        validation_rows=split_provenance["validation_rows"],
        test_rows=split_provenance["test_rows"],
        feature_count=_optional_int(training_metadata.get("feature_count") or feature_metadata.get("n_features")),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        warnings=list(metrics.get("warnings", [])) + list(training_metadata.get("warnings", [])),
        model_artifact_path=_resolve_model_artifact(run_path, model_family),
        tokenizer_path=str(run_path / "tokenizer") if (run_path / "tokenizer").exists() else None,
        inference_metadata_path=str(run_path / "model_config.json") if (run_path / "model_config.json").exists() else None,
        training_metadata=training_metadata,
        feature_metadata=feature_metadata,
        model_config=model_config,
        split_provenance=split_provenance,
        package_versions=package_versions,
    )


def _validate_required_files(run_path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (run_path / name).exists()]
    if missing:
        raise ValueError(f"Model run {run_path} is missing required artifact(s): {', '.join(missing)}.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON artifact {path}: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact {path} must contain an object.")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _required_matching_value(field: str, *payloads: dict[str, Any]) -> str:
    values = {str(payload[field]) for payload in payloads if payload.get(field) not in (None, "")}
    if not values:
        raise ValueError(f"Missing required field '{field}' in model-run artifacts.")
    if len(values) > 1:
        raise ValueError(f"Artifact {field} mismatch: {', '.join(sorted(values))}.")
    return values.pop()


def _required_metrics(metrics: dict[str, Any], split: str, run_path: Path) -> dict[str, Any]:
    split_metrics = metrics.get(split)
    if not isinstance(split_metrics, dict) or not split_metrics:
        raise ValueError(f"Model run {run_path} is missing {split} metrics.")
    return split_metrics


def _validate_prediction_schema(path: Path, task_type: str) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    required = CLASSIFICATION_PREDICTION_COLUMNS if task_type == "binary_classification" else REGRESSION_PREDICTION_COLUMNS
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Prediction file {path} has incompatible schema; missing: {', '.join(missing)}.")


def _stable_run_id(run_path: Path) -> str:
    normalized = str(run_path.resolve()).replace("\\", "/")
    return f"run-{uuid.uuid5(uuid.NAMESPACE_URL, normalized).hex[:12]}"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_model_artifact(run_path: Path, model_family: str) -> str | None:
    if model_family == "classical" and (run_path / "model.joblib").exists():
        return str(run_path / "model.joblib")
    if (run_path / "model").exists():
        return str(run_path / "model")
    return None
