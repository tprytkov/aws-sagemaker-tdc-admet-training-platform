"""Validated configurations for one multitask regressor and one BBB classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_TASKS = {"regression", "binary_classification"}
REGRESSION_ENDPOINTS = (
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "ppbr_az",
    "vdss_lombardo",
)
REQUIRED_MODEL_KEYS = {
    "atom_featurizer", "aggregation", "message_passing_depth", "message_hidden_dim",
    "ffn_num_layers", "ffn_hidden_dim", "dropout",
}
REQUIRED_TRAINING_KEYS = {
    "batch_size", "max_epochs", "early_stopping_patience", "warmup_epochs",
    "init_lr", "max_lr", "final_lr", "gradient_clip_norm", "num_workers", "accelerator",
}


@dataclass(frozen=True)
class ChempropExperimentConfig:
    source_path: Path
    raw: dict[str, Any]
    endpoint: str
    endpoint_id: str
    dataset: str
    dataset_version: str
    task_type: str
    primary_metric: str
    prepared_root: Path
    split_manifest: Path
    split_files: dict[str, str]
    split_hash_keys: dict[str, str]
    tasks: dict[str, dict[str, Any]]
    model: dict[str, Any]
    training: dict[str, Any]


def load_chemprop_config(path: str | Path) -> ChempropExperimentConfig:
    source = Path(path).resolve()
    raw = _load_yaml(source)
    parent_name = raw.pop("extends", None)
    if parent_name:
        parent = _load_yaml((source.parent / str(parent_name)).resolve())
        raw = _deep_merge(parent, raw)
    required = {
        "schema_version", "provider", "model_family", "chemprop_version", "endpoint",
        "endpoint_id", "dataset", "dataset_version", "task_type", "primary_metric",
        "prepared_root", "split_manifest", "split_files", "model", "training",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Chemprop config is missing required fields: {missing}")
    if raw["chemprop_version"] != "2.3.1":
        raise ValueError("Chemprop version must remain pinned to 2.3.1.")
    if raw["task_type"] not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task_type: {raw['task_type']}")
    if set(raw["split_files"]) != {"train", "validation", "test"}:
        raise ValueError("split_files must define train, validation, and test separately.")
    _require_keys(raw["model"], REQUIRED_MODEL_KEYS, "model")
    _require_keys(raw["training"], REQUIRED_TRAINING_KEYS, "training")
    tasks: dict[str, dict[str, Any]] = {}
    split_hash_keys: dict[str, str] = {}
    if raw["task_type"] == "regression":
        tasks = _validate_regression_tasks(raw)
        if raw.get("target_scaling") != "train_only_per_endpoint_standard":
            raise ValueError("Regression must normalize every endpoint from training labels only.")
        if raw.get("task_weighting", {}).get("method") != (
            "inverse_training_label_count_equal_endpoint_contribution"
        ):
            raise ValueError("Regression must use the declared equal-endpoint task weighting.")
    if raw["task_type"] == "binary_classification":
        split_hash_keys = _validate_split_hash_keys(raw.get("split_hash_keys"), "split_hash_keys")
        if raw["endpoint_id"] != "bbb_martins":
            raise ValueError("BBB_Martins is the only supported Chemprop classification task.")
        if raw.get("calibration", {}).get("fit_split") != "validation":
            raise ValueError("BBB calibration must be fitted on validation only.")
        if raw.get("thresholds", {}).get("selected_method") != "maximum_validation_mcc":
            raise ValueError("BBB operating threshold must use maximum validation MCC.")
    return ChempropExperimentConfig(
        source_path=source, raw=raw, endpoint=str(raw["endpoint"]),
        endpoint_id=str(raw["endpoint_id"]), dataset=str(raw["dataset"]),
        dataset_version=str(raw["dataset_version"]), task_type=str(raw["task_type"]),
        primary_metric=str(raw["primary_metric"]),
        prepared_root=(source.parent / str(raw["prepared_root"])).resolve(),
        split_manifest=(source.parent / str(raw["split_manifest"])).resolve(),
        split_files={str(k): str(v) for k, v in raw["split_files"].items()},
        split_hash_keys=split_hash_keys, tasks=tasks,
        model=dict(raw["model"]), training=dict(raw["training"]),
    )


def _validate_regression_tasks(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = raw.get("tasks")
    if not isinstance(value, dict) or tuple(value) != REGRESSION_ENDPOINTS:
        raise ValueError(f"Regression tasks must be exactly and in order: {REGRESSION_ENDPOINTS}")
    tasks: dict[str, dict[str, Any]] = {}
    for endpoint, metadata in value.items():
        _require_keys(
            metadata,
            {"tdc_name", "target_definition", "units", "target_transform", "split_hash_keys"},
            f"tasks.{endpoint}",
        )
        if metadata["target_transform"] not in {"identity", "log10"}:
            raise ValueError(f"Unsupported target transform for {endpoint}.")
        task = dict(metadata)
        task["split_hash_keys"] = _validate_split_hash_keys(
            metadata["split_hash_keys"], f"tasks.{endpoint}.split_hash_keys"
        )
        tasks[str(endpoint)] = task
    return tasks


def _validate_split_hash_keys(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"train", "validation", "test"}:
        raise ValueError(f"{label} must define train, validation, and test separately.")
    return {str(key): str(item) for key, item in value.items()}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _require_keys(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")
