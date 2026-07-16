"""Configuration and prepared-data contracts for multi-task ADMET experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from admet_platform.config import EndpointConfig, _load_yaml_mapping, load_endpoint_config


MULTITASK_SCHEMA_VERSION = "1.0.0"
ALLOWED_SPLIT_TRACKS = {"official_tdc", "coordinated_multitask"}
REQUIRED_SPLITS = ("train", "validation", "test")
REQUIRED_DATA_COLUMNS = ("molecule_id", "canonical_smiles", "target")


@dataclass(frozen=True)
class MultiTaskEndpointConfig:
    """Validated configuration for one endpoint in a multi-task run."""

    endpoint_id: str
    tdc_name: str
    task_group: str
    task_type: str
    primary_metric: str
    endpoint_config_path: Path | None = None


@dataclass(frozen=True)
class MultiTaskAuditConfig:
    """Blocking rules applied before coordinated multi-task training."""

    enforce_exact_smiles_exclusion: bool = True
    enforce_scaffold_exclusion: bool = True
    fail_on_invalid_molecules: bool = True
    fail_on_conflicting_labels: bool = True
    fail_on_duplicates: bool = True


@dataclass(frozen=True)
class MultiTaskTrainingConfig:
    """Platform-independent settings required by the training core."""

    random_seed: int = 42
    encoder_learning_rate: float = 2.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    task_sampling: str = "round_robin"
    task_loss_weights: Mapping[str, float] | None = None


@dataclass(frozen=True)
class MultiTaskConfig:
    """Validated multi-task data-foundation configuration."""

    schema_version: str
    run_name: str
    split_track: str
    prepared_root: Path
    tasks: Mapping[str, MultiTaskEndpointConfig]
    split_files: Mapping[str, str]
    audit: MultiTaskAuditConfig
    training: MultiTaskTrainingConfig
    source_path: Path


@dataclass(frozen=True)
class EndpointDatasetSplits:
    """Separate prepared train, validation, and test frames for one endpoint."""

    endpoint: MultiTaskEndpointConfig
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    paths: Mapping[str, Path]

    def by_name(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def load_multitask_config(path: str | Path) -> MultiTaskConfig:
    """Load and validate a multi-task YAML file without changing endpoint configs."""

    config_path = Path(path).resolve()
    raw = _load_yaml_mapping(config_path.read_text(encoding="utf-8"), source=str(config_path))
    required = {"schema_version", "run_name", "split_track", "prepared_root", "tasks", "split_files", "audit"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Multi-task config {config_path} is missing required field(s): {', '.join(missing)}.")

    schema_version = _nonempty_string(raw["schema_version"], "schema_version", config_path)
    if schema_version != MULTITASK_SCHEMA_VERSION:
        raise ValueError(
            f"Multi-task config {config_path} schema_version must be '{MULTITASK_SCHEMA_VERSION}'."
        )
    run_name = _nonempty_string(raw["run_name"], "run_name", config_path)
    split_track = _nonempty_string(raw["split_track"], "split_track", config_path)
    if split_track not in ALLOWED_SPLIT_TRACKS:
        raise ValueError(f"Multi-task config split_track must be one of: {', '.join(sorted(ALLOWED_SPLIT_TRACKS))}.")

    prepared_root_raw = _nonempty_string(raw["prepared_root"], "prepared_root", config_path)
    prepared_root = (config_path.parent / prepared_root_raw).resolve()
    tasks = _parse_tasks(raw["tasks"], config_path)
    split_files = _parse_split_files(raw["split_files"], config_path)
    audit = _parse_audit(raw["audit"], config_path)
    training = _parse_training(raw.get("training", {}), tasks, config_path)

    return MultiTaskConfig(
        schema_version=schema_version,
        run_name=run_name,
        split_track=split_track,
        prepared_root=prepared_root,
        tasks=tasks,
        split_files=split_files,
        audit=audit,
        training=training,
        source_path=config_path,
    )


def load_endpoint_datasets(
    config: MultiTaskConfig,
    prepared_root: str | Path | None = None,
) -> dict[str, EndpointDatasetSplits]:
    """Load separate prepared split files for every configured endpoint."""

    root = Path(prepared_root).resolve() if prepared_root is not None else config.prepared_root
    datasets: dict[str, EndpointDatasetSplits] = {}
    for task_name, endpoint in config.tasks.items():
        endpoint_root = root / endpoint.endpoint_id
        paths = {split: endpoint_root / config.split_files[split] for split in REQUIRED_SPLITS}
        frames = {
            split: _load_prepared_split(path, endpoint, split)
            for split, path in paths.items()
        }
        datasets[task_name] = EndpointDatasetSplits(
            endpoint=endpoint,
            train=frames["train"],
            validation=frames["validation"],
            test=frames["test"],
            paths=paths,
        )
    return datasets


def _parse_tasks(raw: Any, config_path: Path) -> dict[str, MultiTaskEndpointConfig]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Multi-task config {config_path} field 'tasks' must be a non-empty mapping.")
    tasks: dict[str, MultiTaskEndpointConfig] = {}
    endpoint_ids: set[str] = set()
    for task_name, task_raw in raw.items():
        if not isinstance(task_name, str) or not task_name.strip() or not isinstance(task_raw, dict):
            raise ValueError("Each multi-task task must have a non-empty name and mapping value.")
        fields = ("endpoint_id", "tdc_name", "task_group", "task_type", "primary_metric")
        missing = [field for field in fields if field not in task_raw]
        if missing:
            raise ValueError(f"Task '{task_name}' is missing required field(s): {', '.join(missing)}.")
        values = {field: _nonempty_string(task_raw[field], f"tasks.{task_name}.{field}", config_path) for field in fields}
        if values["task_type"] != "binary_classification":
            raise ValueError(f"Task '{task_name}' must use task_type 'binary_classification' in this milestone.")
        if values["endpoint_id"] in endpoint_ids:
            raise ValueError(f"Duplicate endpoint_id '{values['endpoint_id']}' in multi-task config.")
        endpoint_ids.add(values["endpoint_id"])

        endpoint_config_path: Path | None = None
        if task_raw.get("endpoint_config") is not None:
            endpoint_config_value = _nonempty_string(
                task_raw["endpoint_config"], f"tasks.{task_name}.endpoint_config", config_path
            )
            endpoint_config_path = (config_path.parent / endpoint_config_value).resolve()
            endpoint_config = load_endpoint_config(endpoint_config_path)
            _validate_endpoint_reference(task_name, values, endpoint_config)

        tasks[task_name] = MultiTaskEndpointConfig(
            endpoint_id=values["endpoint_id"],
            tdc_name=values["tdc_name"],
            task_group=values["task_group"],
            task_type=values["task_type"],
            primary_metric=values["primary_metric"],
            endpoint_config_path=endpoint_config_path,
        )
    return tasks


def _validate_endpoint_reference(
    task_name: str,
    values: Mapping[str, str],
    endpoint: EndpointConfig,
) -> None:
    comparisons = {
        "endpoint_id": endpoint.endpoint_id,
        "tdc_name": endpoint.tdc_name,
        "task_group": endpoint.task_group,
        "task_type": endpoint.task_type,
    }
    for field, endpoint_value in comparisons.items():
        if values[field] != endpoint_value:
            raise ValueError(
                f"Task '{task_name}' field '{field}' does not match referenced endpoint config "
                f"('{values[field]}' != '{endpoint_value}')."
            )


def _parse_split_files(raw: Any, config_path: Path) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"Multi-task config {config_path} field 'split_files' must be a mapping.")
    missing = [split for split in REQUIRED_SPLITS if split not in raw]
    if missing:
        raise ValueError(f"Multi-task config split_files is missing: {', '.join(missing)}.")
    return {
        split: _nonempty_string(raw[split], f"split_files.{split}", config_path)
        for split in REQUIRED_SPLITS
    }


def _parse_audit(raw: Any, config_path: Path) -> MultiTaskAuditConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Multi-task config {config_path} field 'audit' must be a mapping.")
    fields = (
        "enforce_exact_smiles_exclusion",
        "enforce_scaffold_exclusion",
        "fail_on_invalid_molecules",
        "fail_on_conflicting_labels",
        "fail_on_duplicates",
    )
    values: dict[str, bool] = {}
    for field in fields:
        value = raw.get(field, True)
        if not isinstance(value, bool):
            raise ValueError(f"Multi-task config audit.{field} must be a boolean.")
        values[field] = value
    return MultiTaskAuditConfig(**values)


def _parse_training(
    raw: Any,
    tasks: Mapping[str, MultiTaskEndpointConfig],
    config_path: Path,
) -> MultiTaskTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Multi-task config {config_path} field 'training' must be a mapping.")
    defaults = MultiTaskTrainingConfig()
    seed = raw.get("random_seed", defaults.random_seed)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Multi-task config training.random_seed must be a non-negative integer.")
    numeric_fields = (
        "encoder_learning_rate",
        "head_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
    )
    values: dict[str, float] = {}
    for field in numeric_fields:
        value = raw.get(field, getattr(defaults, field))
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Multi-task config training.{field} must be numeric.")
        value = float(value)
        if field == "weight_decay" and value < 0:
            raise ValueError("Multi-task config training.weight_decay must be non-negative.")
        if field != "weight_decay" and value <= 0:
            raise ValueError(f"Multi-task config training.{field} must be positive.")
        values[field] = value
    sampling = raw.get("task_sampling", defaults.task_sampling)
    if sampling != "round_robin":
        raise ValueError("Multi-task config training.task_sampling currently supports only 'round_robin'.")
    raw_weights = raw.get("task_loss_weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError("Multi-task config training.task_loss_weights must be a mapping.")
    unknown = sorted(set(raw_weights) - set(tasks))
    if unknown:
        raise ValueError(f"Unknown task_loss_weights task(s): {', '.join(unknown)}.")
    weights = {task: float(raw_weights.get(task, 1.0)) for task in tasks}
    if any(not value > 0 for value in weights.values()):
        raise ValueError("Every task loss weight must be positive.")
    return MultiTaskTrainingConfig(
        random_seed=seed,
        task_sampling=sampling,
        task_loss_weights=weights,
        **values,
    )


def _load_prepared_split(path: Path, endpoint: MultiTaskEndpointConfig, expected_split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prepared {expected_split} CSV for endpoint '{endpoint.endpoint_id}': {path}"
        )
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_DATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prepared CSV {path} is missing required column(s): {', '.join(missing)}.")
    if "split" in frame.columns:
        values = set(frame["split"].dropna().astype(str).str.strip())
        aliases = {"validation", "valid"} if expected_split == "validation" else {expected_split}
        if not values or not values.issubset(aliases):
            raise ValueError(
                f"Prepared CSV {path} contains split values {sorted(values)}; expected {sorted(aliases)}."
            )
    numeric_target = pd.to_numeric(frame["target"], errors="coerce")
    non_null = numeric_target.dropna()
    if not non_null.isin([0, 1]).all():
        raise ValueError(f"Prepared CSV {path} contains non-binary targets for '{endpoint.endpoint_id}'.")
    frame = frame.copy()
    frame["target"] = numeric_target
    return frame


def _nonempty_string(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Multi-task config {source} field '{field}' must be a non-empty string.")
    return value.strip()
