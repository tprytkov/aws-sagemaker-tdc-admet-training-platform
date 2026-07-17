"""Configuration and prepared-data contracts for multi-task ADMET experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from admet_platform.config import EndpointConfig, _load_yaml_mapping, load_endpoint_config


MULTITASK_SCHEMA_VERSION = "1.0.0"
ALLOWED_SPLIT_TRACKS = {"official_tdc", "coordinated_multitask"}
REQUIRED_SPLITS = ("train", "validation", "test")
REQUIRED_DATA_COLUMNS = ("molecule_id", "target")


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
    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    max_sequence_length: int = 128
    model_name_or_path: str = "seyonec/ChemBERTa-zinc-base-v1"
    model_revision: str | None = None
    pooling: str = "masked_mean"
    dropout: float = 0.15
    allow_smiles_fallback: bool = False
    max_steps: int = 100
    evaluation_interval_steps: int = 1
    checkpoint_interval_steps: int = 1
    warmup_steps: int = 0
    warmup_ratio: float | None = None
    scheduler: str = "linear_warmup_decay"
    early_stopping_patience_evaluations: int = 0
    minimum_training_steps_before_stopping: int = 0
    mixed_precision: str = "no"
    endpoint_minimum_roc_auc: Mapping[str, float] | None = None
    negative_transfer_tolerance: Mapping[str, float] | None = None


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


class PreparedSmilesDataset(Dataset[dict[str, Any]]):
    """Deterministic lazy-tokenized view of one prepared endpoint split."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, task_name: str, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row["model_smiles"]), max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(row["target"]), dtype=torch.float32),
            "molecule_id": str(row["molecule_id"]),
            "canonical_smiles": str(row["model_smiles"]),
            "task_name": self.task_name,
        }


class StatefulRandomSampler(Sampler[int]):
    """Random sampler whose permutation, cursor, and generator are checkpointable."""

    def __init__(self, data_source: Dataset[Any], generator: torch.Generator, seed: int) -> None:
        self.data_source = data_source
        self.generator = generator
        self.seed = seed
        self.permutation: list[int] = []
        self.cursor = 0

    def __iter__(self):
        if not self.permutation or self.cursor >= len(self.permutation):
            self.permutation = torch.randperm(
                len(self.data_source), generator=self.generator
            ).tolist()
            self.cursor = 0
        while self.cursor < len(self.permutation):
            index = self.permutation[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "generator_state": self.generator.get_state(),
            "permutation": list(self.permutation), "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("Training sampler seed is incompatible with the checkpoint.")
        self.generator.set_state(state["generator_state"])
        self.permutation = [int(value) for value in state["permutation"]]
        self.cursor = int(state["cursor"])
        if self.cursor < 0 or self.cursor > len(self.permutation):
            raise ValueError("Training sampler checkpoint contains an invalid cursor.")


def class_preserving_subset(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Select a deterministic bounded subset while retaining both binary classes."""
    if limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True)
    classes = sorted(frame["target"].astype(int).unique())
    if classes == [0, 1] and limit < 2:
        raise ValueError("limit_samples_per_task must be at least 2 to preserve both classes.")
    selected: list[int] = []
    if classes == [0, 1]:
        for label in classes:
            selected.append(int(frame[frame["target"].astype(int) == label].sample(n=1, random_state=seed).index[0]))
    remaining = frame.drop(index=selected)
    count = limit - len(selected)
    if count > 0:
        selected.extend(remaining.sample(n=min(count, len(remaining)), random_state=seed).index.tolist())
    return frame.loc[selected].reset_index(drop=True)


def build_task_dataloaders(
    datasets: Mapping[str, EndpointDatasetSplits], tokenizer: Any, *, seed: int,
    train_batch_size: int, evaluation_batch_size: int, max_length: int,
    limit_samples_per_task: int | None = None,
) -> dict[str, dict[str, DataLoader]]:
    """Build one deterministic, non-mixing DataLoader per endpoint and split."""
    result: dict[str, dict[str, DataLoader]] = {}
    for task_index, (task, splits) in enumerate(datasets.items()):
        split_loaders: dict[str, DataLoader] = {}
        for split_name, frame in splits.by_name().items():
            selected = frame
            if limit_samples_per_task is not None and split_name == "train":
                selected = class_preserving_subset(frame, limit_samples_per_task, seed + task_index)
            dataset = PreparedSmilesDataset(selected, tokenizer, task, max_length)
            split_offset = {"train": 0, "validation": 1, "test": 2}[split_name]
            loader_seed = seed + 10_000 + task_index * 100 + split_offset
            loader_generator = torch.Generator().manual_seed(loader_seed)
            if split_name == "train":
                sampler_seed = seed + 1_000 + task_index
                sampler_generator = torch.Generator().manual_seed(sampler_seed)
                sampler = StatefulRandomSampler(dataset, sampler_generator, sampler_seed)
                loader = DataLoader(
                    dataset, batch_size=train_batch_size, sampler=sampler,
                    generator=loader_generator, num_workers=0, drop_last=False,
                )
                loader.reproducibility_metadata = {  # type: ignore[attr-defined]
                    "loader_seed": loader_seed, "sampler_seed": sampler_seed,
                }
            else:
                loader = DataLoader(
                    dataset, batch_size=evaluation_batch_size, shuffle=False,
                    generator=loader_generator, num_workers=0, drop_last=False,
                )
                loader.reproducibility_metadata = {  # type: ignore[attr-defined]
                    "loader_seed": loader_seed, "shuffle": False,
                }
            split_loaders[split_name] = loader
        result[task] = split_loaders
    return result


def build_dataset_manifest(datasets: Mapping[str, EndpointDatasetSplits]) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for task, splits in datasets.items():
        endpoints[task] = {}
        for split, frame in splits.by_name().items():
            path = splits.paths[split]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            key = f"{task}/{split}"
            hashes[key] = digest
            counts = frame["target"].astype(int).value_counts().sort_index()
            endpoints[task][split] = {
                "path": str(path), "sha256": digest, "row_count": len(frame),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
            }
    return {"schema_version": "1.0.0", "endpoints": endpoints, "input_hashes": hashes}


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
            split: _load_prepared_split(path, endpoint, split, config.training.allow_smiles_fallback)
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
    integer_values = {}
    for field, default in (
        ("train_batch_size", 8), ("evaluation_batch_size", 16),
        ("max_sequence_length", 128), ("max_steps", 100),
        ("evaluation_interval_steps", 1), ("checkpoint_interval_steps", 1),
    ):
        value = raw.get(field, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Multi-task config training.{field} must be a positive integer.")
        integer_values[field] = value
    model_name = raw.get("model_name_or_path", "seyonec/ChemBERTa-zinc-base-v1")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Multi-task config training.model_name_or_path must be non-empty.")
    dropout = float(raw.get("dropout", 0.15))
    if not 0 <= dropout < 1:
        raise ValueError("Multi-task config training.dropout must be in [0, 1).")
    pooling = raw.get("pooling", "masked_mean")
    if pooling not in {"masked_mean", "cls"}:
        raise ValueError("Multi-task config training.pooling must be 'masked_mean' or 'cls'.")
    allow_fallback = raw.get("allow_smiles_fallback", False)
    if not isinstance(allow_fallback, bool):
        raise ValueError("Multi-task config training.allow_smiles_fallback must be a boolean.")
    model_revision = raw.get("model_revision")
    if model_revision is not None and (not isinstance(model_revision, str) or not model_revision.strip()):
        raise ValueError("Multi-task config training.model_revision must be null or a non-empty string.")
    warmup_steps = raw.get("warmup_steps", 0)
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError("Multi-task config training.warmup_steps must be non-negative.")
    warmup_ratio = raw.get("warmup_ratio")
    if warmup_ratio is not None:
        warmup_ratio = float(warmup_ratio)
        if not 0 <= warmup_ratio < 1:
            raise ValueError("Multi-task config training.warmup_ratio must be in [0, 1).")
        if warmup_steps:
            raise ValueError("Configure only one of warmup_steps or warmup_ratio.")
    patience = raw.get("early_stopping_patience_evaluations", 0)
    minimum_steps = raw.get("minimum_training_steps_before_stopping", 0)
    for field, value in (("early_stopping_patience_evaluations", patience),
                         ("minimum_training_steps_before_stopping", minimum_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Multi-task config training.{field} must be non-negative.")
    scheduler = raw.get("scheduler", "linear_warmup_decay")
    if scheduler != "linear_warmup_decay":
        raise ValueError("Only training.scheduler='linear_warmup_decay' is supported.")
    precision = raw.get("mixed_precision", "no")
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be no, fp16, or bf16.")
    floors = _parse_task_float_mapping(raw.get("endpoint_minimum_roc_auc", {}), tasks,
                                       "endpoint_minimum_roc_auc", minimum=0.0, maximum=1.0)
    tolerances = _parse_task_float_mapping(raw.get("negative_transfer_tolerance", {}), tasks,
                                            "negative_transfer_tolerance", minimum=0.0)
    return MultiTaskTrainingConfig(
        random_seed=seed,
        task_sampling=sampling,
        task_loss_weights=weights,
        **values, **integer_values, model_name_or_path=model_name,
        model_revision=model_revision, pooling=pooling,
        dropout=dropout, allow_smiles_fallback=allow_fallback,
        warmup_steps=warmup_steps, warmup_ratio=warmup_ratio, scheduler=scheduler,
        early_stopping_patience_evaluations=patience,
        minimum_training_steps_before_stopping=minimum_steps,
        mixed_precision=precision, endpoint_minimum_roc_auc=floors,
        negative_transfer_tolerance=tolerances,
    )


def _parse_task_float_mapping(
    raw: Any, tasks: Mapping[str, MultiTaskEndpointConfig], field: str,
    *, minimum: float, maximum: float | None = None,
) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"Multi-task config training.{field} must be a mapping.")
    unknown = sorted(set(raw) - set(tasks))
    if unknown:
        raise ValueError(f"Unknown {field} task(s): {', '.join(unknown)}.")
    values = {task: float(value) for task, value in raw.items()}
    if any(value < minimum or (maximum is not None and value > maximum) for value in values.values()):
        raise ValueError(f"Multi-task config training.{field} contains an out-of-range value.")
    return values


def _load_prepared_split(
    path: Path, endpoint: MultiTaskEndpointConfig, expected_split: str, allow_smiles_fallback: bool = False
) -> pd.DataFrame:
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
    smiles_column = "canonical_smiles" if "canonical_smiles" in frame.columns else None
    if smiles_column is None and allow_smiles_fallback and "smiles" in frame.columns:
        smiles_column = "smiles"
    if smiles_column is None:
        raise ValueError(
            f"Prepared CSV {path} is missing canonical_smiles; set training.allow_smiles_fallback=true "
            "to explicitly permit the smiles column."
        )
    values = frame[smiles_column].astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise ValueError(f"Prepared CSV {path} contains invalid or empty SMILES in '{smiles_column}'.")
    numeric_target = pd.to_numeric(frame["target"], errors="coerce")
    if numeric_target.isna().any():
        raise ValueError(f"Prepared CSV {path} contains missing or non-numeric targets for '{endpoint.endpoint_id}'.")
    non_null = numeric_target.dropna()
    if not non_null.isin([0, 1]).all():
        raise ValueError(f"Prepared CSV {path} contains non-binary targets for '{endpoint.endpoint_id}'.")
    frame = frame.copy()
    frame["target"] = numeric_target
    frame["model_smiles"] = values
    return frame


def _nonempty_string(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Multi-task config {source} field '{field}' must be a non-empty string.")
    return value.strip()
