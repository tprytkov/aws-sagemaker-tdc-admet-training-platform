"""Prepared-data contracts for separate multi-task regression experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from admet_platform.config import _load_yaml_mapping
from admet_platform.data.multitask import StatefulRandomSampler
from admet_platform.data.regression_transforms import (
    FittedRegressionTransform,
    SUPPORTED_TRANSFORMS,
    fit_regression_target_transform,
)


REGRESSION_TRAINING_SCHEMA_VERSION = "1.0.0"
REGRESSION_TRAINING_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class RegressionEndpointTrainingConfig:
    endpoint_id: str
    tdc_name: str
    units: str
    target_definition: str
    target_transform: str
    provenance_note: str | None = None


@dataclass(frozen=True)
class MultiTaskRegressionTrainingConfig:
    random_seed: int = 42
    encoder_learning_rate: float = 2.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    task_sampling: str = "round_robin"
    task_loss_weights: Mapping[str, float] | None = None
    loss: str = "huber"
    huber_delta: float = 1.0
    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    max_sequence_length: int = 128
    model_name_or_path: str = "seyonec/ChemBERTa-zinc-base-v1"
    model_revision: str | None = None
    pooling: str = "masked_mean"
    dropout: float = 0.15
    max_steps: int = 3000
    evaluation_interval_steps: int = 100
    checkpoint_interval_steps: int = 100
    warmup_steps: int = 0
    warmup_ratio: float | None = 0.1
    scheduler: str = "linear_warmup_decay"
    early_stopping_patience_evaluations: int = 5
    minimum_training_steps_before_stopping: int = 500
    mixed_precision: str = "bf16"


@dataclass(frozen=True)
class MultiTaskRegressionConfig:
    schema_version: str
    run_name: str
    split_track: str
    prepared_root: Path
    tasks: Mapping[str, RegressionEndpointTrainingConfig]
    split_files: Mapping[str, str]
    training: MultiTaskRegressionTrainingConfig
    source_path: Path


@dataclass(frozen=True)
class RegressionEndpointSplits:
    endpoint: RegressionEndpointTrainingConfig
    train: pd.DataFrame
    validation: pd.DataFrame
    paths: Mapping[str, Path]


class PreparedRegressionDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, task_name: str, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row["canonical_smiles"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(row["target_normalized"]), dtype=torch.float32),
            "target_original": torch.tensor(
                float(row["target_original"]), dtype=torch.float64
            ),
            "molecule_id": str(row["molecule_id"]),
            "canonical_smiles": str(row["canonical_smiles"]),
            "task_name": self.task_name,
        }


def load_multitask_regression_config(path: str | Path) -> MultiTaskRegressionConfig:
    source = Path(path).resolve()
    raw = _load_yaml_mapping(source.read_text(encoding="utf-8"), source=str(source))
    required = {
        "schema_version",
        "run_name",
        "split_track",
        "prepared_root",
        "tasks",
        "split_files",
        "training",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("Regression training config is missing: " + ", ".join(missing))
    if raw["schema_version"] != REGRESSION_TRAINING_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be '{REGRESSION_TRAINING_SCHEMA_VERSION}'."
        )
    if raw["split_track"] != "coordinated_multitask_regression":
        raise ValueError("split_track must be 'coordinated_multitask_regression'.")
    tasks = _parse_tasks(raw["tasks"])
    split_files = _parse_split_files(raw["split_files"])
    training = _parse_training(raw["training"], tuple(tasks))
    return MultiTaskRegressionConfig(
        schema_version=raw["schema_version"],
        run_name=_nonempty(raw["run_name"], "run_name"),
        split_track=raw["split_track"],
        prepared_root=(source.parent / _nonempty(raw["prepared_root"], "prepared_root")).resolve(),
        tasks=tasks,
        split_files=split_files,
        training=training,
        source_path=source,
    )


def load_regression_training_datasets(
    config: MultiTaskRegressionConfig,
    prepared_root: str | Path | None = None,
) -> dict[str, RegressionEndpointSplits]:
    """Load train and validation only; the locked test split is never opened."""

    root = Path(prepared_root).resolve() if prepared_root is not None else config.prepared_root
    result: dict[str, RegressionEndpointSplits] = {}
    for task, endpoint in config.tasks.items():
        paths = {
            split: root / endpoint.endpoint_id / config.split_files[split]
            for split in REGRESSION_TRAINING_SPLITS
        }
        frames = {
            split: _load_split(path, split)
            for split, path in paths.items()
        }
        result[task] = RegressionEndpointSplits(
            endpoint=endpoint,
            train=frames["train"],
            validation=frames["validation"],
            paths=paths,
        )
    return result


def fit_training_transforms(
    datasets: Mapping[str, RegressionEndpointSplits],
) -> dict[str, FittedRegressionTransform]:
    return {
        task: fit_regression_target_transform(
            splits.train["target_original"],
            endpoint_id=splits.endpoint.endpoint_id,
            units=splits.endpoint.units,
            transform=splits.endpoint.target_transform,
        )
        for task, splits in datasets.items()
    }


def build_regression_dataloaders(
    datasets: Mapping[str, RegressionEndpointSplits],
    transforms: Mapping[str, FittedRegressionTransform],
    tokenizer: Any,
    *,
    seed: int,
    train_batch_size: int,
    evaluation_batch_size: int,
    max_length: int,
    limit_samples_per_task: int | None = None,
    limit_validation_samples_per_task: int | None = None,
) -> dict[str, dict[str, DataLoader]]:
    if set(datasets) != set(transforms):
        raise ValueError("Target transforms must exactly match regression endpoints.")
    result: dict[str, dict[str, DataLoader]] = {}
    for task_index, (task, splits) in enumerate(datasets.items()):
        split_loaders: dict[str, DataLoader] = {}
        for split_index, (split, frame) in enumerate(
            (("train", splits.train), ("validation", splits.validation))
        ):
            selected = frame
            if split == "train" and limit_samples_per_task is not None:
                if limit_samples_per_task <= 0:
                    raise ValueError("limit_samples_per_task must be positive.")
                selected = frame.sample(
                    n=min(limit_samples_per_task, len(frame)),
                    random_state=seed + task_index,
                ).reset_index(drop=True)
            if split == "validation" and limit_validation_samples_per_task is not None:
                if limit_validation_samples_per_task <= 0:
                    raise ValueError(
                        "limit_validation_samples_per_task must be positive."
                    )
                selected = frame.iloc[:limit_validation_samples_per_task].reset_index(
                    drop=True
                )
            transformed = transforms[task].transform_frame(selected)
            dataset = PreparedRegressionDataset(transformed, tokenizer, task, max_length)
            loader_seed = seed + 10_000 + task_index * 100 + split_index
            generator = torch.Generator().manual_seed(loader_seed)
            if split == "train":
                sampler_seed = seed + 1_000 + task_index
                sampler_generator = torch.Generator().manual_seed(sampler_seed)
                sampler = StatefulRandomSampler(dataset, sampler_generator, sampler_seed)
                loader = DataLoader(
                    dataset,
                    batch_size=train_batch_size,
                    sampler=sampler,
                    generator=generator,
                    num_workers=0,
                    drop_last=False,
                )
                loader.reproducibility_metadata = {  # type: ignore[attr-defined]
                    "loader_seed": loader_seed,
                    "sampler_seed": sampler_seed,
                }
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=evaluation_batch_size,
                    shuffle=False,
                    generator=generator,
                    num_workers=0,
                    drop_last=False,
                )
                loader.reproducibility_metadata = {  # type: ignore[attr-defined]
                    "loader_seed": loader_seed,
                    "shuffle": False,
                }
            split_loaders[split] = loader
        result[task] = split_loaders
    return result


def build_regression_training_manifest(
    datasets: Mapping[str, RegressionEndpointSplits],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for task, splits in datasets.items():
        endpoints[task] = {}
        for split, frame in (("train", splits.train), ("validation", splits.validation)):
            path = splits.paths[split]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[f"{task}/{split}"] = digest
            endpoints[task][split] = {
                "path": str(path),
                "sha256": digest,
                "row_count": int(len(frame)),
            }
    return {
        "schema_version": "1.0.0",
        "loaded_splits": list(REGRESSION_TRAINING_SPLITS),
        "test_data_used": False,
        "endpoints": endpoints,
        "input_hashes": hashes,
    }


def _load_split(path: Path, expected_split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing regression {expected_split} split: {path}")
    frame = pd.read_csv(path)
    required = ("molecule_id", "canonical_smiles", "target_original")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Regression split {path} is missing: {', '.join(missing)}.")
    if "split" in frame:
        observed = set(frame["split"].dropna().astype(str).str.strip())
        aliases = {"validation", "valid"} if expected_split == "validation" else {expected_split}
        if not observed or not observed.issubset(aliases):
            raise ValueError(f"Regression split {path} has unexpected split labels.")
    smiles = frame["canonical_smiles"].astype("string").str.strip()
    targets = pd.to_numeric(frame["target_original"], errors="coerce")
    if smiles.isna().any() or smiles.eq("").any():
        raise ValueError(f"Regression split {path} contains empty canonical SMILES.")
    if targets.isna().any() or not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError(f"Regression split {path} contains non-finite targets.")
    result = frame.copy()
    result["canonical_smiles"] = smiles
    result["target_original"] = targets.astype(float)
    return result


def _parse_tasks(raw: Any) -> dict[str, RegressionEndpointTrainingConfig]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("tasks must be a non-empty mapping.")
    result: dict[str, RegressionEndpointTrainingConfig] = {}
    endpoint_ids: set[str] = set()
    for task, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Task '{task}' must be a mapping.")
        required = (
            "endpoint_id",
            "tdc_name",
            "task_type",
            "units",
            "target_definition",
            "target_transform",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"Task '{task}' is missing: {', '.join(missing)}.")
        if value["task_type"] != "regression":
            raise ValueError(f"Task '{task}' must use task_type: regression.")
        endpoint_id = _nonempty(value["endpoint_id"], f"tasks.{task}.endpoint_id")
        if endpoint_id in endpoint_ids:
            raise ValueError(f"Duplicate endpoint_id '{endpoint_id}'.")
        endpoint_ids.add(endpoint_id)
        target_transform = _nonempty(
            value["target_transform"], f"tasks.{task}.target_transform"
        )
        if target_transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"Task '{task}' target_transform must be one of: "
                + ", ".join(SUPPORTED_TRANSFORMS)
            )
        result[task] = RegressionEndpointTrainingConfig(
            endpoint_id=endpoint_id,
            tdc_name=_nonempty(value["tdc_name"], f"tasks.{task}.tdc_name"),
            units=_nonempty(value["units"], f"tasks.{task}.units"),
            target_definition=_nonempty(
                value["target_definition"], f"tasks.{task}.target_definition"
            ),
            target_transform=target_transform,
            provenance_note=(
                _nonempty(value["provenance_note"], f"tasks.{task}.provenance_note")
                if value.get("provenance_note") is not None
                else None
            ),
        )
    return result


def _parse_split_files(raw: Any) -> dict[str, str]:
    required = {"train", "validation", "test"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("split_files must define train, validation, and test.")
    return {split: _nonempty(raw[split], f"split_files.{split}") for split in required}


def _parse_training(
    raw: Any, tasks: tuple[str, ...]
) -> MultiTaskRegressionTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("training must be a mapping.")
    defaults = MultiTaskRegressionTrainingConfig()
    values = {
        field: raw.get(field, getattr(defaults, field))
        for field in defaults.__dataclass_fields__
    }
    weights = values["task_loss_weights"] or {}
    if not isinstance(weights, dict) or set(weights) - set(tasks):
        raise ValueError("task_loss_weights contains unknown regression tasks.")
    values["task_loss_weights"] = {
        task: float(weights.get(task, 1.0)) for task in tasks
    }
    if any(weight <= 0 for weight in values["task_loss_weights"].values()):
        raise ValueError("Every regression task loss weight must be positive.")
    if values["loss"] not in {"huber", "mse"}:
        raise ValueError("training.loss must be 'huber' or 'mse'.")
    if values["task_sampling"] != "round_robin":
        raise ValueError("Only round_robin task sampling is supported.")
    if values["scheduler"] != "linear_warmup_decay":
        raise ValueError("Only linear_warmup_decay is supported.")
    if values["pooling"] not in {"masked_mean", "cls"}:
        raise ValueError("training.pooling must be masked_mean or cls.")
    if values["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be no, fp16, or bf16.")
    for field in (
        "encoder_learning_rate",
        "head_learning_rate",
        "gradient_clip_norm",
        "huber_delta",
    ):
        values[field] = float(values[field])
        if values[field] <= 0:
            raise ValueError(f"training.{field} must be positive.")
    values["weight_decay"] = float(values["weight_decay"])
    if values["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be non-negative.")
    values["dropout"] = float(values["dropout"])
    if not 0 <= values["dropout"] < 1:
        raise ValueError("training.dropout must be in [0, 1).")
    for field in (
        "random_seed",
        "train_batch_size",
        "evaluation_batch_size",
        "max_sequence_length",
        "max_steps",
        "evaluation_interval_steps",
        "checkpoint_interval_steps",
        "warmup_steps",
        "early_stopping_patience_evaluations",
        "minimum_training_steps_before_stopping",
    ):
        value = values[field]
        minimum = 0 if field in {
            "random_seed",
            "warmup_steps",
            "early_stopping_patience_evaluations",
            "minimum_training_steps_before_stopping",
        } else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"training.{field} is out of range.")
    if values["warmup_ratio"] is not None:
        values["warmup_ratio"] = float(values["warmup_ratio"])
        if not 0 <= values["warmup_ratio"] < 1 or values["warmup_steps"]:
            raise ValueError("Configure a valid warmup_ratio or warmup_steps, not both.")
    values["model_name_or_path"] = _nonempty(
        values["model_name_or_path"], "training.model_name_or_path"
    )
    revision = values["model_revision"]
    if revision is not None:
        values["model_revision"] = _nonempty(revision, "training.model_revision")
    return MultiTaskRegressionTrainingConfig(**values)


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


__all__ = [
    "MultiTaskRegressionConfig",
    "MultiTaskRegressionTrainingConfig",
    "PreparedRegressionDataset",
    "RegressionEndpointSplits",
    "RegressionEndpointTrainingConfig",
    "build_regression_dataloaders",
    "build_regression_training_manifest",
    "fit_training_transforms",
    "load_multitask_regression_config",
    "load_regression_training_datasets",
]
