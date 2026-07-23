"""Evaluation-only orchestration for selected ChemBERTa test checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer

from admet_platform.config import _load_yaml_mapping
from admet_platform.data.multitask import (
    MultiTaskTrainingConfig,
    build_task_dataloaders,
    load_endpoint_datasets,
    load_multitask_config,
)
from admet_platform.models.multitask_chemberta import (
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)
from admet_platform.training.multitask_control import evaluate_split
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss
from admet_platform.training.multitask_trainer import MultiTaskTrainer


FINAL_EVALUATION_SCHEMA_VERSION = "1.0.0"
EXPECTED_EXPERIMENTS = {
    "bbb_single_task": ("single_task", ("bbb_martins",)),
    "herg_single_task": ("single_task", ("herg_karim",)),
    "ames_single_task": ("single_task", ("ames",)),
    "multitask": ("multi_task", ("bbb_martins", "herg_karim", "ames")),
}
COMPARISON_METRICS = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "f1",
    "mcc",
    "sensitivity",
    "specificity",
    "confusion_matrix",
    "row_count",
)


@dataclass(frozen=True)
class FinalTestExperiment:
    name: str
    role: str
    endpoint: str | None
    training_config: Path
    checkpoint: Path


@dataclass(frozen=True)
class FinalTestEvaluationConfig:
    source_path: Path
    project_root: Path
    prepared_root: Path
    coordinated_manifest: Path
    experiments: Mapping[str, FinalTestExperiment]


def load_final_test_evaluation_config(path: str | Path) -> FinalTestEvaluationConfig:
    """Load the fixed selected-checkpoint test-evaluation contract."""

    source = Path(path).resolve()
    project_root = source.parent.parent
    raw = _load_yaml_mapping(source.read_text(encoding="utf-8"), source=str(source))
    if raw.get("schema_version") != FINAL_EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"Final evaluation schema_version must be '{FINAL_EVALUATION_SCHEMA_VERSION}'."
        )
    prepared_root = _project_path(project_root, raw.get("prepared_root"), "prepared_root")
    manifest = _project_path(
        project_root, raw.get("coordinated_manifest"), "coordinated_manifest"
    )
    experiments_raw = raw.get("experiments")
    if not isinstance(experiments_raw, dict) or set(experiments_raw) != set(EXPECTED_EXPERIMENTS):
        raise ValueError(
            "Final evaluation experiments must be exactly: "
            + ", ".join(EXPECTED_EXPERIMENTS)
        )

    experiments: dict[str, FinalTestExperiment] = {}
    for name, (expected_role, expected_tasks) in EXPECTED_EXPERIMENTS.items():
        entry = experiments_raw[name]
        if not isinstance(entry, dict) or entry.get("role") != expected_role:
            raise ValueError(f"Experiment '{name}' must use role '{expected_role}'.")
        endpoint = entry.get("endpoint")
        if expected_role == "single_task" and endpoint != expected_tasks[0]:
            raise ValueError(
                f"Experiment '{name}' must explicitly select endpoint '{expected_tasks[0]}'."
            )
        training_config = _project_path(
            project_root, entry.get("training_config"), f"experiments.{name}.training_config"
        )
        checkpoint = _project_path(
            project_root, entry.get("checkpoint"), f"experiments.{name}.checkpoint"
        )
        parsed_training = load_multitask_config(training_config)
        if tuple(parsed_training.tasks) != expected_tasks:
            raise ValueError(
                f"Experiment '{name}' training config tasks do not match {expected_tasks}."
            )
        experiments[name] = FinalTestExperiment(
            name=name,
            role=expected_role,
            endpoint=endpoint if isinstance(endpoint, str) else None,
            training_config=training_config,
            checkpoint=checkpoint,
        )
    return FinalTestEvaluationConfig(
        source_path=source,
        project_root=project_root,
        prepared_root=prepared_root,
        coordinated_manifest=manifest,
        experiments=experiments,
    )


def verify_test_dataset_hashes(
    *,
    experiment: FinalTestExperiment,
    prepared_root: Path,
    coordinated_manifest: Path,
) -> dict[str, dict[str, Any]]:
    """Verify test CSVs against both coordinated and training-run manifests."""

    if not coordinated_manifest.is_file():
        raise FileNotFoundError(f"Coordinated split manifest does not exist: {coordinated_manifest}")
    coordinated = json.loads(coordinated_manifest.read_text(encoding="utf-8"))
    if coordinated.get("split_track") != "coordinated_multitask":
        raise ValueError("The coordinated manifest does not identify coordinated_multitask data.")
    run_manifest_path = experiment.checkpoint.parents[1] / "dataset_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"Training dataset manifest does not exist for '{experiment.name}': {run_manifest_path}"
        )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    training_config = load_multitask_config(experiment.training_config)
    verified: dict[str, dict[str, Any]] = {}
    for task, endpoint in training_config.tasks.items():
        key = f"{task}/test"
        test_path = prepared_root / endpoint.endpoint_id / training_config.split_files["test"]
        if test_path.name != "test.csv":
            raise ValueError(f"Experiment '{experiment.name}' does not select test.csv for {task}.")
        if not test_path.is_file():
            raise FileNotFoundError(f"Coordinated test split does not exist: {test_path}")
        actual = hashlib.sha256(test_path.read_bytes()).hexdigest()
        coordinated_expected = coordinated.get("output_file_sha256", {}).get(key)
        run_expected = run_manifest.get("input_hashes", {}).get(key)
        if actual != coordinated_expected:
            raise ValueError(f"Coordinated manifest hash mismatch for {key}.")
        if actual != run_expected:
            raise ValueError(f"Selected checkpoint dataset hash mismatch for {key}.")
        verified[task] = {
            "path": str(test_path),
            "sha256": actual,
            "coordinated_manifest": str(coordinated_manifest),
            "training_manifest": str(run_manifest_path),
        }
    return verified


def run_final_test_evaluation(
    *,
    evaluation_config: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate fixed selected checkpoints once on coordinated test data only."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_final_test_evaluation_config(evaluation_config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    verified_hashes: dict[str, Any] = {}
    for name, experiment in config.experiments.items():
        if not experiment.checkpoint.is_file():
            raise FileNotFoundError(
                f"Selected checkpoint for '{name}' does not exist: {experiment.checkpoint}"
            )
        verified_hashes[name] = verify_test_dataset_hashes(
            experiment=experiment,
            prepared_root=config.prepared_root,
            coordinated_manifest=config.coordinated_manifest,
        )

    for name, experiment in config.experiments.items():
        results[name] = _evaluate_experiment(
            experiment=experiment,
            prepared_root=config.prepared_root,
            output_dir=output / name,
            device=device,
        )

    comparison = build_final_test_comparison(results)
    comparison.to_csv(output / "single_task_vs_multitask_test_metrics.csv", index=False)
    payload = {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "source_split": "test",
        "threshold": 0.5,
        "checkpoint_selection_performed": False,
        "checkpoint_selection_state_loaded": False,
        "dataset_hashes": verified_hashes,
        "experiments": results,
        "comparison_csv": "single_task_vs_multitask_test_metrics.csv",
    }
    _write_json(output / "test_metrics.json", payload)
    return payload


def _evaluate_experiment(
    *,
    experiment: FinalTestExperiment,
    prepared_root: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    checkpoint = MultiTaskTrainer.read_checkpoint(experiment.checkpoint, "cpu")
    model_config = MultiTaskChemBERTaConfig.from_dict(checkpoint["model_config"])
    training_config = MultiTaskTrainingConfig(**checkpoint["training_config"])
    run_root = experiment.checkpoint.parents[1]
    encoder_config_path = run_root / "model" / "encoder_config"
    tokenizer_path = run_root / "tokenizer"
    if not encoder_config_path.is_dir() or not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"Selected run '{experiment.name}' is missing its local model config or tokenizer."
        )
    encoder_config = AutoConfig.from_pretrained(encoder_config_path, local_files_only=True)
    encoder = AutoModel.from_config(encoder_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = MultiTaskChemBERTa(model_config, encoder=encoder)
    loss_metadata = checkpoint["loss_metadata"]
    loss_module = MultiTaskBinaryLoss(
        loss_metadata["positive_class_weights"], loss_metadata["task_loss_weights"]
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
    trainer.load_checkpoint_for_evaluation(experiment.checkpoint)
    parsed_config = load_multitask_config(experiment.training_config)
    datasets = load_endpoint_datasets(parsed_config, prepared_root)
    loaders = build_task_dataloaders(
        datasets,
        tokenizer,
        seed=training_config.random_seed,
        train_batch_size=training_config.train_batch_size,
        evaluation_batch_size=training_config.evaluation_batch_size,
        max_length=training_config.max_sequence_length,
        splits=("test",),
    )
    if any(set(task_loaders) != {"test"} for task_loaders in loaders.values()):
        raise RuntimeError("Final evaluation constructed a non-test data loader.")
    evaluation = evaluate_split(
        trainer,
        {task: task_loaders["test"] for task, task_loaders in loaders.items()},
        output_dir,
        trainer.global_step,
        split="test",
    )
    if json.dumps(trainer.control_state, sort_keys=True) != selection_state_before:
        raise RuntimeError("Test evaluation altered checkpoint-selection state.")
    return {
        "role": experiment.role,
        "checkpoint": str(experiment.checkpoint),
        "global_step": trainer.global_step,
        "source_split": "test",
        "checkpoint_selection_state_loaded": False,
        "metrics": evaluation["endpoints"],
    }


def build_final_test_comparison(results: Mapping[str, Any]) -> pd.DataFrame:
    """Build the endpoint-level single-task versus multi-task test comparison."""
    multi = results["multitask"]["metrics"]
    rows: list[dict[str, Any]] = []
    single_names = {
        "bbb_martins": "bbb_single_task",
        "herg_karim": "herg_single_task",
        "ames": "ames_single_task",
    }
    for task, single_name in single_names.items():
        single_metrics = results[single_name]["metrics"][task]
        multi_metrics = multi[task]
        row: dict[str, Any] = {"endpoint": task}
        for metric in COMPARISON_METRICS:
            single_value = single_metrics[metric]
            multi_value = multi_metrics[metric]
            if metric == "confusion_matrix":
                single_value = json.dumps(single_value, sort_keys=True)
                multi_value = json.dumps(multi_value, sort_keys=True)
            row[f"single_task_{metric}"] = single_value
            row[f"multitask_{metric}"] = multi_value
        row["delta_roc_auc_multitask_minus_single_task"] = (
            multi_metrics["roc_auc"] - single_metrics["roc_auc"]
            if multi_metrics["roc_auc"] is not None and single_metrics["roc_auc"] is not None
            else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _project_path(project_root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Final evaluation field '{field}' must be a non-empty path.")
    return (project_root / raw).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FinalTestEvaluationConfig",
    "FinalTestExperiment",
    "build_final_test_comparison",
    "load_final_test_evaluation_config",
    "run_final_test_evaluation",
    "verify_test_dataset_hashes",
]
