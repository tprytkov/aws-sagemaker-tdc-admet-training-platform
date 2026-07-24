"""Offline-capable orchestration for prepared multi-task regression training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer

from admet_platform.data.multitask_regression import (
    build_regression_dataloaders,
    build_regression_training_manifest,
    fit_training_transforms,
    load_multitask_regression_config,
    load_regression_training_datasets,
)
from admet_platform.models.multitask_regression_chemberta import (
    MultiTaskRegressionChemBERTa,
    MultiTaskRegressionChemBERTaConfig,
)
from admet_platform.training.multitask_regression_control import (
    evaluate_regression_validation,
    should_stop_regression_early,
    update_regression_checkpoint_selection,
)
from admet_platform.training.multitask_regression_losses import MultiTaskRegressionLoss
from admet_platform.training.multitask_regression_trainer import (
    MultiTaskRegressionTrainer,
)
from admet_platform.training.reproducibility import seed_everything


def run_multitask_regression_training(
    *,
    config_path: str | Path,
    prepared_root: str | Path | None,
    output_dir: str | Path,
    checkpoint: str | None = None,
    resume_from: str | Path | None = None,
    max_steps: int | None = None,
    limit_samples_per_task: int | None = None,
    limit_validation_samples_per_task: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
    offline: bool = False,
    deterministic_algorithms: bool = False,
    mixed_precision: str | None = None,
    loss: str | None = None,
    evaluation_interval_steps: int | None = None,
    checkpoint_interval_steps: int | None = None,
) -> dict[str, Any]:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_multitask_regression_config(config_path)
    training = replace(
        config.training,
        random_seed=config.training.random_seed if seed is None else seed,
    )
    overrides = {
        "mixed_precision": mixed_precision,
        "loss": loss,
        "evaluation_interval_steps": evaluation_interval_steps,
        "checkpoint_interval_steps": checkpoint_interval_steps,
    }
    training = replace(
        training, **{key: value for key, value in overrides.items() if value is not None}
    )
    if training.loss not in {"huber", "mse"}:
        raise ValueError("loss must be 'huber' or 'mse'.")
    if training.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16.")
    if (
        training.evaluation_interval_steps <= 0
        or training.checkpoint_interval_steps <= 0
    ):
        raise ValueError("Evaluation and checkpoint intervals must be positive.")
    steps_to_run = training.max_steps if max_steps is None else max_steps
    seed_everything(
        training.random_seed, deterministic_algorithms=deterministic_algorithms
    )
    source = checkpoint or training.model_name_or_path
    local_only = offline or Path(source).exists()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=training.model_revision,
            local_files_only=local_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load regression tokenizer '{source}' without fallback."
        ) from exc
    datasets = load_regression_training_datasets(config, prepared_root)
    transforms = fit_training_transforms(datasets)
    loaders = build_regression_dataloaders(
        datasets,
        transforms,
        tokenizer,
        seed=training.random_seed,
        train_batch_size=training.train_batch_size,
        evaluation_batch_size=training.evaluation_batch_size,
        max_length=training.max_sequence_length,
        limit_samples_per_task=limit_samples_per_task,
        limit_validation_samples_per_task=limit_validation_samples_per_task,
    )
    model = MultiTaskRegressionChemBERTa(
        MultiTaskRegressionChemBERTaConfig(
            model_name_or_path=source,
            tasks=tuple(config.tasks),
            pooling=training.pooling,  # type: ignore[arg-type]
            dropout=training.dropout,
            model_revision=training.model_revision,
            local_files_only=local_only,
        )
    )
    loss_module = MultiTaskRegressionLoss(
        tuple(config.tasks),
        loss=training.loss,
        huber_delta=training.huber_delta,
        task_loss_weights=training.task_loss_weights,
    )
    trainer = MultiTaskRegressionTrainer(
        model,
        {task: loaders[task]["train"] for task in config.tasks},
        loss_module,
        training,
        device=device,
    )
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        trainer.load_checkpoint(resume_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "target_transforms.json",
        {
            "schema_version": "1.0.0",
            "fit_split": "train",
            "test_statistics_used": False,
            "endpoints": {
                task: {
                    **transforms[task].to_metadata(),
                    "target_definition": config.tasks[task].target_definition,
                    "provenance_note": config.tasks[task].provenance_note,
                }
                for task in config.tasks
            },
        },
    )
    target_step = trainer.global_step + steps_to_run
    last_validation: dict[str, Any] = {}
    while trainer.global_step < target_step and not trainer.control_state["stopped_early"]:
        trainer.train_step()
        step = trainer.global_step
        if step % training.checkpoint_interval_steps == 0:
            trainer.save_checkpoint(output / "latest" / "checkpoint.pt")
        if step % training.evaluation_interval_steps == 0:
            last_validation = _evaluate_and_select(
                trainer, loaders, transforms, output, training
            )
    if (
        trainer.global_step
        and (
            not last_validation
            or last_validation["global_step"] != trainer.global_step
        )
    ):
        last_validation = _evaluate_and_select(
            trainer, loaders, transforms, output, training
        )
    checkpoint_path = trainer.save_checkpoint(output / "latest" / "checkpoint.pt")
    trainer.save_checkpoint(output / "checkpoint.pt")
    trainer.write_metrics_json(output / "training_metrics.json")
    tokenizer.save_pretrained(output / "tokenizer")
    model.save_model(output / "model")
    _write_json(
        output / "checkpoint_selection.json",
        {
            "primary": "lowest mean validation normalized RMSE across all endpoints",
            "tie_breaker": "highest mean validation Spearman correlation",
            "source_split": "validation",
            "shared_checkpoint_only": True,
            "state": trainer.control_state,
        },
    )
    _write_json(
        output / "task_contributions.json",
        {
            "batch_counts": dict(trainer.sampler.batch_counts),
            "example_counts": dict(trainer.sampler.example_counts),
        },
    )
    manifest = build_regression_training_manifest(datasets)
    _write_json(output / "dataset_manifest.json", manifest)
    _write_json(
        output / "resolved_config.json",
        {
            "config": str(Path(config_path).resolve()),
            "prepared_root": (
                str(Path(prepared_root).resolve())
                if prepared_root is not None
                else str(config.prepared_root)
            ),
            "output_dir": str(output),
            "offline": offline,
            "device": str(torch.device(device)),
            "deterministic_algorithms": deterministic_algorithms,
            "training": asdict(training),
            "loaded_splits": ["train", "validation"],
            "limit_samples_per_task": limit_samples_per_task,
            "limit_validation_samples_per_task": limit_validation_samples_per_task,
            "test_data_used": False,
        },
    )
    _write_json(
        output / "run_manifest.json",
        {
            "schema_version": "1.0.0",
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "package_versions": _package_versions(),
            "seed": training.random_seed,
            "endpoint_names": list(config.tasks),
            "input_hashes": manifest["input_hashes"],
            "checkpoint_sha256": _sha256(checkpoint_path),
            "initial_model_state_hash": trainer.initial_model_state_hash,
            "initial_task_head_hashes": trainer.initial_task_head_hashes,
            "loss": training.loss,
            "test_data_used": False,
        },
    )
    summary = {
        "status": (
            "stopped_early" if trainer.control_state["stopped_early"] else "completed"
        ),
        "global_step": trainer.global_step,
        "best_mean_normalized_rmse": trainer.control_state[
            "best_mean_normalized_rmse"
        ],
        "best_mean_spearman": trainer.control_state["best_mean_spearman"],
        "last_validation": last_validation,
        "test_data_used": False,
    }
    _write_json(output / "final_run_summary.json", summary)
    return {
        "output_dir": str(output),
        **summary,
        "task_contributions": {
            "batch_counts": dict(trainer.sampler.batch_counts),
            "example_counts": dict(trainer.sampler.example_counts),
        },
        "initial_model_state_hash": trainer.initial_model_state_hash,
    }


def _evaluate_and_select(
    trainer: MultiTaskRegressionTrainer,
    loaders: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, Any],
    output: Path,
    training: Any,
) -> dict[str, Any]:
    evaluation = evaluate_regression_validation(
        trainer,
        {task: loaders[task]["validation"] for task in trainer.model.task_names},
        transforms,
        output,
        trainer.global_step,
    )
    trainer.control_state["validation_history"].append(evaluation)
    event = update_regression_checkpoint_selection(trainer.control_state, evaluation)
    if event["composite_improved"]:
        trainer.save_checkpoint(output / "best_composite" / "checkpoint.pt")
        _write_json(
            output / "best_composite" / "selection.json",
            {**event, "selected": event["selections"][0]},
        )
    if should_stop_regression_early(
        trainer.control_state,
        global_step=trainer.global_step,
        patience_evaluations=training.early_stopping_patience_evaluations,
        minimum_training_steps=training.minimum_training_steps_before_stopping,
    ):
        trainer.control_state["stopped_early"] = True
        trainer.control_state["stop_reason"] = (
            "no mean validation normalized RMSE improvement within patience"
        )
    return evaluation


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


__all__ = ["run_multitask_regression_training"]
