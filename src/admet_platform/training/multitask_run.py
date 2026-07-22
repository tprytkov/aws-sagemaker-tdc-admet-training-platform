"""Prepared-data orchestration for a local single-device multi-task run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

from admet_platform.data.multitask import (
    build_dataset_manifest, build_task_dataloaders, load_endpoint_datasets,
    load_multitask_config,
)
from admet_platform.models.multitask_chemberta import MultiTaskChemBERTa, MultiTaskChemBERTaConfig
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss, calculate_positive_class_weights
from admet_platform.training.multitask_trainer import MultiTaskTrainer
from admet_platform.training.reproducibility import seed_everything
from admet_platform.training.multitask_control import (
    build_endpoint_comparison, evaluate_validation, load_baselines,
    should_stop_early, update_checkpoint_selection,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, allow_nan=False, default=str) + "\n" for record in records),
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _identity(path_or_name: str) -> dict[str, Any]:
    path = Path(path_or_name)
    identity: dict[str, Any] = {"source": path_or_name}
    if path.is_file():
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif path.exists():
        identity["resolved_path"] = str(path.resolve())
    return identity


def run_multitask_training(
    *, config_path: str | Path, prepared_root: str | Path | None, output_dir: str | Path,
    checkpoint: str | None = None, resume_from: str | Path | None = None,
    max_steps: int | None = None, limit_samples_per_task: int | None = None, seed: int | None = None,
    device: str = "cpu", offline: bool = False, deterministic_algorithms: bool = False,
    classical_baseline_json: str | Path | None = None,
    single_task_baseline_json: str | Path | None = None,
    mixed_precision: str | None = None,
    evaluation_interval_steps: int | None = None,
    checkpoint_interval_steps: int | None = None,
    warmup_steps: int | None = None,
    warmup_ratio: float | None = None,
    early_stopping_patience_evaluations: int | None = None,
    minimum_training_steps_before_stopping: int | None = None,
) -> dict[str, Any]:
    """Train/evaluate configured prepared endpoints and write reproducibility artifacts."""
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_multitask_config(config_path)
    training = replace(config.training, random_seed=config.training.random_seed if seed is None else seed)
    if mixed_precision is not None:
        training = replace(training, mixed_precision=mixed_precision)
    overrides = {
        "evaluation_interval_steps": evaluation_interval_steps,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "early_stopping_patience_evaluations": early_stopping_patience_evaluations,
        "minimum_training_steps_before_stopping": minimum_training_steps_before_stopping,
    }
    training = replace(training, **{key: value for key, value in overrides.items() if value is not None})
    if warmup_steps is not None and warmup_ratio is not None:
        raise ValueError("Specify only one of warmup_steps or warmup_ratio.")
    if warmup_steps is not None:
        training = replace(training, warmup_steps=warmup_steps, warmup_ratio=None)
    if warmup_ratio is not None:
        training = replace(training, warmup_steps=0, warmup_ratio=warmup_ratio)
    for field in ("evaluation_interval_steps", "checkpoint_interval_steps"):
        if getattr(training, field) <= 0:
            raise ValueError(f"{field} must be positive.")
    if training.warmup_steps < 0 or (
        training.warmup_ratio is not None and not 0 <= training.warmup_ratio < 1
    ):
        raise ValueError("Warmup configuration is out of range.")
    if training.early_stopping_patience_evaluations < 0 or training.minimum_training_steps_before_stopping < 0:
        raise ValueError("Early-stopping controls must be non-negative.")
    steps_to_run = training.max_steps if max_steps is None else max_steps
    # This is intentionally before tokenizer/model/head, subset, sampler,
    # DataLoader iterator, optimizer, and trainer construction.
    seed_everything(
        training.random_seed, deterministic_algorithms=deterministic_algorithms
    )
    source = checkpoint or training.model_name_or_path
    local_only = offline or Path(source).exists()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source, revision=training.model_revision, local_files_only=local_only
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load tokenizer files from '{source}'. In offline mode all files must already exist locally."
        ) from exc
    model = MultiTaskChemBERTa(MultiTaskChemBERTaConfig(
        model_name_or_path=source, tasks=tuple(config.tasks), pooling=training.pooling,
        dropout=training.dropout, model_revision=training.model_revision, local_files_only=local_only,
    ))
    datasets = load_endpoint_datasets(config, prepared_root)
    manifest = build_dataset_manifest(datasets)
    loaders = build_task_dataloaders(
        datasets, tokenizer, seed=training.random_seed,
        train_batch_size=training.train_batch_size,
        evaluation_batch_size=training.evaluation_batch_size,
        max_length=training.max_sequence_length,
        limit_samples_per_task=limit_samples_per_task,
    )
    train_labels = {
        task: loaders[task]["train"].dataset.frame["target"].tolist() for task in config.tasks
    }
    positive_weights = calculate_positive_class_weights(train_labels)
    trainer = MultiTaskTrainer(
        model, {task: loaders[task]["train"] for task in config.tasks},
        MultiTaskBinaryLoss(positive_weights, training.task_loss_weights), training, device=device,
    )
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        trainer.load_checkpoint(resume_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines({
        "classical": classical_baseline_json, "single_task": single_task_baseline_json,
    })
    target_step = trainer.global_step + steps_to_run
    last_validation: dict[str, Any] = (
        trainer.control_state["validation_history"][-1]
        if trainer.control_state.get("validation_history") else {}
    )
    while trainer.global_step < target_step and not trainer.control_state["stopped_early"]:
        trainer.train_step()
        step = trainer.global_step
        if step % training.checkpoint_interval_steps == 0:
            trainer.save_checkpoint(output / "latest" / "checkpoint.pt")
        if step % training.evaluation_interval_steps != 0:
            continue
        last_validation = evaluate_validation(
            trainer, {task: loaders[task]["validation"] for task in config.tasks}, output, step
        )
        trainer.control_state["validation_history"].append(last_validation)
        event = update_checkpoint_selection(
            trainer.control_state, last_validation,
            training.endpoint_minimum_roc_auc or {}, tuple(config.tasks),
        )
        for selection in event["selections"]:
            directory = (
                output / "best_composite" if selection["kind"] == "composite"
                else output / f"best_{selection['task']}"
            )
            trainer.save_checkpoint(directory / "checkpoint.pt")
            _write_json(directory / "selection.json", {**event, "selected": selection})
        patience = training.early_stopping_patience_evaluations
        if should_stop_early(
            trainer.control_state, global_step=step, patience_evaluations=patience,
            minimum_training_steps=training.minimum_training_steps_before_stopping,
        ):
            trainer.control_state["stopped_early"] = True
            trainer.control_state["stop_reason"] = (
                f"no composite validation improvement for {patience} evaluations"
            )

    checkpoint_path = trainer.save_checkpoint(output / "latest" / "checkpoint.pt")
    trainer.save_checkpoint(output / "checkpoint.pt")  # Compatibility alias.
    trainer.write_metrics_json(output / "training_metrics.json")
    tokenizer.save_pretrained(output / "tokenizer")
    model.save_model(output / "model")
    _write_jsonl(output / "training_history.jsonl", trainer.history)
    _write_jsonl(
        output / "validation_history.jsonl", trainer.control_state["validation_history"]
    )
    single_task = len(config.tasks) == 1
    _write_json(output / "checkpoint_selection.json", {
        "selection_metric": (
            "validation ROC-AUC" if single_task
            else "mean validation ROC-AUC across all endpoints"
        ),
        "tie_breaker": None if single_task else "mean validation PR-AUC",
        "source_split": "validation",
        "state": trainer.control_state,
    })
    _write_json(output / "early_stopping.json", {
        "patience_evaluations": training.early_stopping_patience_evaluations,
        "minimum_training_steps": training.minimum_training_steps_before_stopping,
        "stopped_early": trainer.control_state["stopped_early"],
        "stop_reason": trainer.control_state["stop_reason"],
        "evaluations_without_improvement": trainer.control_state["evaluations_without_improvement"],
    })
    comparison_frame, comparison = build_endpoint_comparison(
        last_validation, baselines, training.negative_transfer_tolerance or {}
    ) if last_validation else (pd.DataFrame(), {"source_split": "validation", "endpoints": []})
    comparison_frame.to_csv(output / "endpoint_comparison.csv", index=False)
    _write_json(output / "endpoint_comparison.json", comparison)
    metrics_path = output / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation"] = last_validation.get("endpoints", {})
    _write_json(metrics_path, metrics)
    contributions = {
        "batch_counts": dict(trainer.sampler.batch_counts),
        "example_counts": dict(trainer.sampler.example_counts),
    }
    _write_json(output / "task_contributions.json", contributions)
    _write_json(output / "dataset_manifest.json", manifest)
    resolved = {
        "config": str(Path(config_path).resolve()), "prepared_root": str(Path(prepared_root).resolve()) if prepared_root else str(config.prepared_root),
        "output_dir": str(output), "max_steps_this_invocation": steps_to_run,
        "limit_samples_per_task": limit_samples_per_task,
        "device": str(torch.device(device)), "offline": offline,
        "deterministic_algorithms": deterministic_algorithms, "training": asdict(training),
    }
    _write_json(output / "resolved_config.json", resolved)
    versions = {}
    for package in ("torch", "transformers", "pandas", "scikit-learn"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    run_manifest = {
        "schema_version": "1.0.0", "git_commit": _git_commit(), "seed": training.random_seed,
        "device": str(torch.device(device)), "python": platform.python_version(), "package_versions": versions,
        "endpoint_names": list(config.tasks), "input_hashes": manifest["input_hashes"],
        "base_checkpoint": _identity(source), "output_checkpoint": _identity(str(checkpoint_path)),
        "resumed_from": _identity(str(resume_from)) if resume_from else None,
        "initial_model_state_hash": trainer.initial_model_state_hash,
        "initial_task_head_hashes": trainer.initial_task_head_hashes,
        "loader_states": trainer._json_loader_metadata(),
        "precision_mode": training.mixed_precision,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None,
        "peak_allocated_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(trainer.device)) if trainer.device.type == "cuda" else 0
        ),
    }
    _write_json(output / "run_manifest.json", run_manifest)
    final_summary = {
        "status": "stopped_early" if trainer.control_state["stopped_early"] else "completed",
        "global_step": trainer.global_step, "target_step_this_invocation": target_step,
        "best_composite": trainer.control_state["best_composite"],
        "best_endpoints": trainer.control_state["best_endpoints"],
        "last_validation": last_validation, "test_data_used": False,
        "precision_mode": training.mixed_precision,
        "peak_allocated_gpu_memory_bytes": run_manifest["peak_allocated_gpu_memory_bytes"],
    }
    _write_json(output / "final_run_summary.json", final_summary)
    return {"output_dir": str(output), "global_step": trainer.global_step,
            "task_contributions": contributions,
            "validation": last_validation.get("endpoints", {}),
            "initial_model_state_hash": trainer.initial_model_state_hash,
            "initial_task_head_hashes": trainer.initial_task_head_hashes}


__all__ = ["run_multitask_training"]
