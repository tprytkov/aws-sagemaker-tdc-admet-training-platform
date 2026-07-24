"""Single-endpoint baseline wrapper around the validated regression runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from admet_platform.data.multitask_regression import (
    load_multitask_regression_config,
)
from admet_platform.models.multitask_regression_chemberta import (
    DEFAULT_REGRESSION_ENDPOINTS,
)
from admet_platform.training.multitask_regression_run import (
    run_multitask_regression_training,
)


BASELINE_SUMMARY_COLUMNS = (
    "endpoint",
    "run_name",
    "selected_step",
    "selected_checkpoint",
    "rmse",
    "mae",
    "r2",
    "spearman",
    "pearson",
    "normalized_rmse",
    "normalized_mae",
    "validation_loss",
    "row_count",
    "train_batch_count",
    "train_example_count",
    "initial_model_state_hash",
    "train_split_sha256",
    "validation_split_sha256",
    "scientific_transform",
    "transformed_train_mean",
    "transformed_train_std",
    "test_data_used",
)


def run_single_task_regression_baseline(
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
    evaluation_interval_steps: int | None = None,
    checkpoint_interval_steps: int | None = None,
) -> dict[str, Any]:
    """Run exactly one regression endpoint and materialize selected-step artifacts."""

    config = load_multitask_regression_config(config_path)
    if len(config.tasks) != 1:
        raise ValueError("A single-task regression baseline config must contain one endpoint.")
    task = next(iter(config.tasks))
    if task not in DEFAULT_REGRESSION_ENDPOINTS:
        raise ValueError(f"Unsupported frozen regression baseline endpoint '{task}'.")
    result = run_multitask_regression_training(
        config_path=config_path,
        prepared_root=prepared_root,
        output_dir=output_dir,
        checkpoint=checkpoint,
        resume_from=resume_from,
        max_steps=max_steps,
        limit_samples_per_task=limit_samples_per_task,
        limit_validation_samples_per_task=limit_validation_samples_per_task,
        seed=seed,
        device=device,
        offline=offline,
        deterministic_algorithms=deterministic_algorithms,
        mixed_precision=mixed_precision,
        loss="huber",
        evaluation_interval_steps=evaluation_interval_steps,
        checkpoint_interval_steps=checkpoint_interval_steps,
    )
    output = Path(output_dir).resolve()
    selection = _read_json(output / "checkpoint_selection.json")
    state = selection["state"]
    history = state["validation_history"]
    selected_step = state["best_composite_step"]
    selected = next(
        (
            evaluation
            for evaluation in history
            if evaluation["global_step"] == selected_step
        ),
        None,
    )
    if selected is None:
        raise RuntimeError("Selected validation step is absent from validation history.")
    selected_checkpoint = output / "best_composite" / "checkpoint.pt"
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {selected_checkpoint}")

    manifest = _read_json(output / "dataset_manifest.json")
    transforms = _read_json(output / "target_transforms.json")
    contributions = _read_json(output / "task_contributions.json")
    run_manifest = _read_json(output / "run_manifest.json")
    metrics = selected["endpoints"][task]
    endpoint_manifest = manifest["endpoints"][task]
    transform = transforms["endpoints"][task]
    row = {
        "endpoint": task,
        "run_name": config.run_name,
        "selected_step": int(selected_step),
        "selected_checkpoint": str(selected_checkpoint),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "spearman": metrics["spearman"],
        "pearson": metrics["pearson"],
        "normalized_rmse": metrics["normalized_rmse"],
        "normalized_mae": metrics["normalized_mae"],
        "validation_loss": metrics["validation_loss"],
        "row_count": metrics["row_count"],
        "train_batch_count": contributions["batch_counts"][task],
        "train_example_count": contributions["example_counts"][task],
        "initial_model_state_hash": run_manifest["initial_model_state_hash"],
        "train_split_sha256": endpoint_manifest["train"]["sha256"],
        "validation_split_sha256": endpoint_manifest["validation"]["sha256"],
        "scientific_transform": transform["transform"],
        "transformed_train_mean": transform["transformed_train_mean"],
        "transformed_train_std": transform["transformed_train_std"],
        "test_data_used": False,
    }
    _write_json(
        output / "validation_history.json",
        {
            "schema_version": "1.0.0",
            "endpoint": task,
            "source_split": "validation",
            "test_data_used": False,
            "history": history,
        },
    )
    (output / "validation_history.jsonl").write_text(
        "".join(
            json.dumps(item, allow_nan=False, default=str) + "\n"
            for item in history
        ),
        encoding="utf-8",
    )
    _write_json(
        output / "single_task_baseline_summary.json",
        {
            "schema_version": "1.0.0",
            "selection_primary": "lowest validation normalized RMSE",
            "selection_tie_breaker": "highest validation Spearman",
            "source_split": "validation",
            "row": row,
        },
    )
    pd.DataFrame([row], columns=BASELINE_SUMMARY_COLUMNS).to_csv(
        output / "single_task_baseline_summary.csv", index=False
    )
    return {
        **result,
        "endpoint": task,
        "selected_step": int(selected_step),
        "selected_checkpoint": str(selected_checkpoint),
        "baseline_summary": row,
    }


def build_single_task_regression_comparison(
    run_directories: Mapping[str, str | Path],
    *,
    output_csv: str | Path,
    output_json: str | Path,
) -> pd.DataFrame:
    """Combine five validation-only baseline summaries without dataset access."""

    if set(run_directories) != set(DEFAULT_REGRESSION_ENDPOINTS):
        raise ValueError(
            "Comparison inputs must exactly match the five frozen regression endpoints."
        )
    rows = []
    for task in DEFAULT_REGRESSION_ENDPOINTS:
        payload = _read_json(
            Path(run_directories[task]) / "single_task_baseline_summary.json"
        )
        if payload.get("source_split") != "validation":
            raise ValueError(f"Baseline summary for '{task}' is not validation-only.")
        row = payload.get("row")
        if not isinstance(row, dict) or row.get("endpoint") != task:
            raise ValueError(f"Baseline summary endpoint mismatch for '{task}'.")
        if row.get("test_data_used") is not False:
            raise ValueError(f"Baseline summary for '{task}' used test data.")
        rows.append(row)
    frame = pd.DataFrame(rows, columns=BASELINE_SUMMARY_COLUMNS)
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    _write_json(
        json_path,
        {
            "schema_version": "1.0.0",
            "source_split": "validation",
            "test_data_used": False,
            "selection_primary": "lowest validation normalized RMSE",
            "selection_tie_breaker": "highest validation Spearman",
            "rows": rows,
        },
    )
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BASELINE_SUMMARY_COLUMNS",
    "build_single_task_regression_comparison",
    "run_single_task_regression_baseline",
]
