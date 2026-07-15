"""Compare loaded model runs and write local evaluation artifacts."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from admet_platform.evaluation.loaders import discover_run_dirs, load_model_runs
from admet_platform.evaluation.model_card import build_model_card
from admet_platform.evaluation.registry import build_registry_entry
from admet_platform.evaluation.schemas import (
    CLASSIFICATION_FALLBACKS,
    CLASSIFICATION_PRIMARY,
    REGRESSION_PRIMARY,
    ComparisonOptions,
    ComparisonResult,
    ModelRun,
)
from admet_platform.models.artifacts import to_json_safe, write_json


CLASSIFICATION_SECONDARY = ["pr_auc", "balanced_accuracy", "f1", "matthews_correlation_coefficient"]
REGRESSION_SECONDARY = ["mae", "r2", "spearman_correlation"]
LOWER_IS_BETTER = {"rmse", "mae"}


def evaluate_model_runs(
    run_dirs: list[str | Path] | None,
    output_dir: str | Path,
    *,
    discovery_parent: str | Path | None = None,
    options: ComparisonOptions | None = None,
    explicit_endpoint_id: str | None = None,
) -> ComparisonResult:
    """Load, compare, and write evaluation artifacts for model run directories."""

    selected_options = options or ComparisonOptions()
    discovered = discover_run_dirs(discovery_parent) if discovery_parent is not None else []
    all_dirs = list(run_dirs or []) + discovered
    if not all_dirs:
        raise ValueError("At least one model-run directory or discovery parent is required.")
    runs = load_model_runs(all_dirs)
    result = compare_runs(runs, selected_options, explicit_endpoint_id=explicit_endpoint_id)
    write_evaluation_artifacts(result, output_dir, selected_options)
    return result


def compare_runs(
    runs: list[ModelRun],
    options: ComparisonOptions | None = None,
    *,
    explicit_endpoint_id: str | None = None,
) -> ComparisonResult:
    selected_options = options or ComparisonOptions()
    if not runs:
        raise ValueError("No model runs were provided for comparison.")
    endpoint_ids = {run.endpoint_id for run in runs}
    task_types = {run.task_type for run in runs}
    source_datasets = {run.source_dataset for run in runs}
    split_signatures = {_split_signature(run) for run in runs}
    if explicit_endpoint_id and endpoint_ids != {explicit_endpoint_id}:
        raise ValueError(f"Endpoint mismatch: expected {explicit_endpoint_id}, got {', '.join(sorted(endpoint_ids))}.")
    if len(endpoint_ids) > 1:
        raise ValueError(f"Endpoint mismatch across runs: {', '.join(sorted(endpoint_ids))}.")
    if len(task_types) > 1:
        raise ValueError(f"Task-type mismatch across runs: {', '.join(sorted(task_types))}.")
    if len(source_datasets) > 1:
        raise ValueError(f"Source-dataset mismatch across runs: {', '.join(sorted(source_datasets))}.")
    if len(split_signatures) > 1:
        raise ValueError("Split-provenance mismatch across runs.")

    task_type = runs[0].task_type
    metric = _comparison_metric(runs, task_type, selected_options.primary_metric_override)
    higher_is_better = metric not in LOWER_IS_BETTER
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    eligible: list[ModelRun] = []
    warnings: list[str] = [
        "Models are compared using validation metrics only; test metrics are descriptive.",
        "Single-split metrics do not establish statistical or production superiority.",
    ]
    if selected_options.include_development_runs:
        warnings.append("Development/smoke runs were explicitly included in recommendation eligibility.")

    for run in runs:
        reason = _exclusion_reason(run, metric, selected_options.include_development_runs)
        row = _comparison_row(run, metric, task_type, reason)
        rows.append(row)
        if reason:
            excluded.append({"run_id": run.run_id, "reason": reason})
        else:
            eligible.append(run)

    recommended_run_id, status, near_ties = _select_recommendation(
        eligible,
        metric,
        higher_is_better,
        selected_options.near_tie_tolerance,
    )
    if status == "insufficient_validation_metrics":
        warnings.append(f"No eligible run had validation metric '{metric}'.")
    if status == "near_tie":
        warnings.append("Near-tie detected; do not claim a decisive winner.")

    return ComparisonResult(
        endpoint_id=runs[0].endpoint_id,
        task_type=task_type,
        source_dataset=runs[0].source_dataset,
        comparison_metric=metric,
        higher_is_better=higher_is_better,
        rows=rows,
        evaluated_run_ids=[run.run_id for run in runs],
        eligible_run_ids=[run.run_id for run in eligible],
        excluded_runs=excluded,
        recommended_run_id=recommended_run_id,
        recommendation_status=status,
        near_tie_run_ids=near_ties,
        warnings=warnings,
        runs_by_id={run.run_id: run for run in runs},
    )


def write_evaluation_artifacts(
    result: ComparisonResult,
    output_dir: str | Path,
    options: ComparisonOptions,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat()
    summary = build_evaluation_summary(result, options, created_at)
    comparison_json = {"rows": result.rows}
    recommended = build_recommended_model(result, options)
    registry_entry = build_registry_entry(result, recommended, options.registry_schema_version, created_at)

    write_json(output_path / "evaluation_summary.json", summary)
    write_json(output_path / "model_comparison.json", comparison_json)
    write_json(output_path / "recommended_model.json", recommended)
    write_json(output_path / "evaluation_warnings.json", {"warnings": result.warnings})
    write_json(output_path / "registry_entry.json", registry_entry)
    _write_comparison_csv(output_path / "model_comparison.csv", result.rows)
    (output_path / "model_card.md").write_text(build_model_card(result, recommended, registry_entry), encoding="utf-8")


def build_evaluation_summary(
    result: ComparisonResult,
    options: ComparisonOptions,
    created_at: str | None = None,
) -> dict[str, Any]:
    recommended = result.runs_by_id.get(result.recommended_run_id or "")
    return {
        "evaluation_run_id": _evaluation_run_id(result),
        "endpoint_id": result.endpoint_id,
        "task_type": result.task_type,
        "source_dataset": result.source_dataset,
        "evaluated_run_ids": result.evaluated_run_ids,
        "eligible_run_ids": result.eligible_run_ids,
        "excluded_run_ids": result.excluded_runs,
        "comparison_metric": result.comparison_metric,
        "near_tie_tolerance": options.near_tie_tolerance,
        "recommended_run_id": result.recommended_run_id,
        "recommendation_status": result.recommendation_status,
        "validation_summary": _metrics_summary(result, "validation"),
        "test_summary": _metrics_summary(result, "test"),
        "dataset_and_split_provenance": recommended.split_provenance if recommended else {},
        "warnings": result.warnings,
        "creation_timestamp": created_at or datetime.now(UTC).isoformat(),
        "package_versions": recommended.package_versions if recommended else {},
    }


def build_recommended_model(result: ComparisonResult, options: ComparisonOptions) -> dict[str, Any]:
    run = result.runs_by_id.get(result.recommended_run_id or "")
    if run is None:
        return {
            "recommendation_status": result.recommendation_status,
            "recommended_run_id": None,
            "selection_metric": result.comparison_metric,
            "validation_metric_value": None,
            "secondary_metric_values": {},
            "tie_or_near_tie_information": {"near_tie_run_ids": result.near_tie_run_ids},
            "model_artifact_location": None,
            "inference_metadata_location": None,
            "endpoint": {"endpoint_id": result.endpoint_id, "task_type": result.task_type, "source_dataset": result.source_dataset},
            "scientific_limitations": _scientific_limitations(),
            "test_metrics_descriptive_only": {},
        }
    secondary = CLASSIFICATION_SECONDARY if run.task_type == "binary_classification" else REGRESSION_SECONDARY
    return {
        "recommendation_status": result.recommendation_status,
        "recommended_run_id": run.run_id,
        "selection_metric": result.comparison_metric,
        "validation_metric_value": run.validation_metrics.get(result.comparison_metric),
        "secondary_metric_values": {metric: run.validation_metrics.get(metric) for metric in secondary},
        "tie_or_near_tie_information": {"near_tie_run_ids": result.near_tie_run_ids},
        "model_artifact_location": run.model_artifact_path,
        "inference_metadata_location": run.inference_metadata_path,
        "endpoint": {"endpoint_id": run.endpoint_id, "task_type": run.task_type, "source_dataset": run.source_dataset},
        "scientific_limitations": _scientific_limitations(),
        "test_metrics_descriptive_only": run.test_metrics,
    }


def _comparison_metric(runs: list[ModelRun], task_type: str, override: str | None) -> str:
    if override:
        return override
    if task_type == "binary_classification":
        if any(run.validation_metrics.get(CLASSIFICATION_PRIMARY) is not None for run in runs):
            return CLASSIFICATION_PRIMARY
        for metric in CLASSIFICATION_FALLBACKS:
            if any(run.validation_metrics.get(metric) is not None for run in runs):
                return metric
        return CLASSIFICATION_PRIMARY
    return REGRESSION_PRIMARY


def _exclusion_reason(run: ModelRun, metric: str, include_development: bool) -> str | None:
    if run.development_mode and not include_development:
        return "development_or_smoke_run_excluded"
    if run.validation_metrics.get(metric) is None:
        return f"missing_validation_metric:{metric}"
    return None


def _comparison_row(run: ModelRun, metric: str, task_type: str, exclusion_reason: str | None) -> dict[str, Any]:
    secondary = CLASSIFICATION_SECONDARY if task_type == "binary_classification" else REGRESSION_SECONDARY
    return {
        "run_id": run.run_id,
        "endpoint_id": run.endpoint_id,
        "task_type": run.task_type,
        "model_family": run.model_family,
        "model_type": run.model_type,
        "feature_type": run.feature_type,
        "pretrained_checkpoint": run.pretrained_checkpoint,
        "development_mode": run.development_mode,
        "train_rows": run.train_rows,
        "validation_rows": run.validation_rows,
        "test_rows": run.test_rows,
        "feature_count": run.feature_count,
        "primary_validation_metric": run.validation_metrics.get(metric),
        "secondary_validation_metrics": {name: run.validation_metrics.get(name) for name in secondary},
        "primary_test_metric": run.test_metrics.get(metric),
        "secondary_test_metrics": {name: run.test_metrics.get(name) for name in secondary},
        "eligibility_status": "excluded" if exclusion_reason else "eligible",
        "exclusion_reason": exclusion_reason,
        "model_artifact_path": run.model_artifact_path,
        "warnings": run.warnings,
    }


def _select_recommendation(
    eligible: list[ModelRun],
    metric: str,
    higher_is_better: bool,
    tolerance: float,
) -> tuple[str | None, str, list[str]]:
    if not eligible:
        return None, "no_eligible_model", []
    metric_runs = [run for run in eligible if run.validation_metrics.get(metric) is not None]
    if not metric_runs:
        return None, "insufficient_validation_metrics", []
    metric_runs = sorted(
        metric_runs,
        key=lambda run: float(run.validation_metrics[metric]),
        reverse=higher_is_better,
    )
    best = metric_runs[0]
    best_value = float(best.validation_metrics[metric])
    near = [
        run.run_id
        for run in metric_runs[1:]
        if abs(float(run.validation_metrics[metric]) - best_value) <= tolerance
    ]
    return best.run_id, "near_tie" if near else "recommended", near


def _metrics_summary(result: ComparisonResult, split: str) -> dict[str, Any]:
    return {
        run_id: (
            result.runs_by_id[run_id].validation_metrics if split == "validation" else result.runs_by_id[run_id].test_metrics
        )
        for run_id in result.evaluated_run_ids
    }


def _split_signature(run: ModelRun) -> tuple[Any, ...]:
    return (
        run.source_dataset,
        run.split_provenance.get("split_strategy"),
        run.train_rows,
        run.validation_rows,
        run.test_rows,
    )


def _evaluation_run_id(result: ComparisonResult) -> str:
    joined = "|".join(sorted(result.evaluated_run_ids))
    seed = f"{result.endpoint_id}|{result.task_type}|{joined}"
    return f"evaluation-{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:12]}"


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_id",
        "endpoint_id",
        "task_type",
        "model_family",
        "model_type",
        "feature_type",
        "pretrained_checkpoint",
        "development_mode",
        "train_rows",
        "validation_rows",
        "test_rows",
        "feature_count",
        "primary_validation_metric",
        "secondary_validation_metrics",
        "primary_test_metric",
        "secondary_test_metrics",
        "eligibility_status",
        "exclusion_reason",
        "model_artifact_path",
        "warnings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialized = {key: json.dumps(to_json_safe(row[key])) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in columns}
            writer.writerow(serialized)


def _scientific_limitations() -> list[str]:
    return [
        "ADMET research model only; not a clinical or regulatory decision system.",
        "Single-split metrics do not establish generalization.",
        "Test metrics were not used for model selection.",
        "Applicability-domain analysis has not yet been completed.",
        "Models are not production-ready without additional validation.",
    ]
