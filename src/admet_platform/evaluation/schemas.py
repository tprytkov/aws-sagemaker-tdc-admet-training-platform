"""Typed evaluation records for model-run comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CLASSIFICATION_PRIMARY = "roc_auc"
CLASSIFICATION_FALLBACKS = ["pr_auc", "balanced_accuracy", "f1", "matthews_correlation_coefficient"]
REGRESSION_PRIMARY = "rmse"
SUPPORTED_MODEL_FAMILIES = {"classical", "chemberta"}


@dataclass(frozen=True)
class ModelRun:
    """Normalized view of one completed local model run."""

    run_id: str
    run_dir: Path
    endpoint_id: str
    task_type: str
    source_dataset: str
    model_family: str
    model_type: str
    feature_type: str | None
    pretrained_checkpoint: str | None
    development_mode: bool
    train_rows: int | None
    validation_rows: int | None
    test_rows: int | None
    feature_count: int | None
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    model_artifact_path: str | None = None
    tokenizer_path: str | None = None
    inference_metadata_path: str | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)
    feature_metadata: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    split_provenance: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        return payload


@dataclass(frozen=True)
class ComparisonOptions:
    """Options controlling recommendation behavior."""

    near_tie_tolerance: float = 0.01
    primary_metric_override: str | None = None
    include_development_runs: bool = False
    registry_schema_version: str = "1.0.0"


@dataclass(frozen=True)
class ComparisonResult:
    """Complete comparison result before artifact writing."""

    endpoint_id: str
    task_type: str
    source_dataset: str
    comparison_metric: str
    higher_is_better: bool
    rows: list[dict[str, Any]]
    evaluated_run_ids: list[str]
    eligible_run_ids: list[str]
    excluded_runs: list[dict[str, str]]
    recommended_run_id: str | None
    recommendation_status: str
    near_tie_run_ids: list[str]
    warnings: list[str]
    runs_by_id: dict[str, ModelRun]
