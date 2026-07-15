"""Local evaluation registry-entry generation."""

from __future__ import annotations

from typing import Any

from admet_platform.evaluation.schemas import ComparisonResult


def build_registry_entry(
    result: ComparisonResult,
    recommended: dict[str, Any],
    schema_version: str,
    created_at: str,
) -> dict[str, Any]:
    run = result.runs_by_id.get(result.recommended_run_id or "")
    return {
        "registry_schema_version": schema_version,
        "model_id": f"{result.endpoint_id}-{recommended.get('recommended_run_id') or 'no-model'}",
        "run_id": recommended.get("recommended_run_id"),
        "endpoint_id": result.endpoint_id,
        "task_type": result.task_type,
        "source_dataset": result.source_dataset,
        "model_family": run.model_family if run else None,
        "model_type": run.model_type if run else None,
        "feature_type": run.feature_type if run else None,
        "pretrained_checkpoint": run.pretrained_checkpoint if run else None,
        "model_artifact_path": run.model_artifact_path if run else None,
        "tokenizer_path": run.tokenizer_path if run else None,
        "inference_metadata_path": run.inference_metadata_path if run else None,
        "model_card_path": "model_card.md",
        "validation_metrics": run.validation_metrics if run else {},
        "test_metrics": run.test_metrics if run else {},
        "data_and_split_provenance": run.split_provenance if run else {},
        "package_versions": run.package_versions if run else {},
        "creation_timestamp": created_at,
        "development_mode": run.development_mode if run else None,
        "recommendation_status": recommended["recommendation_status"],
        "approval_status": "pending_review",
        "aws_model_registry_registered": False,
    }
