"""Human-readable model-card generation."""

from __future__ import annotations

from typing import Any

from admet_platform.evaluation.schemas import ComparisonResult


REQUIRED_SECTIONS = [
    "Model overview",
    "Endpoint and intended use",
    "Dataset provenance",
    "Split strategy",
    "Model architecture or feature representation",
    "Training configuration",
    "Validation results",
    "Test results",
    "Baseline comparison",
    "Selection rationale",
    "Known limitations",
    "Class imbalance or target-distribution notes",
    "Data-quality findings",
    "Ethical and scientific-use limitations",
    "Reproducibility information",
    "Artifact inventory",
]


def build_model_card(
    result: ComparisonResult,
    recommended: dict[str, Any],
    registry_entry: dict[str, Any],
) -> str:
    run = result.runs_by_id.get(result.recommended_run_id or "")
    lines = ["# ADMET Model Card", ""]
    lines.extend(_section("Model overview", [
        f"Recommendation status: `{recommended['recommendation_status']}`.",
        f"Recommended run ID: `{recommended.get('recommended_run_id') or 'unavailable'}`.",
        "This is an ADMET research model. It is not a clinical or regulatory decision system.",
    ]))
    lines.extend(_section("Endpoint and intended use", [
        f"Endpoint ID: `{result.endpoint_id}`.",
        f"Task type: `{result.task_type}`.",
        "Intended use: local research comparison of ADMET model candidates.",
    ]))
    lines.extend(_section("Dataset provenance", [
        f"Source dataset: `{result.source_dataset or 'unavailable'}`.",
        "Private molecules, credentials, and clinical claims are out of scope.",
    ]))
    split_strategy = run.split_provenance.get("split_strategy") if run else None
    lines.extend(_section("Split strategy", [
        f"Split strategy: `{split_strategy or 'unavailable'}`.",
        "Test data were not used for model selection.",
    ]))
    lines.extend(_section("Model architecture or feature representation", [
        f"Model family: `{run.model_family if run else 'unavailable'}`.",
        f"Model type: `{run.model_type if run else 'unavailable'}`.",
        f"Feature type: `{run.feature_type if run and run.feature_type else 'unavailable'}`.",
        f"Pretrained checkpoint: `{run.pretrained_checkpoint if run and run.pretrained_checkpoint else 'unavailable'}`.",
    ]))
    lines.extend(_section("Training configuration", [
        _jsonish(run.model_config if run and run.model_config else run.training_metadata if run else None),
    ]))
    lines.extend(_section("Validation results", [
        f"Selection metric: `{result.comparison_metric}`.",
        _jsonish(run.validation_metrics if run else None),
    ]))
    lines.extend(_section("Test results", [
        "Test metrics are descriptive only and were not used for selection.",
        _jsonish(run.test_metrics if run else None),
    ]))
    comparison_lines = [
        f"- `{row['run_id']}`: validation={row['primary_validation_metric']}, test={row['primary_test_metric']}, status={row['eligibility_status']}"
        for row in result.rows
    ]
    lines.extend(_section("Baseline comparison", comparison_lines or ["unavailable"]))
    lines.extend(_section("Selection rationale", [
        f"Recommendation status: `{result.recommendation_status}`.",
        f"Near-tie run IDs: `{', '.join(result.near_tie_run_ids) if result.near_tie_run_ids else 'none'}`.",
        "Do not claim statistical superiority from this single split.",
    ]))
    lines.extend(_section("Known limitations", recommended["scientific_limitations"]))
    lines.extend(_section("Class imbalance or target-distribution notes", [
        "Use available dataset metadata when present; otherwise unavailable.",
        _jsonish(run.training_metadata.get("class_counts") if run else None),
    ]))
    lines.extend(_section("Data-quality findings", [
        _jsonish({"warnings": result.warnings + (run.warnings if run else [])}),
    ]))
    lines.extend(_section("Ethical and scientific-use limitations", [
        "Single-split metrics do not establish generalization.",
        "Applicability-domain analysis has not yet been completed unless separately documented.",
        "Smoke/development metrics are not scientific performance estimates.",
    ]))
    lines.extend(_section("Reproducibility information", [
        _jsonish(run.package_versions if run else None),
    ]))
    lines.extend(_section("Artifact inventory", [
        f"Model artifact: `{registry_entry.get('model_artifact_path') or 'unavailable'}`.",
        f"Tokenizer: `{registry_entry.get('tokenizer_path') or 'unavailable'}`.",
        f"Registry entry approval status: `{registry_entry.get('approval_status')}`.",
    ]))
    return "\n".join(lines).rstrip() + "\n"


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body, ""]


def _jsonish(value: Any) -> str:
    if value in (None, {}, []):
        return "unavailable"
    return f"`{value}`"
