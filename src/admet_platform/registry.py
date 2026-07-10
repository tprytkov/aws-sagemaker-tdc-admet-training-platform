"""Public-safe model registry metadata generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from admet_platform.config import EndpointConfig, load_endpoint_config


def build_model_registry_entry(
    config_path: str | Path,
    metrics_json_path: str | Path,
    artifact_uri: str,
    output_json_path: str | Path,
    validation_status: str = "experimental",
) -> dict[str, Any]:
    """Build a public-safe JSON registry entry for a trained ADMET model artifact."""

    config = load_endpoint_config(config_path)
    metrics_payload = json.loads(Path(metrics_json_path).read_text(encoding="utf-8"))
    _validate_metrics_match_config(config, metrics_payload)

    entry = {
        "model_id": _build_model_id(config, metrics_payload, validation_status),
        "endpoint_id": config.endpoint_id,
        "tdc_name": config.tdc_name,
        "task_group": config.task_group,
        "task_type": config.task_type,
        "model_type": metrics_payload["model_type"],
        "base_model": config.base_model,
        "artifact_uri": artifact_uri,
        "training_source": "local_toy_sample_baseline",
        "metrics": metrics_payload["metrics"],
        "validation_status": validation_status,
        "input_schema": {
            "molecule_id_column": "molecule_id",
            "smiles_column": config.smiles_column,
        },
        "output_schema": {
            "prediction_column": config.output_prediction_column,
            "score_column": config.output_score_column,
            "model_source_column": "model_source",
            "validation_status_column": "validation_status",
        },
        "moloptima_enabled": False,
        "limitations": config.limitations,
        "created_by": "Tatiana Prytkova",
        "notes": (
            "Toy-sample local baseline registry entry for integration scaffolding only. "
            "Not suitable for scientific, clinical, medical, safety, or production claims."
        ),
    }

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry


def _validate_metrics_match_config(config: EndpointConfig, metrics_payload: dict[str, Any]) -> None:
    required_fields = {"endpoint_id", "task_type", "model_type", "metrics"}
    missing_fields = sorted(required_fields - set(metrics_payload))
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Metrics JSON is missing required field(s): {missing}.")

    if metrics_payload["endpoint_id"] != config.endpoint_id:
        raise ValueError(
            "Metrics endpoint_id does not match config endpoint_id: "
            f"{metrics_payload['endpoint_id']} != {config.endpoint_id}."
        )

    if metrics_payload["task_type"] != config.task_type:
        raise ValueError(
            "Metrics task_type does not match config task_type: "
            f"{metrics_payload['task_type']} != {config.task_type}."
        )

    if not isinstance(metrics_payload["metrics"], dict) or not metrics_payload["metrics"]:
        raise ValueError("Metrics JSON field 'metrics' must be a non-empty object.")


def _build_model_id(
    config: EndpointConfig,
    metrics_payload: dict[str, Any],
    validation_status: str,
) -> str:
    model_type = str(metrics_payload["model_type"]).replace("_", "-")
    status = validation_status.replace("_", "-")
    return f"{config.endpoint_id}-{model_type}-{status}"
