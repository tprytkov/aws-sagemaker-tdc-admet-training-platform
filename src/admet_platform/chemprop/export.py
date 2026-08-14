"""Separate staging contracts for five-head regression and single-head BBB artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from admet_platform.chemprop.config import ChempropExperimentConfig


def build_export_manifest(
    config: ChempropExperimentConfig,
    checkpoint: str | Path,
    resolved_config: str | Path,
    split_manifest_sha256: str,
    *,
    scaler: str | Path | None = None,
    calibrator: str | Path | None = None,
    threshold: float | None = None,
    class_counts: dict[str, int] | None = None,
    endpoint_artifacts: dict[str, dict[str, str | Path]] | None = None,
) -> dict[str, Any]:
    if config.task_type == "regression" and scaler is None:
        raise ValueError("Regression export requires a separate target scaler artifact.")
    if config.task_type == "regression" and set(endpoint_artifacts or {}) != set(config.tasks):
        raise ValueError("Regression export requires artifacts for every configured endpoint.")
    if config.task_type == "binary_classification" and (calibrator is None or threshold is None):
        raise ValueError("BBB export requires separate calibrator and threshold artifacts.")
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "provider": "moloptima_internal_chemprop",
        "endpoint": config.endpoint,
        "task_type": config.task_type,
        "model_family": "chemprop_dmpnn",
        "chemprop_version": "2.3.1",
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "split_protocol": "internal_leakage_controlled",
        "split_manifest_sha256": split_manifest_sha256,
        "preprocessing_version": config.raw["preprocessing"]["version"],
        "configuration_sha256": sha256_file(resolved_config),
        "checkpoint_sha256": sha256_file(checkpoint),
        "evidence_status": config.raw["evidence_status"],
        "created_at": datetime.now(UTC).isoformat(),
    }
    if config.task_type == "regression":
        payload.update({
            "checkpoint_contents": "five_regression_outputs_no_classification_outputs",
            "scaler_sha256": sha256_file(scaler),
            "task_weighting": config.raw["task_weighting"],
            "endpoints": {
                endpoint: {
                    "tdc_name": metadata["tdc_name"],
                    "target_definition": metadata["target_definition"],
                    "unit": metadata["units"],
                    "target_transform": metadata["target_transform"],
                    "normalization": "training_only_standard",
                    "artifacts": _endpoint_artifact_hashes(endpoint_artifacts[endpoint]),
                }
                for endpoint, metadata in config.tasks.items()
            },
        })
    else:
        payload.update({
            "positive_class_meaning": config.raw["positive_class_meaning"],
            "probability_calibration_method": "platt_scaling",
            "calibrator_sha256": sha256_file(calibrator),
            "validation_selected_threshold": float(threshold),
            "class_counts": class_counts or {},
        })
    return payload


def _endpoint_artifact_hashes(files: dict[str, str | Path]) -> dict[str, dict[str, str]]:
    required = {"predictions", "metadata", "applicability", "uncertainty"}
    if set(files) != required:
        raise ValueError(f"Endpoint artifacts must be exactly: {sorted(required)}")
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in files.items()
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
