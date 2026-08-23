from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from admet_platform.gmc_mpnn.calibration import GMC_CALIBRATION_VERSION
from admet_platform.gmc_mpnn.geometry import GEOMETRY_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.ggl import GGL_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.model import MODEL_INTERFACE_VERSION, GMCMPNNArchitecture
from admet_platform.gmc_mpnn.model_data import (
    MODEL_DATA_CONTRACT_VERSION,
    VALIDATION_PREPROCESSING_VERSION,
)
from admet_platform.gmc_mpnn import production_manifest as manifest
from admet_platform.gmc_mpnn.scaling import (
    GGL_SCALER_VERSION,
    TRAINING_PREPROCESSING_VERSION,
)
from admet_platform.gmc_mpnn.standardization import GMC_STANDARDIZATION_VERSION


def test_load_valid_production_manifest_verifies_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    manifest_path = tmp_path / "production_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = manifest.load_production_manifest(manifest_path, artifact_root=tmp_path)

    assert loaded == payload
    assert loaded["production_decision"]["threshold"] == 0.5
    assert loaded["production_decision"]["calibration"]["adopted"] is False
    assert loaded["test_artifact_accessed"] is False


def test_inference_manifest_loader_does_not_read_development_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    manifest_path = tmp_path / "production_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    development_records = [
        *payload["preprocessing"]["training"]["artifacts"].values(),
        *payload["preprocessing"]["validation"]["artifacts"].values(),
        *payload["source_provenance"].values(),
        *({"path": checkpoint["training_summary_path"]} for checkpoint in payload["checkpoints"]),
    ]
    for record in development_records:
        (tmp_path / record["path"]).unlink()

    loaded = manifest.load_inference_manifest(manifest_path, artifact_root=tmp_path)

    assert loaded["manifest_version"] == manifest.MANIFEST_VERSION


def test_inference_manifest_loader_still_rejects_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    checkpoint_path = tmp_path / payload["checkpoints"][0]["path"]
    checkpoint_path.write_bytes(b"wrong production checkpoint")
    manifest_path = tmp_path / "production_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(manifest.GMCProductionManifestError, match="SHA-256 mismatch"):
        manifest.load_inference_manifest(manifest_path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "schema mismatch"),
        ("extra", "schema mismatch"),
        ("threshold", "threshold"),
        ("calibration", "adopted"),
        ("oof_threshold", "value"),
        ("seed", "seed"),
        ("test_access", "test_artifact_accessed"),
        ("release", "release_status"),
    ),
)
def test_frozen_schema_and_scientific_decisions_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    if mutation == "missing":
        del payload["environment"]
    elif mutation == "extra":
        payload["notes"] = "not permitted"
    elif mutation == "threshold":
        payload["production_decision"]["threshold"] = 0.42
    elif mutation == "calibration":
        payload["production_decision"]["calibration"]["adopted"] = True
    elif mutation == "oof_threshold":
        payload["production_decision"]["experimental_oof_threshold"]["value"] = 0.5
    elif mutation == "seed":
        payload["checkpoints"][0]["seed"] = 14
    elif mutation == "test_access":
        payload["test_artifact_accessed"] = 0
    else:
        payload["release_status"] = "draft"

    with pytest.raises(manifest.GMCProductionManifestError, match=message):
        manifest.validate_production_manifest(payload, artifact_root=tmp_path)


def test_artifact_tampering_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    checkpoint = tmp_path / payload["checkpoints"][2]["path"]
    checkpoint.write_bytes(b"tampered checkpoint")

    with pytest.raises(manifest.GMCProductionManifestError, match="SHA-256 mismatch"):
        manifest.validate_production_manifest(payload, artifact_root=tmp_path)


def test_rehashed_inconsistent_source_provenance_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    record = payload["source_provenance"]["calibrator"]
    calibrator_path = tmp_path / record["path"]
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    calibrator["selected_threshold"] = 0.5
    calibrator_path.write_text(json.dumps(calibrator), encoding="utf-8")
    record["sha256"] = _sha256(calibrator_path)

    with pytest.raises(manifest.GMCProductionManifestError, match="selected_threshold"):
        manifest.validate_production_manifest(payload, artifact_root=tmp_path)


@pytest.mark.parametrize("bad_path", ("../outside.ckpt", "/absolute.ckpt", "locked_test/x"))
def test_unsafe_or_locked_artifact_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_path: str
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    payload["checkpoints"][0]["path"] = bad_path

    with pytest.raises(manifest.GMCProductionManifestError, match="path"):
        manifest.validate_production_manifest(payload, artifact_root=tmp_path)


def test_missing_package_version_and_wrong_scaler_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_manifest(tmp_path, monkeypatch)
    missing_package = copy.deepcopy(payload)
    del missing_package["environment"]["packages"]["rdkit"]
    with pytest.raises(manifest.GMCProductionManifestError, match="schema mismatch"):
        manifest.validate_production_manifest(missing_package, artifact_root=tmp_path)

    wrong_scaler = copy.deepcopy(payload)
    wrong_scaler["scaler"]["portable_sha256"] = "0" * 64
    with pytest.raises(manifest.GMCProductionManifestError, match="portable_scaler_sha256"):
        manifest.validate_production_manifest(wrong_scaler, artifact_root=tmp_path)


def test_writer_refuses_to_overwrite_frozen_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        manifest.write_production_manifest(output, {})


def _valid_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    scaler_portable_hash = "e" * 64
    scaler_json_path = artifacts / "scaler.json"
    scaler_json_path.write_text(
        json.dumps(
            {
                "scaler_version": GGL_SCALER_VERSION,
                "portable_scaler_sha256": scaler_portable_hash,
                "test_artifact_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_paths = {
        "train_summary": _write(artifacts / "train_summary.json", b"train summary"),
        "train_manifest": _write(artifacts / "train_manifest.csv", b"train manifest"),
        "train_status": _write(artifacts / "train_status.csv", b"train status"),
        "validation_summary": _write(artifacts / "validation_summary.json", b"validation summary"),
        "validation_manifest": _write(
            artifacts / "validation_manifest.csv", b"validation manifest"
        ),
        "validation_status": _write(artifacts / "validation_status.csv", b"validation status"),
        "scaler_json": scaler_json_path,
        "scaler_npz": _write(artifacts / "scaler.npz", b"scaler npz"),
        "scaler_summary": _write(artifacts / "fit_summary.json", b"fit summary"),
        "environment": _write(artifacts / "environment.yml", b"environment"),
        "validation_metrics": _write(artifacts / "validation_metrics.json", b"validation metrics"),
        "ensemble": _write(artifacts / "ensemble_summary.json", b"ensemble"),
        "calibration_summary": _write(
            artifacts / "calibration_summary.json", b"calibration summary"
        ),
        "calibrator": _write(artifacts / "calibrator.json", b"calibrator"),
        "calibration_metrics": _write(
            artifacts / "calibration_validation_metrics.json", b"calibration metrics"
        ),
    }
    checkpoints = []
    for seed in manifest.PRODUCTION_SEEDS:
        checkpoint_path = _write(artifacts / f"seed{seed}.ckpt", f"seed {seed}".encode())
        summary_path = _write(
            artifacts / f"seed{seed}_run_summary.json", f"summary {seed}".encode()
        )
        digest = _sha256(checkpoint_path)
        monkeypatch.setitem(manifest.EXPECTED_CHECKPOINT_SHA256, seed, digest)
        checkpoints.append(
            {
                "seed": seed,
                "path": checkpoint_path.relative_to(tmp_path).as_posix(),
                "sha256": digest,
                "training_summary_path": summary_path.relative_to(tmp_path).as_posix(),
                "training_summary_sha256": _sha256(summary_path),
            }
        )

    train_manifest_hash = _sha256(artifact_paths["train_manifest"])
    train_status_hash = _sha256(artifact_paths["train_status"])
    validation_manifest_hash = _sha256(artifact_paths["validation_manifest"])
    validation_status_hash = _sha256(artifact_paths["validation_status"])
    artifact_paths["train_summary"].write_text(
        json.dumps(
            {
                "loaded_split": "train",
                "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
                "standardization_version": GMC_STANDARDIZATION_VERSION,
                "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
                "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
                "source_row_count": 1561,
                "successful_molecule_count": 1558,
                "validation_artifact_accessed": False,
                "test_artifact_accessed": False,
                "feature_manifest_sha256": train_manifest_hash,
                "molecule_status_sha256": train_status_hash,
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["validation_summary"].write_text(
        json.dumps(
            {
                "loaded_split": "validation",
                "validation_preprocessing_version": VALIDATION_PREPROCESSING_VERSION,
                "standardization_version": GMC_STANDARDIZATION_VERSION,
                "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
                "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
                "source_row_count": 196,
                "successful_molecule_count": 196,
                "validation_artifact_accessed": True,
                "test_artifact_accessed": False,
                "feature_manifest_sha256": validation_manifest_hash,
                "molecule_status_sha256": validation_status_hash,
            }
        ),
        encoding="utf-8",
    )
    packages = {
        "python": "3.11.15",
        "chemprop": "2.1.0",
        "lightning": "2.1.4",
        "torch": "2.1.2+cu121",
        "numpy": "1.26.4",
        "scikit_learn": "1.9.0",
        "rdkit": "2026.3.5",
    }
    for checkpoint in checkpoints:
        summary_path = tmp_path / checkpoint["training_summary_path"]
        summary_path.write_text(
            json.dumps(
                {
                    "seed": checkpoint["seed"],
                    "architecture": asdict(GMCMPNNArchitecture()),
                    "frozen_provenance": {
                        "model_interface_version": MODEL_INTERFACE_VERSION,
                        "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
                        "portable_scaler_sha256": scaler_portable_hash,
                        "validation_feature_manifest_sha256": validation_manifest_hash,
                    },
                    "test_artifact_accessed": False,
                }
            ),
            encoding="utf-8",
        )
        checkpoint["training_summary_sha256"] = _sha256(summary_path)
    artifact_paths["validation_metrics"].write_text(
        json.dumps({"test_artifact_accessed": False}), encoding="utf-8"
    )
    artifact_paths["ensemble"].write_text(
        json.dumps(
            {
                "validation_count": 196,
                "seeds": list(manifest.PRODUCTION_SEEDS),
                "ensemble_method": "unweighted_arithmetic_mean",
                "provenance": {
                    "checkpoints": {
                        str(record["seed"]): {"sha256": record["sha256"]} for record in checkpoints
                    },
                    "frozen_validation": {"feature_manifest_sha256": validation_manifest_hash},
                    "package_versions": packages,
                },
                "test_artifact_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["calibration_summary"].write_text(
        json.dumps(
            {
                "selected_threshold": manifest.EXPERIMENTAL_OOF_THRESHOLD,
                "production_seeds": list(manifest.PRODUCTION_SEEDS),
                "frozen_scaler": {"portable_scaler_sha256": scaler_portable_hash},
                "test_artifact_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["calibrator"].write_text(
        json.dumps(
            {
                "selected_threshold": manifest.EXPERIMENTAL_OOF_THRESHOLD,
                "fit_split": "train_oof",
                "fit_count": 1558,
                "validation_used_for_fitting": False,
                "test_artifact_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["calibration_metrics"].write_text(
        json.dumps(
            {
                "metrics": {
                    "uncalibrated_threshold_0_5": dict(manifest.EXTERNAL_VALIDATION_METRICS)
                },
                "test_artifact_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    def record(key: str) -> dict[str, str]:
        path = artifact_paths[key]
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": _sha256(path)}

    return {
        "manifest_schema_version": manifest.MANIFEST_SCHEMA_VERSION,
        "manifest_version": manifest.MANIFEST_VERSION,
        "endpoint": "BBB",
        "model_family": "GMC-MPNN",
        "model_interface_version": MODEL_INTERFACE_VERSION,
        "architecture": {
            "version": manifest.ARCHITECTURE_VERSION,
            "parameters": asdict(GMCMPNNArchitecture()),
        },
        "preprocessing": {
            "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
            "standardization_version": GMC_STANDARDIZATION_VERSION,
            "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
            "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
            "training": {
                "version": TRAINING_PREPROCESSING_VERSION,
                "source_row_count": 1561,
                "successful_molecule_count": 1558,
                "artifacts": {
                    "summary": record("train_summary"),
                    "feature_manifest": record("train_manifest"),
                    "molecule_status": record("train_status"),
                },
            },
            "validation": {
                "version": VALIDATION_PREPROCESSING_VERSION,
                "source_row_count": 196,
                "successful_molecule_count": 196,
                "artifacts": {
                    "summary": record("validation_summary"),
                    "feature_manifest": record("validation_manifest"),
                    "molecule_status": record("validation_status"),
                },
            },
        },
        "checkpoints": checkpoints,
        "ensemble": {
            "seeds": list(manifest.PRODUCTION_SEEDS),
            "rule": "unweighted_arithmetic_mean",
            "production_probability": "raw_uncalibrated_five_seed_mean",
        },
        "production_decision": {
            "threshold": 0.5,
            "classification_rule": (
                "BBB_positive_if_probability_greater_than_or_equal_to_threshold"
            ),
            "calibration": {
                "method": "platt_scaling",
                "version": GMC_CALIBRATION_VERSION,
                "status": "evaluated_rejected_for_production",
                "adopted": False,
                "reason": "external_validation_did_not_improve_production_metrics",
            },
            "experimental_oof_threshold": {
                "value": manifest.EXPERIMENTAL_OOF_THRESHOLD,
                "selection": "train_oof_maximum_mcc",
                "role": "provenance_only_not_for_production",
            },
        },
        "external_validation": {
            "count": 196,
            "probability": "raw_uncalibrated_five_seed_mean",
            "threshold": 0.5,
            "metrics": dict(manifest.EXTERNAL_VALIDATION_METRICS),
        },
        "scaler": {
            "version": GGL_SCALER_VERSION,
            "portable_sha256": scaler_portable_hash,
            "artifacts": {
                "json": record("scaler_json"),
                "npz": record("scaler_npz"),
                "fit_summary": record("scaler_summary"),
            },
        },
        "environment": {
            "packages": packages,
            "environment_file": record("environment"),
        },
        "source_provenance": {
            "validation_metrics": record("validation_metrics"),
            "validation_ensemble_summary": record("ensemble"),
            "calibration_summary": record("calibration_summary"),
            "calibrator": record("calibrator"),
            "calibration_validation_metrics": record("calibration_metrics"),
        },
        "git_commit": "a" * 40,
        "test_artifact_accessed": False,
        "release_status": "production_manifest_frozen_pending_inference_qualification",
    }


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
