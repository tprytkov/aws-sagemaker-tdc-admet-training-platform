"""Fail-closed production manifest contract for the GMC-MPNN BBB ensemble."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from admet_platform.gmc_mpnn.calibration import GMC_CALIBRATION_VERSION
from admet_platform.gmc_mpnn.geometry import GEOMETRY_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.ggl import GGL_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.model import MODEL_INTERFACE_VERSION, GMCMPNNArchitecture
from admet_platform.gmc_mpnn.model_data import (
    MODEL_DATA_CONTRACT_VERSION,
    VALIDATION_PREPROCESSING_VERSION,
)
from admet_platform.gmc_mpnn.scaling import (
    GGL_SCALER_VERSION,
    TRAINING_PREPROCESSING_VERSION,
)
from admet_platform.gmc_mpnn.standardization import GMC_STANDARDIZATION_VERSION


MANIFEST_SCHEMA_VERSION: Final = "gmc-mpnn-bbb-production-manifest-schema-v1"
MANIFEST_VERSION: Final = "gmc-mpnn-bbb-production-v1"
ARCHITECTURE_VERSION: Final = "gmc-mpnn-bbb-chemprop-2.1.0-architecture-v1"
PRODUCTION_SEEDS: Final = (13, 37, 73, 101, 137)
PRODUCTION_THRESHOLD: Final = 0.5
EXPERIMENTAL_OOF_THRESHOLD: Final = 0.42079211712059594
EXPECTED_CHECKPOINT_SHA256: Final = {
    13: "643b7de6f0f9d07a6ff7a9d5e266d57f6c3f41844f417683ee31b87eaf958dfe",
    37: "84c348f53c29a089b32d115bbf8dc9e25844c533240f5e7cfc5738353e18463d",
    73: "b00b4f8b750ecdd59973171d36ec4362b61b0bd85a39c87df8900127c6ae73ff",
    101: "314d740363efa1e19e3f44cfa6035f3ddcb01afda6bd2f1ade30037ae27eaca3",
    137: "84e65029ec63bd2ee8500903a388189594c2e8c0b8831ef783bb08ef8ae8a0c4",
}
EXTERNAL_VALIDATION_METRICS: Final = {
    "auroc": 0.8707246376811594,
    "auprc": 0.9521514395745343,
    "brier_score": 0.11314795524204886,
    "binary_log_loss": 0.3593691410242609,
    "expected_calibration_error": 0.05388108144647307,
    "balanced_accuracy": 0.7142028985507246,
    "sensitivity": 0.9066666666666666,
    "specificity": 0.5217391304347826,
    "mcc": 0.4592609740410448,
}
REQUIRED_PACKAGE_NAMES: Final = (
    "python",
    "chemprop",
    "lightning",
    "torch",
    "numpy",
    "scikit_learn",
    "rdkit",
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PATH_PARTS: Final = frozenset(
    {"test", "locked", "locked_test", "locked-test", "bbb_test", "bbb-test"}
)


class GMCProductionManifestError(RuntimeError):
    """A missing, malformed, inconsistent, or incorrect production provenance record."""


@dataclass(frozen=True)
class ProductionManifestConfig:
    artifact_root: Path
    checkpoint_paths: tuple[tuple[int, Path], ...]
    training_preprocessing_dir: Path
    validation_preprocessing_dir: Path
    scaler_dir: Path
    validation_evaluation_dir: Path
    calibration_dir: Path
    environment_file: Path
    git_commit: str


def load_production_manifest(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    verify_runtime: bool = False,
) -> dict[str, Any]:
    """Load and fully validate a production manifest and all referenced artifacts."""

    path = Path(manifest_path)
    payload = _read_json(path, "production manifest")
    root = Path(artifact_root) if artifact_root is not None else path.parent
    validate_production_manifest(payload, artifact_root=root, verify_runtime=verify_runtime)
    return payload


def validate_production_manifest(
    payload: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    verify_runtime: bool = False,
) -> None:
    """Validate the exact schema, frozen decisions, and every referenced file hash."""

    root = Path(artifact_root).resolve()
    _exact_keys(
        payload,
        {
            "manifest_schema_version",
            "manifest_version",
            "endpoint",
            "model_family",
            "model_interface_version",
            "architecture",
            "preprocessing",
            "checkpoints",
            "ensemble",
            "production_decision",
            "external_validation",
            "scaler",
            "environment",
            "source_provenance",
            "git_commit",
            "test_artifact_accessed",
            "release_status",
        },
        "manifest",
    )
    _require_equal(payload, "manifest_schema_version", MANIFEST_SCHEMA_VERSION, "manifest")
    _require_equal(payload, "manifest_version", MANIFEST_VERSION, "manifest")
    _require_equal(payload, "endpoint", "BBB", "manifest")
    _require_equal(payload, "model_family", "GMC-MPNN", "manifest")
    _require_equal(payload, "model_interface_version", MODEL_INTERFACE_VERSION, "manifest")
    _require_equal(payload, "test_artifact_accessed", False, "manifest")
    _require_equal(
        payload,
        "release_status",
        "production_manifest_frozen_pending_inference_qualification",
        "manifest",
    )
    if not _HEX_40.fullmatch(str(payload["git_commit"])):
        raise GMCProductionManifestError("manifest git_commit must be a lowercase 40-hex SHA.")

    _validate_architecture(_mapping(payload["architecture"], "architecture"))
    _validate_preprocessing(_mapping(payload["preprocessing"], "preprocessing"), root)
    _validate_checkpoints(payload["checkpoints"], root)
    _validate_ensemble(_mapping(payload["ensemble"], "ensemble"))
    _validate_decision(_mapping(payload["production_decision"], "production_decision"))
    _validate_external_validation(_mapping(payload["external_validation"], "external_validation"))
    _validate_scaler(_mapping(payload["scaler"], "scaler"), root)
    _validate_environment(_mapping(payload["environment"], "environment"), root, verify_runtime)
    _validate_source_provenance(_mapping(payload["source_provenance"], "source_provenance"), root)
    _validate_manifest_source_consistency(payload, root)


def build_production_manifest(config: ProductionManifestConfig) -> dict[str, Any]:
    """Build one manifest from existing frozen artifacts without fitting or inference."""

    root = config.artifact_root.resolve()
    if tuple(seed for seed, _ in config.checkpoint_paths) != PRODUCTION_SEEDS:
        raise GMCProductionManifestError("Checkpoint seeds must be exactly the production seeds.")
    if not _HEX_40.fullmatch(config.git_commit):
        raise GMCProductionManifestError("Release Git commit must be a lowercase 40-hex SHA.")

    train = _preprocessing_record(config.training_preprocessing_dir, "train", root)
    validation = _preprocessing_record(config.validation_preprocessing_dir, "validation", root)
    scaler = _scaler_record(config.scaler_dir, root, train)
    checkpoints = _checkpoint_records(config.checkpoint_paths, root, scaler, validation)
    validation_metrics_path = config.validation_evaluation_dir / "validation_metrics.json"
    ensemble_path = config.validation_evaluation_dir / "ensemble_summary.json"
    validation_metrics = _read_json(validation_metrics_path, "validation metrics")
    ensemble_summary = _read_json(ensemble_path, "validation ensemble summary")
    calibration_summary_path = config.calibration_dir / "calibration_summary.json"
    calibrator_path = config.calibration_dir / "calibrator.json"
    calibration_validation_metrics_path = config.calibration_dir / "validation_metrics.json"
    calibration_summary = _read_json(calibration_summary_path, "calibration summary")
    calibrator = _read_json(calibrator_path, "calibrator")
    calibration_validation_metrics = _read_json(
        calibration_validation_metrics_path, "calibration validation metrics"
    )
    _validate_source_documents(
        validation_metrics,
        ensemble_summary,
        calibration_summary,
        calibrator,
        calibration_validation_metrics,
        checkpoints,
        scaler,
        validation,
    )
    packages = _mapping(ensemble_summary["provenance"], "validation provenance").get(
        "package_versions"
    )
    packages = dict(_mapping(packages, "validation package versions"))

    payload: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "endpoint": "BBB",
        "model_family": "GMC-MPNN",
        "model_interface_version": MODEL_INTERFACE_VERSION,
        "architecture": {
            "version": ARCHITECTURE_VERSION,
            "parameters": asdict(GMCMPNNArchitecture()),
        },
        "preprocessing": {
            "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
            "standardization_version": GMC_STANDARDIZATION_VERSION,
            "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
            "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
            "training": train,
            "validation": validation,
        },
        "checkpoints": checkpoints,
        "ensemble": {
            "seeds": list(PRODUCTION_SEEDS),
            "rule": "unweighted_arithmetic_mean",
            "production_probability": "raw_uncalibrated_five_seed_mean",
        },
        "production_decision": {
            "threshold": PRODUCTION_THRESHOLD,
            "classification_rule": "BBB_positive_if_probability_greater_than_or_equal_to_threshold",
            "calibration": {
                "method": "platt_scaling",
                "version": GMC_CALIBRATION_VERSION,
                "status": "evaluated_rejected_for_production",
                "adopted": False,
                "reason": "external_validation_did_not_improve_production_metrics",
            },
            "experimental_oof_threshold": {
                "value": EXPERIMENTAL_OOF_THRESHOLD,
                "selection": "train_oof_maximum_mcc",
                "role": "provenance_only_not_for_production",
            },
        },
        "external_validation": {
            "count": 196,
            "probability": "raw_uncalibrated_five_seed_mean",
            "threshold": PRODUCTION_THRESHOLD,
            "metrics": dict(EXTERNAL_VALIDATION_METRICS),
        },
        "scaler": scaler,
        "environment": {
            "packages": packages,
            "environment_file": _artifact_record(config.environment_file, root),
        },
        "source_provenance": {
            "validation_metrics": _artifact_record(validation_metrics_path, root),
            "validation_ensemble_summary": _artifact_record(ensemble_path, root),
            "calibration_summary": _artifact_record(calibration_summary_path, root),
            "calibrator": _artifact_record(calibrator_path, root),
            "calibration_validation_metrics": _artifact_record(
                calibration_validation_metrics_path, root
            ),
        },
        "git_commit": config.git_commit,
        "test_artifact_accessed": False,
        "release_status": "production_manifest_frozen_pending_inference_qualification",
    }
    validate_production_manifest(payload, artifact_root=root)
    return payload


def write_production_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a validated manifest once; an existing output is never overwritten."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Production manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_architecture(value: Mapping[str, Any]) -> None:
    _exact_keys(value, {"version", "parameters"}, "architecture")
    _require_equal(value, "version", ARCHITECTURE_VERSION, "architecture")
    if value["parameters"] != asdict(GMCMPNNArchitecture()):
        raise GMCProductionManifestError("architecture parameters are not the frozen architecture.")


def _validate_preprocessing(value: Mapping[str, Any], root: Path) -> None:
    _exact_keys(
        value,
        {
            "model_data_contract_version",
            "standardization_version",
            "geometry_preprocessing_version",
            "ggl_preprocessing_version",
            "training",
            "validation",
        },
        "preprocessing",
    )
    expected = {
        "model_data_contract_version": MODEL_DATA_CONTRACT_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
    }
    for key, expected_value in expected.items():
        _require_equal(value, key, expected_value, "preprocessing")
    _validate_split_record(_mapping(value["training"], "training preprocessing"), "train", root)
    _validate_split_record(
        _mapping(value["validation"], "validation preprocessing"), "validation", root
    )


def _validate_split_record(value: Mapping[str, Any], split: str, root: Path) -> None:
    _exact_keys(
        value,
        {"version", "source_row_count", "successful_molecule_count", "artifacts"},
        f"{split} preprocessing",
    )
    expected_version = (
        TRAINING_PREPROCESSING_VERSION if split == "train" else VALIDATION_PREPROCESSING_VERSION
    )
    expected_counts = (1561, 1558) if split == "train" else (196, 196)
    _require_equal(value, "version", expected_version, f"{split} preprocessing")
    _require_equal(value, "source_row_count", expected_counts[0], f"{split} preprocessing")
    _require_equal(value, "successful_molecule_count", expected_counts[1], f"{split} preprocessing")
    artifacts = _mapping(value["artifacts"], f"{split} preprocessing artifacts")
    _exact_keys(artifacts, {"summary", "feature_manifest", "molecule_status"}, "artifacts")
    for record in artifacts.values():
        _validate_artifact_record(_mapping(record, "artifact record"), root)
    summary_record = _mapping(artifacts["summary"], f"{split} summary artifact")
    summary = _read_json(
        _resolve_artifact_path(summary_record["path"], root, f"{split} summary"),
        f"{split} preprocessing summary",
    )
    version_key = (
        "training_preprocessing_version" if split == "train" else "validation_preprocessing_version"
    )
    expected_summary = {
        "loaded_split": split,
        version_key: expected_version,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "source_row_count": expected_counts[0],
        "successful_molecule_count": expected_counts[1],
        "validation_artifact_accessed": split == "validation",
        "test_artifact_accessed": False,
        "feature_manifest_sha256": artifacts["feature_manifest"]["sha256"],
        "molecule_status_sha256": artifacts["molecule_status"]["sha256"],
    }
    for key, expected in expected_summary.items():
        _require_equal(summary, key, expected, f"{split} preprocessing summary")


def _validate_checkpoints(value: Any, root: Path) -> None:
    if not isinstance(value, list) or len(value) != len(PRODUCTION_SEEDS):
        raise GMCProductionManifestError("checkpoints must contain exactly five records.")
    for index, (record_value, seed) in enumerate(zip(value, PRODUCTION_SEEDS, strict=True)):
        record = _mapping(record_value, f"checkpoint {index}")
        _exact_keys(
            record,
            {"seed", "path", "sha256", "training_summary_path", "training_summary_sha256"},
            f"checkpoint {seed}",
        )
        _require_equal(record, "seed", seed, f"checkpoint {seed}")
        _require_equal(record, "sha256", EXPECTED_CHECKPOINT_SHA256[seed], f"checkpoint {seed}")
        _verify_path_hash(record["path"], record["sha256"], root, f"checkpoint {seed}")
        _verify_path_hash(
            record["training_summary_path"],
            record["training_summary_sha256"],
            root,
            f"checkpoint {seed} training summary",
        )


def _validate_ensemble(value: Mapping[str, Any]) -> None:
    _exact_keys(value, {"seeds", "rule", "production_probability"}, "ensemble")
    _require_equal(value, "seeds", list(PRODUCTION_SEEDS), "ensemble")
    _require_equal(value, "rule", "unweighted_arithmetic_mean", "ensemble")
    _require_equal(value, "production_probability", "raw_uncalibrated_five_seed_mean", "ensemble")


def _validate_decision(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {"threshold", "classification_rule", "calibration", "experimental_oof_threshold"},
        "production_decision",
    )
    _require_equal(value, "threshold", PRODUCTION_THRESHOLD, "production_decision")
    _require_equal(
        value,
        "classification_rule",
        "BBB_positive_if_probability_greater_than_or_equal_to_threshold",
        "production_decision",
    )
    calibration = _mapping(value["calibration"], "calibration decision")
    expected_calibration = {
        "method": "platt_scaling",
        "version": GMC_CALIBRATION_VERSION,
        "status": "evaluated_rejected_for_production",
        "adopted": False,
        "reason": "external_validation_did_not_improve_production_metrics",
    }
    _exact_keys(calibration, set(expected_calibration), "calibration decision")
    for key, expected in expected_calibration.items():
        _require_equal(calibration, key, expected, "calibration decision")
    threshold = _mapping(value["experimental_oof_threshold"], "experimental threshold")
    expected_threshold = {
        "value": EXPERIMENTAL_OOF_THRESHOLD,
        "selection": "train_oof_maximum_mcc",
        "role": "provenance_only_not_for_production",
    }
    _exact_keys(threshold, set(expected_threshold), "experimental threshold")
    for key, expected in expected_threshold.items():
        _require_equal(threshold, key, expected, "experimental threshold")


def _validate_external_validation(value: Mapping[str, Any]) -> None:
    _exact_keys(value, {"count", "probability", "threshold", "metrics"}, "external_validation")
    _require_equal(value, "count", 196, "external_validation")
    _require_equal(value, "probability", "raw_uncalibrated_five_seed_mean", "external_validation")
    _require_equal(value, "threshold", PRODUCTION_THRESHOLD, "external_validation")
    metrics = _mapping(value["metrics"], "external validation metrics")
    _exact_keys(metrics, set(EXTERNAL_VALIDATION_METRICS), "external validation metrics")
    for key, expected in EXTERNAL_VALIDATION_METRICS.items():
        _require_equal(metrics, key, expected, "external validation metrics")


def _validate_scaler(value: Mapping[str, Any], root: Path) -> None:
    _exact_keys(value, {"version", "portable_sha256", "artifacts"}, "scaler")
    _require_equal(value, "version", GGL_SCALER_VERSION, "scaler")
    if not _HEX_64.fullmatch(str(value["portable_sha256"])):
        raise GMCProductionManifestError("scaler portable_sha256 must be lowercase 64-hex.")
    artifacts = _mapping(value["artifacts"], "scaler artifacts")
    _exact_keys(artifacts, {"json", "npz", "fit_summary"}, "scaler artifacts")
    for record in artifacts.values():
        _validate_artifact_record(_mapping(record, "scaler artifact"), root)
    scaler_json = _read_json(
        _resolve_artifact_path(artifacts["json"]["path"], root, "scaler JSON"), "scaler JSON"
    )
    _require_equal(scaler_json, "scaler_version", GGL_SCALER_VERSION, "scaler JSON")
    _require_equal(scaler_json, "portable_scaler_sha256", value["portable_sha256"], "scaler JSON")
    _require_equal(scaler_json, "test_artifact_accessed", False, "scaler JSON")


def _validate_environment(value: Mapping[str, Any], root: Path, verify_runtime: bool) -> None:
    _exact_keys(value, {"packages", "environment_file"}, "environment")
    packages = _mapping(value["packages"], "environment packages")
    _exact_keys(packages, set(REQUIRED_PACKAGE_NAMES), "environment packages")
    for name, version in packages.items():
        if not isinstance(version, str) or not version or version == "unavailable":
            raise GMCProductionManifestError(f"environment package {name!r} has no version.")
    _validate_artifact_record(_mapping(value["environment_file"], "environment file"), root)
    if verify_runtime:
        observed = {
            "python": platform.python_version(),
            **{
                name: _installed_version(name)
                for name in REQUIRED_PACKAGE_NAMES
                if name != "python"
            },
        }
        mismatches = {
            name: {"expected": packages[name], "observed": observed[name]}
            for name in REQUIRED_PACKAGE_NAMES
            if packages[name] != observed[name]
        }
        if mismatches:
            raise GMCProductionManifestError(f"runtime package versions mismatch: {mismatches}")


def _validate_source_provenance(value: Mapping[str, Any], root: Path) -> None:
    expected = {
        "validation_metrics",
        "validation_ensemble_summary",
        "calibration_summary",
        "calibrator",
        "calibration_validation_metrics",
    }
    _exact_keys(value, expected, "source_provenance")
    for record in value.values():
        _validate_artifact_record(_mapping(record, "source provenance artifact"), root)


def _validate_manifest_source_consistency(payload: Mapping[str, Any], root: Path) -> None:
    source = _mapping(payload["source_provenance"], "source_provenance")

    def source_json(key: str, label: str) -> dict[str, Any]:
        record = _mapping(source[key], f"{label} artifact")
        return _read_json(_resolve_artifact_path(record["path"], root, label), label)

    checkpoints = payload["checkpoints"]
    preprocessing = _mapping(payload["preprocessing"], "preprocessing")
    validation = _mapping(preprocessing["validation"], "validation preprocessing")
    scaler = _mapping(payload["scaler"], "scaler")
    _validate_source_documents(
        source_json("validation_metrics", "validation metrics"),
        source_json("validation_ensemble_summary", "validation ensemble summary"),
        source_json("calibration_summary", "calibration summary"),
        source_json("calibrator", "calibrator"),
        source_json("calibration_validation_metrics", "calibration validation metrics"),
        checkpoints,
        scaler,
        validation,
    )
    validation_artifacts = _mapping(validation["artifacts"], "validation artifacts")
    for record_value in checkpoints:
        record = _mapping(record_value, "checkpoint record")
        summary = _read_json(
            _resolve_artifact_path(
                record["training_summary_path"], root, "checkpoint training summary"
            ),
            "checkpoint training summary",
        )
        seed = record["seed"]
        _require_equal(summary, "seed", seed, f"seed {seed} training summary")
        _require_equal(summary, "architecture", asdict(GMCMPNNArchitecture()), "training summary")
        _require_equal(summary, "test_artifact_accessed", False, "training summary")
        frozen = _mapping(summary.get("frozen_provenance"), "checkpoint frozen provenance")
        _require_equal(frozen, "model_interface_version", MODEL_INTERFACE_VERSION, "provenance")
        _require_equal(
            frozen, "model_data_contract_version", MODEL_DATA_CONTRACT_VERSION, "provenance"
        )
        _require_equal(frozen, "portable_scaler_sha256", scaler["portable_sha256"], "provenance")
        _require_equal(
            frozen,
            "validation_feature_manifest_sha256",
            validation_artifacts["feature_manifest"]["sha256"],
            "provenance",
        )


def _preprocessing_record(directory: Path, split: str, root: Path) -> dict[str, Any]:
    summary_path = directory / "preprocessing_summary.json"
    manifest_path = directory / "feature_manifest.csv"
    status_path = directory / "molecule_status.csv"
    summary = _read_json(summary_path, f"{split} preprocessing summary")
    expected_version_key = (
        "training_preprocessing_version" if split == "train" else "validation_preprocessing_version"
    )
    expected_version = (
        TRAINING_PREPROCESSING_VERSION if split == "train" else VALIDATION_PREPROCESSING_VERSION
    )
    expected = {
        "loaded_split": split,
        expected_version_key: expected_version,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "test_artifact_accessed": False,
    }
    expected_counts = (1561, 1558) if split == "train" else (196, 196)
    expected.update(
        {"source_row_count": expected_counts[0], "successful_molecule_count": expected_counts[1]}
    )
    for key, expected_value in expected.items():
        _require_equal(summary, key, expected_value, f"{split} preprocessing summary")
    if split == "train":
        _require_equal(
            summary, "validation_artifact_accessed", False, "train preprocessing summary"
        )
    else:
        _require_equal(
            summary, "validation_artifact_accessed", True, "validation preprocessing summary"
        )
    records = {
        "summary": _artifact_record(summary_path, root),
        "feature_manifest": _artifact_record(manifest_path, root),
        "molecule_status": _artifact_record(status_path, root),
    }
    _require_equal(
        summary, "feature_manifest_sha256", records["feature_manifest"]["sha256"], "summary"
    )
    _require_equal(
        summary, "molecule_status_sha256", records["molecule_status"]["sha256"], "summary"
    )
    return {
        "version": expected_version,
        "source_row_count": expected_counts[0],
        "successful_molecule_count": expected_counts[1],
        "artifacts": records,
    }


def _scaler_record(directory: Path, root: Path, train: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "json": directory / "scaler.json",
        "npz": directory / "scaler.npz",
        "fit_summary": directory / "fit_summary.json",
    }
    scaler_json = _read_json(paths["json"], "scaler JSON")
    fit_summary = _read_json(paths["fit_summary"], "scaler fit summary")
    for key in (
        "scaler_version",
        "portable_scaler_sha256",
        "feature_manifest_sha256",
        "molecule_status_sha256",
        "test_artifact_accessed",
    ):
        if fit_summary.get(key) != scaler_json.get(key):
            raise GMCProductionManifestError(f"scaler JSON and fit summary disagree for {key}.")
    _require_equal(scaler_json, "scaler_version", GGL_SCALER_VERSION, "scaler JSON")
    _require_equal(scaler_json, "test_artifact_accessed", False, "scaler JSON")
    train_artifacts = _mapping(train["artifacts"], "train artifacts")
    _require_equal(
        scaler_json,
        "feature_manifest_sha256",
        train_artifacts["feature_manifest"]["sha256"],
        "scaler JSON",
    )
    _require_equal(
        scaler_json,
        "molecule_status_sha256",
        train_artifacts["molecule_status"]["sha256"],
        "scaler JSON",
    )
    records = {name: _artifact_record(path, root) for name, path in paths.items()}
    _require_equal(fit_summary, "scaler_json_sha256", records["json"]["sha256"], "fit summary")
    _require_equal(fit_summary, "scaler_npz_sha256", records["npz"]["sha256"], "fit summary")
    portable = scaler_json.get("portable_scaler_sha256")
    if not _HEX_64.fullmatch(str(portable)):
        raise GMCProductionManifestError("Scaler portable hash is malformed.")
    return {"version": GGL_SCALER_VERSION, "portable_sha256": portable, "artifacts": records}


def _checkpoint_records(
    checkpoint_paths: Sequence[tuple[int, Path]],
    root: Path,
    scaler: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    validation_artifacts = _mapping(validation["artifacts"], "validation artifacts")
    for seed, checkpoint_path in checkpoint_paths:
        digest = _sha256_file(checkpoint_path)
        if digest != EXPECTED_CHECKPOINT_SHA256[seed]:
            raise GMCProductionManifestError(
                f"Seed {seed} is not the frozen production checkpoint."
            )
        summary_path = checkpoint_path.parent / "run_summary.json"
        summary = _read_json(summary_path, f"seed {seed} training summary")
        _require_equal(summary, "seed", seed, f"seed {seed} training summary")
        _require_equal(summary, "test_artifact_accessed", False, f"seed {seed} training summary")
        _require_equal(summary, "architecture", asdict(GMCMPNNArchitecture()), "training summary")
        provenance = _mapping(summary.get("frozen_provenance"), "checkpoint provenance")
        _require_equal(provenance, "model_interface_version", MODEL_INTERFACE_VERSION, "provenance")
        _require_equal(
            provenance, "model_data_contract_version", MODEL_DATA_CONTRACT_VERSION, "provenance"
        )
        _require_equal(
            provenance, "portable_scaler_sha256", scaler["portable_sha256"], "provenance"
        )
        _require_equal(
            provenance,
            "validation_feature_manifest_sha256",
            validation_artifacts["feature_manifest"]["sha256"],
            "provenance",
        )
        records.append(
            {
                "seed": seed,
                "path": _relative_artifact_path(checkpoint_path, root),
                "sha256": digest,
                "training_summary_path": _relative_artifact_path(summary_path, root),
                "training_summary_sha256": _sha256_file(summary_path),
            }
        )
    return records


def _validate_source_documents(
    validation_metrics: Mapping[str, Any],
    ensemble_summary: Mapping[str, Any],
    calibration_summary: Mapping[str, Any],
    calibrator: Mapping[str, Any],
    calibration_validation_metrics: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    for value, label in (
        (validation_metrics, "validation metrics"),
        (ensemble_summary, "validation ensemble"),
        (calibration_summary, "calibration summary"),
        (calibrator, "calibrator"),
        (calibration_validation_metrics, "calibration validation metrics"),
    ):
        _require_equal(value, "test_artifact_accessed", False, label)
    _require_equal(ensemble_summary, "validation_count", 196, "validation ensemble")
    _require_equal(ensemble_summary, "seeds", list(PRODUCTION_SEEDS), "validation ensemble")
    _require_equal(
        ensemble_summary, "ensemble_method", "unweighted_arithmetic_mean", "validation ensemble"
    )
    raw = _mapping(
        _mapping(calibration_validation_metrics["metrics"], "calibration metrics")[
            "uncalibrated_threshold_0_5"
        ],
        "raw external metrics",
    )
    for key, expected in EXTERNAL_VALIDATION_METRICS.items():
        _require_equal(raw, key, expected, "raw external validation metrics")
    _require_equal(calibrator, "selected_threshold", EXPERIMENTAL_OOF_THRESHOLD, "calibrator")
    _require_equal(calibrator, "fit_split", "train_oof", "calibrator")
    _require_equal(calibrator, "fit_count", 1558, "calibrator")
    _require_equal(calibrator, "validation_used_for_fitting", False, "calibrator")
    _require_equal(calibration_summary, "selected_threshold", EXPERIMENTAL_OOF_THRESHOLD, "summary")
    _require_equal(calibration_summary, "production_seeds", list(PRODUCTION_SEEDS), "summary")
    _require_equal(
        _mapping(calibration_summary["frozen_scaler"], "calibration scaler"),
        "portable_scaler_sha256",
        scaler["portable_sha256"],
        "calibration scaler",
    )
    provenance = _mapping(ensemble_summary["provenance"], "validation provenance")
    provenance_checkpoints = _mapping(provenance["checkpoints"], "validation checkpoints")
    for record in checkpoints:
        _require_equal(
            _mapping(provenance_checkpoints[str(record["seed"])], "checkpoint provenance"),
            "sha256",
            record["sha256"],
            "checkpoint provenance",
        )
    frozen_validation = _mapping(provenance["frozen_validation"], "frozen validation")
    validation_artifacts = _mapping(validation["artifacts"], "validation artifacts")
    _require_equal(
        frozen_validation,
        "feature_manifest_sha256",
        validation_artifacts["feature_manifest"]["sha256"],
        "frozen validation",
    )


def _artifact_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": _relative_artifact_path(path, root), "sha256": _sha256_file(path)}


def _validate_artifact_record(value: Mapping[str, Any], root: Path) -> None:
    _exact_keys(value, {"path", "sha256"}, "artifact record")
    _verify_path_hash(value["path"], value["sha256"], root, "artifact")


def _verify_path_hash(path_value: Any, digest: Any, root: Path, label: str) -> None:
    if not _HEX_64.fullmatch(str(digest)):
        raise GMCProductionManifestError(f"{label} SHA-256 must be lowercase 64-hex.")
    path = _resolve_artifact_path(path_value, root, label)
    if _sha256_file(path) != digest:
        raise GMCProductionManifestError(f"{label} SHA-256 mismatch.")


def _resolve_artifact_path(path_value: Any, root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value or "\\" in path_value:
        raise GMCProductionManifestError(f"{label} path must be a nonempty POSIX relative path.")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise GMCProductionManifestError(f"{label} path must remain beneath artifact_root.")
    if _FORBIDDEN_PATH_PARTS.intersection(part.lower() for part in relative.parts):
        raise GMCProductionManifestError(f"{label} path points to a prohibited test artifact.")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GMCProductionManifestError(f"{label} path escapes artifact_root.") from exc
    if not resolved.is_file():
        raise GMCProductionManifestError(f"{label} artifact is missing: {path_value}")
    return resolved


def _relative_artifact_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise GMCProductionManifestError(f"Artifact is outside artifact_root: {path}") from exc
    if _FORBIDDEN_PATH_PARTS.intersection(part.lower() for part in relative.parts):
        raise GMCProductionManifestError(f"Artifact path is prohibited: {relative.as_posix()}")
    if not resolved.is_file():
        raise GMCProductionManifestError(f"Required artifact is missing: {path}")
    return relative.as_posix()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GMCProductionManifestError(f"Cannot hash required artifact: {path.name}") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GMCProductionManifestError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GMCProductionManifestError(f"{label} must be a JSON object.")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GMCProductionManifestError(f"{label} must be an object.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise GMCProductionManifestError(
            f"{label} schema mismatch; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}."
        )


def _require_equal(value: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if key not in value or type(value[key]) is not type(expected) or value[key] != expected:
        raise GMCProductionManifestError(f"{label} {key!r} is incompatible; expected {expected!r}.")


def _installed_version(name: str) -> str:
    distribution = "scikit-learn" if name == "scikit_learn" else name
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise GMCProductionManifestError(f"Required runtime package is missing: {name}") from exc


__all__ = [
    "ARCHITECTURE_VERSION",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPERIMENTAL_OOF_THRESHOLD",
    "EXTERNAL_VALIDATION_METRICS",
    "GMCProductionManifestError",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_VERSION",
    "PRODUCTION_SEEDS",
    "PRODUCTION_THRESHOLD",
    "ProductionManifestConfig",
    "build_production_manifest",
    "load_production_manifest",
    "validate_production_manifest",
    "write_production_manifest",
]
