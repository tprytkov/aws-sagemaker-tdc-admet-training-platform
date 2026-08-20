"""Train-only fitting and portable application of the GMC-MPNN GGL scaler."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.preprocessing import StandardScaler

from admet_platform.gmc_mpnn.geometry import (
    GEOMETRY_PREPROCESSING_VERSION,
    MMFF94S_RETRY_METHOD,
)
from admet_platform.gmc_mpnn.ggl import GGL_FEATURE_NAMES, GGL_PREPROCESSING_VERSION
from admet_platform.gmc_mpnn.standardization import GMC_STANDARDIZATION_VERSION


GGL_SCALER_VERSION: Final = "gmc-mpnn-ggl-standard-scaler-v1"
TRAINING_PREPROCESSING_VERSION: Final = "gmc-mpnn-training-raw-ggl-v1"
PREPROCESSING_SUMMARY_FILENAME: Final = "preprocessing_summary.json"
MANIFEST_FILENAME: Final = "feature_manifest.csv"
STATUS_FILENAME: Final = "molecule_status.csv"
SCALER_JSON_FILENAME: Final = "scaler.json"
SCALER_NPZ_FILENAME: Final = "scaler.npz"
FIT_SUMMARY_FILENAME: Final = "fit_summary.json"
RAW_GGL_DIRECTORY: Final = "raw_ggl"

RAW_NPZ_KEYS: Final = frozenset(
    {
        "raw_ggl_features",
        "heavy_atom_atomic_numbers",
        "heavy_atom_rdkit_indices",
        "ggl_feature_names",
        "record_key",
        "training_preprocessing_version",
        "standardization_version",
        "geometry_preprocessing_version",
        "ggl_preprocessing_version",
        "geometry_smiles",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
        "artifact_content_sha256",
    }
)
SCALER_NPZ_KEYS: Final = frozenset(
    {
        "scaler_version",
        "feature_order",
        "mean_",
        "var_",
        "scale_",
        "n_features_in_",
        "n_samples_seen_",
        "portable_scaler_sha256",
    }
)
REQUIRED_MANIFEST_COLUMNS: Final = frozenset(
    {
        "record_key",
        "molecule_id",
        "split",
        "status",
        "raw_ggl_path",
        "heavy_atom_count",
        "raw_ggl_rows",
        "raw_ggl_columns",
        "standardization_version",
        "geometry_smiles",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
    }
)
STATUS_MATCH_COLUMNS: Final = (
    "record_key",
    "molecule_id",
    "split",
    "status",
    "raw_ggl_path",
)


class GGLScalingError(RuntimeError):
    """A frozen-artifact or scaler contract violation."""


@dataclass(frozen=True)
class FrozenGGLScaler:
    """Portable, fit-free representation of a frozen six-feature scaler."""

    mean_: np.ndarray
    var_: np.ndarray
    scale_: np.ndarray
    n_features_in_: int
    n_samples_seen_: int
    feature_order: tuple[str, ...]
    scaler_version: str
    portable_scaler_sha256: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply frozen mean/scale statistics without constructing or fitting sklearn."""

        if self.scaler_version != GGL_SCALER_VERSION:
            raise GGLScalingError("Unsupported GGL scaler version.")
        if self.feature_order != tuple(GGL_FEATURE_NAMES):
            raise GGLScalingError("Frozen scaler feature order is incompatible.")
        matrix = _validate_transform_input(values)
        transformed = (matrix - self.mean_) / self.scale_
        if transformed.dtype != np.float64 or not np.isfinite(transformed).all():
            raise GGLScalingError("Scaled GGL features must be finite float64 values.")
        return transformed


@dataclass(frozen=True)
class ValidatedTrainingPool:
    """Validated pooled training atoms and their immutable provenance."""

    features: np.ndarray
    training_molecule_count: int
    training_atom_count: int
    source_row_count: int
    ordered_input_artifact_sha256: str
    feature_manifest_sha256: str
    molecule_status_sha256: str
    preprocessing_summary: Mapping[str, Any]


def fit_training_ggl_scaler(
    preprocessing_dir: str | Path,
    output_dir: str | Path,
    *,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Validate frozen TRAIN artifacts, fit StandardScaler, and write portable outputs."""

    source = Path(preprocessing_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Scaler output path already exists: {destination}")

    pool = validate_and_pool_training_artifacts(source)
    scaler = StandardScaler(with_mean=True, with_std=True)
    fitted = scaler.fit(pool.features)
    frozen = _freeze_fitted_scaler(fitted, pool.training_atom_count)
    resolved_git_commit = git_commit if git_commit is not None else _git_commit()
    portable_payload = _portable_payload(
        frozen,
        pool,
        git_commit=resolved_git_commit,
    )
    scaler_content_sha256 = _portable_scaler_sha256(portable_payload)
    portable_payload["portable_scaler_sha256"] = scaler_content_sha256

    npz_payload = {
        "scaler_version": np.asarray(GGL_SCALER_VERSION),
        "feature_order": np.asarray(GGL_FEATURE_NAMES),
        "mean_": frozen.mean_,
        "var_": frozen.var_,
        "scale_": frozen.scale_,
        "n_features_in_": np.asarray(frozen.n_features_in_, dtype=np.int64),
        "n_samples_seen_": np.asarray(frozen.n_samples_seen_, dtype=np.int64),
        "portable_scaler_sha256": np.asarray(scaler_content_sha256),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_json(temporary_directory / SCALER_JSON_FILENAME, portable_payload)
        _write_deterministic_npz(temporary_directory / SCALER_NPZ_FILENAME, npz_payload)
        fit_summary = dict(portable_payload)
        fit_summary.update(
            {
                "scaler_json_sha256": _sha256_file(temporary_directory / SCALER_JSON_FILENAME),
                "scaler_npz_sha256": _sha256_file(temporary_directory / SCALER_NPZ_FILENAME),
            }
        )
        _write_json(temporary_directory / FIT_SUMMARY_FILENAME, fit_summary)
        os.replace(temporary_directory, destination)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return fit_summary


def validate_and_pool_training_artifacts(
    preprocessing_dir: str | Path,
) -> ValidatedTrainingPool:
    """Validate a completed raw-GGL TRAIN run and pool successful atom rows."""

    source = Path(preprocessing_dir)
    summary_path = source / PREPROCESSING_SUMMARY_FILENAME
    manifest_path = source / MANIFEST_FILENAME
    status_path = source / STATUS_FILENAME
    summary = _read_json(summary_path, "preprocessing summary")
    _validate_preprocessing_summary(summary)

    manifest_sha256 = _sha256_file(manifest_path)
    status_sha256 = _sha256_file(status_path)
    if manifest_sha256 != summary["feature_manifest_sha256"]:
        raise GGLScalingError("feature_manifest.csv SHA-256 does not match the summary.")
    if status_sha256 != summary["molecule_status_sha256"]:
        raise GGLScalingError("molecule_status.csv SHA-256 does not match the summary.")

    manifest = _read_csv(manifest_path, "feature manifest")
    status = _read_csv(status_path, "molecule status")
    _validate_tables(manifest, status, summary)

    pooled: list[np.ndarray] = []
    ordered_digest = hashlib.sha256()
    successful_count = 0
    atom_count = 0
    for source_row_index, row in manifest.iterrows():
        row_status = str(row["status"])
        raw_path = str(row["raw_ggl_path"])
        if row_status != "success":
            if raw_path:
                raise GGLScalingError("Excluded or failed rows must not reference raw GGL files.")
            continue

        record_key = str(row["record_key"])
        if not _is_sha256(record_key):
            raise GGLScalingError("Feature manifest contains an invalid record key.")
        expected_relative_path = f"{RAW_GGL_DIRECTORY}/{record_key}.npz"
        if raw_path.replace("\\", "/") != expected_relative_path:
            raise GGLScalingError(f"Unexpected raw GGL path for record {record_key}: {raw_path!r}.")
        artifact_path = source / RAW_GGL_DIRECTORY / f"{record_key}.npz"
        features, artifact_content_sha256 = _validate_raw_artifact(
            artifact_path,
            row,
            summary,
        )
        file_sha256 = _sha256_file(artifact_path)
        ordered_entry = {
            "artifact_content_sha256": artifact_content_sha256,
            "artifact_file_sha256": file_sha256,
            "raw_ggl_path": expected_relative_path,
            "record_key": record_key,
            "source_row_index": int(source_row_index),
        }
        ordered_digest.update(_canonical_json_bytes(ordered_entry))
        ordered_digest.update(b"\n")
        pooled.append(features)
        successful_count += 1
        atom_count += int(features.shape[0])

    if successful_count == 0 or atom_count == 0:
        raise GGLScalingError("No successful TRAIN atom features are available for scaler fitting.")
    if successful_count != int(summary["finite_ggl_molecule_count"]):
        raise GGLScalingError("Finite-GGL count does not match successful TRAIN artifacts.")
    if atom_count != int(summary["total_heavy_atom_count_among_successes"]):
        raise GGLScalingError("TRAIN heavy-atom count does not match the preprocessing summary.")
    features = np.concatenate(pooled, axis=0)
    if features.dtype != np.float64 or features.shape != (atom_count, len(GGL_FEATURE_NAMES)):
        raise GGLScalingError("Pooled TRAIN feature matrix violates the float64 [N,6] contract.")
    return ValidatedTrainingPool(
        features=features,
        training_molecule_count=successful_count,
        training_atom_count=atom_count,
        source_row_count=len(manifest),
        ordered_input_artifact_sha256=ordered_digest.hexdigest(),
        feature_manifest_sha256=manifest_sha256,
        molecule_status_sha256=status_sha256,
        preprocessing_summary=summary,
    )


def load_frozen_ggl_scaler(output_dir: str | Path) -> FrozenGGLScaler:
    """Load and cross-check portable scaler files without calling sklearn fit."""

    source = Path(output_dir)
    json_path = source / SCALER_JSON_FILENAME
    npz_path = source / SCALER_NPZ_FILENAME
    summary_path = source / FIT_SUMMARY_FILENAME
    payload = _read_json(json_path, "scaler JSON")
    summary = _read_json(summary_path, "fit summary")
    if _sha256_file(json_path) != summary.get("scaler_json_sha256"):
        raise GGLScalingError("scaler.json SHA-256 does not match fit_summary.json.")
    if _sha256_file(npz_path) != summary.get("scaler_npz_sha256"):
        raise GGLScalingError("scaler.npz SHA-256 does not match fit_summary.json.")
    for key, value in payload.items():
        if summary.get(key) != value:
            raise GGLScalingError(f"Scaler JSON and fit summary disagree for {key!r}.")
    _validate_portable_payload(payload)

    try:
        with np.load(npz_path, allow_pickle=False) as artifact:
            if set(artifact.files) != SCALER_NPZ_KEYS:
                raise GGLScalingError("scaler.npz has an unexpected schema.")
            npz_values = {key: artifact[key] for key in artifact.files}
    except (OSError, ValueError, EOFError) as exc:
        raise GGLScalingError("scaler.npz is unreadable.") from exc

    frozen = _frozen_from_payload(payload)
    _validate_npz_against_frozen(npz_values, frozen)
    return frozen


def transform_frozen_ggl(values: np.ndarray, scaler: FrozenGGLScaler) -> np.ndarray:
    """Apply a loaded scaler; this API has no fitting path."""

    return scaler.transform(values)


def _validate_preprocessing_summary(summary: Mapping[str, Any]) -> None:
    required = {
        "dataset",
        "dataset_version",
        "source_row_count",
        "successful_molecule_count",
        "policy_excluded_molecule_count",
        "failed_molecule_count",
        "finite_ggl_molecule_count",
        "total_heavy_atom_count_among_successes",
        "feature_manifest_sha256",
        "molecule_status_sha256",
    }
    missing = required.difference(summary)
    if missing:
        raise GGLScalingError(f"Training preprocessing summary is missing: {sorted(missing)}")
    expected = {
        "loaded_split": "train",
        "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_feature_order": list(GGL_FEATURE_NAMES),
        "ggl_scaled": False,
        "rdkit_version": rdBase.rdkitVersion,
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise GGLScalingError(
            f"Training preprocessing summary is incompatible: {_canonical_json(mismatches)}"
        )
    for key in ("feature_manifest_sha256", "molecule_status_sha256"):
        if not _is_sha256(summary.get(key)):
            raise GGLScalingError(f"Training preprocessing summary has invalid {key}.")


def _validate_tables(
    manifest: pd.DataFrame,
    status: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise GGLScalingError(f"Feature manifest is missing columns: {sorted(missing)}")
    status_missing = set(STATUS_MATCH_COLUMNS).difference(status.columns)
    if status_missing:
        raise GGLScalingError(f"Molecule status is missing columns: {sorted(status_missing)}")
    if len(manifest) != len(status) or len(manifest) != int(summary["source_row_count"]):
        raise GGLScalingError("Manifest/status row counts do not match the preprocessing summary.")
    if manifest["record_key"].duplicated().any():
        raise GGLScalingError("Feature manifest record keys must be unique.")
    for column in STATUS_MATCH_COLUMNS:
        if manifest[column].astype(str).tolist() != status[column].astype(str).tolist():
            raise GGLScalingError(f"Manifest and status disagree in column {column!r}.")
    if set(manifest["split"].astype(str)) != {"train"}:
        raise GGLScalingError("Feature manifest must contain TRAIN rows only.")
    statuses = manifest["status"].astype(str)
    if not set(statuses).issubset({"success", "excluded", "failed"}):
        raise GGLScalingError("Feature manifest contains an unsupported row status.")
    observed = {
        "successful_molecule_count": int((statuses == "success").sum()),
        "policy_excluded_molecule_count": int((statuses == "excluded").sum()),
        "failed_molecule_count": int((statuses == "failed").sum()),
    }
    for key, value in observed.items():
        if value != int(summary[key]):
            raise GGLScalingError(f"Manifest {key} does not match the preprocessing summary.")


def _validate_raw_artifact(
    path: Path,
    manifest_row: pd.Series,
    summary: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise GGLScalingError(f"Missing raw GGL artifact: {path.name}")
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != RAW_NPZ_KEYS:
                raise GGLScalingError(f"Raw GGL artifact has an unexpected schema: {path.name}")
            expected_scalars = {
                "record_key": str(manifest_row["record_key"]),
                "training_preprocessing_version": summary["training_preprocessing_version"],
                "standardization_version": summary["standardization_version"],
                "geometry_preprocessing_version": summary["geometry_preprocessing_version"],
                "ggl_preprocessing_version": summary["ggl_preprocessing_version"],
                "geometry_smiles": str(manifest_row["geometry_smiles"]),
            }
            if str(manifest_row["standardization_version"]) != summary["standardization_version"]:
                raise GGLScalingError(
                    f"Manifest standardization version is invalid for {path.name}."
                )
            for key, expected in expected_scalars.items():
                if _scalar_string(artifact[key]) != expected:
                    raise GGLScalingError(f"Raw GGL artifact {path.name} has invalid {key}.")
            feature_order = tuple(str(value) for value in artifact["ggl_feature_names"].tolist())
            if feature_order != tuple(GGL_FEATURE_NAMES):
                raise GGLScalingError(f"Raw GGL artifact {path.name} has wrong feature order.")
            features = artifact["raw_ggl_features"]
            atomic_numbers = artifact["heavy_atom_atomic_numbers"]
            rdkit_indices = artifact["heavy_atom_rdkit_indices"]
            if features.dtype != np.float64:
                raise GGLScalingError(f"Raw GGL artifact {path.name} is not float64.")
            if atomic_numbers.dtype != np.int64 or rdkit_indices.dtype != np.int64:
                raise GGLScalingError(
                    f"Raw GGL artifact {path.name} has invalid atom metadata dtype."
                )
            _validate_raw_arrays(features, atomic_numbers, rdkit_indices, path.name)
            expected_count = int(manifest_row["heavy_atom_count"])
            if not (
                features.shape[0] == expected_count == int(manifest_row["raw_ggl_rows"])
                and features.shape[1] == int(manifest_row["raw_ggl_columns"])
            ):
                raise GGLScalingError(
                    f"Raw GGL artifact {path.name} disagrees with manifest shape."
                )
            for key in ("geometry_fingerprint", "ggl_fingerprint", "optimization_method"):
                observed = _scalar_string(artifact[key])
                if not observed or observed != str(manifest_row[key]):
                    raise GGLScalingError(f"Raw GGL artifact {path.name} has invalid {key}.")
            optimization_method = _scalar_string(artifact["optimization_method"])
            if optimization_method not in {"MMFF94s", MMFF94S_RETRY_METHOD, "UFF"}:
                raise GGLScalingError(
                    f"Raw GGL artifact {path.name} has unknown optimization method."
                )
            artifact_rdkit = _scalar_string(artifact["rdkit_version"])
            if artifact_rdkit != summary["rdkit_version"] or artifact_rdkit != str(
                manifest_row["rdkit_version"]
            ):
                raise GGLScalingError(f"Raw GGL artifact {path.name} has invalid RDKit version.")
            stored_checksum = _scalar_string(artifact["artifact_content_sha256"])
            content = {
                key: artifact[key] for key in artifact.files if key != "artifact_content_sha256"
            }
            if stored_checksum != _npz_content_sha256(content):
                raise GGLScalingError(f"Raw GGL artifact checksum failed: {path.name}")
            sorted_order = np.argsort(rdkit_indices, kind="stable")
            return np.asarray(features[sorted_order], dtype=np.float64), stored_checksum
    except GGLScalingError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise GGLScalingError(f"Raw GGL artifact is unreadable: {path.name}") from exc


def _validate_raw_arrays(
    features: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
    artifact_name: str,
) -> None:
    if features.ndim != 2 or features.shape[1] != len(GGL_FEATURE_NAMES):
        raise GGLScalingError(f"Raw GGL artifact {artifact_name} must have shape [n_atoms,6].")
    if features.shape[0] == 0:
        raise GGLScalingError(f"Raw GGL artifact {artifact_name} has no atom rows.")
    if atomic_numbers.shape != (features.shape[0],) or rdkit_indices.shape != (features.shape[0],):
        raise GGLScalingError(f"Raw GGL artifact {artifact_name} has misaligned atom metadata.")
    if not np.isfinite(features).all():
        raise GGLScalingError(f"Raw GGL artifact {artifact_name} contains NaN or Inf.")
    if np.any(atomic_numbers <= 1):
        raise GGLScalingError(f"Raw GGL artifact {artifact_name} contains a non-heavy atom.")
    expected_indices = np.arange(features.shape[0], dtype=np.int64)
    if not np.array_equal(rdkit_indices, expected_indices):
        raise GGLScalingError(
            f"Raw GGL artifact {artifact_name} is not in ascending RDKit atom order."
        )


def _freeze_fitted_scaler(scaler: StandardScaler, expected_samples: int) -> FrozenGGLScaler:
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    variance = np.asarray(scaler.var_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if mean.shape != (len(GGL_FEATURE_NAMES),):
        raise GGLScalingError("Fitted scaler mean has the wrong shape.")
    if variance.shape != mean.shape or scale.shape != mean.shape:
        raise GGLScalingError("Fitted scaler statistics have inconsistent shapes.")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise GGLScalingError("Fitted scaler statistics contain NaN or Inf.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise GGLScalingError("Fitted scaler scale values must be finite and positive.")
    n_features = int(scaler.n_features_in_)
    samples = np.asarray(scaler.n_samples_seen_)
    if samples.shape != ():
        raise GGLScalingError("Finite unweighted scaler fitting must yield scalar n_samples_seen_.")
    n_samples = int(samples.item())
    if n_features != len(GGL_FEATURE_NAMES) or n_samples != expected_samples:
        raise GGLScalingError("Fitted scaler dimensions do not match the validated TRAIN pool.")
    return FrozenGGLScaler(
        mean_=mean,
        var_=variance,
        scale_=scale,
        n_features_in_=n_features,
        n_samples_seen_=n_samples,
        feature_order=tuple(GGL_FEATURE_NAMES),
        scaler_version=GGL_SCALER_VERSION,
        portable_scaler_sha256="",
    )


def _portable_payload(
    frozen: FrozenGGLScaler,
    pool: ValidatedTrainingPool,
    *,
    git_commit: str,
) -> dict[str, Any]:
    summary = pool.preprocessing_summary
    return {
        "scaler_version": GGL_SCALER_VERSION,
        "scaler_type": "sklearn.preprocessing.StandardScaler",
        "with_mean": True,
        "with_std": True,
        "transformation": "(X - mean_) / scale_",
        "calculation_dtype": "float64",
        "feature_order": list(GGL_FEATURE_NAMES),
        "mean_": frozen.mean_.tolist(),
        "var_": frozen.var_.tolist(),
        "scale_": frozen.scale_.tolist(),
        "n_features_in_": frozen.n_features_in_,
        "n_samples_seen_": frozen.n_samples_seen_,
        "training_molecule_count": pool.training_molecule_count,
        "training_atom_count": pool.training_atom_count,
        "source_row_count": pool.source_row_count,
        "feature_manifest_sha256": pool.feature_manifest_sha256,
        "molecule_status_sha256": pool.molecule_status_sha256,
        "ordered_input_artifact_sha256": pool.ordered_input_artifact_sha256,
        "dataset": summary["dataset"],
        "dataset_version": summary["dataset_version"],
        "loaded_split": "train",
        "training_preprocessing_version": summary["training_preprocessing_version"],
        "standardization_version": summary["standardization_version"],
        "geometry_preprocessing_version": summary["geometry_preprocessing_version"],
        "ggl_preprocessing_version": summary["ggl_preprocessing_version"],
        "rdkit_version": summary["rdkit_version"],
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "git_commit": git_commit,
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }


def _portable_scaler_sha256(payload: Mapping[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "portable_scaler_sha256"}
    return hashlib.sha256(_canonical_json_bytes(content)).hexdigest()


def _validate_portable_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("scaler_version") != GGL_SCALER_VERSION:
        raise GGLScalingError("Unsupported GGL scaler version.")
    if payload.get("feature_order") != list(GGL_FEATURE_NAMES):
        raise GGLScalingError("Frozen scaler feature order is incompatible.")
    if payload.get("validation_artifact_accessed") is not False:
        raise GGLScalingError("Frozen scaler provenance indicates validation access.")
    if payload.get("test_artifact_accessed") is not False:
        raise GGLScalingError("Frozen scaler provenance indicates test access.")
    stored_hash = payload.get("portable_scaler_sha256")
    if not _is_sha256(stored_hash) or stored_hash != _portable_scaler_sha256(payload):
        raise GGLScalingError("Frozen scaler portable content SHA-256 is invalid.")
    _frozen_from_payload(payload)


def _frozen_from_payload(payload: Mapping[str, Any]) -> FrozenGGLScaler:
    try:
        mean = np.asarray(payload["mean_"], dtype=np.float64)
        variance = np.asarray(payload["var_"], dtype=np.float64)
        scale = np.asarray(payload["scale_"], dtype=np.float64)
        n_features = int(payload["n_features_in_"])
        n_samples = int(payload["n_samples_seen_"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GGLScalingError("Frozen scaler statistics are invalid.") from exc
    expected_shape = (len(GGL_FEATURE_NAMES),)
    if (
        mean.shape != expected_shape
        or variance.shape != expected_shape
        or scale.shape != expected_shape
    ):
        raise GGLScalingError("Frozen scaler statistics have the wrong shape.")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise GGLScalingError("Frozen scaler statistics contain NaN or Inf.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise GGLScalingError("Frozen scaler scale values must be finite and positive.")
    if n_features != len(GGL_FEATURE_NAMES) or n_samples < 1:
        raise GGLScalingError("Frozen scaler sample/feature counts are invalid.")
    return FrozenGGLScaler(
        mean_=mean,
        var_=variance,
        scale_=scale,
        n_features_in_=n_features,
        n_samples_seen_=n_samples,
        feature_order=tuple(GGL_FEATURE_NAMES),
        scaler_version=GGL_SCALER_VERSION,
        portable_scaler_sha256=str(payload["portable_scaler_sha256"]),
    )


def _validate_npz_against_frozen(
    payload: Mapping[str, np.ndarray], frozen: FrozenGGLScaler
) -> None:
    if _scalar_string(payload["scaler_version"]) != frozen.scaler_version:
        raise GGLScalingError("scaler.npz has the wrong scaler version.")
    if tuple(str(value) for value in payload["feature_order"].tolist()) != frozen.feature_order:
        raise GGLScalingError("scaler.npz has the wrong feature order.")
    for key, expected in (
        ("mean_", frozen.mean_),
        ("var_", frozen.var_),
        ("scale_", frozen.scale_),
    ):
        observed = payload[key]
        if observed.dtype != np.float64 or not np.array_equal(observed, expected):
            raise GGLScalingError(f"scaler.npz disagrees with scaler.json for {key}.")
    if int(payload["n_features_in_"].item()) != frozen.n_features_in_:
        raise GGLScalingError("scaler.npz has the wrong n_features_in_.")
    if int(payload["n_samples_seen_"].item()) != frozen.n_samples_seen_:
        raise GGLScalingError("scaler.npz has the wrong n_samples_seen_.")
    if _scalar_string(payload["portable_scaler_sha256"]) != frozen.portable_scaler_sha256:
        raise GGLScalingError("scaler.npz has the wrong portable scaler SHA-256.")


def _validate_transform_input(values: np.ndarray) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.dtype != np.float64:
        raise GGLScalingError("GGL transform input must be a NumPy float64 array.")
    if values.ndim != 2 or values.shape[1] != len(GGL_FEATURE_NAMES):
        raise GGLScalingError("GGL transform input must have shape [n_atoms,6].")
    if not np.isfinite(values).all():
        raise GGLScalingError("GGL transform input contains NaN or Inf.")
    return values


def _npz_content_sha256(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = np.ascontiguousarray(payload[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as handle, zipfile.ZipFile(handle, mode="w") as archive:
        for key in sorted(payload):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(payload[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GGLScalingError(f"The {label} is missing or unreadable.") from exc
    if not isinstance(value, dict):
        raise GGLScalingError(f"The {label} must be a JSON object.")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise GGLScalingError(f"The {label} is missing or unreadable.") from exc


def _scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise GGLScalingError("Expected a scalar string provenance value.")
    return str(value.item())


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GGLScalingError(f"Required artifact is missing or unreadable: {path.name}") from exc


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


__all__ = [
    "FIT_SUMMARY_FILENAME",
    "GGL_SCALER_VERSION",
    "SCALER_JSON_FILENAME",
    "SCALER_NPZ_FILENAME",
    "FrozenGGLScaler",
    "GGLScalingError",
    "ValidatedTrainingPool",
    "fit_training_ggl_scaler",
    "load_frozen_ggl_scaler",
    "transform_frozen_ggl",
    "validate_and_pool_training_artifacts",
]
