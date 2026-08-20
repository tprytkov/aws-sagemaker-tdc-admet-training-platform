"""Validation-only GMC-MPNN geometry, raw-GGL, and scaled-GGL preprocessing."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import (  # noqa: E402
    ChempropExperimentConfig,
    load_chemprop_config,
)
from admet_platform.gmc_mpnn.data import load_bbb_development_split  # noqa: E402
from admet_platform.gmc_mpnn.geometry import (  # noqa: E402
    GEOMETRY_PREPROCESSING_VERSION,
    MMFF94S_RETRY_METHOD,
    GeometryConfig,
    GeometryError,
    GeometryResult,
    generate_deterministic_geometry,
)
from admet_platform.gmc_mpnn.ggl import (  # noqa: E402
    GGL_FEATURE_NAMES,
    GGL_PREPROCESSING_VERSION,
    GGLConfig,
    GGLPreprocessingError,
    GGLResult,
    compute_ggl_features,
)
from admet_platform.gmc_mpnn.scaling import (  # noqa: E402
    FIT_SUMMARY_FILENAME,
    GGL_SCALER_VERSION,
    TRAINING_PREPROCESSING_VERSION as SCALER_TRAINING_PREPROCESSING_VERSION,
    FrozenGGLScaler,
    GGLScalingError,
    load_frozen_ggl_scaler,
    transform_frozen_ggl,
)
from admet_platform.gmc_mpnn.standardization import (  # noqa: E402
    EXCLUDED_BY_POLICY,
    GMC_STANDARDIZATION_VERSION,
    PARENT_SELECTED,
    UNCHANGED,
    StandardizationResult,
    standardize_for_gmc_geometry,
)
from scripts.pilot_gmc_mpnn_geometry import FAILURE_CATEGORIES, _failure_category, _git_commit  # noqa: E402
from scripts.preprocess_gmc_mpnn_training import (  # noqa: E402
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_npz,
    _npz_content_sha256,
    _serialize,
    stable_record_key,
)


VALIDATION_PREPROCESSING_VERSION = "gmc-mpnn-validation-raw-scaled-ggl-v1"
EXPECTED_VALIDATION_SOURCE_ROWS = 196
DEFAULT_CONFIG = ROOT / "configs" / "chemprop" / "bbb_martins.yaml"
DEFAULT_SCALER_DIR = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_ggl_scaler_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_validation_preprocessing_v1"
SUMMARY_FILENAME = "preprocessing_summary.json"
STATUS_FILENAME = "molecule_status.csv"
MANIFEST_FILENAME = "feature_manifest.csv"
RAW_GGL_DIRECTORY = "raw_ggl"
SCALED_GGL_DIRECTORY = "scaled_ggl"

MANIFEST_COLUMNS = (
    "source_row_index",
    "record_key",
    "molecule_id",
    "canonical_smiles",
    "target",
    "split",
    "standardization_action",
    "standardization_version",
    "geometry_smiles",
    "source_fragment_count",
    "source_heavy_atom_count",
    "source_formal_charge",
    "parent_heavy_atom_count",
    "parent_formal_charge",
    "removed_fragment_smiles",
    "removed_fragment_heavy_atom_counts",
    "removed_fragment_formal_charges",
    "exclusion_reason",
    "status",
    "failure_category",
    "failure_message",
    "raw_ggl_path",
    "scaled_ggl_path",
    "heavy_atom_count",
    "raw_ggl_rows",
    "raw_ggl_columns",
    "scaled_ggl_rows",
    "scaled_ggl_columns",
    "geometry_fingerprint",
    "ggl_fingerprint",
    "optimization_method",
    "raw_artifact_content_sha256",
    "scaled_artifact_content_sha256",
    "scaler_version",
    "portable_scaler_sha256",
    "rdkit_version",
)
STATUS_COLUMNS = (
    "source_row_index",
    "record_key",
    "molecule_id",
    "canonical_smiles",
    "target",
    "split",
    "standardization_action",
    "geometry_smiles",
    "status",
    "failure_category",
    "failure_message",
    "exclusion_reason",
    "raw_ggl_path",
    "scaled_ggl_path",
)
RAW_NPZ_KEYS = frozenset(
    {
        "raw_ggl_features",
        "heavy_atom_atomic_numbers",
        "heavy_atom_rdkit_indices",
        "ggl_feature_names",
        "record_key",
        "validation_preprocessing_version",
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
SCALED_NPZ_KEYS = frozenset(
    {
        "scaled_ggl_features",
        "heavy_atom_atomic_numbers",
        "heavy_atom_rdkit_indices",
        "ggl_feature_names",
        "record_key",
        "validation_preprocessing_version",
        "standardization_version",
        "geometry_preprocessing_version",
        "ggl_preprocessing_version",
        "scaler_version",
        "portable_scaler_sha256",
        "raw_artifact_content_sha256",
        "geometry_smiles",
        "geometry_fingerprint",
        "ggl_fingerprint",
        "optimization_method",
        "rdkit_version",
        "artifact_content_sha256",
    }
)


class ValidationPreprocessingError(RuntimeError):
    """A validation-only preprocessing contract violation."""


class ProhibitedArtifactAccessError(ValidationPreprocessingError):
    """Raised before TRAIN or locked-test split artifacts can be inspected."""


@dataclass
class ValidationAccessGuard:
    """Track validation access and reject TRAIN/test artifact inspection."""

    validation_path: Path
    train_path: Path
    test_path: Path
    validation_artifact_accessed: bool = False
    train_artifact_accessed: bool = False
    test_artifact_accessed: bool = False

    def inspect(self, candidate: object) -> None:
        try:
            candidate_path = Path(os.path.abspath(os.fspath(candidate)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if candidate_path == self.validation_path:
            self.validation_artifact_accessed = True
        elif candidate_path == self.train_path:
            self.train_artifact_accessed = True
            raise ProhibitedArtifactAccessError(
                "TRAIN split artifact access is prohibited in validation-only preprocessing."
            )
        elif candidate_path == self.test_path:
            self.test_artifact_accessed = True
            raise ProhibitedArtifactAccessError(
                "Locked BBB test artifact access is prohibited in validation-only preprocessing."
            )


@dataclass(frozen=True)
class ScalerProvenance:
    """Validated immutable TRAIN-scaler provenance recorded into validation outputs."""

    portable_scaler_sha256: str
    scaler_json_sha256: str
    scaler_npz_sha256: str
    fit_summary_sha256: str
    training_feature_manifest_sha256: str
    training_molecule_status_sha256: str
    training_ordered_input_artifact_sha256: str
    training_preprocessing_version: str
    training_rdkit_version: str
    scaler_numpy_version: str
    scaler_scikit_learn_version: str


@dataclass(frozen=True)
class StoredArtifactPair:
    """Validated metadata for one raw/scaled validation artifact pair."""

    heavy_atom_count: int
    geometry_fingerprint: str
    ggl_fingerprint: str
    optimization_method: str
    rdkit_version: str
    raw_content_sha256: str
    scaled_content_sha256: str


ConfigLoader = Callable[[str | Path], ChempropExperimentConfig]
ValidationLoader = Callable[..., pd.DataFrame]
ScalerLoader = Callable[[str | Path], FrozenGGLScaler]
StandardizationFunction = Callable[[object, str], StandardizationResult]
GeometryFunction = Callable[..., GeometryResult]
GGLFunction = Callable[..., GGLResult]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess only BBB_Martins validation rows using frozen geometry/GGL contracts "
            "and a previously fitted TRAIN scaler."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def run_validation_preprocessing(
    config_path: str | Path,
    *,
    scaler_dir: str | Path = DEFAULT_SCALER_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    expected_source_row_count: int = EXPECTED_VALIDATION_SOURCE_ROWS,
    config_loader: ConfigLoader = load_chemprop_config,
    validation_loader: ValidationLoader = load_bbb_development_split,
    scaler_loader: ScalerLoader = load_frozen_ggl_scaler,
    standardization_function: StandardizationFunction = standardize_for_gmc_geometry,
    geometry_function: GeometryFunction = generate_deterministic_geometry,
    ggl_function: GGLFunction = compute_ggl_features,
) -> dict[str, Any]:
    """Create a complete validation-only raw/scaled preprocessing dataset."""

    started = time.perf_counter()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Validation preprocessing output already exists: {destination}")
    _reject_test_named_inputs(config_path, destination)
    config = config_loader(config_path)
    _validate_validation_only_config(config)

    frozen_scaler = scaler_loader(scaler_dir)
    scaler_provenance = _load_scaler_provenance(Path(scaler_dir), frozen_scaler)
    validation_path = config.prepared_root / config.split_files["validation"]
    with _validation_only_artifact_access(config) as access_guard:
        validation = validation_loader(validation_path, split="validation")
        validation_source_sha256 = _sha256_file(validation_path)
        expected_validation_sha256 = _load_expected_validation_sha256(config)
    if access_guard.train_artifact_accessed or access_guard.test_artifact_accessed:
        raise ProhibitedArtifactAccessError(
            "A prohibited TRAIN/test artifact access attempt was detected."
        )
    if validation_source_sha256 != expected_validation_sha256:
        raise ValidationPreprocessingError(
            "BBB_Martins validation SHA-256 does not match the frozen split manifest."
        )
    _validate_loaded_validation(validation)
    if len(validation) != expected_source_row_count:
        raise ValidationPreprocessingError(
            "BBB_Martins validation source-row count mismatch: "
            f"expected {expected_source_row_count}, observed {len(validation)}."
        )
    if not access_guard.validation_artifact_accessed:
        raise ValidationPreprocessingError(
            "Validation loader did not access its validation artifact."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        raw_directory = temporary_directory / RAW_GGL_DIRECTORY
        scaled_directory = temporary_directory / SCALED_GGL_DIRECTORY
        raw_directory.mkdir()
        scaled_directory.mkdir()
        with _validation_only_artifact_access(config, guard=access_guard):
            records = _process_validation_rows(
                validation,
                raw_directory=raw_directory,
                scaled_directory=scaled_directory,
                scaler=frozen_scaler,
                standardization_function=standardization_function,
                geometry_function=geometry_function,
                ggl_function=ggl_function,
            )
        if access_guard.train_artifact_accessed or access_guard.test_artifact_accessed:
            raise ProhibitedArtifactAccessError(
                "A prohibited TRAIN/test artifact access attempt was detected."
            )
        _validate_complete_records(records, expected_source_row_count)
        manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
        status = pd.DataFrame(records, columns=STATUS_COLUMNS)
        manifest_path = temporary_directory / MANIFEST_FILENAME
        status_path = temporary_directory / STATUS_FILENAME
        _atomic_write_csv(manifest_path, manifest)
        _atomic_write_csv(status_path, status)
        summary = _build_summary(
            config=config,
            records=records,
            elapsed_seconds=time.perf_counter() - started,
            access_guard=access_guard,
            scaler_provenance=scaler_provenance,
            validation_source_sha256=validation_source_sha256,
            manifest_sha256=_sha256_file(manifest_path),
            status_sha256=_sha256_file(status_path),
        )
        _atomic_write_json(temporary_directory / SUMMARY_FILENAME, summary)
        os.replace(temporary_directory, destination)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return summary


def _process_validation_rows(
    validation: pd.DataFrame,
    *,
    raw_directory: Path,
    scaled_directory: Path,
    scaler: FrozenGGLScaler,
    standardization_function: StandardizationFunction,
    geometry_function: GeometryFunction,
    ggl_function: GGLFunction,
) -> list[dict[str, Any]]:
    geometry_config = GeometryConfig()
    ggl_config = GGLConfig()
    records: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(validation.itertuples(index=False)):
        identity = {
            "source_row_index": source_row_index,
            "molecule_id": str(row.molecule_id),
            "canonical_smiles": str(row.canonical_smiles),
            "target": int(row.target),
            "split": "validation",
        }
        record_key = stable_record_key(
            identity["molecule_id"],
            identity["canonical_smiles"],
            identity["target"],
            identity["split"],
            preprocessing_version=VALIDATION_PREPROCESSING_VERSION,
        )
        identity["record_key"] = record_key
        provenance = _empty_standardization_provenance()
        raw_path = raw_directory / f"{record_key}.npz"
        scaled_path = scaled_directory / f"{record_key}.npz"
        try:
            standardized = standardization_function(
                identity["molecule_id"], identity["canonical_smiles"]
            )
            provenance = _standardization_provenance(standardized)
            if standardized.action == EXCLUDED_BY_POLICY:
                records.append(
                    _excluded_record(identity, provenance, standardized.exclusion_reason)
                )
                continue
            geometry_smiles = standardized.geometry_canonical_smiles
            if geometry_smiles is None:  # pragma: no cover - defensive policy boundary
                raise RuntimeError("Non-excluded standardization produced no geometry SMILES.")
            geometry = geometry_function(geometry_smiles, config=geometry_config)
            if geometry.canonical_isomeric_smiles != geometry_smiles:
                raise GeometryError(
                    "atom_alignment_failed",
                    "Geometry canonical identity differs from the standardized geometry identity.",
                )
            ggl = ggl_function(
                geometry.coordinates,
                np.asarray(geometry.heavy_atom_atomic_numbers, dtype=np.int64),
                geometry_fingerprint=geometry.geometry_fingerprint,
                config=ggl_config,
            )
            raw_features = np.asarray(ggl.features)
            atomic_numbers = np.asarray(geometry.heavy_atom_atomic_numbers, dtype=np.int64)
            rdkit_indices = np.asarray(geometry.heavy_atom_rdkit_indices, dtype=np.int64)
            _validate_generated_result(geometry, ggl, raw_features, atomic_numbers, rdkit_indices)
            scaled_features = transform_frozen_ggl(raw_features, scaler)
            _validate_scaled_features(scaled_features, raw_features.shape[0])

            raw_payload = _raw_artifact_payload(
                record_key,
                geometry_smiles,
                geometry,
                ggl,
                raw_features,
                atomic_numbers,
                rdkit_indices,
            )
            raw_checksum = str(raw_payload["artifact_content_sha256"].item())
            scaled_payload = _scaled_artifact_payload(
                record_key,
                geometry_smiles,
                geometry,
                ggl,
                scaled_features,
                atomic_numbers,
                rdkit_indices,
                scaler,
                raw_checksum,
            )
            _atomic_write_npz(raw_path, raw_payload)
            _atomic_write_npz(scaled_path, scaled_payload)
            stored = _validate_artifact_pair(
                raw_path,
                scaled_path,
                expected_record_key=record_key,
                expected_geometry_smiles=geometry_smiles,
                scaler=scaler,
            )
            records.append(_success_record(identity, provenance, stored, scaler))
        except Exception as exc:  # every validation source row remains represented
            raw_path.unlink(missing_ok=True)
            scaled_path.unlink(missing_ok=True)
            records.append(_failed_record(identity, provenance, exc))
    return records


def _validate_generated_result(
    geometry: GeometryResult,
    ggl: GGLResult,
    features: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
) -> None:
    if tuple(ggl.feature_names) != tuple(GGL_FEATURE_NAMES):
        raise GGLPreprocessingError("feature_order_mismatch", "GGL feature order is not frozen.")
    if ggl.geometry_fingerprint != geometry.geometry_fingerprint or not ggl.ggl_fingerprint:
        raise GGLPreprocessingError("provenance_mismatch", "GGL geometry provenance is invalid.")
    if features.dtype != np.float64:
        raise GGLPreprocessingError("invalid_dtype", "Raw GGL features must be float64.")
    if features.ndim != 2 or features.shape[1] != len(GGL_FEATURE_NAMES):
        raise GGLPreprocessingError(
            "invalid_coordinate_shape", "Raw GGL features must have shape [n_atoms,6]."
        )
    if features.shape[0] == 0 or not np.isfinite(features).all():
        raise GGLPreprocessingError(
            "nonfinite_features", "Raw GGL features must be nonempty and finite."
        )
    if atomic_numbers.dtype != np.int64 or rdkit_indices.dtype != np.int64:
        raise GGLPreprocessingError("invalid_atom_dtype", "Heavy-atom metadata must be int64.")
    if atomic_numbers.shape != (features.shape[0],) or rdkit_indices.shape != (features.shape[0],):
        raise GGLPreprocessingError("atom_count_mismatch", "Raw GGL rows and atoms are misaligned.")
    if geometry.heavy_atom_count != features.shape[0] or np.any(atomic_numbers <= 1):
        raise GeometryError("atom_alignment_failed", "Geometry heavy atoms and GGL rows differ.")
    if not np.array_equal(rdkit_indices, np.arange(features.shape[0], dtype=np.int64)):
        raise GeometryError("atom_alignment_failed", "RDKit atom indices are not ascending.")


def _validate_scaled_features(features: np.ndarray, expected_rows: int) -> None:
    if features.dtype != np.float64 or features.shape != (expected_rows, len(GGL_FEATURE_NAMES)):
        raise GGLScalingError("Scaled GGL features must be float64 with shape [n_atoms,6].")
    if not np.isfinite(features).all():
        raise GGLScalingError("Scaled GGL features contain NaN or Inf.")


def _common_artifact_payload(
    record_key: str,
    geometry_smiles: str,
    geometry: GeometryResult,
    ggl: GGLResult,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "heavy_atom_atomic_numbers": atomic_numbers,
        "heavy_atom_rdkit_indices": rdkit_indices,
        "ggl_feature_names": np.asarray(GGL_FEATURE_NAMES),
        "record_key": np.asarray(record_key),
        "validation_preprocessing_version": np.asarray(VALIDATION_PREPROCESSING_VERSION),
        "standardization_version": np.asarray(GMC_STANDARDIZATION_VERSION),
        "geometry_preprocessing_version": np.asarray(GEOMETRY_PREPROCESSING_VERSION),
        "ggl_preprocessing_version": np.asarray(GGL_PREPROCESSING_VERSION),
        "geometry_smiles": np.asarray(geometry_smiles),
        "geometry_fingerprint": np.asarray(geometry.geometry_fingerprint),
        "ggl_fingerprint": np.asarray(ggl.ggl_fingerprint),
        "optimization_method": np.asarray(geometry.optimization_method),
        "rdkit_version": np.asarray(geometry.rdkit_version),
    }


def _raw_artifact_payload(
    record_key: str,
    geometry_smiles: str,
    geometry: GeometryResult,
    ggl: GGLResult,
    features: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    payload = _common_artifact_payload(
        record_key, geometry_smiles, geometry, ggl, atomic_numbers, rdkit_indices
    )
    payload["raw_ggl_features"] = features
    payload["artifact_content_sha256"] = np.asarray(_npz_content_sha256(payload))
    return payload


def _scaled_artifact_payload(
    record_key: str,
    geometry_smiles: str,
    geometry: GeometryResult,
    ggl: GGLResult,
    features: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
    scaler: FrozenGGLScaler,
    raw_checksum: str,
) -> dict[str, np.ndarray]:
    payload = _common_artifact_payload(
        record_key, geometry_smiles, geometry, ggl, atomic_numbers, rdkit_indices
    )
    payload.update(
        {
            "scaled_ggl_features": features,
            "scaler_version": np.asarray(scaler.scaler_version),
            "portable_scaler_sha256": np.asarray(scaler.portable_scaler_sha256),
            "raw_artifact_content_sha256": np.asarray(raw_checksum),
        }
    )
    payload["artifact_content_sha256"] = np.asarray(_npz_content_sha256(payload))
    return payload


def _validate_artifact_pair(
    raw_path: Path,
    scaled_path: Path,
    *,
    expected_record_key: str,
    expected_geometry_smiles: str,
    scaler: FrozenGGLScaler,
) -> StoredArtifactPair:
    try:
        with (
            np.load(raw_path, allow_pickle=False) as raw,
            np.load(scaled_path, allow_pickle=False) as scaled,
        ):
            if set(raw.files) != RAW_NPZ_KEYS or set(scaled.files) != SCALED_NPZ_KEYS:
                raise ValidationPreprocessingError("Validation GGL artifact schema is invalid.")
            expected = {
                "record_key": expected_record_key,
                "validation_preprocessing_version": VALIDATION_PREPROCESSING_VERSION,
                "standardization_version": GMC_STANDARDIZATION_VERSION,
                "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
                "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
                "geometry_smiles": expected_geometry_smiles,
            }
            for key, value in expected.items():
                if _scalar_string(raw[key]) != value or _scalar_string(scaled[key]) != value:
                    raise ValidationPreprocessingError(f"Validation artifact has invalid {key}.")
            raw_features = raw["raw_ggl_features"]
            scaled_features = scaled["scaled_ggl_features"]
            atomic_numbers = raw["heavy_atom_atomic_numbers"]
            rdkit_indices = raw["heavy_atom_rdkit_indices"]
            _validate_stored_arrays(raw_features, atomic_numbers, rdkit_indices, scaled_features)
            feature_order = tuple(str(value) for value in raw["ggl_feature_names"].tolist())
            if feature_order != tuple(GGL_FEATURE_NAMES):
                raise ValidationPreprocessingError("Validation artifact feature order is invalid.")
            for key in (
                "heavy_atom_atomic_numbers",
                "heavy_atom_rdkit_indices",
                "ggl_feature_names",
                "geometry_fingerprint",
                "ggl_fingerprint",
                "optimization_method",
                "rdkit_version",
            ):
                if not np.array_equal(raw[key], scaled[key]):
                    raise ValidationPreprocessingError(f"Raw/scaled artifacts disagree for {key}.")
            if _scalar_string(scaled["scaler_version"]) != GGL_SCALER_VERSION:
                raise ValidationPreprocessingError("Scaled artifact has the wrong scaler version.")
            if _scalar_string(scaled["portable_scaler_sha256"]) != scaler.portable_scaler_sha256:
                raise ValidationPreprocessingError("Scaled artifact has the wrong scaler hash.")
            raw_checksum = _validated_content_checksum(raw)
            scaled_checksum = _validated_content_checksum(scaled)
            if _scalar_string(scaled["raw_artifact_content_sha256"]) != raw_checksum:
                raise ValidationPreprocessingError(
                    "Scaled artifact has the wrong raw-artifact hash."
                )
            expected_scaled = transform_frozen_ggl(raw_features, scaler)
            if not np.array_equal(scaled_features, expected_scaled):
                raise ValidationPreprocessingError("Scaled artifact differs from frozen transform.")
            optimization_method = _scalar_string(raw["optimization_method"])
            if optimization_method not in {"MMFF94s", MMFF94S_RETRY_METHOD, "UFF"}:
                raise ValidationPreprocessingError(
                    "Validation artifact optimization method is invalid."
                )
            stored_rdkit_version = _scalar_string(raw["rdkit_version"])
            if stored_rdkit_version != rdBase.rdkitVersion:
                raise ValidationPreprocessingError("Validation artifact RDKit version is invalid.")
            return StoredArtifactPair(
                heavy_atom_count=raw_features.shape[0],
                geometry_fingerprint=_scalar_string(raw["geometry_fingerprint"]),
                ggl_fingerprint=_scalar_string(raw["ggl_fingerprint"]),
                optimization_method=optimization_method,
                rdkit_version=stored_rdkit_version,
                raw_content_sha256=raw_checksum,
                scaled_content_sha256=scaled_checksum,
            )
    except ValidationPreprocessingError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValidationPreprocessingError("Validation GGL artifact pair is unreadable.") from exc


def _validate_stored_arrays(
    raw: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
    scaled: np.ndarray,
) -> None:
    if raw.dtype != np.float64 or scaled.dtype != np.float64:
        raise ValidationPreprocessingError("Raw/scaled GGL artifacts must be float64.")
    if raw.ndim != 2 or raw.shape[1] != len(GGL_FEATURE_NAMES) or scaled.shape != raw.shape:
        raise ValidationPreprocessingError("Raw/scaled GGL shapes or alignment are invalid.")
    if raw.shape[0] == 0 or not np.isfinite(raw).all() or not np.isfinite(scaled).all():
        raise ValidationPreprocessingError("Raw/scaled GGL artifacts must be nonempty and finite.")
    if atomic_numbers.dtype != np.int64 or rdkit_indices.dtype != np.int64:
        raise ValidationPreprocessingError("Validation atom metadata must be int64.")
    if atomic_numbers.shape != (raw.shape[0],) or rdkit_indices.shape != (raw.shape[0],):
        raise ValidationPreprocessingError("Validation atom metadata is misaligned.")
    if np.any(atomic_numbers <= 1) or not np.array_equal(
        rdkit_indices, np.arange(raw.shape[0], dtype=np.int64)
    ):
        raise ValidationPreprocessingError("Validation heavy-atom alignment is invalid.")


def _validated_content_checksum(artifact: Mapping[str, np.ndarray]) -> str:
    stored = _scalar_string(artifact["artifact_content_sha256"])
    content = {key: artifact[key] for key in artifact if key != "artifact_content_sha256"}
    if stored != _npz_content_sha256(content):
        raise ValidationPreprocessingError("Validation artifact content checksum failed.")
    return stored


def _load_scaler_provenance(path: Path, scaler: FrozenGGLScaler) -> ScalerProvenance:
    summary_path = path / FIT_SUMMARY_FILENAME
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GGLScalingError("Frozen scaler fit summary is unreadable.") from exc
    expected = {
        "scaler_version": GGL_SCALER_VERSION,
        "portable_scaler_sha256": scaler.portable_scaler_sha256,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "training_preprocessing_version": SCALER_TRAINING_PREPROCESSING_VERSION,
        "loaded_split": "train",
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
        "rdkit_version": rdBase.rdkitVersion,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise GGLScalingError(
            f"Frozen TRAIN scaler provenance is incompatible: {_serialize(mismatches)}"
        )
    required_hashes = (
        "scaler_json_sha256",
        "scaler_npz_sha256",
        "feature_manifest_sha256",
        "molecule_status_sha256",
        "ordered_input_artifact_sha256",
    )
    for key in required_hashes:
        if not _is_sha256(summary.get(key)):
            raise GGLScalingError(f"Frozen TRAIN scaler has invalid {key}.")
    return ScalerProvenance(
        portable_scaler_sha256=scaler.portable_scaler_sha256,
        scaler_json_sha256=str(summary["scaler_json_sha256"]),
        scaler_npz_sha256=str(summary["scaler_npz_sha256"]),
        fit_summary_sha256=_sha256_file(summary_path),
        training_feature_manifest_sha256=str(summary["feature_manifest_sha256"]),
        training_molecule_status_sha256=str(summary["molecule_status_sha256"]),
        training_ordered_input_artifact_sha256=str(summary["ordered_input_artifact_sha256"]),
        training_preprocessing_version=str(summary["training_preprocessing_version"]),
        training_rdkit_version=str(summary["rdkit_version"]),
        scaler_numpy_version=_required_nonempty_string(summary, "numpy_version"),
        scaler_scikit_learn_version=_required_nonempty_string(summary, "scikit_learn_version"),
    )


def _standardization_provenance(result: StandardizationResult) -> dict[str, Any]:
    return {
        "standardization_action": result.action,
        "standardization_version": result.standardization_version,
        "geometry_smiles": result.geometry_canonical_smiles or "",
        "source_fragment_count": result.fragment_count,
        "source_heavy_atom_count": result.source_heavy_atom_count,
        "source_formal_charge": result.source_formal_charge,
        "parent_heavy_atom_count": result.parent_heavy_atom_count,
        "parent_formal_charge": result.parent_formal_charge,
        "removed_fragment_smiles": _serialize(result.removed_fragment_smiles),
        "removed_fragment_heavy_atom_counts": _serialize(result.removed_fragment_heavy_atom_counts),
        "removed_fragment_formal_charges": _serialize(result.removed_fragment_formal_charges),
        "exclusion_reason": result.exclusion_reason,
    }


def _empty_standardization_provenance() -> dict[str, Any]:
    return {
        "standardization_action": "",
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_smiles": "",
        "source_fragment_count": None,
        "source_heavy_atom_count": None,
        "source_formal_charge": None,
        "parent_heavy_atom_count": None,
        "parent_formal_charge": None,
        "removed_fragment_smiles": "[]",
        "removed_fragment_heavy_atom_counts": "[]",
        "removed_fragment_formal_charges": "[]",
        "exclusion_reason": "",
    }


def _base_record(identity: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **identity,
        **provenance,
        "failure_message": "",
        "raw_ggl_path": "",
        "scaled_ggl_path": "",
        "heavy_atom_count": None,
        "raw_ggl_rows": None,
        "raw_ggl_columns": None,
        "scaled_ggl_rows": None,
        "scaled_ggl_columns": None,
        "geometry_fingerprint": "",
        "ggl_fingerprint": "",
        "optimization_method": "",
        "raw_artifact_content_sha256": "",
        "scaled_artifact_content_sha256": "",
        "scaler_version": "",
        "portable_scaler_sha256": "",
        "rdkit_version": rdBase.rdkitVersion,
    }


def _excluded_record(
    identity: Mapping[str, Any], provenance: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    return {
        **_base_record(identity, provenance),
        "status": "excluded",
        "failure_category": "excluded_by_policy",
        "failure_message": reason,
    }


def _failed_record(
    identity: Mapping[str, Any], provenance: Mapping[str, Any], error: Exception
) -> dict[str, Any]:
    return {
        **_base_record(identity, provenance),
        "status": "failed",
        "failure_category": _failure_category(error),
        "failure_message": str(error),
    }


def _success_record(
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    artifact: StoredArtifactPair,
    scaler: FrozenGGLScaler,
) -> dict[str, Any]:
    record_key = str(identity["record_key"])
    return {
        **_base_record(identity, provenance),
        "status": "success",
        "failure_category": "",
        "raw_ggl_path": f"{RAW_GGL_DIRECTORY}/{record_key}.npz",
        "scaled_ggl_path": f"{SCALED_GGL_DIRECTORY}/{record_key}.npz",
        "heavy_atom_count": artifact.heavy_atom_count,
        "raw_ggl_rows": artifact.heavy_atom_count,
        "raw_ggl_columns": len(GGL_FEATURE_NAMES),
        "scaled_ggl_rows": artifact.heavy_atom_count,
        "scaled_ggl_columns": len(GGL_FEATURE_NAMES),
        "geometry_fingerprint": artifact.geometry_fingerprint,
        "ggl_fingerprint": artifact.ggl_fingerprint,
        "optimization_method": artifact.optimization_method,
        "raw_artifact_content_sha256": artifact.raw_content_sha256,
        "scaled_artifact_content_sha256": artifact.scaled_content_sha256,
        "scaler_version": GGL_SCALER_VERSION,
        "portable_scaler_sha256": scaler.portable_scaler_sha256,
        "rdkit_version": artifact.rdkit_version,
    }


def _build_summary(
    *,
    config: ChempropExperimentConfig,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
    access_guard: ValidationAccessGuard,
    scaler_provenance: ScalerProvenance,
    validation_source_sha256: str,
    manifest_sha256: str,
    status_sha256: str,
) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] == "failed"]
    exclusions = [record for record in records if record["status"] == "excluded"]
    failure_counter = Counter(str(record["failure_category"]) for record in failures)
    failure_counter["excluded_by_policy"] = len(exclusions)
    failure_counts = {
        category: int(failure_counter.get(category, 0)) for category in FAILURE_CATEGORIES
    }
    for category, count in sorted(failure_counter.items()):
        failure_counts.setdefault(category, int(count))
    action_counts = Counter(
        str(record["standardization_action"])
        for record in records
        if record["standardization_action"]
    )
    optimization_counts = Counter(str(record["optimization_method"]) for record in successes)
    return {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "git_commit": _git_commit(),
        "loaded_split": "validation",
        "validation_source_sha256": validation_source_sha256,
        "source_row_count": len(records),
        "successful_molecule_count": len(successes),
        "failed_molecule_count": len(failures),
        "policy_excluded_molecule_count": len(exclusions),
        "counts_by_standardization_action": {
            action: int(action_counts.get(action, 0))
            for action in (UNCHANGED, PARENT_SELECTED, EXCLUDED_BY_POLICY)
        },
        "counts_by_failure_category": failure_counts,
        "counts_by_optimization_method": dict(sorted(optimization_counts.items())),
        "finite_raw_ggl_molecule_count": len(successes),
        "finite_scaled_ggl_molecule_count": len(successes),
        "total_heavy_atom_count_among_successes": sum(
            int(record["heavy_atom_count"]) for record in successes
        ),
        "total_elapsed_seconds": elapsed_seconds,
        "validation_preprocessing_version": VALIDATION_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_scaler_version": GGL_SCALER_VERSION,
        "ggl_feature_order": list(GGL_FEATURE_NAMES),
        "raw_ggl_dtype": "float64",
        "scaled_ggl_dtype": "float64",
        "scaler_transformation": "(X - mean_) / scale_",
        "portable_scaler_sha256": scaler_provenance.portable_scaler_sha256,
        "scaler_json_sha256": scaler_provenance.scaler_json_sha256,
        "scaler_npz_sha256": scaler_provenance.scaler_npz_sha256,
        "scaler_fit_summary_sha256": scaler_provenance.fit_summary_sha256,
        "training_preprocessing_version": scaler_provenance.training_preprocessing_version,
        "training_feature_manifest_sha256": scaler_provenance.training_feature_manifest_sha256,
        "training_molecule_status_sha256": scaler_provenance.training_molecule_status_sha256,
        "training_ordered_input_artifact_sha256": (
            scaler_provenance.training_ordered_input_artifact_sha256
        ),
        "training_rdkit_version": scaler_provenance.training_rdkit_version,
        "scaler_numpy_version": scaler_provenance.scaler_numpy_version,
        "scaler_scikit_learn_version": scaler_provenance.scaler_scikit_learn_version,
        "validation_numpy_version": np.__version__,
        "ordered_raw_artifact_sha256": _ordered_artifact_hash(
            successes, "raw_artifact_content_sha256"
        ),
        "ordered_scaled_artifact_sha256": _ordered_artifact_hash(
            successes, "scaled_artifact_content_sha256"
        ),
        "rdkit_version": rdBase.rdkitVersion,
        "feature_manifest_sha256": manifest_sha256,
        "molecule_status_sha256": status_sha256,
        "validation_artifact_accessed": access_guard.validation_artifact_accessed,
        "test_artifact_accessed": access_guard.test_artifact_accessed,
    }


def _ordered_artifact_hash(records: list[dict[str, Any]], checksum_field: str) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = [record["source_row_index"], record["record_key"], record[checksum_field]]
        digest.update(json.dumps(payload, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def _validate_loaded_validation(frame: pd.DataFrame) -> None:
    required = {"molecule_id", "canonical_smiles", "target", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValidationPreprocessingError(
            f"Validation loader returned rows missing columns: {sorted(missing)}"
        )
    observed = set(frame["split"].astype(str).str.strip().str.lower())
    if not observed or not observed.issubset({"validation", "valid", "val"}):
        raise ProhibitedArtifactAccessError(
            f"Validation loader returned non-validation split labels: {sorted(observed)}"
        )


def _validate_complete_records(records: list[dict[str, Any]], expected_count: int) -> None:
    if len(records) != expected_count:
        raise ValidationPreprocessingError(
            "Validation preprocessing did not retain every source row."
        )
    indices = [record["source_row_index"] for record in records]
    if indices != list(range(expected_count)):
        raise ValidationPreprocessingError(
            "Validation source-row indices are incomplete or reordered."
        )
    record_keys = [str(record["record_key"]) for record in records]
    if len(set(record_keys)) != expected_count:
        raise ValidationPreprocessingError("Validation record keys are not unique.")
    if any(record["split"] != "validation" for record in records):
        raise ValidationPreprocessingError("Validation output contains a non-validation row.")
    if any(record["status"] not in {"success", "excluded", "failed"} for record in records):
        raise ValidationPreprocessingError("Validation output contains an invalid row status.")


def _load_expected_validation_sha256(config: ChempropExperimentConfig) -> str:
    try:
        manifest_bytes = config.split_manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationPreprocessingError("Frozen split manifest is unreadable.") from exc
    if not isinstance(manifest, dict):
        raise ValidationPreprocessingError("Frozen split manifest must be a JSON object.")
    observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    configured_manifest_sha256 = config.raw.get("split_manifest_sha256")
    if configured_manifest_sha256 and configured_manifest_sha256 != observed_manifest_sha256:
        raise ValidationPreprocessingError("Frozen split-manifest SHA-256 is incompatible.")
    configured_manifest_id = config.raw.get("split_manifest_id")
    if configured_manifest_id and manifest.get("split_manifest_id") != configured_manifest_id:
        raise ValidationPreprocessingError("Frozen split-manifest identity is incompatible.")
    try:
        value: object = manifest
        for part in config.split_hash_keys["validation"].split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(part)
            value = value[part]
    except (KeyError, AttributeError) as exc:
        raise ValidationPreprocessingError(
            "Frozen split manifest has no configured validation hash."
        ) from exc
    if not _is_sha256(value):
        raise ValidationPreprocessingError("Frozen split manifest has an invalid validation hash.")
    return str(value)


def _validate_validation_only_config(config: ChempropExperimentConfig) -> None:
    if config.endpoint_id != "bbb_martins" or config.task_type != "binary_classification":
        raise ValidationPreprocessingError(
            "Validation preprocessing supports only binary BBB_Martins."
        )
    names = {split: config.split_files.get(split) for split in ("train", "validation", "test")}
    if any(not name for name in names.values()) or len(set(names.values())) != 3:
        raise ProhibitedArtifactAccessError(
            "Config must identify distinct TRAIN, validation, and locked-test artifacts."
        )


def _reject_test_named_inputs(config_path: str | Path, output_dir: Path) -> None:
    if Path(config_path).name.lower() in {"test.csv", "locked.csv"}:
        raise ProhibitedArtifactAccessError("A test artifact cannot be used as validation config.")
    if output_dir.name.lower() in {"test", "test.csv", "locked", "locked.csv"}:
        raise ProhibitedArtifactAccessError("Validation output cannot target a test-named path.")


@contextmanager
def _validation_only_artifact_access(
    config: ChempropExperimentConfig,
    *,
    guard: ValidationAccessGuard | None = None,
) -> Iterator[ValidationAccessGuard]:
    if guard is None:
        guard = ValidationAccessGuard(
            validation_path=Path(
                os.path.abspath(config.prepared_root / config.split_files["validation"])
            ),
            train_path=Path(os.path.abspath(config.prepared_root / config.split_files["train"])),
            test_path=Path(os.path.abspath(config.prepared_root / config.split_files["test"])),
        )
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_path_open = Path.open
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir
    original_stat = Path.stat
    original_builtin_open = builtins.open
    original_read_csv = pd.read_csv

    def inspect(candidate: object) -> None:
        guard.inspect(candidate)

    def guarded_read_bytes(path: Path) -> bytes:
        inspect(path)
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        inspect(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        inspect(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_exists(path: Path) -> bool:
        inspect(path)
        return original_exists(path)

    def guarded_is_file(path: Path) -> bool:
        inspect(path)
        return original_is_file(path)

    def guarded_is_dir(path: Path) -> bool:
        inspect(path)
        return original_is_dir(path)

    def guarded_stat(path: Path, *args: Any, **kwargs: Any):
        inspect(path)
        return original_stat(path, *args, **kwargs)

    def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any):
        inspect(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_read_csv(source: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        inspect(source)
        return original_read_csv(source, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Path, "read_bytes", guarded_read_bytes))
        stack.enter_context(patch.object(Path, "read_text", guarded_read_text))
        stack.enter_context(patch.object(Path, "open", guarded_path_open))
        stack.enter_context(patch.object(Path, "exists", guarded_exists))
        stack.enter_context(patch.object(Path, "is_file", guarded_is_file))
        stack.enter_context(patch.object(Path, "is_dir", guarded_is_dir))
        stack.enter_context(patch.object(Path, "stat", guarded_stat))
        stack.enter_context(patch.object(builtins, "open", guarded_builtin_open))
        stack.enter_context(patch.object(pd, "read_csv", guarded_read_csv))
        yield guard


def _scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError("Expected scalar NPZ provenance value.")
    return str(value.item())


def _required_nonempty_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GGLScalingError(f"Frozen TRAIN scaler has invalid {key}.")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_validation_preprocessing(
        args.config,
        scaler_dir=args.scaler_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "EXPECTED_VALIDATION_SOURCE_ROWS",
    "ProhibitedArtifactAccessError",
    "VALIDATION_PREPROCESSING_VERSION",
    "ValidationPreprocessingError",
    "build_parser",
    "run_validation_preprocessing",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
