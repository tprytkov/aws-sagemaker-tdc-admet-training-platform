"""Resumable, training-only GMC-MPNN geometry and raw-GGL preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from admet_platform.gmc_mpnn.standardization import (  # noqa: E402
    EXCLUDED_BY_POLICY,
    GMC_STANDARDIZATION_VERSION,
    PARENT_SELECTED,
    StandardizationResult,
    UNCHANGED,
    standardize_for_gmc_geometry,
)
from scripts.pilot_gmc_mpnn_geometry import (  # noqa: E402
    FAILURE_CATEGORIES,
    ArtifactAccessGuard,
    NonTrainingArtifactAccessError,
    _deny_nontraining_artifact_access,
    _failure_category,
    _git_commit,
    _reject_test_named_config,
    _validate_training_only_config,
)


TRAINING_PREPROCESSING_VERSION = "gmc-mpnn-training-raw-ggl-v1"
DEFAULT_CONFIG = ROOT / "configs" / "chemprop" / "bbb_martins.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_training_preprocessing"
SUMMARY_FILENAME = "preprocessing_summary.json"
STATUS_FILENAME = "molecule_status.csv"
MANIFEST_FILENAME = "feature_manifest.csv"
RAW_GGL_DIRECTORY = "raw_ggl"
MANIFEST_COLUMNS = (
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
    "heavy_atom_count",
    "raw_ggl_rows",
    "raw_ggl_columns",
    "geometry_fingerprint",
    "ggl_fingerprint",
    "optimization_method",
    "rdkit_version",
)
STATUS_COLUMNS = (
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
)
NPZ_KEYS = frozenset(
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


class IncompatibleResumeError(RuntimeError):
    """Existing run provenance is incompatible with this preprocessing version."""


@dataclass(frozen=True)
class StoredArtifact:
    """Validated metadata read from one raw-GGL NPZ."""

    heavy_atom_count: int
    geometry_fingerprint: str
    ggl_fingerprint: str
    optimization_method: str
    rdkit_version: str


ConfigLoader = Callable[[str | Path], ChempropExperimentConfig]
TrainingLoader = Callable[..., pd.DataFrame]
StandardizationFunction = Callable[[object, str], StandardizationResult]
GeometryFunction = Callable[..., GeometryResult]
GGLFunction = Callable[..., GGLResult]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the complete BBB_Martins training split into deterministic geometry "
            "and raw, unscaled six-column GMC GGL feature artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="BBB Chemprop configuration (default: configs/chemprop/bbb_martins.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Training-preprocessing output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only fully validated compatible successful NPZ artifacts.",
    )
    return parser


def stable_record_key(
    molecule_id: object,
    canonical_smiles: object,
    target: object,
    split: object,
    *,
    preprocessing_version: str = TRAINING_PREPROCESSING_VERSION,
) -> str:
    """Hash the original source composite identity and preprocessing version."""

    payload = {
        "canonical_smiles": str(canonical_smiles),
        "molecule_id": str(molecule_id),
        "preprocessing_version": preprocessing_version,
        "split": str(split),
        "target": int(target),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def run_training_preprocessing(
    config_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    resume: bool = False,
    config_loader: ConfigLoader = load_chemprop_config,
    training_loader: TrainingLoader = load_bbb_development_split,
    standardization_function: StandardizationFunction = standardize_for_gmc_geometry,
    geometry_function: GeometryFunction = generate_deterministic_geometry,
    ggl_function: GGLFunction = compute_ggl_features,
) -> dict[str, Any]:
    """Preprocess every training source row while retaining failures and exclusions."""

    started = time.perf_counter()
    destination = Path(output_dir)
    _prepare_output_directory(destination, resume=resume)
    _reject_test_named_config(config_path)
    config = config_loader(config_path)
    _validate_training_only_config(config)
    if resume:
        _validate_resume_summary(destination, config)

    geometry_config = GeometryConfig()
    ggl_config = GGLConfig()
    raw_directory = destination / RAW_GGL_DIRECTORY
    raw_directory.mkdir(parents=True, exist_ok=True)
    train_path = config.prepared_root / config.split_files["train"]

    with _deny_nontraining_artifact_access(config) as access_guard:
        training = training_loader(train_path, split="train")
        records, reused_count = _process_training_rows(
            training,
            raw_directory=raw_directory,
            resume=resume,
            geometry_config=geometry_config,
            ggl_config=ggl_config,
            standardization_function=standardization_function,
            geometry_function=geometry_function,
            ggl_function=ggl_function,
        )

    elapsed = time.perf_counter() - started
    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    status = pd.DataFrame(records, columns=STATUS_COLUMNS)
    manifest_path = destination / MANIFEST_FILENAME
    status_path = destination / STATUS_FILENAME
    _atomic_write_csv(manifest_path, manifest)
    _atomic_write_csv(status_path, status)
    summary = _build_summary(
        config=config,
        records=records,
        elapsed_seconds=elapsed,
        reused_count=reused_count,
        access_guard=access_guard,
        manifest_sha256=_sha256_file(manifest_path),
        status_sha256=_sha256_file(status_path),
    )
    _atomic_write_json(destination / SUMMARY_FILENAME, summary)
    return summary


def _process_training_rows(
    training: pd.DataFrame,
    *,
    raw_directory: Path,
    resume: bool,
    geometry_config: GeometryConfig,
    ggl_config: GGLConfig,
    standardization_function: StandardizationFunction,
    geometry_function: GeometryFunction,
    ggl_function: GGLFunction,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    reused_count = 0
    for row in training.itertuples(index=False):
        identity = {
            "molecule_id": str(row.molecule_id),
            "canonical_smiles": str(row.canonical_smiles),
            "target": int(row.target),
            "split": "train",
        }
        record_key = stable_record_key(
            identity["molecule_id"],
            identity["canonical_smiles"],
            identity["target"],
            identity["split"],
        )
        identity["record_key"] = record_key
        artifact_path = raw_directory / f"{record_key}.npz"
        relative_artifact_path = f"{RAW_GGL_DIRECTORY}/{record_key}.npz"
        provenance = _empty_standardization_provenance()
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

            stored = None
            if resume and artifact_path.is_file():
                stored = _validate_stored_artifact(
                    artifact_path,
                    expected_record_key=record_key,
                    expected_geometry_smiles=geometry_smiles,
                )
            if stored is not None:
                reused_count += 1
                records.append(
                    _success_record(
                        identity,
                        provenance,
                        relative_artifact_path,
                        stored,
                    )
                )
                continue

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
            if tuple(ggl.feature_names) != tuple(GGL_FEATURE_NAMES):
                raise RuntimeError("GGL feature order differs from the frozen six-feature order.")
            if ggl.geometry_fingerprint != geometry.geometry_fingerprint:
                raise RuntimeError("GGL geometry provenance differs from the geometry result.")
            if not isinstance(ggl.ggl_fingerprint, str) or not ggl.ggl_fingerprint:
                raise RuntimeError("Finite GGL output has no provenance fingerprint.")
            features = np.asarray(ggl.features, dtype=np.float64)
            atomic_numbers = np.asarray(geometry.heavy_atom_atomic_numbers, dtype=np.int64)
            rdkit_indices = np.asarray(geometry.heavy_atom_rdkit_indices, dtype=np.int64)
            _validate_generated_arrays(features, atomic_numbers, rdkit_indices)
            if geometry.heavy_atom_count != features.shape[0]:
                raise GeometryError(
                    "atom_alignment_failed",
                    "Geometry heavy-atom count differs from the raw GGL row count.",
                )
            payload = _artifact_payload(
                record_key=record_key,
                geometry_smiles=geometry_smiles,
                geometry=geometry,
                ggl=ggl,
                features=features,
                atomic_numbers=atomic_numbers,
                rdkit_indices=rdkit_indices,
            )
            _atomic_write_npz(artifact_path, payload)
            stored = _validate_stored_artifact(
                artifact_path,
                expected_record_key=record_key,
                expected_geometry_smiles=geometry_smiles,
            )
            if stored is None:  # pragma: no cover - write/read defensive boundary
                raise RuntimeError("New raw-GGL artifact failed provenance validation.")
            records.append(_success_record(identity, provenance, relative_artifact_path, stored))
        except Exception as exc:  # every source row remains represented
            records.append(_failed_record(identity, provenance, exc))
    return records, reused_count


def _artifact_payload(
    *,
    record_key: str,
    geometry_smiles: str,
    geometry: GeometryResult,
    ggl: GGLResult,
    features: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    payload = {
        "raw_ggl_features": features,
        "heavy_atom_atomic_numbers": atomic_numbers,
        "heavy_atom_rdkit_indices": rdkit_indices,
        "ggl_feature_names": np.asarray(GGL_FEATURE_NAMES),
        "record_key": np.asarray(record_key),
        "training_preprocessing_version": np.asarray(TRAINING_PREPROCESSING_VERSION),
        "standardization_version": np.asarray(GMC_STANDARDIZATION_VERSION),
        "geometry_preprocessing_version": np.asarray(GEOMETRY_PREPROCESSING_VERSION),
        "ggl_preprocessing_version": np.asarray(GGL_PREPROCESSING_VERSION),
        "geometry_smiles": np.asarray(geometry_smiles),
        "geometry_fingerprint": np.asarray(geometry.geometry_fingerprint),
        "ggl_fingerprint": np.asarray(ggl.ggl_fingerprint),
        "optimization_method": np.asarray(geometry.optimization_method),
        "rdkit_version": np.asarray(geometry.rdkit_version),
    }
    payload["artifact_content_sha256"] = np.asarray(_npz_content_sha256(payload))
    return payload


def _validate_generated_arrays(
    features: np.ndarray,
    atomic_numbers: np.ndarray,
    rdkit_indices: np.ndarray,
) -> None:
    if features.ndim != 2 or features.shape[1] != len(GGL_FEATURE_NAMES):
        raise GGLPreprocessingError(
            "invalid_coordinate_shape",
            "Raw GGL features must have shape [n_heavy_atoms, 6].",
        )
    if features.shape[0] != len(atomic_numbers) or features.shape[0] != len(rdkit_indices):
        raise GGLPreprocessingError(
            "atom_count_mismatch", "Raw GGL rows do not align with heavy-atom metadata."
        )
    if not np.isfinite(features).all():
        raise GGLPreprocessingError("nonfinite_features", "Raw GGL features contain NaN or Inf.")
    if np.any(atomic_numbers <= 1):
        raise GGLPreprocessingError(
            "hydrogen_not_allowed", "Raw GGL artifacts may contain heavy atoms only."
        )
    if not np.array_equal(rdkit_indices, np.arange(features.shape[0], dtype=np.int64)):
        raise GeometryError(
            "atom_alignment_failed",
            "Stored heavy-atom RDKit indices are not canonical-order aligned.",
        )


def _validate_stored_artifact(
    path: Path,
    *,
    expected_record_key: str,
    expected_geometry_smiles: str,
) -> StoredArtifact | None:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != NPZ_KEYS:
                return None
            expected_scalars = {
                "record_key": expected_record_key,
                "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
                "standardization_version": GMC_STANDARDIZATION_VERSION,
                "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
                "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
                "geometry_smiles": expected_geometry_smiles,
            }
            if any(
                _scalar_string(artifact[key]) != value for key, value in expected_scalars.items()
            ):
                return None
            features = artifact["raw_ggl_features"]
            atomic_numbers = artifact["heavy_atom_atomic_numbers"]
            rdkit_indices = artifact["heavy_atom_rdkit_indices"]
            if tuple(str(value) for value in artifact["ggl_feature_names"].tolist()) != tuple(
                GGL_FEATURE_NAMES
            ):
                return None
            if features.dtype != np.float64:
                return None
            if atomic_numbers.dtype != np.int64 or rdkit_indices.dtype != np.int64:
                return None
            _validate_generated_arrays(features, atomic_numbers, rdkit_indices)
            if np.any(atomic_numbers <= 1):
                return None
            geometry_fingerprint = _scalar_string(artifact["geometry_fingerprint"])
            ggl_fingerprint = _scalar_string(artifact["ggl_fingerprint"])
            optimization_method = _scalar_string(artifact["optimization_method"])
            stored_rdkit_version = _scalar_string(artifact["rdkit_version"])
            stored_content_sha256 = _scalar_string(artifact["artifact_content_sha256"])
            if not all(
                (geometry_fingerprint, ggl_fingerprint, optimization_method, stored_rdkit_version)
            ):
                return None
            if optimization_method not in {"MMFF94s", MMFF94S_RETRY_METHOD, "UFF"}:
                return None
            if stored_rdkit_version != rdBase.rdkitVersion:
                return None
            content = {
                key: artifact[key] for key in artifact.files if key != "artifact_content_sha256"
            }
            if stored_content_sha256 != _npz_content_sha256(content):
                return None
            return StoredArtifact(
                heavy_atom_count=int(features.shape[0]),
                geometry_fingerprint=geometry_fingerprint,
                ggl_fingerprint=ggl_fingerprint,
                optimization_method=optimization_method,
                rdkit_version=stored_rdkit_version,
            )
    except (OSError, ValueError, KeyError, EOFError, GGLPreprocessingError, GeometryError):
        return None


def _scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError("Expected scalar NPZ provenance value.")
    return str(value.item())


def _npz_content_sha256(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = np.ascontiguousarray(payload[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


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
        "heavy_atom_count": None,
        "raw_ggl_rows": None,
        "raw_ggl_columns": None,
        "geometry_fingerprint": "",
        "ggl_fingerprint": "",
        "optimization_method": "",
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
    relative_artifact_path: str,
    artifact: StoredArtifact,
) -> dict[str, Any]:
    return {
        **_base_record(identity, provenance),
        "status": "success",
        "failure_category": "",
        "raw_ggl_path": relative_artifact_path,
        "heavy_atom_count": artifact.heavy_atom_count,
        "raw_ggl_rows": artifact.heavy_atom_count,
        "raw_ggl_columns": len(GGL_FEATURE_NAMES),
        "geometry_fingerprint": artifact.geometry_fingerprint,
        "ggl_fingerprint": artifact.ggl_fingerprint,
        "optimization_method": artifact.optimization_method,
        "rdkit_version": artifact.rdkit_version,
    }


def _build_summary(
    *,
    config: ChempropExperimentConfig,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
    reused_count: int,
    access_guard: ArtifactAccessGuard,
    manifest_sha256: str,
    status_sha256: str,
) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] == "failed"]
    exclusions = [record for record in records if record["status"] == "excluded"]
    actions = Counter(
        str(record["standardization_action"])
        for record in records
        if record["standardization_action"]
    )
    action_counts = {
        action: int(actions.get(action, 0))
        for action in (UNCHANGED, PARENT_SELECTED, EXCLUDED_BY_POLICY)
    }
    failure_counter = Counter(str(record["failure_category"]) for record in failures)
    failure_counter["excluded_by_policy"] = len(exclusions)
    failure_counts = {
        category: int(failure_counter.get(category, 0)) for category in FAILURE_CATEGORIES
    }
    for category, count in sorted(failure_counter.items()):
        failure_counts.setdefault(category, int(count))
    optimization_counts = Counter(str(record["optimization_method"]) for record in successes)
    heavy_counts = [int(record["heavy_atom_count"]) for record in successes]
    source_count = len(records)
    return {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "git_commit": _git_commit(),
        "loaded_split": "train",
        "source_row_count": source_count,
        "successful_molecule_count": len(successes),
        "failed_molecule_count": len(failures),
        "policy_excluded_molecule_count": len(exclusions),
        "counts_by_standardization_action": action_counts,
        "counts_by_failure_category": failure_counts,
        "counts_by_optimization_method": dict(sorted(optimization_counts.items())),
        "finite_ggl_molecule_count": len(successes),
        "total_heavy_atom_count_among_successes": sum(heavy_counts),
        "minimum_heavy_atom_count": min(heavy_counts) if heavy_counts else None,
        "maximum_heavy_atom_count": max(heavy_counts) if heavy_counts else None,
        "total_elapsed_seconds": elapsed_seconds,
        "mean_seconds_per_processed_molecule": (
            elapsed_seconds / source_count if source_count else None
        ),
        "reused_successful_artifact_count": reused_count,
        "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_feature_order": list(GGL_FEATURE_NAMES),
        "ggl_scaled": False,
        "rdkit_version": rdBase.rdkitVersion,
        "feature_manifest_sha256": manifest_sha256,
        "molecule_status_sha256": status_sha256,
        "validation_artifact_accessed": access_guard.validation_artifact_accessed,
        "test_artifact_accessed": access_guard.test_artifact_accessed,
    }


def _prepare_output_directory(output_dir: Path, *, resume: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Preprocessing output path is not a directory: {output_dir}")
    if resume:
        if not output_dir.exists():
            raise FileNotFoundError("--resume requires an existing preprocessing output directory.")
        return
    if output_dir.exists():
        raise FileExistsError(
            f"Preprocessing output directory already exists: {output_dir}. Use --resume explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=False)


def _validate_resume_summary(output_dir: Path, config: ChempropExperimentConfig) -> None:
    summary_path = output_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncompatibleResumeError("Existing preprocessing summary is unreadable.") from exc
    expected = {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "loaded_split": "train",
        "training_preprocessing_version": TRAINING_PREPROCESSING_VERSION,
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "ggl_preprocessing_version": GGL_PREPROCESSING_VERSION,
        "ggl_scaled": False,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise IncompatibleResumeError(
            f"Existing preprocessing summary is incompatible: {_serialize(mismatches)}"
        )


def _atomic_write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialize(values: Any) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_training_preprocessing(
        args.config,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "IncompatibleResumeError",
    "NonTrainingArtifactAccessError",
    "TRAINING_PREPROCESSING_VERSION",
    "build_parser",
    "run_training_preprocessing",
    "stable_record_key",
]


if __name__ == "__main__":
    raise SystemExit(main())
