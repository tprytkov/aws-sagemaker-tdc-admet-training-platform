"""Run an auditable, training-only GMC-MPNN geometry/GGL feasibility pilot.

This script deterministically selects training rows and records preprocessing
reliability and runtime only. Its output must not be used to select a prediction
threshold, calibration method, model, train/validation membership, or any choice
based on validation/test performance. Any later geometry-setting change prompted
by pilot failures is training-only preprocessing development and requires review.

No validation or test artifact is loaded. No model, feature scaling, Chemprop
``V_f`` integration, conformer file, or bulk feature cache is created.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import (  # noqa: E402
    ChempropExperimentConfig,
    load_chemprop_config,
)
from admet_platform.gmc_mpnn.data import load_bbb_development_split  # noqa: E402
from admet_platform.gmc_mpnn.geometry import (  # noqa: E402
    GeometryConfig,
    GeometryError,
    GeometryResult,
    generate_deterministic_geometry,
)
from admet_platform.gmc_mpnn.ggl import (  # noqa: E402
    ELEMENT_RADIUS_MAPPING_VERSION,
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


DEFAULT_CONFIG = ROOT / "configs" / "chemprop" / "bbb_martins.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_geometry_pilot"
DEFAULT_PILOT_SIZE = 25
PILOT_SELECTION_SEED = 13
PILOT_SELECTION_VERSION = "sha256-composite-v1"
SELECTION_COLUMNS = ("molecule_id", "canonical_smiles", "target")
FAILURE_CATEGORIES = (
    "excluded_by_policy",
    "invalid_smiles",
    "disconnected_fragment",
    "embedding_failed",
    "no_conformer",
    "optimization_failed",
    "atom_alignment_failed",
    "unsupported_element",
    "invalid_coordinates",
    "nonfinite_ggl",
    "unexpected_error",
)
STATUS_COLUMNS = (
    "pilot_index",
    "molecule_id",
    "canonical_smiles",
    "target",
    "source_split",
    "selection_hash",
    "standardization_action",
    "standardization_version",
    "source_fragment_count",
    "source_heavy_atom_count",
    "source_formal_charge",
    "geometry_smiles",
    "parent_heavy_atom_count",
    "parent_formal_charge",
    "removed_fragment_smiles",
    "removed_fragment_heavy_atom_counts",
    "removed_fragment_formal_charges",
    "exclusion_reason",
    "status",
    "failure_category",
    "failure_message",
    "heavy_atom_count",
    "formal_charge_sum",
    "charged_heavy_atom_count",
    "generated_conformer_count",
    "optimization_method",
    "selected_conformer_id",
    "selected_energy",
    "effective_embedding_seed",
    "geometry_time_seconds",
    "ggl_time_seconds",
    "total_preprocessing_time_seconds",
    "geometry_fingerprint",
    "ggl_fingerprint",
    "raw_ggl_rows",
    "raw_ggl_columns",
    "all_ggl_values_finite",
    "rdkit_version",
)


class NonTrainingArtifactAccessError(RuntimeError):
    """Raised before validation or locked-test data can be inspected."""


@dataclass
class ArtifactAccessGuard:
    """Tracks prohibited access attempts; successful pilots leave both flags false."""

    validation_path: Path
    test_path: Path
    validation_artifact_accessed: bool = False
    test_artifact_accessed: bool = False

    def reject(self, candidate: object) -> None:
        try:
            candidate_path = Path(os.path.abspath(os.fspath(candidate)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if candidate_path == self.validation_path:
            self.validation_artifact_accessed = True
            raise NonTrainingArtifactAccessError(
                "Validation artifact access is prohibited in the training-only pilot."
            )
        if candidate_path == self.test_path:
            self.test_artifact_accessed = True
            raise NonTrainingArtifactAccessError(
                "Locked BBB test artifact access is prohibited in the training-only pilot."
            )


GeometryFunction = Callable[..., GeometryResult]
GGLFunction = Callable[..., GGLResult]
StandardizationFunction = Callable[[object, str], StandardizationResult]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic, training-only BBB_Martins geometry/GGL pilot for "
            "preprocessing feasibility and runtime characterization."
        ),
        epilog=(
            "This pilot must not be used for model selection, thresholding, calibration, "
            "split changes, or validation/test-driven tuning."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="BBB Chemprop configuration (default: configs/chemprop/bbb_martins.yaml).",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help="Number of deterministically selected training molecules (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pilot output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the three known pilot files in an existing exact output directory.",
    )
    return parser


def stable_selection_hash(
    molecule_id: object,
    canonical_smiles: object,
    target: object,
    *,
    selection_seed: int = PILOT_SELECTION_SEED,
) -> str:
    """Hash one composite identity without Python's process-dependent ``hash()``."""

    payload = {
        "canonical_smiles": str(canonical_smiles),
        "molecule_id": str(molecule_id),
        "selection_seed": int(selection_seed),
        "selection_version": PILOT_SELECTION_VERSION,
        "target": int(target),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def select_training_pilot(
    training_frame: pd.DataFrame,
    pilot_size: int = DEFAULT_PILOT_SIZE,
    *,
    selection_seed: int = PILOT_SELECTION_SEED,
) -> pd.DataFrame:
    """Select rows by stable composite hash, independent of source row order."""

    if pilot_size < 1:
        raise ValueError("Pilot size must be positive.")
    missing = sorted(set((*SELECTION_COLUMNS, "split")) - set(training_frame.columns))
    if missing:
        raise ValueError(f"Training pilot input is missing columns: {missing}")
    observed_splits = set(training_frame["split"].astype(str).str.strip().str.lower())
    if observed_splits != {"train"}:
        raise ValueError(
            f"Training-only pilot received non-training split labels: {sorted(observed_splits)}"
        )

    selected = training_frame.loc[:, [*SELECTION_COLUMNS, "split"]].copy()
    selected["selection_hash"] = [
        stable_selection_hash(
            row.molecule_id,
            row.canonical_smiles,
            row.target,
            selection_seed=selection_seed,
        )
        for row in selected.itertuples(index=False)
    ]
    selected = selected.sort_values(
        ["selection_hash", "molecule_id", "canonical_smiles", "target"],
        kind="mergesort",
    ).head(pilot_size)
    selected = selected.reset_index(drop=True)
    selected.insert(0, "pilot_index", np.arange(1, len(selected) + 1, dtype=np.int64))
    return selected


def run_geometry_pilot(
    config_path: str | Path,
    *,
    pilot_size: int = DEFAULT_PILOT_SIZE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    standardization_function: StandardizationFunction = standardize_for_gmc_geometry,
    geometry_function: GeometryFunction = generate_deterministic_geometry,
    ggl_function: GGLFunction = compute_ggl_features,
) -> dict[str, Any]:
    """Load only training, run the pilot, and write compact audit artifacts."""

    requested_output = Path(output_dir)
    _check_output_target(requested_output, overwrite=overwrite)
    if pilot_size < 1:
        raise ValueError("Pilot size must be positive.")
    started = time.perf_counter()
    _reject_test_named_config(config_path)
    config = load_chemprop_config(config_path)
    _validate_training_only_config(config)
    train_path = config.prepared_root / config.split_files["train"]
    geometry_config = GeometryConfig()
    ggl_config = GGLConfig()

    with _deny_nontraining_artifact_access(config) as access_guard:
        training = load_bbb_development_split(train_path, split="train")
        selected = select_training_pilot(training, pilot_size)
        records = _process_selected_rows(
            selected,
            geometry_config=geometry_config,
            ggl_config=ggl_config,
            standardization_function=standardization_function,
            geometry_function=geometry_function,
            ggl_function=ggl_function,
        )

    total_elapsed = time.perf_counter() - started
    summary = _build_summary(
        config=config,
        requested_pilot_size=pilot_size,
        records=records,
        total_elapsed_seconds=total_elapsed,
        geometry_config=geometry_config,
        ggl_config=ggl_config,
        access_guard=access_guard,
    )
    selection_manifest = _selection_manifest(config, pilot_size, selected)
    _write_outputs(
        requested_output,
        records=records,
        summary=summary,
        selection_manifest=selection_manifest,
        overwrite=overwrite,
    )
    return summary


def _process_selected_rows(
    selected: pd.DataFrame,
    *,
    geometry_config: GeometryConfig,
    ggl_config: GGLConfig,
    standardization_function: StandardizationFunction,
    geometry_function: GeometryFunction,
    ggl_function: GGLFunction,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        identity = {
            "pilot_index": int(row.pilot_index),
            "molecule_id": str(row.molecule_id),
            "canonical_smiles": str(row.canonical_smiles),
            "target": int(row.target),
            "source_split": "train",
            "selection_hash": str(row.selection_hash),
        }
        molecule_started = time.perf_counter()
        geometry_elapsed = 0.0
        ggl_elapsed = 0.0
        provenance = _empty_standardization_provenance()
        try:
            standardized = standardization_function(
                identity["molecule_id"], identity["canonical_smiles"]
            )
            provenance = _standardization_provenance(standardized)
            if standardized.action == EXCLUDED_BY_POLICY:
                record = {
                    **identity,
                    **provenance,
                    "status": "excluded",
                    "failure_category": "excluded_by_policy",
                    "failure_message": standardized.exclusion_reason,
                    **_empty_geometry_ggl_fields(),
                }
                record["geometry_time_seconds"] = geometry_elapsed
                record["ggl_time_seconds"] = ggl_elapsed
                record["total_preprocessing_time_seconds"] = time.perf_counter() - molecule_started
                records.append({column: record[column] for column in STATUS_COLUMNS})
                continue
            geometry_smiles = standardized.geometry_canonical_smiles
            if geometry_smiles is None:  # pragma: no cover - defensive policy boundary
                raise RuntimeError("Non-excluded standardization produced no geometry SMILES.")
            geometry_started = time.perf_counter()
            geometry = geometry_function(geometry_smiles, config=geometry_config)
            geometry_elapsed = time.perf_counter() - geometry_started
            if geometry.canonical_isomeric_smiles != geometry_smiles:
                raise GeometryError(
                    "atom_alignment_failed",
                    "Geometry canonical identity differs from the standardized geometry identity.",
                )
            ggl_started = time.perf_counter()
            ggl = ggl_function(
                geometry.coordinates,
                np.asarray(geometry.heavy_atom_atomic_numbers, dtype=np.int64),
                geometry_fingerprint=geometry.geometry_fingerprint,
                config=ggl_config,
            )
            ggl_elapsed = time.perf_counter() - ggl_started
            finite = bool(np.isfinite(ggl.features).all())
            if not finite:
                raise GGLPreprocessingError(
                    "nonfinite_features", "Pilot GGL matrix contains NaN or Inf."
                )
            record = {
                **identity,
                **provenance,
                "status": "success",
                "failure_category": "",
                "failure_message": "",
                "heavy_atom_count": geometry.heavy_atom_count,
                "formal_charge_sum": int(sum(geometry.heavy_atom_formal_charges)),
                "charged_heavy_atom_count": int(
                    sum(charge != 0 for charge in geometry.heavy_atom_formal_charges)
                ),
                "generated_conformer_count": geometry.generated_conformer_count,
                "optimization_method": geometry.optimization_method,
                "selected_conformer_id": geometry.selected_conformer_id,
                "selected_energy": geometry.selected_energy,
                "effective_embedding_seed": geometry.embedding_seed,
                "geometry_fingerprint": geometry.geometry_fingerprint,
                "ggl_fingerprint": ggl.ggl_fingerprint,
                "raw_ggl_rows": int(ggl.features.shape[0]),
                "raw_ggl_columns": int(ggl.features.shape[1]),
                "all_ggl_values_finite": finite,
                "rdkit_version": geometry.rdkit_version,
            }
        except Exception as exc:  # every selected row must remain in the audit table
            record = {
                **identity,
                **provenance,
                "status": "failed",
                "failure_category": _failure_category(exc),
                "failure_message": str(exc),
                **_empty_geometry_ggl_fields(),
            }
        record["geometry_time_seconds"] = geometry_elapsed
        record["ggl_time_seconds"] = ggl_elapsed
        record["total_preprocessing_time_seconds"] = time.perf_counter() - molecule_started
        records.append({column: record[column] for column in STATUS_COLUMNS})
    return records


def _standardization_provenance(result: StandardizationResult) -> dict[str, Any]:
    return {
        "standardization_action": result.action,
        "standardization_version": result.standardization_version,
        "source_fragment_count": result.fragment_count,
        "source_heavy_atom_count": result.source_heavy_atom_count,
        "source_formal_charge": result.source_formal_charge,
        "geometry_smiles": result.geometry_canonical_smiles or "",
        "parent_heavy_atom_count": result.parent_heavy_atom_count,
        "parent_formal_charge": result.parent_formal_charge,
        "removed_fragment_smiles": _serialize_list(result.removed_fragment_smiles),
        "removed_fragment_heavy_atom_counts": _serialize_list(
            result.removed_fragment_heavy_atom_counts
        ),
        "removed_fragment_formal_charges": _serialize_list(result.removed_fragment_formal_charges),
        "exclusion_reason": result.exclusion_reason,
    }


def _empty_standardization_provenance() -> dict[str, Any]:
    return {
        "standardization_action": "",
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "source_fragment_count": None,
        "source_heavy_atom_count": None,
        "source_formal_charge": None,
        "geometry_smiles": "",
        "parent_heavy_atom_count": None,
        "parent_formal_charge": None,
        "removed_fragment_smiles": "[]",
        "removed_fragment_heavy_atom_counts": "[]",
        "removed_fragment_formal_charges": "[]",
        "exclusion_reason": "",
    }


def _empty_geometry_ggl_fields() -> dict[str, Any]:
    return {
        "heavy_atom_count": None,
        "formal_charge_sum": None,
        "charged_heavy_atom_count": None,
        "generated_conformer_count": None,
        "optimization_method": "",
        "selected_conformer_id": None,
        "selected_energy": None,
        "effective_embedding_seed": None,
        "geometry_fingerprint": "",
        "ggl_fingerprint": "",
        "raw_ggl_rows": None,
        "raw_ggl_columns": None,
        "all_ggl_values_finite": False,
        "rdkit_version": rdBase.rdkitVersion,
    }


def _serialize_list(values: Sequence[object]) -> str:
    return json.dumps(list(values), separators=(",", ":"), allow_nan=False)


def _failure_category(error: Exception) -> str:
    status = getattr(error, "status", "")
    if status == "invalid_smiles":
        return "invalid_smiles"
    if status == "disconnected_fragment":
        return "disconnected_fragment"
    if status in {"embedding_failed", "etkdgv3_unavailable"}:
        return "embedding_failed"
    if status in {"no_conformers_generated", "no_conformers"}:
        return "no_conformer"
    if status in {"optimization_failed", "mmff_unavailable", "uff_unavailable"}:
        return "optimization_failed"
    if status == "atom_alignment_failed":
        return "atom_alignment_failed"
    if status == "unsupported_element":
        return "unsupported_element"
    if status in {
        "invalid_coordinate_shape",
        "nonfinite_coordinates",
        "atom_count_mismatch",
        "invalid_atomic_number",
        "hydrogen_not_allowed",
    }:
        return "invalid_coordinates"
    if status == "nonfinite_features":
        return "nonfinite_ggl"
    return "unexpected_error"


def _build_summary(
    *,
    config: ChempropExperimentConfig,
    requested_pilot_size: int,
    records: list[dict[str, Any]],
    total_elapsed_seconds: float,
    geometry_config: GeometryConfig,
    ggl_config: GGLConfig,
    access_guard: ArtifactAccessGuard,
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
    optimization_counts = Counter(str(record["optimization_method"]) for record in successes)
    action_counter = Counter(
        str(record["standardization_action"])
        for record in records
        if record["standardization_action"]
    )
    action_counts = {
        action: int(action_counter.get(action, 0))
        for action in (UNCHANGED, PARENT_SELECTED, EXCLUDED_BY_POLICY)
    }
    total_times = [float(record["total_preprocessing_time_seconds"]) for record in records]
    heavy_counts = [int(record["heavy_atom_count"]) for record in successes]
    selected_count = len(records)
    return {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "split": "train",
        "loaded_split": "train",
        "requested_pilot_size": requested_pilot_size,
        "selected_pilot_size": selected_count,
        "successful_molecule_count": len(successes),
        "failed_molecule_count": len(failures),
        "policy_excluded_molecule_count": len(exclusions),
        "success_fraction": len(successes) / selected_count if selected_count else 0.0,
        "counts_by_failure_category": failure_counts,
        "counts_by_standardization_action": action_counts,
        "parent_standardized_molecule_count": action_counts[PARENT_SELECTED],
        "standardization_version": GMC_STANDARDIZATION_VERSION,
        "counts_by_optimization_method": dict(sorted(optimization_counts.items())),
        "total_elapsed_seconds": total_elapsed_seconds,
        "mean_seconds_per_selected_molecule": statistics.fmean(total_times)
        if total_times
        else None,
        "median_seconds_per_selected_molecule": statistics.median(total_times)
        if total_times
        else None,
        "mean_geometry_seconds_among_successes": (
            statistics.fmean(float(record["geometry_time_seconds"]) for record in successes)
            if successes
            else None
        ),
        "mean_ggl_seconds_among_successes": (
            statistics.fmean(float(record["ggl_time_seconds"]) for record in successes)
            if successes
            else None
        ),
        "minimum_heavy_atom_count": min(heavy_counts) if heavy_counts else None,
        "maximum_heavy_atom_count": max(heavy_counts) if heavy_counts else None,
        "finite_ggl_molecule_count": sum(
            bool(record["all_ggl_values_finite"]) for record in records
        ),
        "selection_method": "SHA-256 over molecule_id, canonical_smiles, target, seed, and version; sort ascending",
        "selection_seed": PILOT_SELECTION_SEED,
        "selection_version": PILOT_SELECTION_VERSION,
        "geometry_settings": {
            **asdict(geometry_config),
            "etkdg_version": "ETKDGv3",
            "optimization_primary": "MMFF94s",
            "optimization_fallback": "UFF only when MMFF94s parameters are unavailable",
            "selection_rule": "lowest finite-energy converged conformer; conformer-ID tie break",
        },
        "ggl_settings": {
            **asdict(ggl_config),
            "feature_order": list(GGL_FEATURE_NAMES),
            "preprocessing_version": GGL_PREPROCESSING_VERSION,
            "element_radius_mapping_version": ELEMENT_RADIUS_MAPPING_VERSION,
            "scaled": False,
        },
        "rdkit_version": rdBase.rdkitVersion,
        "git_commit": _git_commit(),
        "status_columns": list(STATUS_COLUMNS),
        "validation_artifact_accessed": access_guard.validation_artifact_accessed,
        "test_artifact_accessed": access_guard.test_artifact_accessed,
    }


def _selection_manifest(
    config: ChempropExperimentConfig,
    requested_pilot_size: int,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    identities = [
        {
            "pilot_index": int(row.pilot_index),
            "molecule_id": str(row.molecule_id),
            "canonical_smiles": str(row.canonical_smiles),
            "target": int(row.target),
            "source_split": "train",
            "selection_hash": str(row.selection_hash),
        }
        for row in selected.itertuples(index=False)
    ]
    return {
        "dataset": config.dataset,
        "loaded_split": "train",
        "requested_pilot_size": requested_pilot_size,
        "selected_pilot_size": len(identities),
        "selection_seed": PILOT_SELECTION_SEED,
        "selection_version": PILOT_SELECTION_VERSION,
        "molecules": identities,
    }


def _check_output_target(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Pilot output path is not a directory: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Pilot output directory already exists: {output_dir}. Use --overwrite explicitly."
        )
    if overwrite:
        resolved = output_dir.resolve()
        forbidden = {Path(resolved.anchor), ROOT.resolve(), (ROOT / "outputs").resolve()}
        if resolved in forbidden:
            raise ValueError("Refusing to overwrite a filesystem, repository, or outputs root.")


def _write_outputs(
    output_dir: Path,
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    selection_manifest: dict[str, Any],
    overwrite: bool,
) -> None:
    # Re-check immediately before writing. With --overwrite only these exact known
    # files are replaced; no directory or unrelated output is deleted.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not overwrite:
        raise FileExistsError(f"Pilot output directory appeared during execution: {output_dir}")
    pd.DataFrame(records, columns=STATUS_COLUMNS).to_csv(
        output_dir / "molecule_status.csv", index=False
    )
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pilot_selection.json").write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_training_only_config(config: ChempropExperimentConfig) -> None:
    if config.endpoint_id != "bbb_martins" or config.task_type != "binary_classification":
        raise ValueError("Geometry pilot supports only binary BBB_Martins.")
    names = {split: config.split_files.get(split) for split in ("train", "validation", "test")}
    if any(not name for name in names.values()):
        raise NonTrainingArtifactAccessError(
            "Config must identify train, validation, and test files."
        )
    if len(set(names.values())) != 3:
        raise NonTrainingArtifactAccessError(
            "Train, validation, and locked-test artifacts must be distinct."
        )


def _reject_test_named_config(config_path: str | Path) -> None:
    if Path(config_path).name.lower() in {"test.csv", "locked.csv"}:
        raise NonTrainingArtifactAccessError("A test artifact cannot be used as pilot config.")


@contextmanager
def _deny_nontraining_artifact_access(
    config: ChempropExperimentConfig,
) -> Iterator[ArtifactAccessGuard]:
    """Block common inspection paths for validation and locked-test artifacts."""

    guard = ArtifactAccessGuard(
        validation_path=Path(
            os.path.abspath(config.prepared_root / config.split_files["validation"])
        ),
        test_path=Path(os.path.abspath(config.prepared_root / config.split_files["test"])),
    )
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_path_open = Path.open
    original_resolve = Path.resolve
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir
    original_stat = Path.stat
    original_builtin_open = builtins.open
    original_read_csv = pd.read_csv

    def guarded_read_bytes(path: Path) -> bytes:
        guard.reject(path)
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        guard.reject(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        guard.reject(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        guard.reject(path)
        return original_resolve(path, *args, **kwargs)

    def guarded_exists(path: Path) -> bool:
        guard.reject(path)
        return original_exists(path)

    def guarded_is_file(path: Path) -> bool:
        guard.reject(path)
        return original_is_file(path)

    def guarded_is_dir(path: Path) -> bool:
        guard.reject(path)
        return original_is_dir(path)

    def guarded_stat(path: Path, *args: Any, **kwargs: Any):
        guard.reject(path)
        return original_stat(path, *args, **kwargs)

    def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any):
        guard.reject(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_read_csv(source: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        guard.reject(source)
        return original_read_csv(source, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Path, "read_bytes", guarded_read_bytes))
        stack.enter_context(patch.object(Path, "read_text", guarded_read_text))
        stack.enter_context(patch.object(Path, "open", guarded_path_open))
        stack.enter_context(patch.object(Path, "resolve", guarded_resolve))
        stack.enter_context(patch.object(Path, "exists", guarded_exists))
        stack.enter_context(patch.object(Path, "is_file", guarded_is_file))
        stack.enter_context(patch.object(Path, "is_dir", guarded_is_dir))
        stack.enter_context(patch.object(Path, "stat", guarded_stat))
        stack.enter_context(patch.object(builtins, "open", guarded_builtin_open))
        stack.enter_context(patch.object(pd, "read_csv", guarded_read_csv))
        yield guard


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_geometry_pilot(
        args.config,
        pilot_size=args.pilot_size,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
