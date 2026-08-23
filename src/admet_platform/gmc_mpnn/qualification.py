"""Fixed release qualification for GMC-MPNN BBB production inference."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

import numpy as np
from rdkit import Chem

from admet_platform.gmc_mpnn.inference import (
    FAILED,
    OUTPUT_FIELDS,
    SUCCESS,
    GMCInferenceInput,
    GMCProductionPredictor,
)
from admet_platform.gmc_mpnn.production_manifest import (
    PRODUCTION_SEEDS,
    PRODUCTION_THRESHOLD,
)
from admet_platform.gmc_mpnn.standardization import standardize_for_gmc_geometry


QUALIFICATION_SCHEMA_VERSION: Final = "gmc-mpnn-bbb-production-qualification-schema-v1"
QUALIFICATION_VERSION: Final = "gmc-mpnn-bbb-production-qualification-v1"
DETERMINISM_ABSOLUTE_TOLERANCE: Final = 0.0
DETERMINISM_RELATIVE_TOLERANCE: Final = 0.0
QUALIFICATION_INPUT_FIELDS: Final = (
    "molecule_id",
    "source_smiles",
    "expected_status",
    "expected_error_code",
    "expected_preprocessing_status",
    "expected_parent_standardization_status",
    "expected_parent_smiles",
    "qualification_categories",
)
REQUIRED_QUALIFICATION_CATEGORIES: Final = frozenset(
    {
        "ordinary_connected_neutral",
        "chemical_diversity",
        "simple_alcohol_small_polar",
        "aromatic",
        "heterocycle",
        "larger_drug_like",
        "salt_multicomponent_parent_selection",
        "disconnected_parent_policy",
        "challenging_3d_geometry",
        "invalid_smiles",
        "empty_input",
        "unsupported_disconnected",
        "duplicate_structure",
    }
)
OUTPUT_FILENAMES: Final = {
    "input": "qualification_input.csv",
    "run1": "qualification_predictions_run1.csv",
    "run2": "qualification_predictions_run2.csv",
    "summary": "qualification_summary.json",
    "checksums": "qualification_sha256.txt",
}
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")


class GMCQualificationError(RuntimeError):
    """A fixed-input qualification contract failure."""


@dataclass(frozen=True)
class QualificationInput:
    """One fixed qualification record and its expected software behavior."""

    molecule_id: str
    source_smiles: str
    expected_status: str
    expected_error_code: str
    expected_preprocessing_status: str
    expected_parent_standardization_status: str
    expected_parent_smiles: str
    qualification_categories: tuple[str, ...]


class QualificationPredictor(Protocol):
    manifest: Mapping[str, Any]

    def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
        """Return ordered production predictions."""


PredictorFactory = Callable[..., QualificationPredictor]


def load_qualification_inputs(path: str | Path) -> tuple[list[QualificationInput], bytes]:
    """Load and fail closed on the fixed qualification input schema."""

    source = Path(path)
    try:
        payload = source.read_bytes()
        text = payload.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise GMCQualificationError(f"Cannot read qualification input: {source}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != QUALIFICATION_INPUT_FIELDS:
        raise GMCQualificationError("Qualification input CSV schema is incompatible.")
    records: list[QualificationInput] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise GMCQualificationError(f"Qualification row {row_number} has extra columns.")
        records.append(_qualification_input_from_row(row, row_number))
    if not 12 <= len(records) <= 20:
        raise GMCQualificationError("Qualification input must contain 12 to 20 records.")
    identifiers = [record.molecule_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise GMCQualificationError("Qualification molecule IDs must be unique.")
    categories = {category for record in records for category in record.qualification_categories}
    missing = REQUIRED_QUALIFICATION_CATEGORIES.difference(categories)
    if missing:
        raise GMCQualificationError(f"Qualification categories are missing: {sorted(missing)}")
    smiles_to_ids: dict[str, list[str]] = {}
    for record in records:
        if record.source_smiles:
            smiles_to_ids.setdefault(record.source_smiles, []).append(record.molecule_id)
    if not any(len(ids) > 1 for ids in smiles_to_ids.values()):
        raise GMCQualificationError("Qualification input must include duplicate structures.")
    return records, payload


def run_production_qualification(
    *,
    manifest_path: str | Path,
    artifact_root: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    git_commit: str,
    num_workers: int = 0,
    predictor_factory: PredictorFactory = GMCProductionPredictor,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and freeze two exact-repeat qualification passes without overwriting outputs."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Qualification output directory already exists: {destination}")
    if not _HEX_40.fullmatch(git_commit):
        raise GMCQualificationError("Qualification Git commit must be a lowercase 40-hex SHA.")

    records, input_bytes = load_qualification_inputs(input_path)
    manifest_file = Path(manifest_path)
    try:
        manifest_sha256 = _sha256_bytes(manifest_file.read_bytes())
    except OSError as exc:
        raise GMCQualificationError(f"Cannot hash production manifest: {manifest_file}") from exc

    predictor = predictor_factory(
        manifest_file,
        artifact_root=artifact_root,
        verify_runtime=True,
        num_workers=num_workers,
    )
    manifest = predictor.manifest
    inference_inputs = [
        GMCInferenceInput(record.molecule_id, record.source_smiles) for record in records
    ]
    run1 = predictor.predict_batch(inference_inputs)
    run2 = predictor.predict_batch(inference_inputs)
    parent_cases = _validate_qualification_results(records, run1, run2)

    run1_bytes = _predictions_csv_bytes(run1)
    run2_bytes = _predictions_csv_bytes(run2)
    artifact_hashes = {
        OUTPUT_FILENAMES["input"]: _sha256_bytes(input_bytes),
        OUTPUT_FILENAMES["run1"]: _sha256_bytes(run1_bytes),
        OUTPUT_FILENAMES["run2"]: _sha256_bytes(run2_bytes),
    }
    observed_runtime = dict(runtime_provenance or collect_runtime_provenance(manifest))
    summary = _build_summary(
        records=records,
        run1=run1,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        git_commit=git_commit,
        runtime_provenance=observed_runtime,
        artifact_hashes=artifact_hashes,
        parent_cases=parent_cases,
    )
    summary_bytes = _json_bytes(summary)
    checksum_records = {
        **artifact_hashes,
        OUTPUT_FILENAMES["summary"]: _sha256_bytes(summary_bytes),
    }
    checksum_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in checksum_records.items()
    ).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    (destination / OUTPUT_FILENAMES["input"]).write_bytes(input_bytes)
    (destination / OUTPUT_FILENAMES["run1"]).write_bytes(run1_bytes)
    (destination / OUTPUT_FILENAMES["run2"]).write_bytes(run2_bytes)
    (destination / OUTPUT_FILENAMES["summary"]).write_bytes(summary_bytes)
    (destination / OUTPUT_FILENAMES["checksums"]).write_bytes(checksum_bytes)
    _verify_written_hashes(destination, checksum_records)
    return summary


def collect_runtime_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Record the package/runtime identity already required by the manifest validator."""

    expected_packages = manifest.get("environment", {}).get("packages", {})
    if not isinstance(expected_packages, Mapping):
        raise GMCQualificationError("Manifest environment package provenance is malformed.")
    packages: dict[str, str] = {}
    for name in expected_packages:
        if name == "python":
            packages[name] = platform.python_version()
            continue
        distribution = "scikit-learn" if name == "scikit_learn" else name
        try:
            packages[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise GMCQualificationError(f"Required runtime package is missing: {name}") from exc
    return {
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": packages,
    }


def _qualification_input_from_row(
    row: Mapping[str, str | None], row_number: int
) -> QualificationInput:
    values = {key: (row.get(key) or "").strip() for key in QUALIFICATION_INPUT_FIELDS}
    molecule_id = values["molecule_id"]
    if not molecule_id:
        raise GMCQualificationError(f"Qualification row {row_number} has no molecule_id.")
    expected_status = values["expected_status"]
    if expected_status not in {SUCCESS, FAILED}:
        raise GMCQualificationError(f"Qualification row {row_number} has invalid expected_status.")
    error_code = values["expected_error_code"]
    if (expected_status == SUCCESS and error_code) or (
        expected_status == FAILED and not error_code
    ):
        raise GMCQualificationError(
            f"Qualification row {row_number} has inconsistent expected_error_code."
        )
    preprocessing_status = values["expected_preprocessing_status"]
    if preprocessing_status not in {SUCCESS, FAILED}:
        raise GMCQualificationError(
            f"Qualification row {row_number} has invalid expected preprocessing status."
        )
    parent_status = values["expected_parent_standardization_status"]
    if parent_status not in {"unchanged", "parent_selected", "not_started"}:
        raise GMCQualificationError(
            f"Qualification row {row_number} has invalid parent standardization status."
        )
    parent_smiles = values["expected_parent_smiles"]
    if (parent_status == "parent_selected") != bool(parent_smiles):
        raise GMCQualificationError(
            f"Qualification row {row_number} has inconsistent expected parent SMILES."
        )
    categories = tuple(
        category.strip()
        for category in values["qualification_categories"].split(";")
        if category.strip()
    )
    if not categories or len(set(categories)) != len(categories):
        raise GMCQualificationError(f"Qualification row {row_number} has invalid categories.")
    return QualificationInput(
        molecule_id=molecule_id,
        source_smiles=values["source_smiles"],
        expected_status=expected_status,
        expected_error_code=error_code,
        expected_preprocessing_status=preprocessing_status,
        expected_parent_standardization_status=parent_status,
        expected_parent_smiles=parent_smiles,
        qualification_categories=categories,
    )


def _validate_qualification_results(
    records: Sequence[QualificationInput],
    run1: Sequence[Mapping[str, Any]],
    run2: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if len(run1) != len(records) or len(run2) != len(records):
        raise GMCQualificationError("Prediction count differs from qualification input count.")
    expected_ids = [record.molecule_id for record in records]
    if [row.get("molecule_id") for row in run1] != expected_ids or [
        row.get("molecule_id") for row in run2
    ] != expected_ids:
        raise GMCQualificationError("Prediction output ordering or molecule identity changed.")
    if any(tuple(row) != OUTPUT_FIELDS for row in (*run1, *run2)):
        raise GMCQualificationError("Prediction output schema is incompatible.")

    parent_cases: list[dict[str, str]] = []
    for record, first, second in zip(records, run1, run2, strict=True):
        _validate_expected_behavior(record, first)
        _validate_expected_behavior(record, second)
        _validate_repeat(record.molecule_id, first, second)
        if record.expected_status == SUCCESS:
            _validate_five_seed_contract(record.molecule_id, first)
            _validate_five_seed_contract(record.molecule_id, second)
        if record.expected_parent_standardization_status == "parent_selected":
            source_molecule = Chem.MolFromSmiles(record.source_smiles)
            if source_molecule is None:
                raise GMCQualificationError(
                    f"{record.molecule_id}: parent case source SMILES is invalid."
                )
            source_canonical_smiles = Chem.MolToSmiles(
                source_molecule, canonical=True, isomericSmiles=True
            )
            standardized = standardize_for_gmc_geometry(record.molecule_id, source_canonical_smiles)
            if first.get("canonical_smiles") not in {None, source_canonical_smiles}:
                raise GMCQualificationError(
                    f"{record.molecule_id}: source canonical SMILES differs from inference output."
                )
            if standardized.geometry_canonical_smiles != record.expected_parent_smiles:
                raise GMCQualificationError(
                    f"{record.molecule_id}: selected parent SMILES differs from expectation."
                )
            parent_cases.append(
                {
                    "molecule_id": record.molecule_id,
                    "source_smiles": record.source_smiles,
                    "source_canonical_smiles": source_canonical_smiles,
                    "selected_parent_smiles": str(standardized.geometry_canonical_smiles),
                    "parent_standardization_status": str(first["parent_standardization_status"]),
                }
            )
    _validate_duplicate_structure_contract(records, run1)
    _validate_duplicate_structure_contract(records, run2)
    return parent_cases


def _validate_expected_behavior(record: QualificationInput, result: Mapping[str, Any]) -> None:
    if result.get("source_smiles") != record.source_smiles:
        raise GMCQualificationError(f"{record.molecule_id}: source SMILES identity changed.")
    if result.get("status") != record.expected_status:
        raise GMCQualificationError(f"{record.molecule_id}: unexpected inference status.")
    if result.get("preprocessing_status") != record.expected_preprocessing_status:
        raise GMCQualificationError(f"{record.molecule_id}: unexpected preprocessing status.")
    if result.get("parent_standardization_status") != record.expected_parent_standardization_status:
        raise GMCQualificationError(
            f"{record.molecule_id}: unexpected parent standardization status."
        )
    if record.expected_status == FAILED:
        if result.get("error_code") != record.expected_error_code:
            raise GMCQualificationError(f"{record.molecule_id}: unstable or unexpected error code.")
        absent = [f"seed{seed}_probability" for seed in PRODUCTION_SEEDS]
        absent.extend(["ensemble_probability", "ensemble_standard_deviation", "prediction"])
        if any(result.get(field) is not None for field in absent):
            raise GMCQualificationError(
                f"{record.molecule_id}: failed input received fabricated prediction values."
            )
    elif result.get("error_code") is not None:
        raise GMCQualificationError(f"{record.molecule_id}: successful result contains an error.")
    elif not isinstance(result.get("canonical_smiles"), str) or not result["canonical_smiles"]:
        raise GMCQualificationError(
            f"{record.molecule_id}: successful result has no canonical SMILES."
        )


def _validate_five_seed_contract(molecule_id: str, result: Mapping[str, Any]) -> None:
    probabilities: list[float] = []
    for seed in PRODUCTION_SEEDS:
        value = result.get(f"seed{seed}_probability")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise GMCQualificationError(f"{molecule_id}: seed {seed} probability is not numeric.")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise GMCQualificationError(f"{molecule_id}: seed {seed} probability is outside [0,1].")
        probabilities.append(probability)
    values = np.asarray(probabilities, dtype=np.float64)
    expected_mean = float(values.mean(dtype=np.float64))
    expected_sd = float(values.std(ddof=0, dtype=np.float64))
    if result.get("ensemble_probability") != expected_mean:
        raise GMCQualificationError(f"{molecule_id}: ensemble is not the exact five-seed mean.")
    if result.get("ensemble_standard_deviation") != expected_sd:
        raise GMCQualificationError(f"{molecule_id}: ensemble SD is not population SD (ddof=0).")
    if result.get("threshold") != PRODUCTION_THRESHOLD:
        raise GMCQualificationError(f"{molecule_id}: production threshold changed.")
    expected_class = "BBB+" if expected_mean >= PRODUCTION_THRESHOLD else "BBB-"
    if result.get("prediction") != expected_class:
        raise GMCQualificationError(f"{molecule_id}: threshold/class contract failed.")


def _validate_repeat(molecule_id: str, first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    exact_fields = (
        "canonical_smiles",
        "preprocessing_status",
        "parent_standardization_status",
        "status",
        "error_code",
        "prediction",
    )
    for field in exact_fields:
        if first.get(field) != second.get(field):
            raise GMCQualificationError(f"{molecule_id}: repeat mismatch in {field}.")
    numeric_fields = tuple(f"seed{seed}_probability" for seed in PRODUCTION_SEEDS) + (
        "ensemble_probability",
        "ensemble_standard_deviation",
    )
    for field in numeric_fields:
        left = first.get(field)
        right = second.get(field)
        if left is None or right is None:
            if left is not right:
                raise GMCQualificationError(f"{molecule_id}: repeat mismatch in {field}.")
            continue
        if not math.isclose(
            float(left),
            float(right),
            rel_tol=DETERMINISM_RELATIVE_TOLERANCE,
            abs_tol=DETERMINISM_ABSOLUTE_TOLERANCE,
        ):
            raise GMCQualificationError(f"{molecule_id}: repeat mismatch in {field}.")


def _validate_duplicate_structure_contract(
    records: Sequence[QualificationInput], results: Sequence[Mapping[str, Any]]
) -> None:
    duplicate_indices = [
        index
        for index, record in enumerate(records)
        if "duplicate_structure" in record.qualification_categories
    ]
    if len(duplicate_indices) < 2:
        raise GMCQualificationError("Qualification duplicate-structure cases are missing.")
    reference = results[duplicate_indices[0]]
    scientific_fields = (
        "canonical_smiles",
        "preprocessing_status",
        "parent_standardization_status",
        *(f"seed{seed}_probability" for seed in PRODUCTION_SEEDS),
        "ensemble_probability",
        "ensemble_standard_deviation",
        "threshold",
        "prediction",
        "status",
    )
    for index in duplicate_indices[1:]:
        candidate = results[index]
        if any(reference.get(field) != candidate.get(field) for field in scientific_fields):
            raise GMCQualificationError(
                "Duplicate structures produced different scientific prediction outputs."
            )


def _build_summary(
    *,
    records: Sequence[QualificationInput],
    run1: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    git_commit: str,
    runtime_provenance: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    parent_cases: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    successes = sum(row.get("status") == SUCCESS for row in run1)
    observed_classes = sorted(
        {str(row["prediction"]) for row in run1 if row.get("status") == SUCCESS}
    )
    checkpoint_identities = [
        {
            "seed": record.get("seed"),
            "path": record.get("path"),
            "sha256": record.get("sha256"),
        }
        for record in manifest.get("checkpoints", [])
    ]
    return {
        "qualification_schema_version": QUALIFICATION_SCHEMA_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "qualification_type": "software_and_production_inference_behavior_not_predictive_validation",
        "release_status": "production_inference_qualification_passed",
        "manifest": {
            "version": manifest.get("manifest_version"),
            "sha256": manifest_sha256,
            "frozen_manifest_git_commit": manifest.get("git_commit"),
        },
        "qualification_git_commit": git_commit,
        "production_contract": {
            "seeds": list(PRODUCTION_SEEDS),
            "ensemble_rule": "unweighted_arithmetic_mean",
            "standard_deviation_ddof": 0,
            "threshold": PRODUCTION_THRESHOLD,
            "classification_rule": "BBB+ iff ensemble_probability >= 0.5",
            "checkpoints": checkpoint_identities,
        },
        "deterministic_repeat": {
            "run_count": 2,
            "comparison": "exact_numeric_equality",
            "absolute_tolerance": DETERMINISM_ABSOLUTE_TOLERANCE,
            "relative_tolerance": DETERMINISM_RELATIVE_TOLERANCE,
            "passed": True,
        },
        "counts": {
            "input_records": len(records),
            "successful_records": successes,
            "failed_records": len(records) - successes,
        },
        "observed_prediction_classes": observed_classes,
        "behavior_verification": {
            "input_order_and_identity_preserved": True,
            "expected_success_and_failure_behavior": True,
            "five_seed_contract": True,
            "parent_fragment_policy": True,
            "failed_records_have_no_prediction_values": True,
            "remaining_batch_survives_failed_records": True,
            "duplicate_structure_identity_and_prediction": True,
            "model_training_performed": False,
            "scaler_fitting_performed": False,
            "calibration_performed": False,
        },
        "parent_fragment_cases": list(parent_cases),
        "artifact_sha256": dict(artifact_hashes),
        "runtime_provenance": {
            "manifest_expected_packages": dict(manifest.get("environment", {}).get("packages", {})),
            "observed": dict(runtime_provenance),
        },
        "artifact_access": {
            "production_manifest": True,
            "frozen_scaler_artifacts": True,
            "environment_provenance": True,
            "production_checkpoints": list(PRODUCTION_SEEDS),
            "external_validation_artifacts": False,
            "calibration_fitting_artifacts": False,
            "oof_prediction_artifacts": False,
            "locked_test_artifacts": False,
        },
        "external_predictive_validation": {
            "performed": False,
            "qualification_inputs_have_biological_ground_truth_labels": False,
            "purpose": "software_and_model_release_qualification_only",
        },
        "test_artifact_accessed": False,
    }


def _predictions_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(OUTPUT_FIELDS), lineterminator="\n")
    writer.writeheader()
    try:
        writer.writerows(rows)
    except (ValueError, TypeError) as exc:
        raise GMCQualificationError("Prediction output schema is incompatible.") from exc
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_written_hashes(directory: Path, expected: Mapping[str, str]) -> None:
    for filename, expected_digest in expected.items():
        try:
            observed_digest = _sha256_bytes((directory / filename).read_bytes())
        except OSError as exc:
            raise GMCQualificationError(
                f"Cannot verify frozen qualification artifact: {filename}"
            ) from exc
        if observed_digest != expected_digest:
            raise GMCQualificationError(
                f"Frozen qualification artifact SHA-256 mismatch: {filename}"
            )


__all__ = [
    "DETERMINISM_ABSOLUTE_TOLERANCE",
    "DETERMINISM_RELATIVE_TOLERANCE",
    "GMCQualificationError",
    "OUTPUT_FILENAMES",
    "QUALIFICATION_INPUT_FIELDS",
    "QUALIFICATION_SCHEMA_VERSION",
    "QUALIFICATION_VERSION",
    "QualificationInput",
    "collect_runtime_provenance",
    "load_qualification_inputs",
    "run_production_qualification",
]
