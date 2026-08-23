from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from rdkit import Chem

from admet_platform.gmc_mpnn.inference import OUTPUT_FIELDS, GMCInferenceInput
from admet_platform.gmc_mpnn.production_manifest import (
    MANIFEST_VERSION,
    PRODUCTION_SEEDS,
)
from admet_platform.gmc_mpnn.qualification import (
    DETERMINISM_ABSOLUTE_TOLERANCE,
    DETERMINISM_RELATIVE_TOLERANCE,
    GMCQualificationError,
    OUTPUT_FILENAMES,
    QUALIFICATION_INPUT_FIELDS,
    QUALIFICATION_SCHEMA_VERSION,
    load_qualification_inputs,
    run_production_qualification,
)
from admet_platform.gmc_mpnn.standardization import standardize_for_gmc_geometry


ROOT = Path(__file__).resolve().parents[2]
FIXED_INPUT = ROOT / "configs" / "gmc_mpnn_bbb_qualification_input.csv"
GIT_COMMIT = "a" * 40
RUNTIME = {
    "python_implementation": "CPython",
    "platform_system": "test",
    "platform_machine": "test",
    "packages": {"python": "3.11.test"},
}


class _FakePredictor:
    calls = 0

    def __init__(self, manifest_path: Path, **kwargs: Any) -> None:
        self.manifest = {
            "manifest_version": MANIFEST_VERSION,
            "git_commit": "b" * 40,
            "environment": {"packages": {"python": "3.11.test"}},
            "checkpoints": [
                {"seed": seed, "path": f"models/seed{seed}.ckpt", "sha256": str(seed) * 64}
                for seed in PRODUCTION_SEEDS
            ],
            "source_provenance": {
                "external_validation": "missing/development/validation.json",
                "calibration": "missing/development/calibration.json",
                "oof": "missing/development/oof.csv",
                "locked_test": "missing/locked/test.csv",
            },
        }
        self._run = 0

    def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
        type(self).calls += 1
        self._run += 1
        return [_result(item, index) for index, item in enumerate(inputs)]


def test_fixed_qualification_input_schema_and_coverage() -> None:
    records, payload = load_qualification_inputs(FIXED_INPUT)

    assert payload == FIXED_INPUT.read_bytes()
    assert 12 <= len(records) <= 20
    assert len({record.molecule_id for record in records}) == len(records)
    assert {record.expected_status for record in records} == {"success", "failed"}
    assert any(
        record.expected_parent_standardization_status == "parent_selected" for record in records
    )
    duplicate_ids = [record.molecule_id for record in records if record.source_smiles == "CCO"]
    assert duplicate_ids == ["QLT_001", "QLT_017"]
    assert all("label" not in field for field in QUALIFICATION_INPUT_FIELDS)


def test_qualification_freezes_ordered_runs_summary_hashes_and_isolation(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "production_manifest.json"
    manifest.write_text('{"identity":"frozen"}\n', encoding="utf-8")
    output = tmp_path / "qualification"
    _FakePredictor.calls = 0

    summary = run_production_qualification(
        manifest_path=manifest,
        artifact_root=tmp_path,
        input_path=FIXED_INPUT,
        output_dir=output,
        git_commit=GIT_COMMIT,
        predictor_factory=_FakePredictor,
        runtime_provenance=RUNTIME,
    )

    assert _FakePredictor.calls == 2
    assert set(path.name for path in output.iterdir()) == set(OUTPUT_FILENAMES.values())
    assert summary["qualification_schema_version"] == QUALIFICATION_SCHEMA_VERSION
    assert summary["qualification_type"].endswith("not_predictive_validation")
    assert summary["deterministic_repeat"] == {
        "run_count": 2,
        "comparison": "exact_numeric_equality",
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "passed": True,
    }
    assert summary["counts"] == {
        "input_records": 17,
        "successful_records": 14,
        "failed_records": 3,
    }
    assert summary["observed_prediction_classes"] == ["BBB+", "BBB-"]
    assert summary["test_artifact_accessed"] is False
    assert summary["artifact_access"]["external_validation_artifacts"] is False
    assert summary["artifact_access"]["calibration_fitting_artifacts"] is False
    assert summary["artifact_access"]["oof_prediction_artifacts"] is False
    assert summary["artifact_access"]["locked_test_artifacts"] is False
    assert summary["external_predictive_validation"]["performed"] is False
    assert (
        summary["external_predictive_validation"][
            "qualification_inputs_have_biological_ground_truth_labels"
        ]
        is False
    )
    assert summary["manifest"]["sha256"] == _sha256(manifest)

    run1 = _read_csv(output / OUTPUT_FILENAMES["run1"])
    run2 = _read_csv(output / OUTPUT_FILENAMES["run2"])
    assert run1 == run2
    assert [row["molecule_id"] for row in run1] == [f"QLT_{index:03d}" for index in range(1, 18)]
    assert run1[0]["molecule_id"] != run1[-1]["molecule_id"]
    assert run1[0]["canonical_smiles"] == run1[-1]["canonical_smiles"]
    assert run1[12]["status"] == "failed"
    assert run1[12]["seed13_probability"] == ""
    assert run1[13]["ensemble_probability"] == ""
    assert run1[14]["error_code"] == "geometry_no_heavy_atoms"
    assert run1[15]["status"] == "success"

    parent_cases = summary["parent_fragment_cases"]
    assert [(case["molecule_id"], case["selected_parent_smiles"]) for case in parent_cases] == [
        ("QLT_011", "CC[NH+](CC)CC"),
        ("QLT_012", "CCO"),
        ("QLT_015", "[H]"),
    ]
    for name, digest in summary["artifact_sha256"].items():
        assert digest == _sha256(output / name)
    checksum_lines = (
        (output / OUTPUT_FILENAMES["checksums"]).read_text(encoding="utf-8").splitlines()
    )
    checksums = {name: digest for digest, name in (line.split("  ", 1) for line in checksum_lines)}
    assert checksums[OUTPUT_FILENAMES["summary"]] == _sha256(output / OUTPUT_FILENAMES["summary"])


def test_success_contract_is_exact_five_seed_mean_population_sd_and_threshold() -> None:
    item = GMCInferenceInput("molecule", "CC")
    row = _result(item, 0)
    values = np.asarray(
        [float(row[f"seed{seed}_probability"]) for seed in PRODUCTION_SEEDS],
        dtype=np.float64,
    )

    assert row["ensemble_probability"] == float(values.mean(dtype=np.float64))
    assert row["ensemble_standard_deviation"] == float(values.std(ddof=0, dtype=np.float64))
    assert row["threshold"] == 0.5
    assert row["prediction"] == ("BBB+" if row["ensemble_probability"] >= 0.5 else "BBB-")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("ensemble_probability", 0.123, "exact five-seed mean"),
        ("ensemble_standard_deviation", 0.123, "population SD"),
        ("threshold", 0.51, "production threshold"),
        ("prediction", "WRONG", "threshold/class contract"),
    ],
)
def test_qualification_fails_closed_on_five_seed_contract_changes(
    tmp_path: Path, field: str, replacement: Any, message: str
) -> None:
    class BadPredictor(_FakePredictor):
        def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
            rows = super().predict_batch(inputs)
            rows[0][field] = replacement
            return rows

    with pytest.raises(GMCQualificationError, match=message):
        _run(tmp_path, BadPredictor)


def test_deterministic_repeat_rejects_any_numeric_difference(tmp_path: Path) -> None:
    class NondeterministicPredictor(_FakePredictor):
        def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
            rows = super().predict_batch(inputs)
            if self._run == 2:
                rows[0]["seed13_probability"] += 1e-15
            return rows

    assert DETERMINISM_ABSOLUTE_TOLERANCE == 0.0
    assert DETERMINISM_RELATIVE_TOLERANCE == 0.0
    with pytest.raises(GMCQualificationError, match="repeat mismatch in seed13_probability"):
        _run(tmp_path, NondeterministicPredictor)


def test_invalid_input_must_fail_without_fabricated_probabilities(tmp_path: Path) -> None:
    class FabricatingPredictor(_FakePredictor):
        def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
            rows = super().predict_batch(inputs)
            rows[12]["seed13_probability"] = 0.4
            return rows

    with pytest.raises(GMCQualificationError, match="fabricated prediction values"):
        _run(tmp_path, FabricatingPredictor)


def test_parent_selection_must_match_frozen_policy(tmp_path: Path) -> None:
    class WrongParentStatusPredictor(_FakePredictor):
        def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
            rows = super().predict_batch(inputs)
            rows[10]["parent_standardization_status"] = "unchanged"
            return rows

    with pytest.raises(GMCQualificationError, match="parent standardization status"):
        _run(tmp_path, WrongParentStatusPredictor)


def test_output_order_and_duplicate_ids_must_be_preserved(tmp_path: Path) -> None:
    class ReorderingPredictor(_FakePredictor):
        def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
            return list(reversed(super().predict_batch(inputs)))

    with pytest.raises(GMCQualificationError, match="ordering or molecule identity"):
        _run(tmp_path, ReorderingPredictor)


def test_duplicate_structures_must_have_identical_scientific_predictions(
    tmp_path: Path,
) -> None:
    class InconsistentDuplicatePredictor(_FakePredictor):
        def predict_batch(self, inputs: Sequence[GMCInferenceInput]) -> list[dict[str, Any]]:
            rows = super().predict_batch(inputs)
            probabilities = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
            for seed, probability in zip(PRODUCTION_SEEDS, probabilities, strict=True):
                rows[16][f"seed{seed}_probability"] = float(probability)
            rows[16]["ensemble_probability"] = float(probabilities.mean(dtype=np.float64))
            rows[16]["ensemble_standard_deviation"] = float(
                probabilities.std(ddof=0, dtype=np.float64)
            )
            rows[16]["prediction"] = "BBB-"
            return rows

    with pytest.raises(GMCQualificationError, match="Duplicate structures produced different"):
        _run(tmp_path, InconsistentDuplicatePredictor)


def test_existing_qualification_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "qualification"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    called = False

    def factory(*args: Any, **kwargs: Any) -> _FakePredictor:
        nonlocal called
        called = True
        return _FakePredictor(*args, **kwargs)

    manifest = tmp_path / "production_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_production_qualification(
            manifest_path=manifest,
            artifact_root=tmp_path,
            input_path=FIXED_INPUT,
            output_dir=output,
            git_commit=GIT_COMMIT,
            predictor_factory=factory,
            runtime_provenance=RUNTIME,
        )
    assert called is False
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_input_schema_fails_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "input.csv"
    malformed.write_text("molecule_id,source_smiles\nonly,CC\n", encoding="utf-8")
    with pytest.raises(GMCQualificationError, match="schema"):
        load_qualification_inputs(malformed)


def _run(tmp_path: Path, predictor: type[_FakePredictor]) -> dict[str, Any]:
    manifest = tmp_path / "production_manifest.json"
    manifest.write_text('{"identity":"frozen"}\n', encoding="utf-8")
    return run_production_qualification(
        manifest_path=manifest,
        artifact_root=tmp_path,
        input_path=FIXED_INPUT,
        output_dir=tmp_path / "qualification",
        git_commit=GIT_COMMIT,
        predictor_factory=predictor,
        runtime_provenance=RUNTIME,
    )


def _result(item: GMCInferenceInput, index: int) -> dict[str, Any]:
    row = dict.fromkeys(OUTPUT_FIELDS)
    row.update(
        {
            "molecule_id": item.molecule_id,
            "source_smiles": item.source_smiles,
            "model_family": "GMC-MPNN",
            "manifest_version": MANIFEST_VERSION,
            "model_interface_version": "gmc-mpnn-chemprop-interface-v1",
            "threshold": 0.5,
        }
    )
    if index in {12, 13, 14}:
        row.update(
            {
                "preprocessing_status": "failed",
                "parent_standardization_status": (
                    "parent_selected" if index == 14 else "not_started"
                ),
                "status": "failed",
                "error_code": "geometry_no_heavy_atoms" if index == 14 else "invalid_smiles",
                "error_message": "synthetic invalid input",
            }
        )
        return row
    molecule = Chem.MolFromSmiles(item.source_smiles)
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    standardized = standardize_for_gmc_geometry(item.molecule_id, canonical)
    center = 0.25 if index % 2 else 0.75
    probabilities = [center - 0.02, center - 0.01, center, center + 0.01, center + 0.02]
    values = np.asarray(probabilities, dtype=np.float64)
    row.update(
        {
            "canonical_smiles": canonical,
            "preprocessing_status": "success",
            "parent_standardization_status": standardized.action,
            "ensemble_probability": float(values.mean(dtype=np.float64)),
            "ensemble_standard_deviation": float(values.std(ddof=0, dtype=np.float64)),
            "prediction": "BBB+" if center >= 0.5 else "BBB-",
            "status": "success",
        }
    )
    row.update(
        {
            f"seed{seed}_probability": probability
            for seed, probability in zip(PRODUCTION_SEEDS, probabilities, strict=True)
        }
    )
    return row


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
