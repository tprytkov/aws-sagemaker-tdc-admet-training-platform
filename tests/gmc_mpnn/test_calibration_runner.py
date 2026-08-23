from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import calibrate_gmc_mpnn_bbb_oof as runner


def test_calibration_fit_and_threshold_are_independent_of_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oof = _oof_inputs()
    validation_a = _validation_inputs(offset=0.0)
    validation_b = _validation_inputs(offset=0.08)
    validation_b.predictions["true_label"] = 1 - validation_b.predictions["true_label"]
    scaler = SimpleNamespace(portable_scaler_sha256="e" * 64)
    scaler_summary = _scaler_summary()
    monkeypatch.setattr(runner, "_load_scaler", lambda _: (scaler, scaler_summary))
    monkeypatch.setattr(runner, "_load_oof_inputs", lambda *args: oof)
    loaded_validation = iter((validation_a, validation_b))
    monkeypatch.setattr(
        runner, "_load_validation_inputs", lambda *args, **kwargs: next(loaded_validation)
    )

    output_a = tmp_path / "calibration-a"
    output_b = tmp_path / "calibration-b"
    runner.run_calibration(_config(output_a), git_commit="a" * 40)
    runner.run_calibration(_config(output_b), git_commit="a" * 40)

    calibrator_a = (output_a / runner.CALIBRATOR_FILENAME).read_bytes()
    calibrator_b = (output_b / runner.CALIBRATOR_FILENAME).read_bytes()
    assert calibrator_a == calibrator_b
    payload = json.loads(calibrator_a)
    assert payload["fit_split"] == "train_oof"
    assert payload["fit_count"] == 1558
    assert payload["validation_used_for_fitting"] is False
    assert payload["validation_artifact_accessed_during_fit"] is False
    assert payload["test_artifact_accessed"] is False
    assert payload["threshold_selection_method"] == "maximum_mcc"
    assert payload["frozen_scaler"]["fully_nested_preprocessing_claimed"] is False
    assert (
        payload["frozen_scaler"]["outer_holdout_contributed_to_train_only_scaling_statistics"]
        is True
    )
    assert payload["coefficient"] > 0
    assert 0 <= payload["selected_threshold"] <= 1
    assert (output_a / runner.SHA256SUMS_FILENAME).is_file()
    assert len((output_a / runner.OOF_CALIBRATED_FILENAME).read_text().splitlines()) == 1559
    assert len((output_a / runner.VALIDATION_CALIBRATED_FILENAME).read_text().splitlines()) == 197
    validation_metrics = json.loads((output_a / runner.VALIDATION_METRICS_FILENAME).read_text())
    assert validation_metrics["validation_used_for_threshold_selection"] is False
    assert validation_metrics["ranking_invariance"]["platt_coefficient_positive"] is True


@pytest.mark.parametrize("invalid", (True, None, 0, "false", "missing"))
@pytest.mark.parametrize("field", ("validation_artifact_accessed", "test_artifact_accessed"))
def test_isolation_flags_must_exist_and_be_literal_false(field: str, invalid: object) -> None:
    payload: dict[str, Any] = {
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }
    if invalid == "missing":
        del payload[field]
    else:
        payload[field] = invalid

    with pytest.raises(runner.GMCCalibrationRunnerError, match="exactly false"):
        runner._require_false(payload, field, "synthetic")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("count", "exactly 1,558"),
        ("duplicate", "unique"),
        ("missing_seed", "missing"),
        ("ensemble", "five-seed mean"),
    ),
)
def test_invalid_oof_prediction_contract_fails(mutation: str, message: str) -> None:
    frame = _oof_frame()
    if mutation == "count":
        frame = frame.iloc[:-1].copy()
    elif mutation == "duplicate":
        frame.loc[1, "record_key"] = frame.loc[0, "record_key"]
    elif mutation == "missing_seed":
        frame = frame.drop(columns="probability_seed137")
    else:
        frame.loc[0, "ensemble_probability"] += 0.1

    with pytest.raises(runner.GMCCalibrationRunnerError, match=message):
        runner._validate_oof_predictions(frame)


def test_wrong_oof_manifest_hash_fails(tmp_path: Path) -> None:
    for name in (
        runner.OUTER_MANIFEST_FILENAME,
        runner.INNER_MANIFEST_FILENAME,
        runner.SPLIT_SUMMARY_FILENAME,
    ):
        (tmp_path / name).write_text("{}\n" if name.endswith("json") else "x\n")
    summary = {
        "manifest_hashes": {
            "outer_fold_manifest_sha256": "0" * 64,
            "inner_split_manifest_sha256": "0" * 64,
            "split_summary_sha256": "0" * 64,
        }
    }
    with pytest.raises(runner.GMCCalibrationRunnerError, match="manifest hash mismatch"):
        runner._validate_oof_manifest(tmp_path, summary)


def test_wrong_aug22_validation_checkpoint_hash_fails(tmp_path: Path) -> None:
    config = _write_validation_fixture(tmp_path)
    ensemble_path = config.validation_evaluation_dir / runner.VALIDATION_ENSEMBLE_INPUT_FILENAME
    ensemble = json.loads(ensemble_path.read_text())
    ensemble["provenance"]["checkpoints"]["13"]["sha256"] = "0" * 64
    ensemble_path.write_text(json.dumps(ensemble))

    with pytest.raises(runner.GMCCalibrationRunnerError, match="Aug-22 current model"):
        runner._load_validation_inputs(config, expected_scaler_sha256="e" * 64)


def test_output_collision_fails_before_any_input_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("No input may be accessed after an output collision.")

    monkeypatch.setattr(runner, "_load_scaler", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_calibration(_config(output))


def test_locked_test_input_path_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path / "output")
    config = runner.CalibrationConfig(
        oof_training_dir=config.oof_training_dir,
        oof_manifest_dir=config.oof_manifest_dir,
        validation_evaluation_dir=tmp_path / "locked_test",
        validation_preprocessing_dir=config.validation_preprocessing_dir,
        scaler_dir=config.scaler_dir,
        output_dir=config.output_dir,
    )
    with pytest.raises(runner.GMCCalibrationRunnerError, match="prohibited test artifact"):
        runner.run_calibration(config)


def test_artifacts_are_deterministic_from_identical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oof = _oof_inputs()
    validation = _validation_inputs()
    scaler = SimpleNamespace(portable_scaler_sha256="e" * 64)
    monkeypatch.setattr(runner, "_load_scaler", lambda _: (scaler, _scaler_summary()))
    monkeypatch.setattr(runner, "_load_oof_inputs", lambda *args: oof)
    monkeypatch.setattr(runner, "_load_validation_inputs", lambda *args, **kwargs: validation)
    first = tmp_path / "first"
    second = tmp_path / "second"
    runner.run_calibration(_config(first), git_commit="b" * 40)
    runner.run_calibration(_config(second), git_commit="b" * 40)

    for name in (
        runner.CALIBRATOR_FILENAME,
        runner.OOF_CALIBRATED_FILENAME,
        runner.OOF_METRICS_FILENAME,
        runner.VALIDATION_CALIBRATED_FILENAME,
        runner.VALIDATION_METRICS_FILENAME,
        runner.CALIBRATION_SUMMARY_FILENAME,
        runner.SHA256SUMS_FILENAME,
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _config(output: Path) -> runner.CalibrationConfig:
    root = output.parent
    return runner.CalibrationConfig(
        oof_training_dir=root / "oof-training",
        oof_manifest_dir=root / "oof-manifest",
        validation_evaluation_dir=root / "validation-evaluation-current",
        validation_preprocessing_dir=root / "validation-preprocessing",
        scaler_dir=root / "scaler",
        output_dir=output,
    )


def _oof_frame() -> pd.DataFrame:
    index = np.arange(runner.EXPECTED_OOF_COUNT)
    labels = index % 2
    base = np.where(labels == 1, 0.65, 0.35) + ((index % 17) - 8) * 0.005
    seeds = {
        seed: np.clip(base + (position - 2) * 0.004, 0.001, 0.999)
        for position, seed in enumerate(runner.EXPECTED_SEEDS)
    }
    matrix = np.column_stack([seeds[seed] for seed in runner.EXPECTED_SEEDS])
    return pd.DataFrame(
        {
            "record_key": [f"record-{value:04d}" for value in index],
            "source_row_index": index,
            "label": labels,
            "outer_fold": index % 5,
            **{f"probability_seed{seed}": seeds[seed] for seed in runner.EXPECTED_SEEDS},
            "ensemble_probability": matrix.mean(axis=1),
            "seed_probability_std": matrix.std(axis=1),
        }
    )


def _validation_frame(offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(runner.EXPECTED_VALIDATION_COUNT)
    labels = index % 2
    base = np.clip(np.where(labels == 1, 0.62, 0.38) + offset, 0.01, 0.99)
    seeds = {
        seed: np.clip(base + (position - 2) * 0.003, 0.001, 0.999)
        for position, seed in enumerate(runner.EXPECTED_SEEDS)
    }
    matrix = np.column_stack([seeds[seed] for seed in runner.EXPECTED_SEEDS])
    return pd.DataFrame(
        {
            "record_key": [f"validation-{value:03d}" for value in index],
            "molecule_id": [f"molecule-{value:03d}" for value in index],
            "canonical_smiles": ["C" if value % 2 == 0 else "CC" for value in index],
            "true_label": labels,
            **{f"probability_seed{seed}": seeds[seed] for seed in runner.EXPECTED_SEEDS},
            "ensemble_probability": matrix.mean(axis=1),
            "probability_standard_deviation": matrix.std(axis=1),
        }
    )


def _oof_inputs() -> runner.OOFInputs:
    return runner.OOFInputs(
        predictions=_oof_frame(),
        prediction_sha256="a" * 64,
        oof_summary={},
        oof_summary_sha256="b" * 64,
        manifest_hashes={
            "outer_fold_manifest_sha256": "c" * 64,
            "inner_split_manifest_sha256": "d" * 64,
            "split_summary_sha256": "f" * 64,
        },
        scaler_sha256="e" * 64,
    )


def _validation_inputs(offset: float = 0.0) -> runner.ValidationInputs:
    return runner.ValidationInputs(
        predictions=_validation_frame(offset),
        prediction_sha256="1" * 64,
        metrics_sha256="2" * 64,
        ensemble_sha256="3" * 64,
        validation_manifest_sha256="4" * 64,
        validation_status_sha256="6" * 64,
        validation_summary_sha256="5" * 64,
        scaler_sha256="e" * 64,
        checkpoint_hashes=runner.EXPECTED_CHECKPOINT_HASHES,
    )


def _scaler_summary() -> dict[str, Any]:
    return {
        "portable_scaler_sha256": "e" * 64,
        "scaler_version": "gmc-mpnn-ggl-standard-scaler-v1",
        "training_molecule_count": 1558,
        "training_atom_count": 38245,
        "validation_artifact_accessed": False,
        "test_artifact_accessed": False,
    }


def _write_validation_fixture(root: Path) -> runner.CalibrationConfig:
    config = _config(root / "output")
    config.validation_evaluation_dir.mkdir(parents=True)
    config.validation_preprocessing_dir.mkdir(parents=True)
    predictions = _validation_frame()
    predictions.to_csv(
        config.validation_evaluation_dir / runner.VALIDATION_PREDICTIONS_FILENAME,
        index=False,
    )
    provenance = {
        "validation_count": 196,
        "seeds": list(runner.EXPECTED_SEEDS),
        "checkpoints": {
            str(seed): {"sha256": digest}
            for seed, digest in runner.EXPECTED_CHECKPOINT_HASHES.items()
        },
        "frozen_validation": {
            "portable_scaler_sha256": "e" * 64,
            "feature_manifest_sha256": "pending",
        },
        "test_artifact_accessed": False,
    }
    metrics = {"test_artifact_accessed": False}
    ensemble = {"provenance": provenance, "test_artifact_accessed": False}
    (config.validation_evaluation_dir / runner.VALIDATION_METRICS_INPUT_FILENAME).write_text(
        json.dumps(metrics)
    )
    ensemble_path = config.validation_evaluation_dir / runner.VALIDATION_ENSEMBLE_INPUT_FILENAME
    ensemble_path.write_text(json.dumps(ensemble))
    manifest_path = config.validation_preprocessing_dir / runner.FEATURE_MANIFEST_FILENAME
    manifest_path.write_text("record_key\nsynthetic\n")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    status_path = config.validation_preprocessing_dir / runner.MOLECULE_STATUS_FILENAME
    status_path.write_text("record_key\nsynthetic\n")
    status_hash = hashlib.sha256(status_path.read_bytes()).hexdigest()
    preprocessing = {
        "feature_manifest_sha256": manifest_hash,
        "molecule_status_sha256": status_hash,
        "source_row_count": 196,
        "validation_artifact_accessed": True,
        "test_artifact_accessed": False,
    }
    (
        config.validation_preprocessing_dir / runner.VALIDATION_PREPROCESSING_SUMMARY_FILENAME
    ).write_text(json.dumps(preprocessing))
    ensemble["provenance"]["frozen_validation"]["feature_manifest_sha256"] = manifest_hash
    ensemble["provenance"]["frozen_validation"]["molecule_status_sha256"] = status_hash
    ensemble_path.write_text(json.dumps(ensemble))
    return config
