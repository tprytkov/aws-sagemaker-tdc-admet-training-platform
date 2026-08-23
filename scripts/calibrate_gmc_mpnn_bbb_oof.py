"""Fit TRAIN-OOF GMC calibration and evaluate it once on external validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.gmc_mpnn.calibration import (  # noqa: E402
    CALIBRATION_METHOD,
    FIT_SPLIT,
    GMC_CALIBRATION_VERSION,
    LOGISTIC_REGRESSION_C,
    LOGISTIC_REGRESSION_RANDOM_STATE,
    LOGISTIC_REGRESSION_SOLVER,
    PROBABILITY_CLIP_EPSILON,
    THRESHOLD_CANDIDATE_POLICY,
    THRESHOLD_SELECTION_METHOD,
    THRESHOLD_TIE_BREAK_POLICY,
    FrozenGMCPlattCalibrator,
    complete_binary_metrics,
    fit_train_oof_platt_calibrator,
)
from admet_platform.gmc_mpnn.scaling import (  # noqa: E402
    FIT_SUMMARY_FILENAME,
    load_frozen_ggl_scaler,
)


CALIBRATION_RUNNER_VERSION: Final = "gmc-mpnn-bbb-oof-calibration-validation-v1"
EXPECTED_OOF_COUNT: Final = 1558
EXPECTED_VALIDATION_COUNT: Final = 196
EXPECTED_FOLDS: Final = (0, 1, 2, 3, 4)
EXPECTED_SEEDS: Final = (13, 37, 73, 101, 137)
EXPECTED_CHECKPOINT_HASHES: Final = {
    13: "643b7de6f0f9d07a6ff7a9d5e266d57f6c3f41844f417683ee31b87eaf958dfe",
    37: "84c348f53c29a089b32d115bbf8dc9e25844c533240f5e7cfc5738353e18463d",
    73: "b00b4f8b750ecdd59973171d36ec4362b61b0bd85a39c87df8900127c6ae73ff",
    101: "314d740363efa1e19e3f44cfa6035f3ddcb01afda6bd2f1ade30037ae27eaca3",
    137: "84e65029ec63bd2ee8500903a388189594c2e8c0b8831ef783bb08ef8ae8a0c4",
}
OOF_PREDICTIONS_FILENAME: Final = "oof_predictions.csv"
OOF_SUMMARY_FILENAME: Final = "oof_summary.json"
OUTER_MANIFEST_FILENAME: Final = "outer_fold_manifest.csv"
INNER_MANIFEST_FILENAME: Final = "inner_split_manifest.csv"
SPLIT_SUMMARY_FILENAME: Final = "split_summary.json"
VALIDATION_PREDICTIONS_FILENAME: Final = "validation_predictions.csv"
VALIDATION_METRICS_INPUT_FILENAME: Final = "validation_metrics.json"
VALIDATION_ENSEMBLE_INPUT_FILENAME: Final = "ensemble_summary.json"
VALIDATION_PREPROCESSING_SUMMARY_FILENAME: Final = "preprocessing_summary.json"
FEATURE_MANIFEST_FILENAME: Final = "feature_manifest.csv"
MOLECULE_STATUS_FILENAME: Final = "molecule_status.csv"

CALIBRATOR_FILENAME: Final = "calibrator.json"
OOF_CALIBRATED_FILENAME: Final = "oof_calibrated_predictions.csv"
OOF_METRICS_FILENAME: Final = "oof_metrics.json"
VALIDATION_CALIBRATED_FILENAME: Final = "validation_calibrated_predictions.csv"
VALIDATION_METRICS_FILENAME: Final = "validation_metrics.json"
CALIBRATION_SUMMARY_FILENAME: Final = "calibration_summary.json"
SHA256SUMS_FILENAME: Final = "SHA256SUMS"

DEFAULT_OOF_TRAINING_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_bbb_oof_training_v1"
)
DEFAULT_OOF_MANIFEST_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_bbb_oof_manifest_v1"
)
DEFAULT_VALIDATION_EVALUATION_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_bbb_validation_evaluation_aug22_current_v1"
)
DEFAULT_VALIDATION_PREPROCESSING_DIR: Final = (
    ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_validation_preprocessing_v1"
)
DEFAULT_SCALER_DIR: Final = ROOT / "outputs" / "gpu" / "pilot" / "gmc_mpnn_ggl_scaler_v1"


class GMCCalibrationRunnerError(RuntimeError):
    """A calibration input, leakage, provenance, or publication violation."""


@dataclass(frozen=True)
class CalibrationConfig:
    oof_training_dir: Path
    oof_manifest_dir: Path
    validation_evaluation_dir: Path
    validation_preprocessing_dir: Path
    scaler_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class OOFInputs:
    predictions: pd.DataFrame
    prediction_sha256: str
    oof_summary: Mapping[str, Any]
    oof_summary_sha256: str
    manifest_hashes: Mapping[str, str]
    scaler_sha256: str


@dataclass(frozen=True)
class ValidationInputs:
    predictions: pd.DataFrame
    prediction_sha256: str
    metrics_sha256: str
    ensemble_sha256: str
    validation_manifest_sha256: str
    validation_status_sha256: str
    validation_summary_sha256: str
    scaler_sha256: str
    checkpoint_hashes: Mapping[int, str]


def run_calibration(
    config: CalibrationConfig,
    *,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Fit only on TRAIN OOF and apply once to frozen external validation."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Calibration output directory already exists: {output_dir}")
    for path, label in (
        (config.oof_training_dir, "OOF training"),
        (config.oof_manifest_dir, "OOF manifest"),
        (config.validation_evaluation_dir, "validation evaluation"),
        (config.validation_preprocessing_dir, "validation preprocessing"),
        (config.scaler_dir, "scaler"),
    ):
        _reject_test_path(path, label)

    scaler, scaler_summary = _load_scaler(config.scaler_dir)
    oof = _load_oof_inputs(config, scaler.portable_scaler_sha256, scaler_summary)
    oof_labels = oof.predictions["label"].to_numpy(dtype=np.int64)
    oof_probabilities = oof.predictions["ensemble_probability"].to_numpy(dtype=np.float64)
    calibrator = fit_train_oof_platt_calibrator(oof_labels, oof_probabilities)
    if calibrator.coefficient <= 0.0:
        raise GMCCalibrationRunnerError(
            "TRAIN-OOF Platt coefficient must be positive to preserve probability ranking."
        )
    calibrated_oof = calibrator.transform(oof_probabilities)
    resolved_git_commit = git_commit or _git_commit()
    calibrator_payload = _calibrator_payload(
        calibrator,
        oof,
        scaler_summary,
        resolved_git_commit,
    )
    oof_output = _calibrated_frame(
        oof.predictions,
        labels_column="label",
        calibrator=calibrator,
        calibrated_probabilities=calibrated_oof,
    )
    oof_metrics = _metrics_payload(
        dataset="train_oof_calibration_development",
        labels=oof_labels,
        uncalibrated=oof_probabilities,
        calibrated=calibrated_oof,
        calibrator=calibrator,
    )

    validation = _load_validation_inputs(
        config,
        expected_scaler_sha256=scaler.portable_scaler_sha256,
    )
    validation_labels = validation.predictions["true_label"].to_numpy(dtype=np.int64)
    validation_probabilities = validation.predictions["ensemble_probability"].to_numpy(
        dtype=np.float64
    )
    calibrated_validation = calibrator.transform(validation_probabilities)
    validation_output = _calibrated_frame(
        validation.predictions,
        labels_column="true_label",
        calibrator=calibrator,
        calibrated_probabilities=calibrated_validation,
    )
    validation_metrics = _metrics_payload(
        dataset="external_validation_evaluation_only",
        labels=validation_labels,
        uncalibrated=validation_probabilities,
        calibrated=calibrated_validation,
        calibrator=calibrator,
    )
    summary = {
        "calibration_runner_version": CALIBRATION_RUNNER_VERSION,
        "calibration_version": GMC_CALIBRATION_VERSION,
        "fit_split": FIT_SPLIT,
        "fit_count": EXPECTED_OOF_COUNT,
        "external_validation_count": EXPECTED_VALIDATION_COUNT,
        "selected_threshold": calibrator.selected_threshold,
        "production_seeds": list(EXPECTED_SEEDS),
        "oof_input": {
            "prediction_sha256": oof.prediction_sha256,
            "oof_summary_sha256": oof.oof_summary_sha256,
            "manifest_hashes": dict(oof.manifest_hashes),
        },
        "validation_input": {
            "prediction_sha256": validation.prediction_sha256,
            "metrics_sha256": validation.metrics_sha256,
            "ensemble_sha256": validation.ensemble_sha256,
            "validation_manifest_sha256": validation.validation_manifest_sha256,
            "validation_status_sha256": validation.validation_status_sha256,
            "validation_summary_sha256": validation.validation_summary_sha256,
            "checkpoint_hashes": {
                str(seed): validation.checkpoint_hashes[seed] for seed in EXPECTED_SEEDS
            },
        },
        "frozen_scaler": calibrator_payload["frozen_scaler"],
        "scientific_scope": {
            "oof_metrics_are_calibration_development_metrics": True,
            "external_validation_is_evaluation_only": True,
            "validation_used_for_calibrator_fit": False,
            "validation_used_for_threshold_selection": False,
            "test_artifact_accessed": False,
        },
        "git_commit": resolved_git_commit,
        "package_versions": _package_versions(),
        "validation_artifact_accessed_during_fit": False,
        "validation_artifact_accessed_during_evaluation": True,
        "test_artifact_accessed": False,
    }
    _publish(
        output_dir,
        calibrator_payload,
        oof_output,
        oof_metrics,
        validation_output,
        validation_metrics,
        summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit GMC calibration on TRAIN OOF and evaluate external validation once."
    )
    parser.add_argument("--oof-training-dir", type=Path, default=DEFAULT_OOF_TRAINING_DIR)
    parser.add_argument("--oof-manifest-dir", type=Path, default=DEFAULT_OOF_MANIFEST_DIR)
    parser.add_argument(
        "--validation-evaluation-dir", type=Path, default=DEFAULT_VALIDATION_EVALUATION_DIR
    )
    parser.add_argument(
        "--validation-preprocessing-dir",
        type=Path,
        default=DEFAULT_VALIDATION_PREPROCESSING_DIR,
    )
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_calibration(
        CalibrationConfig(
            oof_training_dir=args.oof_training_dir,
            oof_manifest_dir=args.oof_manifest_dir,
            validation_evaluation_dir=args.validation_evaluation_dir,
            validation_preprocessing_dir=args.validation_preprocessing_dir,
            scaler_dir=args.scaler_dir,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _load_scaler(directory: Path) -> tuple[Any, dict[str, Any]]:
    scaler = load_frozen_ggl_scaler(directory)
    summary = _read_json(directory / FIT_SUMMARY_FILENAME, "frozen scaler fit summary")
    _require_false(summary, "validation_artifact_accessed", "scaler fit summary")
    _require_false(summary, "test_artifact_accessed", "scaler fit summary")
    if summary.get("portable_scaler_sha256") != scaler.portable_scaler_sha256:
        raise GMCCalibrationRunnerError("Frozen scaler summary and portable scaler disagree.")
    if summary.get("training_molecule_count") != EXPECTED_OOF_COUNT:
        raise GMCCalibrationRunnerError("Frozen scaler was not fitted on 1,558 TRAIN molecules.")
    return scaler, summary


def _load_oof_inputs(
    config: CalibrationConfig,
    scaler_sha256: str,
    scaler_summary: Mapping[str, Any],
) -> OOFInputs:
    predictions_path = config.oof_training_dir / OOF_PREDICTIONS_FILENAME
    summary_path = config.oof_training_dir / OOF_SUMMARY_FILENAME
    predictions = _read_csv(predictions_path, "TRAIN OOF predictions")
    summary = _read_json(summary_path, "TRAIN OOF summary")
    _require_false(summary, "validation_artifact_accessed", "TRAIN OOF summary")
    _require_false(summary, "test_artifact_accessed", "TRAIN OOF summary")
    prediction_sha256 = _sha256_file(predictions_path)
    if summary.get("oof_predictions_sha256") != prediction_sha256:
        raise GMCCalibrationRunnerError("TRAIN OOF prediction SHA-256 disagrees with its summary.")
    _validate_oof_predictions(predictions)
    if summary.get("train_count") != EXPECTED_OOF_COUNT:
        raise GMCCalibrationRunnerError("TRAIN OOF summary count is incompatible.")
    if summary.get("outer_fold_count") != len(EXPECTED_FOLDS):
        raise GMCCalibrationRunnerError("TRAIN OOF summary outer-fold count is incompatible.")
    if summary.get("model_count") != len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS):
        raise GMCCalibrationRunnerError("TRAIN OOF summary must prove exactly 25 model runs.")
    if summary.get("seeds") != list(EXPECTED_SEEDS):
        raise GMCCalibrationRunnerError("TRAIN OOF summary seeds are incompatible.")
    checks = summary.get("checks")
    required_checks = (
        "exactly_1558_unique_oof_records",
        "five_seed_predictions_per_record",
        "outer_holdout_excluded_from_model_fit_early_stopping_checkpoint_selection",
        "outer_scaffold_overlap_absent",
        "inner_scaffold_overlap_absent",
        "scaler_refit_absent",
    )
    if not isinstance(checks, Mapping) or any(
        checks.get(field) is not True for field in required_checks
    ):
        raise GMCCalibrationRunnerError("TRAIN OOF summary does not prove its safety checks.")
    frozen_scaler = summary.get("frozen_global_scaler")
    if not isinstance(frozen_scaler, Mapping):
        raise GMCCalibrationRunnerError("TRAIN OOF summary lacks frozen scaler provenance.")
    if frozen_scaler.get("portable_scaler_sha256") != scaler_sha256:
        raise GMCCalibrationRunnerError("TRAIN OOF and frozen scaler SHA-256 values disagree.")
    if frozen_scaler.get("refit_during_oof_training") is not False:
        raise GMCCalibrationRunnerError(
            "TRAIN OOF summary does not prove scaler refitting was absent."
        )
    if frozen_scaler.get("outer_holdout_included_in_preexisting_scaler_fit") is not True:
        raise GMCCalibrationRunnerError("TRAIN OOF summary lacks global-scaler scope provenance.")
    if scaler_summary.get("training_atom_count") != 38245:
        raise GMCCalibrationRunnerError("Frozen scaler atom count is incompatible.")

    manifest_hashes = _validate_oof_manifest(config.oof_manifest_dir, summary)
    return OOFInputs(
        predictions=predictions,
        prediction_sha256=prediction_sha256,
        oof_summary=summary,
        oof_summary_sha256=_sha256_file(summary_path),
        manifest_hashes=manifest_hashes,
        scaler_sha256=scaler_sha256,
    )


def _validate_oof_predictions(frame: pd.DataFrame) -> None:
    required = {
        "record_key",
        "source_row_index",
        "label",
        "outer_fold",
        "ensemble_probability",
        "seed_probability_std",
        *(f"probability_seed{seed}" for seed in EXPECTED_SEEDS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise GMCCalibrationRunnerError(f"TRAIN OOF predictions are missing: {sorted(missing)}")
    if len(frame) != EXPECTED_OOF_COUNT:
        raise GMCCalibrationRunnerError("TRAIN OOF predictions must contain exactly 1,558 rows.")
    if frame["record_key"].astype(str).eq("").any() or frame["record_key"].duplicated().any():
        raise GMCCalibrationRunnerError("TRAIN OOF record_key values must be nonempty and unique.")
    if set(frame["outer_fold"]) != set(EXPECTED_FOLDS):
        raise GMCCalibrationRunnerError("TRAIN OOF outer folds must be exactly 0 through 4.")
    labels = _binary_labels(frame["label"], "TRAIN OOF")
    if len(np.unique(labels)) != 2:
        raise GMCCalibrationRunnerError("TRAIN OOF labels must contain both classes.")
    seed_columns = [f"probability_seed{seed}" for seed in EXPECTED_SEEDS]
    if frame[list(required)].isna().any().any():
        raise GMCCalibrationRunnerError("TRAIN OOF predictions contain missing values.")
    seed_matrix = _probability_matrix(frame, seed_columns, "TRAIN OOF seed")
    ensemble = _probabilities(frame["ensemble_probability"], "TRAIN OOF ensemble")
    expected = seed_matrix.mean(axis=1, dtype=np.float64)
    if not np.allclose(ensemble, expected, rtol=0.0, atol=1e-12):
        raise GMCCalibrationRunnerError("TRAIN OOF ensemble does not equal the five-seed mean.")


def _validate_oof_manifest(
    directory: Path,
    oof_summary: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "outer_fold_manifest_sha256": directory / OUTER_MANIFEST_FILENAME,
        "inner_split_manifest_sha256": directory / INNER_MANIFEST_FILENAME,
        "split_summary_sha256": directory / SPLIT_SUMMARY_FILENAME,
    }
    observed = {field: _sha256_file(path) for field, path in paths.items()}
    expected = oof_summary.get("manifest_hashes")
    if not isinstance(expected, Mapping):
        raise GMCCalibrationRunnerError("TRAIN OOF summary lacks manifest hashes.")
    for field, digest in observed.items():
        if expected.get(field) != digest:
            raise GMCCalibrationRunnerError(f"TRAIN OOF manifest hash mismatch: {field}.")
    split_summary = _read_json(paths["split_summary_sha256"], "OOF split summary")
    _require_false(split_summary, "validation_artifact_accessed", "OOF split summary")
    _require_false(split_summary, "test_artifact_accessed", "OOF split summary")
    return observed


def _load_validation_inputs(
    config: CalibrationConfig,
    *,
    expected_scaler_sha256: str,
) -> ValidationInputs:
    directory = config.validation_evaluation_dir
    predictions_path = directory / VALIDATION_PREDICTIONS_FILENAME
    metrics_path = directory / VALIDATION_METRICS_INPUT_FILENAME
    ensemble_path = directory / VALIDATION_ENSEMBLE_INPUT_FILENAME
    predictions = _read_csv(predictions_path, "external validation predictions")
    metrics = _read_json(metrics_path, "external validation metrics provenance")
    ensemble = _read_json(ensemble_path, "external validation ensemble provenance")
    _require_false(metrics, "test_artifact_accessed", "validation metrics")
    _require_false(ensemble, "test_artifact_accessed", "validation ensemble")
    _validate_validation_predictions(predictions)
    provenance = ensemble.get("provenance")
    if not isinstance(provenance, Mapping):
        raise GMCCalibrationRunnerError("Validation ensemble lacks provenance.")
    if provenance.get("validation_count") != EXPECTED_VALIDATION_COUNT:
        raise GMCCalibrationRunnerError("Validation provenance count is incompatible.")
    if provenance.get("seeds") != list(EXPECTED_SEEDS):
        raise GMCCalibrationRunnerError("Validation ensemble seeds are incompatible.")
    _require_false(provenance, "test_artifact_accessed", "validation provenance")
    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise GMCCalibrationRunnerError("Validation provenance lacks checkpoint hashes.")
    observed_checkpoints: dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        record = checkpoints.get(str(seed))
        if not isinstance(record, Mapping):
            raise GMCCalibrationRunnerError(f"Validation provenance lacks seed {seed} checkpoint.")
        digest = record.get("sha256")
        if digest != EXPECTED_CHECKPOINT_HASHES[seed]:
            raise GMCCalibrationRunnerError(
                f"Validation seed {seed} checkpoint SHA-256 is not the Aug-22 current model."
            )
        observed_checkpoints[seed] = str(digest)
    frozen_validation = provenance.get("frozen_validation")
    if not isinstance(frozen_validation, Mapping):
        raise GMCCalibrationRunnerError("Validation provenance lacks frozen-data identity.")
    if frozen_validation.get("portable_scaler_sha256") != expected_scaler_sha256:
        raise GMCCalibrationRunnerError("Validation and frozen scaler SHA-256 values disagree.")

    preprocessing_summary_path = (
        config.validation_preprocessing_dir / VALIDATION_PREPROCESSING_SUMMARY_FILENAME
    )
    validation_manifest_path = config.validation_preprocessing_dir / FEATURE_MANIFEST_FILENAME
    validation_status_path = config.validation_preprocessing_dir / MOLECULE_STATUS_FILENAME
    preprocessing_summary = _read_json(
        preprocessing_summary_path, "validation preprocessing summary"
    )
    if preprocessing_summary.get("validation_artifact_accessed") is not True:
        raise GMCCalibrationRunnerError(
            "Validation preprocessing summary must record validation artifact access."
        )
    _require_false(preprocessing_summary, "test_artifact_accessed", "validation preprocessing")
    manifest_sha256 = _sha256_file(validation_manifest_path)
    status_sha256 = _sha256_file(validation_status_path)
    if preprocessing_summary.get("feature_manifest_sha256") != manifest_sha256:
        raise GMCCalibrationRunnerError("Validation preprocessing manifest SHA-256 is invalid.")
    if frozen_validation.get("feature_manifest_sha256") != manifest_sha256:
        raise GMCCalibrationRunnerError(
            "Validation evaluation and manifest SHA-256 values disagree."
        )
    if preprocessing_summary.get("molecule_status_sha256") != status_sha256:
        raise GMCCalibrationRunnerError("Validation preprocessing status SHA-256 is invalid.")
    if frozen_validation.get("molecule_status_sha256") != status_sha256:
        raise GMCCalibrationRunnerError("Validation evaluation and status SHA-256 values disagree.")
    if preprocessing_summary.get("source_row_count") != EXPECTED_VALIDATION_COUNT:
        raise GMCCalibrationRunnerError("Validation preprocessing count is incompatible.")
    return ValidationInputs(
        predictions=predictions,
        prediction_sha256=_sha256_file(predictions_path),
        metrics_sha256=_sha256_file(metrics_path),
        ensemble_sha256=_sha256_file(ensemble_path),
        validation_manifest_sha256=manifest_sha256,
        validation_status_sha256=status_sha256,
        validation_summary_sha256=_sha256_file(preprocessing_summary_path),
        scaler_sha256=expected_scaler_sha256,
        checkpoint_hashes=observed_checkpoints,
    )


def _validate_validation_predictions(frame: pd.DataFrame) -> None:
    required = {
        "record_key",
        "molecule_id",
        "canonical_smiles",
        "true_label",
        "ensemble_probability",
        *(f"probability_seed{seed}" for seed in EXPECTED_SEEDS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise GMCCalibrationRunnerError(
            f"External validation predictions are missing: {sorted(missing)}"
        )
    if len(frame) != EXPECTED_VALIDATION_COUNT:
        raise GMCCalibrationRunnerError("External validation must contain exactly 196 rows.")
    if frame[list(required)].isna().any().any() or frame["record_key"].duplicated().any():
        raise GMCCalibrationRunnerError(
            "External validation identities/probabilities are incomplete."
        )
    _binary_labels(frame["true_label"], "external validation")
    seed_columns = [f"probability_seed{seed}" for seed in EXPECTED_SEEDS]
    seed_matrix = _probability_matrix(frame, seed_columns, "external validation seed")
    ensemble = _probabilities(frame["ensemble_probability"], "external validation ensemble")
    if not np.allclose(ensemble, seed_matrix.mean(axis=1), rtol=0.0, atol=1e-12):
        raise GMCCalibrationRunnerError(
            "External validation ensemble does not equal the five-seed mean."
        )


def _calibrator_payload(
    calibrator: FrozenGMCPlattCalibrator,
    oof: OOFInputs,
    scaler_summary: Mapping[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    return {
        "calibration_version": GMC_CALIBRATION_VERSION,
        "calibration_method": CALIBRATION_METHOD,
        "fit_split": FIT_SPLIT,
        "fit_count": EXPECTED_OOF_COUNT,
        "fit_probability": "five_seed_oof_ensemble_probability",
        "coefficient": calibrator.coefficient,
        "intercept": calibrator.intercept,
        "probability_clip_epsilon": PROBABILITY_CLIP_EPSILON,
        "logistic_regression_C": LOGISTIC_REGRESSION_C,
        "logistic_regression_solver": LOGISTIC_REGRESSION_SOLVER,
        "logistic_regression_random_state": LOGISTIC_REGRESSION_RANDOM_STATE,
        "threshold_selection_method": THRESHOLD_SELECTION_METHOD,
        "threshold_candidate_policy": THRESHOLD_CANDIDATE_POLICY,
        "threshold_prediction_rule": "probability >= threshold",
        "threshold_tie_break_policy": THRESHOLD_TIE_BREAK_POLICY,
        "selected_threshold": calibrator.selected_threshold,
        "oof_prediction_file_sha256": oof.prediction_sha256,
        "oof_manifest_hashes": dict(oof.manifest_hashes),
        "oof_summary_sha256": oof.oof_summary_sha256,
        "production_seeds": list(EXPECTED_SEEDS),
        "frozen_scaler": {
            "portable_scaler_sha256": oof.scaler_sha256,
            "scaler_version": scaler_summary.get("scaler_version"),
            "fit_population": "complete frozen TRAIN successful atom rows",
            "training_molecule_count": scaler_summary.get("training_molecule_count"),
            "training_atom_count": scaler_summary.get("training_atom_count"),
            "unsupervised_global_preprocessing_component": True,
            "outer_holdout_contributed_to_train_only_scaling_statistics": True,
            "fully_nested_preprocessing_claimed": False,
            "validation_included_in_scaler_fit": False,
            "refitted_for_calibration": False,
        },
        "git_commit": git_commit,
        "package_versions": _package_versions(),
        "validation_artifact_accessed_during_fit": False,
        "validation_used_for_fitting": False,
        "test_artifact_accessed": False,
    }


def _calibrated_frame(
    source: pd.DataFrame,
    *,
    labels_column: str,
    calibrator: FrozenGMCPlattCalibrator,
    calibrated_probabilities: np.ndarray,
) -> pd.DataFrame:
    frame = source.copy()
    frame["uncalibrated_probability"] = frame["ensemble_probability"].to_numpy(dtype=np.float64)
    frame["calibrated_probability"] = calibrated_probabilities
    frame["prediction_uncalibrated_threshold_0_5"] = (
        frame["uncalibrated_probability"] >= 0.5
    ).astype(np.int64)
    frame["prediction_calibrated_threshold_0_5"] = (frame["calibrated_probability"] >= 0.5).astype(
        np.int64
    )
    frame["prediction_calibrated_frozen_oof_mcc_threshold"] = (
        frame["calibrated_probability"] >= calibrator.selected_threshold
    ).astype(np.int64)
    if not np.array_equal(
        frame[labels_column].to_numpy(dtype=np.int64),
        _binary_labels(frame[labels_column], labels_column),
    ):
        raise GMCCalibrationRunnerError("Output labels changed during calibration.")
    return frame


def _metrics_payload(
    *,
    dataset: str,
    labels: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    calibrator: FrozenGMCPlattCalibrator,
) -> dict[str, Any]:
    uncalibrated_metrics = complete_binary_metrics(labels, uncalibrated, threshold=0.5)
    calibrated_half = complete_binary_metrics(labels, calibrated, threshold=0.5)
    calibrated_frozen = complete_binary_metrics(
        labels, calibrated, threshold=calibrator.selected_threshold
    )
    auroc_difference = calibrated_half["auroc"] - uncalibrated_metrics["auroc"]
    auprc_difference = calibrated_half["auprc"] - uncalibrated_metrics["auprc"]
    if not np.isclose(auroc_difference, 0.0, rtol=0.0, atol=1e-12):
        raise GMCCalibrationRunnerError("Positive-slope Platt scaling changed AUROC ranking.")
    if not np.isclose(auprc_difference, 0.0, rtol=0.0, atol=1e-12):
        raise GMCCalibrationRunnerError("Positive-slope Platt scaling changed AUPRC ranking.")
    return {
        "dataset": dataset,
        "count": int(len(labels)),
        "metrics": {
            "uncalibrated_threshold_0_5": uncalibrated_metrics,
            "calibrated_threshold_0_5": calibrated_half,
            "calibrated_frozen_train_oof_maximum_mcc_threshold": calibrated_frozen,
        },
        "ranking_invariance": {
            "platt_coefficient_positive": calibrator.coefficient > 0.0,
            "auroc_difference_calibrated_minus_uncalibrated": auroc_difference,
            "auprc_difference_calibrated_minus_uncalibrated": auprc_difference,
            "assertion_tolerance": 1e-12,
            "interpretation": "positive-slope Platt scaling is monotonic and preserves ranking",
        },
        "selected_threshold_source": "calibrated TRAIN OOF maximum MCC",
        "validation_used_for_threshold_selection": False,
        "test_artifact_accessed": False,
    }


def _publish(
    output_dir: Path,
    calibrator: Mapping[str, Any],
    oof_predictions: pd.DataFrame,
    oof_metrics: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
    validation_metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_json(temporary_dir / CALIBRATOR_FILENAME, calibrator)
        oof_predictions.to_csv(
            temporary_dir / OOF_CALIBRATED_FILENAME,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        _write_json(temporary_dir / OOF_METRICS_FILENAME, oof_metrics)
        validation_predictions.to_csv(
            temporary_dir / VALIDATION_CALIBRATED_FILENAME,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        _write_json(temporary_dir / VALIDATION_METRICS_FILENAME, validation_metrics)
        _write_json(temporary_dir / CALIBRATION_SUMMARY_FILENAME, summary)
        artifact_names = (
            CALIBRATOR_FILENAME,
            OOF_CALIBRATED_FILENAME,
            OOF_METRICS_FILENAME,
            VALIDATION_CALIBRATED_FILENAME,
            VALIDATION_METRICS_FILENAME,
            CALIBRATION_SUMMARY_FILENAME,
        )
        checksum_lines = [
            f"{_sha256_file(temporary_dir / name)}  {name}" for name in artifact_names
        ]
        (temporary_dir / SHA256SUMS_FILENAME).write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Calibration output directory appeared during run: {output_dir}")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _binary_labels(values: Any, label: str) -> np.ndarray:
    try:
        numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise GMCCalibrationRunnerError(f"{label} labels must be numeric binary values.") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise GMCCalibrationRunnerError(f"{label} labels must be finite binary values.")
    return numeric.astype(np.int64)


def _probabilities(values: Any, label: str) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise GMCCalibrationRunnerError(f"{label} probabilities must be numeric.") from exc
    if not np.isfinite(probabilities).all():
        raise GMCCalibrationRunnerError(f"{label} probabilities contain NaN or infinity.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise GMCCalibrationRunnerError(f"{label} probabilities fall outside [0, 1].")
    return probabilities


def _probability_matrix(frame: pd.DataFrame, columns: list[str], label: str) -> np.ndarray:
    try:
        matrix = frame[columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GMCCalibrationRunnerError(f"{label} probabilities must be numeric.") from exc
    _probabilities(matrix.reshape(-1), label)
    return matrix


def _require_false(payload: Mapping[str, Any], field: str, label: str) -> None:
    if field not in payload or payload[field] is not False:
        raise GMCCalibrationRunnerError(f"{label} {field!r} must exist and be exactly false.")


def _reject_test_path(path: Path, label: str) -> None:
    forbidden = {"test", "locked_test", "locked-test", "bbb_test", "bbb-test"}
    if any(part.lower() in forbidden for part in path.parts):
        raise GMCCalibrationRunnerError(f"{label} path points to a prohibited test artifact.")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GMCCalibrationRunnerError(f"Cannot read {label}: {path.name}") from exc
    if not isinstance(value, dict):
        raise GMCCalibrationRunnerError(f"{label} must be a JSON object.")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise GMCCalibrationRunnerError(f"Cannot read {label}: {path.name}") from exc


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GMCCalibrationRunnerError(f"Cannot hash required file: {path.name}") from exc


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": metadata.version("scikit-learn"),
        "rdkit": metadata.version("rdkit"),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GMCCalibrationRunnerError("Unable to record the Git commit.") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
