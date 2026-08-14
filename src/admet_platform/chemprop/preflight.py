"""Locked-test-safe real train/validation preflight audit and conventional baselines."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from admet_platform.chemprop.baselines import (
    run_classification_baselines,
    run_regression_baselines,
)
from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.chemprop.data import VerifiedDevelopmentData, load_verified_development_data
from admet_platform.chemprop.losses import (
    SELECTION_METRIC_EQUATION,
    TRAINING_LOSS_EQUATION,
    inverse_count_equal_endpoint_weights,
)


def build_real_data_preflight_audit(
    regression_config: ChempropExperimentConfig,
    bbb_config: ChempropExperimentConfig,
) -> tuple[dict[str, Any], VerifiedDevelopmentData, VerifiedDevelopmentData]:
    """Read manifests plus train/validation CSVs; never open or hash locked-test CSVs."""

    regression = load_verified_development_data(regression_config)
    bbb = load_verified_development_data(bbb_config)
    regression_manifest = _read_json(regression_config.split_manifest)
    bbb_manifest = _read_json(bbb_config.split_manifest)
    endpoint_order = list(regression_config.tasks)
    train_counts = [int(regression.train[task].notna().sum()) for task in endpoint_order]
    weights = inverse_count_equal_endpoint_weights(train_counts)

    endpoint_audits: dict[str, Any] = {}
    for index, endpoint in enumerate(endpoint_order):
        metadata = regression_config.tasks[endpoint]
        train_values = regression.train[endpoint].dropna().to_numpy(dtype=float)
        transformed = _scientific_transform(train_values, metadata["target_transform"])
        source = regression_manifest["endpoints"][endpoint]
        endpoint_audits[endpoint] = {
            "order_index": index,
            "tdc_name": metadata["tdc_name"],
            "unit": metadata["units"],
            "stored_label_representation": metadata.get(
                "stored_label_representation", metadata["units"]
            ),
            "underlying_physical_unit": metadata.get("underlying_physical_unit"),
            "label_contract_version": metadata.get("label_contract_version"),
            "scientific_transform": metadata["target_transform"],
            "training_only_scaler": {
                "fit_split": "train",
                "mean": float(np.mean(transformed)),
                "scale": float(np.std(transformed, ddof=0)),
                "validation_statistics_used": False,
                "test_statistics_used": False,
            },
            "labels": {
                split: {
                    "observed": int(frame[endpoint].notna().sum()),
                    "missing_in_sparse_matrix": int(frame[endpoint].isna().sum()),
                }
                for split, frame in (("train", regression.train), ("validation", regression.validation))
            },
            "locked_test": {
                "availability": "present_and_locked",
                "row_count_from_manifest": int(source["splits"]["test"]["row_count"]),
                "sha256_from_manifest": regression.expected_hashes[f"{endpoint}/test"],
                "csv_opened": False,
            },
            "source_and_cleaning_accounting": {
                key: source[key]
                for key in (
                    "source_row_count", "valid_canonicalized_records", "invalid_records",
                    "exact_duplicate_groups", "identical_label_duplicate_groups_collapsed",
                    "identical_duplicate_rows_removed",
                    "conflicting_label_duplicate_groups_quarantined",
                    "conflicting_duplicate_rows_quarantined", "retained_rows",
                )
            },
            "hashes": {
                "raw_source": regression_manifest["input_file_sha256"][f"{endpoint}/raw"],
                "train_expected": regression.expected_hashes[f"{endpoint}/train"],
                "train_verified": regression.actual_hashes[f"{endpoint}/train"],
                "validation_expected": regression.expected_hashes[f"{endpoint}/validation"],
                "validation_verified": regression.actual_hashes[f"{endpoint}/validation"],
            },
            "task_weight": float(weights[index]),
            "task_weight_times_training_labels": float(weights[index] * train_counts[index]),
        }

    audit = {
        "schema_version": "1.0.0",
        "protocol": "chemprop_real_train_validation_preflight",
        "locked_test_csv_opened": False,
        "endpoint_order": endpoint_order,
        "regression": {
            "endpoints": endpoint_audits,
            "sparse_matrix": {
                split: _sparse_matrix_summary(frame, endpoint_order)
                for split, frame in (("train", regression.train), ("validation", regression.validation))
            },
            "pairwise_observed_label_overlap": {
                split: _pairwise_overlap(frame, endpoint_order)
                for split, frame in (("train", regression.train), ("validation", regression.validation))
            },
            "task_weighting": {
                "equation": "w_t=(1/n_t)/mean_j(1/n_j)",
                "training_loss_equation": TRAINING_LOSS_EQUATION,
                "derivation": "w_t*n_t is constant across endpoints",
                "effective_training_label_counts": dict(zip(endpoint_order, train_counts)),
                "resolved_weights": dict(zip(endpoint_order, map(float, weights))),
            },
            "checkpoint_selection": {
                "metric": "equal-task mean standardized MAE",
                "equation": SELECTION_METRIC_EQUATION,
                "split": "validation",
                "mode": "min",
            },
            "leakage": {
                "train_validation_canonical_overlap": 0,
                "train_validation_murcko_scaffold_overlap": 0,
                "manifest_audit": regression_manifest["leakage_audit"],
            },
            "split_manifest_sha256": regression.split_manifest_sha256,
        },
        "bbb": {
            "positive_class_meaning": bbb_config.raw["positive_class_meaning"],
            "evidence_status": bbb_config.raw["evidence_status"],
            "class_counts": {
                "train": _class_counts(bbb.train),
                "validation": _class_counts(bbb.validation),
                "locked_test_from_manifest": bbb_manifest["endpoints"]["bbb_martins"]["splits"]["test"]["class_counts"],
            },
            "hashes": {
                "train_expected": bbb.expected_hashes["train"],
                "train_verified": bbb.actual_hashes["train"],
                "validation_expected": bbb.expected_hashes["validation"],
                "validation_verified": bbb.actual_hashes["validation"],
                "locked_test_expected_from_manifest": bbb.expected_hashes["test"],
            },
            "locked_test_csv_opened": False,
            "calibration_fit_split": "validation",
            "threshold_selection_split": "validation",
            "leakage": {
                "train_validation_canonical_overlap": 0,
                "train_validation_murcko_scaffold_overlap": 0,
                "manifest_audit": bbb_manifest["audit_summary"],
            },
            "split_manifest_id": bbb_manifest["split_manifest_id"],
            "split_manifest_sha256": bbb.split_manifest_sha256,
        },
    }
    return audit, regression, bbb


def run_real_validation_baselines(
    regression_config: ChempropExperimentConfig,
    regression: VerifiedDevelopmentData,
    bbb: VerifiedDevelopmentData,
) -> dict[str, Any]:
    regression_results = {}
    for endpoint, metadata in regression_config.tasks.items():
        train = regression.train.loc[
            regression.train[endpoint].notna(), ["canonical_smiles", endpoint]
        ].rename(columns={endpoint: "target"})
        validation = regression.validation.loc[
            regression.validation[endpoint].notna(), ["canonical_smiles", endpoint]
        ].rename(columns={endpoint: "target"})
        transformed_train = train.copy()
        transformed_validation = validation.copy()
        transformed_train["target"] = _scientific_transform(
            train["target"].to_numpy(dtype=float), metadata["target_transform"]
        )
        transformed_validation["target"] = _scientific_transform(
            validation["target"].to_numpy(dtype=float), metadata["target_transform"]
        )
        metrics = run_regression_baselines(
            transformed_train,
            transformed_validation,
            include_esol=endpoint == "solubility_aqsoldb",
            evaluation_targets=validation["target"].to_numpy(dtype=float),
            inverse_prediction=(lambda values: np.power(10.0, values))
            if metadata["target_transform"] == "log10" else None,
        )
        regression_results[endpoint] = {
            "unit": metadata["units"],
            "evaluation_space": "original endpoint unit",
            "metrics": metrics,
        }
    return {
        "protocol": "train_fit_validation_evaluation_only",
        "locked_test_csv_opened": False,
        "regression": regression_results,
        "bbb": run_classification_baselines(bbb.train, bbb.validation),
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scientific_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.copy()
    if transform == "log10":
        if np.any(values <= 0):
            raise ValueError("log10 scientific transforms require positive values.")
        return np.log10(values)
    raise ValueError(f"Unsupported scientific transform: {transform}")


def _sparse_matrix_summary(frame, tasks: list[str]) -> dict[str, Any]:
    counts = frame[tasks].notna().sum(axis=1)
    return {
        "unique_canonical_molecules": int(frame["canonical_smiles"].nunique()),
        "rows": int(len(frame)),
        "molecules_by_observed_target_count": {
            str(number): int((counts == number).sum()) for number in range(1, len(tasks) + 1)
        },
        "zero_target_rows": int((counts == 0).sum()),
    }


def _pairwise_overlap(frame, tasks: list[str]) -> dict[str, int]:
    return {
        f"{left}|{right}": int((frame[left].notna() & frame[right].notna()).sum())
        for left, right in combinations(tasks, 2)
    }


def _class_counts(frame) -> dict[str, int]:
    counts = frame["target"].astype(int).value_counts()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "build_real_data_preflight_audit",
    "run_real_validation_baselines",
    "write_json",
]
