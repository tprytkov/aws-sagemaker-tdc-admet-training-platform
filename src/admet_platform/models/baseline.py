"""Split-based local classical ML baselines for ADMET datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from admet_platform.config import load_endpoint_config
from admet_platform.features import DESCRIPTOR_NAMES, FeatureConfig, featurize_dataframe
from admet_platform.models.artifacts import write_json
from admet_platform.models.metrics import classification_metrics, regression_metrics


def train_local_baseline(
    train_csv: str | Path,
    validation_csv: str | Path,
    test_csv: str | Path,
    config_path: str | Path,
    feature_type: str,
    output_dir: str | Path,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train a local baseline from prepared train/validation/test CSV files."""

    endpoint_config = load_endpoint_config(config_path)
    feature_config = FeatureConfig(
        feature_type=feature_type,  # type: ignore[arg-type]
        morgan_radius=morgan_radius,
        morgan_bits=morgan_bits,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(train_csv)
    validation_raw = pd.read_csv(validation_csv)
    test_raw = pd.read_csv(test_csv)

    train_features, train_rejected, feature_metadata = featurize_dataframe(train_raw, feature_config)
    validation_features, validation_rejected, _ = featurize_dataframe(validation_raw, feature_config)
    test_features, test_rejected, _ = featurize_dataframe(test_raw, feature_config)

    warnings_list = _rejection_warnings(
        train_rejected=train_rejected,
        validation_rejected=validation_rejected,
        test_rejected=test_rejected,
    )
    if train_features.empty:
        raise ValueError("Training split has no valid featurized rows.")

    feature_columns = _feature_columns(feature_type, morgan_bits)
    x_train = train_features[feature_columns]
    x_validation = validation_features[feature_columns]
    x_test = test_features[feature_columns]
    y_train = _target_array(train_features, endpoint_config.task_type)
    y_validation = _target_array(validation_features, endpoint_config.task_type)
    y_test = _target_array(test_features, endpoint_config.task_type)

    model, model_type = _build_model(endpoint_config.task_type, feature_type, random_seed)
    model.fit(x_train, y_train)

    validation_predictions = _predict(model, x_validation, y_validation, validation_features, endpoint_config.task_type)
    test_predictions = _predict(model, x_test, y_test, test_features, endpoint_config.task_type)

    validation_metrics, validation_warnings = _evaluate(
        endpoint_config.task_type,
        y_validation,
        validation_predictions,
    )
    test_metrics, test_warnings = _evaluate(endpoint_config.task_type, y_test, test_predictions)
    warnings_list.extend([f"validation: {warning}" for warning in validation_warnings])
    warnings_list.extend([f"test: {warning}" for warning in test_warnings])

    artifact_payload = {
        "model": model,
        "feature_config": feature_config.to_dict(),
        "feature_columns": feature_columns,
        "endpoint_id": endpoint_config.endpoint_id,
        "task_type": endpoint_config.task_type,
        "model_type": model_type,
    }
    joblib.dump(artifact_payload, output_path / "model.joblib")

    validation_predictions["dataframe"].to_csv(output_path / "predictions_validation.csv", index=False)
    test_predictions["dataframe"].to_csv(output_path / "predictions_test.csv", index=False)

    metrics_payload = {
        "endpoint_id": endpoint_config.endpoint_id,
        "task_type": endpoint_config.task_type,
        "feature_type": feature_type,
        "model_type": model_type,
        "validation": validation_metrics,
        "test": test_metrics,
        "warnings": warnings_list,
    }
    training_metadata = {
        "endpoint_id": endpoint_config.endpoint_id,
        "task_type": endpoint_config.task_type,
        "feature_type": feature_type,
        "model_type": model_type,
        "random_seed": int(random_seed),
        "training_row_count": int(len(train_features)),
        "validation_row_count": int(len(validation_features)),
        "test_row_count": int(len(test_features)),
        "feature_count": int(len(feature_columns)),
        "package_versions": {
            "python_packages": {
                "rdkit": getattr(rdkit, "__version__", None),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            }
        },
        "creation_timestamp": datetime.now(UTC).isoformat(),
        "warnings": warnings_list,
    }
    feature_metadata = {
        **feature_metadata,
        "feature_type": feature_type,
        "feature_columns": feature_columns,
    }

    write_json(output_path / "metrics.json", metrics_payload)
    write_json(output_path / "training_metadata.json", training_metadata)
    write_json(output_path / "feature_metadata.json", feature_metadata)

    return {
        "metrics": metrics_payload,
        "training_metadata": training_metadata,
        "feature_metadata": feature_metadata,
        "output_dir": str(output_path),
    }


def _build_model(task_type: str, feature_type: str, random_seed: int) -> tuple[Pipeline | Ridge, str]:
    if task_type == "binary_classification":
        classifier = LogisticRegression(max_iter=1000, random_state=random_seed, class_weight="balanced")
        if feature_type == "descriptors":
            return (
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", classifier),
                    ]
                ),
                "descriptor_logistic_regression",
            )
        return classifier, "morgan_logistic_regression"

    if task_type == "regression":
        regressor = Ridge(alpha=1.0, random_state=random_seed)
        if feature_type == "descriptors":
            return (
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", regressor),
                    ]
                ),
                "descriptor_ridge_regression",
            )
        return regressor, "morgan_ridge_regression"

    raise ValueError(f"Unsupported task_type '{task_type}'.")


def _predict(
    model: Any,
    x_values: pd.DataFrame,
    y_true: np.ndarray,
    feature_df: pd.DataFrame,
    task_type: str,
) -> dict[str, Any]:
    preserved = feature_df[[column for column in ["molecule_id", "canonical_smiles"] if column in feature_df]].copy()
    preserved["observed_target"] = y_true

    if task_type == "binary_classification":
        predicted_class = model.predict(x_values).astype(int)
        predicted_probability = model.predict_proba(x_values)[:, 1]
        preserved["predicted_class"] = predicted_class
        preserved["predicted_probability"] = predicted_probability
        return {
            "dataframe": preserved,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_probability,
        }

    predicted_value = model.predict(x_values).astype(float)
    preserved["predicted_value"] = predicted_value
    preserved["residual"] = y_true.astype(float) - predicted_value
    return {"dataframe": preserved, "predicted_value": predicted_value}


def _evaluate(task_type: str, y_true: np.ndarray, predictions: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if task_type == "binary_classification":
        return classification_metrics(
            y_true=y_true.astype(int),
            y_pred=predictions["predicted_class"].astype(int),
            y_probability=predictions["predicted_probability"].astype(float),
        )
    return regression_metrics(y_true.astype(float), predictions["predicted_value"].astype(float))


def _target_array(df: pd.DataFrame, task_type: str) -> np.ndarray:
    target = pd.to_numeric(df["target"], errors="raise")
    if task_type == "binary_classification":
        return target.astype(int).to_numpy()
    return target.astype(float).to_numpy()


def _feature_columns(feature_type: str, morgan_bits: int) -> list[str]:
    if feature_type == "descriptors":
        return list(DESCRIPTOR_NAMES)
    if feature_type == "morgan":
        return [f"morgan_{index:04d}" for index in range(morgan_bits)]
    raise ValueError("feature_type must be either 'descriptors' or 'morgan'.")


def _rejection_warnings(
    train_rejected: pd.DataFrame,
    validation_rejected: pd.DataFrame,
    test_rejected: pd.DataFrame,
) -> list[str]:
    warnings_list: list[str] = []
    for split_name, rejected in {
        "train": train_rejected,
        "validation": validation_rejected,
        "test": test_rejected,
    }.items():
        if not rejected.empty:
            warnings_list.append(f"{split_name}: {len(rejected)} invalid molecule row(s) were rejected.")
    return warnings_list
