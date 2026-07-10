"""Local classical ML baselines for prepared ADMET datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.schema import (
    validate_dataset_columns,
    validate_smiles_column,
    validate_split_column,
    validate_target_column,
)


def train_baseline_model(
    input_csv: str | Path,
    config_path: str | Path,
    model_output_path: str | Path,
    metrics_json_path: str | Path,
) -> dict[str, Any]:
    """Train and evaluate a simple local baseline model from a prepared CSV."""

    config = load_endpoint_config(config_path)
    df = pd.read_csv(input_csv)
    _validate_prepared_dataset(df, config)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty:
        raise ValueError("Prepared dataset must contain at least one train row.")
    if test_df.empty:
        raise ValueError("Prepared dataset must contain at least one test row.")

    pipeline, model_type = _build_pipeline(config)
    x_train = train_df["smiles"].astype(str)
    x_test = test_df["smiles"].astype(str)
    y_train = _coerce_target(train_df["target"], config)
    y_test = _coerce_target(test_df["target"], config)

    pipeline.fit(x_train, y_train)
    metrics = _evaluate_model(pipeline, x_test, y_test, config)

    result = {
        "endpoint_id": config.endpoint_id,
        "task_type": config.task_type,
        "model_type": model_type,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": metrics,
    }

    model_path = Path(model_output_path)
    metrics_path = Path(metrics_json_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _validate_prepared_dataset(df: pd.DataFrame, config: EndpointConfig) -> None:
    validate_dataset_columns(df, config)
    validate_smiles_column(df, config)
    validate_target_column(df, config)
    validate_split_column(df)


def _build_pipeline(config: EndpointConfig) -> tuple[Pipeline, str]:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False)

    if config.task_type == "binary_classification":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        return Pipeline([("tfidf", vectorizer), ("model", model)]), "tfidf_logistic_regression"

    if config.task_type == "regression":
        model = Ridge(alpha=1.0, random_state=42)
        return Pipeline([("tfidf", vectorizer), ("model", model)]), "tfidf_ridge_regression"

    raise ValueError(f"Unsupported task_type '{config.task_type}'.")


def _coerce_target(target: pd.Series, config: EndpointConfig) -> pd.Series:
    numeric_target = pd.to_numeric(target, errors="raise")
    if config.task_type == "binary_classification":
        return numeric_target.astype(int)
    if config.task_type == "regression":
        return numeric_target.astype(float)
    raise ValueError(f"Unsupported task_type '{config.task_type}'.")


def _evaluate_model(
    pipeline: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
    config: EndpointConfig,
) -> dict[str, float]:
    predictions = pipeline.predict(x_test)

    if config.task_type == "binary_classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        if len(set(y_test)) > 1 and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            metrics["auroc"] = float(roc_auc_score(y_test, probabilities))
        return metrics

    if config.task_type == "regression":
        return {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r2": float(r2_score(y_test, predictions)),
        }

    raise ValueError(f"Unsupported task_type '{config.task_type}'.")
