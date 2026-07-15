"""Local ChemBERTa fine-tuning for prepared ADMET datasets."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from admet_platform.config import load_endpoint_config
from admet_platform.models.artifacts import write_json
from admet_platform.models.metrics import classification_metrics, regression_metrics


DEFAULT_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


@dataclass(frozen=True)
class ChemBERTaTrainingConfig:
    """Serializable ChemBERTa local training configuration."""

    model_name: str = DEFAULT_CHEMBERTA_MODEL
    max_sequence_length: int = 128
    learning_rate: float = 2e-5
    training_epochs: int = 3
    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    random_seed: int = 42
    development_row_limit: int | None = None
    cache_dir: str | None = None
    local_files_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_chemberta_model(
    train_csv: str | Path,
    validation_csv: str | Path,
    test_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    training_config: ChemBERTaTrainingConfig | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune ChemBERTa locally on prepared train/validation/test CSV files."""

    start = time.perf_counter()
    endpoint_config = load_endpoint_config(config_path)
    chemberta_config = training_config or ChemBERTaTrainingConfig()
    _set_seed(chemberta_config.random_seed)

    train_df = _load_prepared_split(train_csv, chemberta_config.development_row_limit)
    validation_df = _load_prepared_split(validation_csv, chemberta_config.development_row_limit)
    test_df = _load_prepared_split(test_csv, chemberta_config.development_row_limit)
    warnings_list: list[str] = []
    if chemberta_config.development_row_limit is not None:
        warnings_list.append("development row limit was used; this is a smoke/development run")

    components = _load_transformers_components(endpoint_config.task_type, chemberta_config)
    tokenizer = components["tokenizer"]
    model = components["model"]
    torch = components["torch"]
    _set_torch_seed(torch, chemberta_config.random_seed)

    train_dataset = _build_torch_dataset(train_df, endpoint_config.task_type, tokenizer, chemberta_config, torch)
    validation_dataset = _build_torch_dataset(
        validation_df,
        endpoint_config.task_type,
        tokenizer,
        chemberta_config,
        torch,
    )
    test_dataset = _build_torch_dataset(test_df, endpoint_config.task_type, tokenizer, chemberta_config, torch)

    history, best_state, best_checkpoint, best_validation_metric, train_warnings = _fit_model(
        model=model,
        torch=torch,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        validation_df=validation_df,
        task_type=endpoint_config.task_type,
        training_config=chemberta_config,
    )
    warnings_list.extend(train_warnings)
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_predictions, validation_metrics, validation_warnings = _predict_and_evaluate(
        model,
        torch,
        validation_dataset,
        validation_df,
        endpoint_config.task_type,
        chemberta_config.evaluation_batch_size,
    )
    test_predictions, test_metrics, test_warnings = _predict_and_evaluate(
        model,
        torch,
        test_dataset,
        test_df,
        endpoint_config.task_type,
        chemberta_config.evaluation_batch_size,
    )
    warnings_list.extend([f"validation: {warning}" for warning in validation_warnings])
    warnings_list.extend([f"test: {warning}" for warning in test_warnings])

    output_path = Path(output_dir)
    model_root = Path(model_dir) if model_dir is not None else output_path
    model_path = model_root / "model"
    tokenizer_path = model_root / "tokenizer"
    output_path.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    validation_predictions.to_csv(output_path / "predictions_validation.csv", index=False)
    test_predictions.to_csv(output_path / "predictions_test.csv", index=False)

    runtime_seconds = time.perf_counter() - start
    metrics_payload = {
        "endpoint_id": endpoint_config.endpoint_id,
        "task_type": endpoint_config.task_type,
        "validation": validation_metrics,
        "test": test_metrics,
        "warnings": warnings_list,
    }
    model_config_payload = chemberta_config.to_dict()
    training_metadata = {
        "endpoint_id": endpoint_config.endpoint_id,
        "task_type": endpoint_config.task_type,
        "source_dataset": endpoint_config.tdc_name,
        "pretrained_model_name": chemberta_config.model_name,
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "random_seed": chemberta_config.random_seed,
        "training_row_count": int(len(train_df)),
        "validation_row_count": int(len(validation_df)),
        "test_row_count": int(len(test_df)),
        "development_row_limit": chemberta_config.development_row_limit,
        "max_sequence_length": chemberta_config.max_sequence_length,
        "hyperparameters": model_config_payload,
        "package_versions": _package_versions(components),
        "creation_timestamp": datetime.now(UTC).isoformat(),
        "best_checkpoint": best_checkpoint,
        "best_validation_metric": best_validation_metric,
        "runtime_seconds": runtime_seconds,
        "warnings": warnings_list,
        "model_output_dir": str(model_path),
        "tokenizer_output_dir": str(tokenizer_path),
    }

    write_json(output_path / "metrics.json", metrics_payload)
    write_json(output_path / "training_metadata.json", training_metadata)
    write_json(output_path / "model_config.json", model_config_payload)
    write_json(output_path / "training_history.json", {"history": history})
    write_json(output_path / "warnings.json", {"warnings": warnings_list})
    return {
        "metrics": metrics_payload,
        "training_metadata": training_metadata,
        "model_config": model_config_payload,
        "history": history,
        "output_dir": str(output_path),
    }


def _load_transformers_components(task_type: str, config: ChemBERTaTrainingConfig) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # noqa: BLE001 - torch can fail from DLL loading on Windows.
        raise RuntimeError(
            "ChemBERTa training requires torch and transformers. Install project dependencies "
            "from requirements.txt before running local fine-tuning. If torch fails to import on "
            "Windows, reinstall a compatible CPU build in the active conda environment."
        ) from exc

    num_labels = 1 if task_type == "regression" else 2
    problem_type = "regression" if task_type == "regression" else "single_label_classification"
    try:
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            problem_type=problem_type,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=model_config,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    except Exception as exc:  # noqa: BLE001 - convert HF errors into a project-level message.
        mode = " with local-files-only mode enabled" if config.local_files_only else ""
        raise RuntimeError(
            f"Unable to load ChemBERTa checkpoint '{config.model_name}'{mode}. "
            "Use a valid cached model or allow/download the requested checkpoint explicitly."
        ) from exc
    return {"torch": torch, "tokenizer": tokenizer, "model": model}


def _build_torch_dataset(
    df: pd.DataFrame,
    task_type: str,
    tokenizer: Any,
    config: ChemBERTaTrainingConfig,
    torch: Any,
) -> Any:
    smiles = df["canonical_smiles"].astype(str).tolist()
    encoded = tokenizer(
        smiles,
        padding=True,
        truncation=True,
        max_length=config.max_sequence_length,
        return_tensors="pt",
    )
    target = pd.to_numeric(df["target"], errors="raise")
    label_dtype = torch.float32 if task_type == "regression" else torch.long
    labels = torch.tensor(target.to_numpy(), dtype=label_dtype)
    if task_type == "regression":
        labels = labels.reshape(-1, 1)
    return torch.utils.data.TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def _fit_model(
    model: Any,
    torch: Any,
    train_dataset: Any,
    validation_dataset: Any,
    validation_df: pd.DataFrame,
    task_type: str,
    training_config: ChemBERTaTrainingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None, float | None, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
    )
    best_score = -math.inf
    best_state = None
    best_checkpoint = None
    patience_remaining = training_config.early_stopping_patience
    history: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for epoch in range(1, training_config.training_epochs + 1):
        model.train()
        losses: list[float] = []
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            losses.append(float(outputs.loss.detach().cpu()))

        predictions, metrics, metric_warnings = _predict_and_evaluate(
            model,
            torch,
            validation_dataset,
            validation_df,
            task_type,
            training_config.evaluation_batch_size,
        )
        del predictions
        score, selected_metric = _checkpoint_score(task_type, metrics)
        warnings_list.extend([f"epoch {epoch}: {warning}" for warning in metric_warnings])
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)) if losses else None,
                "validation_metrics": metrics,
                "checkpoint_metric": selected_metric,
                "checkpoint_score": score,
            }
        )
        if score is not None and score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint = f"epoch_{epoch}"
            patience_remaining = training_config.early_stopping_patience
        else:
            patience_remaining -= 1
            if patience_remaining < 0:
                warnings_list.append(f"early stopping triggered after epoch {epoch}")
                break

    if best_score == -math.inf:
        best_score = None
    return history, best_state, best_checkpoint, best_score, warnings_list


def _predict_and_evaluate(
    model: Any,
    torch: Any,
    dataset: Any,
    df: pd.DataFrame,
    task_type: str,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    logits_batches = []
    with torch.no_grad():
        for input_ids, attention_mask, _labels in loader:
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            logits_batches.append(outputs.logits.detach().cpu())
    logits = torch.cat(logits_batches, dim=0)
    y_true = pd.to_numeric(df["target"], errors="raise").to_numpy()
    predictions = df[[column for column in ["molecule_id", "canonical_smiles"] if column in df.columns]].copy()
    predictions["observed_target"] = y_true

    if task_type == "binary_classification":
        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
        predicted_class = (probabilities >= 0.5).astype(int)
        predictions["predicted_class"] = predicted_class
        predictions["predicted_probability"] = probabilities
        metrics, warnings_list = classification_metrics(y_true.astype(int), predicted_class, probabilities)
    else:
        predicted_value = logits.reshape(-1).numpy().astype(float)
        predictions["predicted_value"] = predicted_value
        predictions["residual"] = y_true.astype(float) - predicted_value
        metrics, warnings_list = regression_metrics(y_true.astype(float), predicted_value)
    return predictions, metrics, warnings_list


def _checkpoint_score(task_type: str, metrics: dict[str, Any]) -> tuple[float | None, str]:
    if task_type == "binary_classification":
        if metrics.get("roc_auc") is not None:
            return float(metrics["roc_auc"]), "roc_auc"
        if metrics.get("accuracy") is not None:
            return float(metrics["accuracy"]), "accuracy_fallback"
        return None, "unavailable"
    if metrics.get("rmse") is not None:
        return -float(metrics["rmse"]), "negative_rmse"
    return None, "unavailable"


def _load_prepared_split(path: str | Path, row_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Prepared split is empty: {path}")
    if "canonical_smiles" not in df.columns:
        raise ValueError("Prepared split must include canonical_smiles.")
    if row_limit is not None:
        df = df.head(row_limit).copy()
    return df


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _set_torch_seed(torch: Any, seed: int) -> None:
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def _package_versions(components: dict[str, Any]) -> dict[str, Any]:
    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    torch = components.get("torch")
    if torch is not None:
        versions["torch"] = getattr(torch, "__version__", None)
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = None
    return versions
