"""Lazy Chemprop 2.3.1 graph-only training and prediction."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from admet_platform.chemprop.calibration import fit_platt_calibrator, select_maximum_mcc_threshold
from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.chemprop.losses import (
    SELECTION_METRIC_EQUATION,
    TRAINING_LOSS_EQUATION,
    equal_task_mean_standardized_mae_metric,
    inverse_count_equal_endpoint_weights,
)
from admet_platform.chemprop.metrics import classification_metrics, regression_metrics


def train_graph_model(
    config: ChempropExperimentConfig,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int,
    smoke: bool = False,
    accelerator_override: str | None = None,
) -> dict[str, Any]:
    """Train one random-initialized graph-only D-MPNN without opening test data."""

    # Heavy dependencies are intentionally local to the training call.
    import torch
    from chemprop import data, featurizers, models, nn
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    if seed not in config.raw["seeds"] and not smoke:
        raise ValueError(f"Seed {seed} is not in the frozen experiment seed set.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    if accelerator_override not in {None, "cpu", "cuda", "auto"}:
        raise ValueError("accelerator_override must be cpu, cuda, auto, or None.")
    if accelerator_override == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    task_ids = list(config.tasks) if config.tasks else [config.endpoint_id]
    target_columns = task_ids if config.tasks else ["target"]
    transformed_train = _transform_regression_targets(train, config) if config.tasks else train
    transformed_validation = (
        _transform_regression_targets(validation, config) if config.tasks else validation
    )
    train_points = _datapoints(transformed_train, data, target_columns)
    validation_points = _datapoints(transformed_validation, data, target_columns)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(train_points, featurizer=featurizer, n_workers=0)
    validation_dataset = data.MoleculeDataset(validation_points, featurizer=featurizer, n_workers=0)

    scaler_payload = None
    output_transform = None
    if config.task_type == "regression":
        scaler = train_dataset.normalize_targets()
        validation_dataset.normalize_targets(scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        label_counts = {
            task: int(train[task].notna().sum()) if config.tasks else len(train)
            for task in task_ids
        }
        scaler_payload = {
            "fit_split": "train",
            "per_endpoint": {
                task: {
                    "scientific_transform": config.tasks[task]["target_transform"]
                    if config.tasks else "identity",
                    "mean": float(scaler.mean_[index]),
                    "scale": float(scaler.scale_[index]),
                    "train_label_count": label_counts[task],
                }
                for index, task in enumerate(task_ids)
            },
            "validation_statistics_used": False,
            "test_statistics_used": False,
        }
        (destination / "target_scaler.json").write_text(
            json.dumps(scaler_payload, indent=2) + "\n", encoding="utf-8"
        )

    training = dict(config.training)
    model_config = dict(config.model)
    if smoke:
        training.update({"batch_size": 4, "max_epochs": 2, "early_stopping_patience": 2})
        model_config.update({"message_hidden_dim": 32, "ffn_hidden_dim": 32, "message_passing_depth": 2})
    train_loader = data.build_dataloader(
        train_dataset, batch_size=int(training["batch_size"]), num_workers=0, seed=seed, shuffle=True
    )
    validation_loader = data.build_dataloader(
        validation_dataset, batch_size=int(training["batch_size"]), num_workers=0, seed=seed, shuffle=False
    )

    message_passing = nn.BondMessagePassing(
        d_h=int(model_config["message_hidden_dim"]),
        depth=int(model_config["message_passing_depth"]),
        dropout=float(model_config["dropout"]),
    )
    aggregation = nn.MeanAggregation()
    if config.task_type == "regression":
        task_weights = _equal_endpoint_task_weights(train, task_ids)
        predictor = nn.RegressionFFN(
            n_tasks=len(task_ids),
            input_dim=int(model_config["message_hidden_dim"]),
            hidden_dim=int(model_config["ffn_hidden_dim"]),
            n_layers=int(model_config["ffn_num_layers"]),
            dropout=float(model_config["dropout"]),
            criterion=nn.MSE(task_weights=torch.tensor(task_weights, dtype=torch.float32)),
            output_transform=output_transform,
        )
        metrics = [equal_task_mean_standardized_mae_metric(len(task_ids))]
        monitor, mode = "val/mean_standardized_mae", "min"
    else:
        predictor = nn.BinaryClassificationFFN(
            input_dim=int(model_config["message_hidden_dim"]),
            hidden_dim=int(model_config["ffn_hidden_dim"]),
            n_layers=int(model_config["ffn_num_layers"]),
            dropout=float(model_config["dropout"]),
        )
        metrics = [nn.BinaryAUROC()]
        monitor, mode = "val/roc", "max"
    model = models.MPNN(
        message_passing, aggregation, predictor, metrics=metrics,
        warmup_epochs=int(training["warmup_epochs"]), init_lr=float(training["init_lr"]),
        max_lr=float(training["max_lr"]), final_lr=float(training["final_lr"]),
    )
    checkpoint = ModelCheckpoint(
        dirpath=destination / "checkpoints", filename="best", monitor=monitor, mode=mode,
        save_top_k=1, save_last=True,
    )
    callbacks = [checkpoint, EarlyStopping(
        monitor=monitor, mode=mode, patience=int(training["early_stopping_patience"])
    )]
    accelerator = accelerator_override or ("cpu" if smoke else training["accelerator"])
    if accelerator == "cuda":
        torch.cuda.reset_peak_memory_stats()
    csv_logger = CSVLogger(save_dir=destination / "lightning_logs", name="training")
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1,
        max_epochs=int(training["max_epochs"]), callbacks=callbacks,
        gradient_clip_val=float(training["gradient_clip_norm"]), deterministic=True,
        logger=csv_logger, enable_progress_bar=False, enable_model_summary=False,
        log_every_n_steps=1,
        default_root_dir=destination, num_sanity_val_steps=0,
    )
    started = time.perf_counter()
    trainer.fit(model, train_loader, validation_loader)
    training_seconds = time.perf_counter() - started
    # The checkpoint was created by this process and includes Chemprop metric objects in addition
    # to tensor weights; PyTorch 2.13 otherwise rejects those trusted local objects by default.
    predictions = trainer.predict(model, validation_loader, ckpt_path="best", weights_only=False)
    values = np.concatenate([np.asarray(item) for item in predictions], axis=0)
    values = values.reshape(len(validation), len(task_ids))
    result: dict[str, Any] = {
        "endpoint": config.endpoint, "task_type": config.task_type, "seed": seed,
        "checkpoint": checkpoint.best_model_path, "monitor": monitor,
        "target_scaler": scaler_payload,
        "runtime": {
            "training_seconds": training_seconds,
            "accelerator_requested": accelerator,
            "device": str(trainer.strategy.root_device),
            "epochs_completed": int(trainer.current_epoch),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
            if trainer.strategy.root_device.type == "cuda" else 0,
            "learning_curve_csv": str(Path(csv_logger.log_dir) / "metrics.csv"),
        },
    }
    if config.task_type == "regression":
        endpoint_metrics: dict[str, Any] = {}
        endpoint_prediction_files: dict[str, str] = {}
        endpoint_metadata_files: dict[str, str] = {}
        prediction_root = destination / "validation_predictions"
        metadata_root = destination / "endpoint_metadata"
        prediction_root.mkdir(exist_ok=True)
        metadata_root.mkdir(exist_ok=True)
        for index, endpoint in enumerate(task_ids):
            observed = validation[endpoint].to_numpy(dtype=float)
            present = np.isfinite(observed)
            predicted = _inverse_scientific_transform(
                values[:, index], config.tasks[endpoint]["target_transform"]
            )
            endpoint_frame = validation.loc[
                present, ["molecule_id", "canonical_smiles", endpoint]
            ].rename(columns={endpoint: "target"})
            endpoint_frame["prediction"] = predicted[present]
            endpoint_path = prediction_root / f"{endpoint}.csv"
            endpoint_frame.to_csv(endpoint_path, index=False)
            endpoint_metrics[endpoint] = {
                "unit": config.tasks[endpoint]["units"],
                "target_transform": config.tasks[endpoint]["target_transform"],
                "label_count": int(present.sum()),
                **regression_metrics(observed[present], predicted[present]),
            }
            endpoint_prediction_files[endpoint] = str(endpoint_path)
            metadata_path = metadata_root / f"{endpoint}.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "endpoint": endpoint,
                        "tdc_name": config.tasks[endpoint]["tdc_name"],
                        "target_definition": config.tasks[endpoint]["target_definition"],
                        "unit": config.tasks[endpoint]["units"],
                        "scientific_transform": config.tasks[endpoint]["target_transform"],
                        "normalization": scaler_payload["per_endpoint"][endpoint],
                        "task_weight": float(task_weights[index]),
                        "validation_metrics_original_units": endpoint_metrics[endpoint],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            endpoint_metadata_files[endpoint] = str(metadata_path)
        result["validation_metrics"] = endpoint_metrics
        result["validation_prediction_files"] = endpoint_prediction_files
        result["endpoint_metadata_files"] = endpoint_metadata_files
        result["task_weights"] = {
            endpoint: float(weight) for endpoint, weight in zip(task_ids, task_weights)
        }
        result["training_loss_contract"] = {
            "equation": TRAINING_LOSS_EQUATION,
            "missing_label_mask": "isfinite(target)",
            "reduction": "weighted sum over observed cells divided by observed-cell count",
            "weight_derivation": "inverse training label count, normalized to mean one",
            "effective_training_label_counts": label_counts,
        }
        result["checkpoint_selection_contract"] = {
            "equation": SELECTION_METRIC_EQUATION,
            "metric": "equal-task mean standardized MAE",
            "split": "validation",
            "mode": "min",
            "test_statistics_used": False,
        }
    else:
        observed = validation["target"].to_numpy()
        prediction_frame = validation[["molecule_id", "canonical_smiles", "target"]].copy()
        values = values[:, 0]
        prediction_frame["probability_uncalibrated"] = values
        calibrator = fit_platt_calibrator(observed.astype(int), values)
        calibrated = calibrator.transform(values)
        threshold = select_maximum_mcc_threshold(observed.astype(int), calibrated)
        prediction_frame["probability_calibrated"] = calibrated
        result["calibrator"] = {
            "method": "platt_scaling", "fit_split": "validation",
            "coefficient": calibrator.coefficient, "intercept": calibrator.intercept,
        }
        result["validation_selected_threshold"] = threshold
        result["validation_metrics_uncalibrated_0_5"] = classification_metrics(observed, values, 0.5)
        result["validation_metrics_calibrated_0_5"] = classification_metrics(observed, calibrated, 0.5)
        result["validation_metrics_calibrated_selected"] = classification_metrics(observed, calibrated, threshold)
        (destination / "calibrator.json").write_text(
            json.dumps(result["calibrator"], indent=2) + "\n", encoding="utf-8"
        )
        (destination / "validation_threshold.json").write_text(
            json.dumps({"fit_split": "validation", "method": "maximum_mcc", "threshold": threshold}, indent=2) + "\n",
            encoding="utf-8",
        )
        prediction_frame.to_csv(destination / "validation_predictions.csv", index=False)
    resolved = {**config.raw, "resolved_seed": seed, "smoke": smoke, "resolved_model": model_config, "resolved_training": training}
    (destination / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _datapoints(frame: pd.DataFrame, data_module: Any, target_columns: list[str]) -> list[Any]:
    return [
        data_module.MoleculeDatapoint.from_smi(
            str(row["canonical_smiles"]),
            y=np.asarray([float(row[column]) for column in target_columns], dtype=np.float32),
            name=str(row["molecule_id"]),
        )
        for row in frame.to_dict(orient="records")
    ]


def _transform_regression_targets(
    frame: pd.DataFrame, config: ChempropExperimentConfig
) -> pd.DataFrame:
    output = frame.copy()
    for endpoint, metadata in config.tasks.items():
        values = output[endpoint].to_numpy(dtype=float, copy=True)
        present = np.isfinite(values)
        if metadata["target_transform"] == "log10":
            if np.any(values[present] <= 0):
                raise ValueError(f"{endpoint} contains non-positive values for log10 transform.")
            values[present] = np.log10(values[present])
        output[endpoint] = values
    return output


def _inverse_scientific_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values
    if transform == "log10":
        return np.power(10.0, values)
    raise ValueError(f"Unsupported scientific transform: {transform}")


def _equal_endpoint_task_weights(train: pd.DataFrame, task_ids: list[str]) -> np.ndarray:
    counts = [int(train[task].notna().sum()) for task in task_ids]
    return inverse_count_equal_endpoint_weights(counts)
