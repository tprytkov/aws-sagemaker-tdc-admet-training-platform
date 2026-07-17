"""Validation, checkpoint selection, early stopping, and baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score,
)

from admet_platform.training.multitask_trainer import MultiTaskTrainer


def evaluate_validation(
    trainer: MultiTaskTrainer, validation_loaders: Mapping[str, Any],
    output_dir: Path, global_step: int,
) -> dict[str, Any]:
    """Evaluate every configured endpoint on its complete validation loader."""
    evaluation_dir = output_dir / "validation" / f"step_{global_step:08d}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    endpoints: dict[str, Any] = {}
    for task in trainer.model.task_names:
        rows: list[dict[str, Any]] = []
        weighted_loss = 0.0
        example_count = 0
        for batch in validation_loaders[task]:
            record = trainer.evaluation_step(task, batch)
            probabilities = torch.sigmoid(record["logits"]).numpy()
            count = int(record["example_count"])
            weighted_loss += float(record["combined_loss"]) * count
            example_count += count
            for molecule_id, smiles, label, probability in zip(
                batch["molecule_id"], batch["canonical_smiles"],
                batch["labels"].numpy(), probabilities,
            ):
                rows.append({
                    "molecule_id": molecule_id, "canonical_smiles": smiles,
                    "target": int(label), "probability": float(probability),
                    "prediction": int(probability >= 0.5),
                })
        predictions = pd.DataFrame(rows)
        prediction_name = f"validation_predictions_{task}.csv"
        predictions.to_csv(evaluation_dir / prediction_name, index=False)
        predictions.to_csv(output_dir / prediction_name, index=False)
        endpoints[task] = classification_metrics(
            predictions["target"].to_numpy(), predictions["probability"].to_numpy()
        )
        endpoints[task]["validation_loss"] = (
            weighted_loss / example_count if example_count else None
        )
        endpoints[task]["row_count"] = example_count
        endpoints[task]["prediction_file"] = str(
            Path("validation") / evaluation_dir.name / prediction_name
        )
    roc_values = [endpoints[task]["roc_auc"] for task in trainer.model.task_names]
    pr_values = [endpoints[task]["pr_auc"] for task in trainer.model.task_names]
    all_valid = all(value is not None for value in roc_values)
    return {
        "global_step": global_step, "split": "validation", "endpoints": endpoints,
        "all_endpoint_roc_auc_valid": all_valid,
        "mean_roc_auc": float(np.mean(roc_values)) if all_valid else None,
        "mean_pr_auc": float(np.mean(pr_values)) if all(value is not None for value in pr_values) else None,
    }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    both = len(np.unique(labels)) == 2
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if both else None,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if both else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)) if both else None,
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def update_checkpoint_selection(
    state: dict[str, Any], evaluation: Mapping[str, Any],
    floors: Mapping[str, float], tasks: tuple[str, ...],
) -> dict[str, Any]:
    """Update validation-only composite and endpoint selection state."""
    state["evaluation_count"] += 1
    selections: list[dict[str, Any]] = []
    endpoints = evaluation["endpoints"]
    for task in tasks:
        score = endpoints[task]["roc_auc"]
        previous = state["best_endpoints"].get(task)
        if score is not None and (previous is None or score > previous["roc_auc"]):
            state["best_endpoints"][task] = {"roc_auc": score, "global_step": evaluation["global_step"]}
            selections.append({"kind": "endpoint", "task": task, "reason": "higher_validation_roc_auc"})
    floor_failures = {
        task: {"observed": endpoints[task]["roc_auc"], "required": floor}
        for task, floor in floors.items()
        if endpoints[task]["roc_auc"] is None or endpoints[task]["roc_auc"] < floor
    }
    composite_improved = False
    if evaluation["all_endpoint_roc_auc_valid"] and not floor_failures:
        score = evaluation["mean_roc_auc"]
        tie = evaluation["mean_pr_auc"]
        best = state["best_composite"]
        best_tie = state["best_mean_pr_auc"]
        composite_improved = best is None or score > best or (score == best and tie > best_tie)
        if composite_improved:
            reason = "higher_mean_validation_roc_auc" if best is None or score > best else "mean_pr_auc_tiebreaker"
            state["best_composite"] = score
            state["best_mean_pr_auc"] = tie
            state["best_composite_step"] = evaluation["global_step"]
            selections.append({"kind": "composite", "reason": reason})
    state["evaluations_without_improvement"] = (
        0 if composite_improved else state["evaluations_without_improvement"] + 1
    )
    event = {
        "global_step": evaluation["global_step"], "source_split": "validation",
        "mean_roc_auc": evaluation["mean_roc_auc"], "mean_pr_auc": evaluation["mean_pr_auc"],
        "floor_failures": floor_failures, "selections": selections,
        "composite_improved": composite_improved,
    }
    state["selection_events"].append(event)
    return event


def load_baselines(paths: Mapping[str, str | Path | None]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for baseline_type, path in paths.items():
        if path is None:
            continue
        source = Path(path)
        if not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        endpoints = payload.get("endpoints", payload)
        if isinstance(endpoints, dict):
            loaded[baseline_type] = endpoints
    return loaded


def build_endpoint_comparison(
    evaluation: Mapping[str, Any], baselines: Mapping[str, Mapping[str, Any]],
    tolerances: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, metrics in evaluation["endpoints"].items():
        row: dict[str, Any] = {"endpoint": task, "multitask_roc_auc": metrics["roc_auc"]}
        negative = False
        for baseline_type in ("classical", "single_task"):
            entry = baselines.get(baseline_type, {}).get(task)
            baseline = entry.get("roc_auc") if isinstance(entry, dict) else entry
            row[f"{baseline_type}_roc_auc"] = baseline
            delta = metrics["roc_auc"] - float(baseline) if baseline is not None and metrics["roc_auc"] is not None else None
            row[f"delta_vs_{baseline_type}"] = delta
            if delta is not None and delta < -float(tolerances.get(task, 0.0)):
                negative = True
        row["negative_transfer_flag"] = negative
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, {
        "source_split": "validation", "baseline_types_available": sorted(baselines),
        "endpoints": rows,
        "claim": "No multi-task improvement claim is made without endpoint-level baseline comparison.",
    }


def should_stop_early(
    state: Mapping[str, Any], *, global_step: int, patience_evaluations: int,
    minimum_training_steps: int,
) -> bool:
    return bool(
        patience_evaluations > 0
        and global_step >= minimum_training_steps
        and state["evaluations_without_improvement"] >= patience_evaluations
    )


__all__ = [
    "build_endpoint_comparison", "classification_metrics", "evaluate_validation",
    "load_baselines", "update_checkpoint_selection",
    "should_stop_early",
]
