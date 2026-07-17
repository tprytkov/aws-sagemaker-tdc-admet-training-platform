import json
from pathlib import Path

import numpy as np
import pytest

from admet_platform.training.multitask_control import (
    build_endpoint_comparison, classification_metrics, should_stop_early,
    update_checkpoint_selection,
)


TASKS = ("bbb_martins", "herg_karim", "ames")


def _evaluation(step: int, scores=(0.7, 0.8, 0.9), pr=(0.6, 0.7, 0.8)) -> dict:
    endpoints = {
        task: {"roc_auc": scores[index], "pr_auc": pr[index]}
        for index, task in enumerate(TASKS)
    }
    return {
        "global_step": step, "split": "validation", "endpoints": endpoints,
        "all_endpoint_roc_auc_valid": all(value is not None for value in scores),
        "mean_roc_auc": float(np.mean(scores)) if all(value is not None for value in scores) else None,
        "mean_pr_auc": float(np.mean(pr)),
    }


def _state() -> dict:
    return {
        "best_composite": None, "best_mean_pr_auc": None,
        "best_endpoints": {task: None for task in TASKS},
        "evaluations_without_improvement": 0, "evaluation_count": 0,
        "selection_events": [],
    }


def test_complete_binary_validation_metrics() -> None:
    metrics = classification_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.8, 0.4, 0.9])
    )
    assert set(metrics) == {
        "roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc",
        "sensitivity", "specificity", "confusion_matrix",
    }
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5


def test_composite_endpoint_selection_and_pr_tiebreaker_are_validation_only() -> None:
    state = _state()
    first = update_checkpoint_selection(state, _evaluation(2), {}, TASKS)
    tied = update_checkpoint_selection(
        state, _evaluation(4, pr=(0.7, 0.8, 0.9)), {}, TASKS
    )
    assert first["source_split"] == "validation"
    assert {event["task"] for event in first["selections"] if event["kind"] == "endpoint"} == set(TASKS)
    assert any(event["kind"] == "composite" for event in first["selections"])
    assert tied["selections"] == [{"kind": "composite", "reason": "mean_pr_auc_tiebreaker"}]
    assert state["best_composite_step"] == 4


def test_invalid_endpoint_auc_and_metric_floor_block_composite_only() -> None:
    state = _state()
    invalid = _evaluation(1, scores=(0.7, None, 0.9))
    event = update_checkpoint_selection(state, invalid, {}, TASKS)
    assert not any(item["kind"] == "composite" for item in event["selections"])
    floored = update_checkpoint_selection(
        state, _evaluation(2), {"bbb_martins": 0.75}, TASKS
    )
    assert floored["floor_failures"]["bbb_martins"]["required"] == 0.75
    assert not any(item["kind"] == "composite" for item in floored["selections"])


def test_early_stopping_respects_minimum_step_protection() -> None:
    state = {"evaluations_without_improvement": 3}
    assert not should_stop_early(
        state, global_step=9, patience_evaluations=2, minimum_training_steps=10
    )
    assert should_stop_early(
        state, global_step=10, patience_evaluations=2, minimum_training_steps=10
    )
    assert not should_stop_early(
        state, global_step=100, patience_evaluations=0, minimum_training_steps=0
    )


def test_baseline_deltas_and_negative_transfer_flags() -> None:
    frame, report = build_endpoint_comparison(
        _evaluation(3),
        {
            "classical": {task: {"roc_auc": 0.75} for task in TASKS},
            "single_task": {task: {"roc_auc": 0.85} for task in TASKS},
        },
        {task: 0.02 for task in TASKS},
    )
    bbb = frame.set_index("endpoint").loc["bbb_martins"]
    assert bbb["delta_vs_classical"] == pytest.approx(-0.05)
    assert bool(bbb["negative_transfer_flag"])
    assert "No multi-task improvement claim" in report["claim"]
