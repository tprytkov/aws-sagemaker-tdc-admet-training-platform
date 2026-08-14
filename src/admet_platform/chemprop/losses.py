"""Explicit multitask loss and validation-selection contracts for Chemprop 2.3.1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torchmetrics import Metric


TRAINING_LOSS_EQUATION = (
    "sum_{rows i,tasks t}(mask_it * task_weight_t * squared_error_it) / "
    "sum_{rows i,tasks t}(mask_it)"
)
SELECTION_METRIC_EQUATION = (
    "mean_t(sum_i(mask_it * abs(standardized_prediction_it - standardized_target_it)) "
    "/ sum_i(mask_it))"
)


def inverse_count_equal_endpoint_weights(label_counts: Sequence[int]) -> np.ndarray:
    """Weights whose count-weighted aggregate is equal across tasks, normalized to mean one."""

    counts = np.asarray(label_counts, dtype=float)
    if counts.ndim != 1 or not len(counts) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("Every endpoint must have a finite positive training-label count.")
    inverse = 1.0 / counts
    return inverse / inverse.mean()


class EqualTaskMeanStandardizedMAE(Metric):
    """Mean of independently reduced per-task MAEs in standardized target space."""

    alias = "mean_standardized_mae"
    higher_is_better = False
    full_state_update = False

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.add_state("absolute_error_sum", default=torch.zeros(n_tasks), dist_reduce_fx="sum")
        self.add_state("observed_count", default=torch.zeros(n_tasks), dist_reduce_fx="sum")

    def update(
        self,
        preds,
        targets,
        mask=None,
        weights=None,
        lt_mask=None,
        gt_mask=None,
    ) -> None:
        del weights, lt_mask, gt_mask
        observed = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        self.absolute_error_sum += (preds - targets).abs().mul(observed).sum(dim=0)
        self.observed_count += observed.sum(dim=0)

    def compute(self):
        per_task = torch.where(
            self.observed_count > 0,
            self.absolute_error_sum / self.observed_count,
            torch.full_like(self.observed_count, float("inf")),
        )
        return per_task.mean()


def equal_task_mean_standardized_mae_metric(n_tasks: int) -> EqualTaskMeanStandardizedMAE:
    return EqualTaskMeanStandardizedMAE(n_tasks)


__all__ = [
    "SELECTION_METRIC_EQUATION",
    "TRAINING_LOSS_EQUATION",
    "EqualTaskMeanStandardizedMAE",
    "equal_task_mean_standardized_mae_metric",
    "inverse_count_equal_endpoint_weights",
]
