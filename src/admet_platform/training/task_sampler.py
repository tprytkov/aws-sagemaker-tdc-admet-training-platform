"""Deterministic and resumable task scheduling."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping


class RoundRobinTaskSampler:
    """Give every task an equal, stable sequence of optimization steps."""

    strategy = "round_robin"

    def __init__(self, task_names: Iterable[str]) -> None:
        self.task_names = tuple(task_names)
        if not self.task_names or any(not isinstance(name, str) or not name for name in self.task_names):
            raise ValueError("task_names must contain at least one non-empty name.")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must not contain duplicates.")
        self.next_index = 0
        self.logical_pass = 0
        self.batch_counts = {task: 0 for task in self.task_names}
        self.example_counts = {task: 0 for task in self.task_names}

    def next_task(self) -> str:
        task = self.task_names[self.next_index]
        self.next_index += 1
        if self.next_index == len(self.task_names):
            self.next_index = 0
            self.logical_pass += 1
        return task

    def record_batch(self, task_name: str, example_count: int) -> None:
        if task_name not in self.batch_counts:
            raise ValueError(f"Unknown sampler task '{task_name}'.")
        if not isinstance(example_count, int) or example_count < 0:
            raise ValueError("example_count must be a non-negative integer.")
        self.batch_counts[task_name] += 1
        self.example_counts[task_name] += example_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task_names": list(self.task_names),
            "next_index": self.next_index,
            "logical_pass": self.logical_pass,
            "batch_counts": dict(self.batch_counts),
            "example_counts": dict(self.example_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("strategy") != self.strategy or tuple(state.get("task_names", ())) != self.task_names:
            raise ValueError("Sampler checkpoint is incompatible with the configured strategy or task order.")
        next_index = state.get("next_index")
        if not isinstance(next_index, int) or not 0 <= next_index < len(self.task_names):
            raise ValueError("Sampler checkpoint contains an invalid next_index.")
        self.next_index = next_index
        self.logical_pass = int(state.get("logical_pass", 0))
        for field, target in (("batch_counts", self.batch_counts), ("example_counts", self.example_counts)):
            values = state.get(field)
            if not isinstance(values, Mapping) or set(values) != set(self.task_names):
                raise ValueError(f"Sampler checkpoint contains invalid {field}.")
            target.update({task: int(values[task]) for task in self.task_names})


class ProbabilisticTaskSampler:
    """Deterministic, resumable sampling from configured task probabilities."""

    def __init__(
        self,
        task_names: Iterable[str],
        probabilities: Mapping[str, float],
        *,
        strategy: str,
        seed: int,
        alpha: float | None = None,
    ) -> None:
        self.task_names = tuple(task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must contain unique non-empty names.")
        if set(probabilities) != set(self.task_names):
            raise ValueError("Sampler probabilities must exactly match task names.")
        values = [float(probabilities[task]) for task in self.task_names]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Sampler probabilities must be finite and positive.")
        total = sum(values)
        self.probabilities = {
            task: value / total for task, value in zip(self.task_names, values, strict=True)
        }
        self.strategy = strategy
        self.seed = int(seed)
        self.alpha = alpha
        self._rng = random.Random(self.seed)
        self.draw_count = 0
        self.logical_pass = 0
        self.batch_counts = {task: 0 for task in self.task_names}
        self.example_counts = {task: 0 for task in self.task_names}

    def next_task(self) -> str:
        task = self._rng.choices(
            self.task_names,
            weights=[self.probabilities[name] for name in self.task_names],
            k=1,
        )[0]
        self.draw_count += 1
        self.logical_pass = self.draw_count // len(self.task_names)
        return task

    def record_batch(self, task_name: str, example_count: int) -> None:
        if task_name not in self.batch_counts:
            raise ValueError(f"Unknown sampler task '{task_name}'.")
        if not isinstance(example_count, int) or example_count < 0:
            raise ValueError("example_count must be a non-negative integer.")
        self.batch_counts[task_name] += 1
        self.example_counts[task_name] += example_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task_names": list(self.task_names),
            "probabilities": dict(self.probabilities),
            "seed": self.seed,
            "alpha": self.alpha,
            "rng_state": self._rng.getstate(),
            "draw_count": self.draw_count,
            "logical_pass": self.logical_pass,
            "batch_counts": dict(self.batch_counts),
            "example_counts": dict(self.example_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("strategy") != self.strategy
            or tuple(state.get("task_names", ())) != self.task_names
            or state.get("probabilities") != self.probabilities
            or int(state.get("seed", -1)) != self.seed
            or state.get("alpha") != self.alpha
        ):
            raise ValueError("Sampler checkpoint is incompatible with configured sampling.")
        self._rng.setstate(state["rng_state"])
        self.draw_count = int(state["draw_count"])
        self.logical_pass = int(state["logical_pass"])
        for field, target in (
            ("batch_counts", self.batch_counts),
            ("example_counts", self.example_counts),
        ):
            values = state.get(field)
            if not isinstance(values, Mapping) or set(values) != set(self.task_names):
                raise ValueError(f"Sampler checkpoint contains invalid {field}.")
            target.update({task: int(values[task]) for task in self.task_names})


def build_task_sampler(
    strategy: str,
    task_names: Iterable[str],
    train_rows: Mapping[str, int],
    *,
    seed: int,
    alpha: float = 1.0,
) -> RoundRobinTaskSampler | ProbabilisticTaskSampler:
    """Construct the configured task sampler from train-only row counts."""

    tasks = tuple(task_names)
    if strategy == "round_robin":
        return RoundRobinTaskSampler(tasks)
    if set(train_rows) != set(tasks) or any(int(train_rows[task]) <= 0 for task in tasks):
        raise ValueError("Train row counts must be positive and exactly match task names.")
    if strategy == "uniform":
        weights = {task: 1.0 for task in tasks}
        exponent = None
    elif strategy == "proportional":
        weights = {task: float(train_rows[task]) for task in tasks}
        exponent = 1.0
    elif strategy == "temperature":
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Temperature sampling alpha must be finite and non-negative.")
        weights = {task: float(train_rows[task]) ** alpha for task in tasks}
        exponent = float(alpha)
    else:
        raise ValueError(f"Unsupported task sampling strategy '{strategy}'.")
    total = sum(weights.values())
    probabilities = {task: weights[task] / total for task in tasks}
    return ProbabilisticTaskSampler(
        tasks,
        probabilities,
        strategy=strategy,
        seed=seed,
        alpha=exponent,
    )


__all__ = [
    "ProbabilisticTaskSampler",
    "RoundRobinTaskSampler",
    "build_task_sampler",
]
