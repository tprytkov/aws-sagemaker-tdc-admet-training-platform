"""Deterministic and resumable task scheduling."""

from __future__ import annotations

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


__all__ = ["RoundRobinTaskSampler"]
