"""Platform-independent single-device multi-task training core."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from admet_platform.data.multitask import MultiTaskTrainingConfig
from admet_platform.models.multitask_chemberta import MultiTaskChemBERTa
from admet_platform.training.multitask_losses import MultiTaskBinaryLoss, TaskLossOutput
from admet_platform.training.task_sampler import RoundRobinTaskSampler


class MultiTaskTrainer:
    """Train a shared encoder and selected task head on CPU or one CUDA device."""

    def __init__(
        self,
        model: MultiTaskChemBERTa,
        train_loaders: Mapping[str, Iterable[Mapping[str, Any]]],
        loss_module: MultiTaskBinaryLoss,
        config: MultiTaskTrainingConfig,
        device: str | torch.device = "cpu",
        sampler: RoundRobinTaskSampler | None = None,
    ) -> None:
        self.model = model
        self.train_loaders = dict(train_loaders)
        self.loss_module = loss_module
        self.config = config
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("MultiTaskTrainer supports only CPU or one CUDA device.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if set(self.train_loaders) != set(model.task_names):
            raise ValueError("train_loaders must exactly match the model task names.")
        if set(loss_module.task_names) != set(model.task_names):
            raise ValueError("loss_module tasks must exactly match the model task names.")
        if config.task_sampling != "round_robin":
            raise ValueError("Only round_robin task sampling is implemented.")

        self._set_seed(config.random_seed)
        self.model.to(self.device)
        self.loss_module.to(self.device)
        self.sampler = sampler or RoundRobinTaskSampler(model.task_names)
        if self.sampler.task_names != model.task_names:
            raise ValueError("Sampler task order must match the model task order.")
        self.optimizer = torch.optim.AdamW(
            model.parameter_groups(config.encoder_learning_rate, config.head_learning_rate),
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self._loader_iterators = {task: iter(loader) for task, loader in self.train_loaders.items()}

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _next_batch(self, task_name: str) -> Mapping[str, Any]:
        iterator = self._loader_iterators[task_name]
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(self.train_loaders[task_name])
            self._loader_iterators[task_name] = iterator
            try:
                return next(iterator)
            except StopIteration as exc:
                raise ValueError(f"Training loader for task '{task_name}' is empty.") from exc

    def _prepare_batch(self, task_name: str, batch: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if batch.get("task_name", task_name) != task_name:
            raise ValueError(f"Batch task identity does not match selected task '{task_name}'.")
        missing = [key for key in ("input_ids", "attention_mask", "labels") if key not in batch]
        if missing:
            raise ValueError(f"Batch for task '{task_name}' is missing: {', '.join(missing)}.")
        inputs = {
            "input_ids": torch.as_tensor(batch["input_ids"], device=self.device),
            "attention_mask": torch.as_tensor(batch["attention_mask"], device=self.device),
        }
        labels = torch.as_tensor(batch["labels"], device=self.device, dtype=torch.float32).reshape(-1)
        return inputs, labels

    def train_step(
        self,
        task_name: str | None = None,
        batch: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one optimizer step for the scheduled or explicitly selected task."""

        if task_name is None:
            task_name = self.sampler.next_task()
        elif task_name not in self.model.task_names:
            raise ValueError(f"Unknown training task '{task_name}'.")
        selected_batch = batch if batch is not None else self._next_batch(task_name)
        inputs, labels = self._prepare_batch(task_name, selected_batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(**inputs, task_name=task_name)
        loss_output = self.loss_module(task_name, logits, labels)
        loss_output.combined_loss.backward()
        self._reject_nonfinite_gradients(task_name)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm detected for task '{task_name}'.")
        self.optimizer.step()
        self.global_step += 1
        self.sampler.record_batch(task_name, int(labels.numel()))
        record = self._loss_record("train", task_name, loss_output, labels.numel())
        record["gradient_norm_before_clipping"] = float(gradient_norm)
        record["gradient_clip_norm"] = self.config.gradient_clip_norm
        self.history.append(record)
        return record

    @torch.no_grad()
    def evaluation_step(self, task_name: str, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate one task batch without changing optimizer or sampler state."""

        inputs, labels = self._prepare_batch(task_name, batch)
        self.model.eval()
        logits = self.model(**inputs, task_name=task_name)
        output = self.loss_module(task_name, logits, labels)
        record = self._loss_record("evaluation", task_name, output, labels.numel())
        record["logits"] = logits.detach().cpu()
        return record

    def _reject_nonfinite_gradients(self, task_name: str) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradients detected for task '{task_name}'.")

    def _loss_record(
        self, phase: str, task_name: str, output: TaskLossOutput, example_count: int
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "global_step": self.global_step,
            "logical_pass": self.sampler.logical_pass,
            "task_name": task_name,
            "example_count": int(example_count),
            "raw_losses": {task: float(loss.detach().cpu()) for task, loss in output.raw_losses.items()},
            "combined_loss": float(output.combined_loss.detach().cpu()),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save all state needed for deterministic continuation."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_version": 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "sampler_state": self.sampler.state_dict(),
            "global_step": self.global_step,
            "logical_pass": self.sampler.logical_pass,
            "rng_state": self._rng_state_dict(),
            "training_config": asdict(self.config),
            "model_config": self.model.multitask_config.to_dict(),
            "loss_metadata": {
                "positive_class_weights": {
                    task: float(getattr(self.loss_module, f"_pos_weight_{task}").item())
                    for task in self.loss_module.task_names
                },
                "task_loss_weights": dict(self.loss_module.task_loss_weights),
            },
            "history": self.history,
        }
        torch.save(checkpoint, destination)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Validate and restore model, optimizer, sampler, loader, and RNG state."""

        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_version") != 1:
            raise ValueError("Unsupported multi-task checkpoint version.")
        if checkpoint.get("model_config") != self.model.multitask_config.to_dict():
            raise ValueError("Checkpoint model configuration is incompatible with this trainer.")
        if checkpoint.get("training_config") != asdict(self.config):
            raise ValueError("Checkpoint training configuration is incompatible with this trainer.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.sampler.load_state_dict(checkpoint["sampler_state"])
        self.global_step = int(checkpoint["global_step"])
        if int(checkpoint["logical_pass"]) != self.sampler.logical_pass:
            raise ValueError("Checkpoint logical pass is inconsistent with sampler state.")
        self.history = list(checkpoint.get("history", []))
        self._restore_loader_positions()
        self._load_rng_state_dict(checkpoint["rng_state"])

    def _restore_loader_positions(self) -> None:
        self._loader_iterators = {task: iter(loader) for task, loader in self.train_loaders.items()}
        for task, consumed in self.sampler.batch_counts.items():
            for _ in range(consumed):
                self._next_batch(task)

    @staticmethod
    def _rng_state_dict() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _load_rng_state_dict(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def write_metrics_json(self, path: str | Path) -> Path:
        """Write finite, structured training history and contribution counts."""

        for record in self.history:
            for key in ("combined_loss", "gradient_norm_before_clipping"):
                if key in record and not math.isfinite(record[key]):
                    raise FloatingPointError(f"Cannot serialize non-finite training metric '{key}'.")
        payload = {
            "schema_version": "1.0.0",
            "global_step": self.global_step,
            "logical_pass": self.sampler.logical_pass,
            "task_sampling": self.sampler.strategy,
            "batch_counts": dict(self.sampler.batch_counts),
            "example_counts": dict(self.sampler.example_counts),
            "history": self.history,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return destination


__all__ = ["MultiTaskTrainer"]
