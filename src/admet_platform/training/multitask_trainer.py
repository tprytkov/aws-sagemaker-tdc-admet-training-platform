"""Platform-independent single-device multi-task training core."""

from __future__ import annotations

import json
import hashlib
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
from admet_platform.training.reproducibility import tensor_mapping_hash


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
        if config.mixed_precision != "no" and self.device.type != "cuda":
            raise ValueError("Mixed precision is supported only on a CUDA device; use mixed_precision='no' on CPU.")
        if set(self.train_loaders) != set(model.task_names):
            raise ValueError("train_loaders must exactly match the model task names.")
        if set(loss_module.task_names) != set(model.task_names):
            raise ValueError("loss_module tasks must exactly match the model task names.")
        if config.task_sampling != "round_robin":
            raise ValueError("Only round_robin task sampling is implemented.")

        self.model.to(self.device)
        self.loss_module.to(self.device)
        self.sampler = sampler or RoundRobinTaskSampler(model.task_names)
        if self.sampler.task_names != model.task_names:
            raise ValueError("Sampler task order must match the model task order.")
        self.optimizer = torch.optim.AdamW(
            model.parameter_groups(config.encoder_learning_rate, config.head_learning_rate),
            weight_decay=config.weight_decay,
        )
        warmup_steps = config.warmup_steps
        if config.warmup_ratio is not None:
            warmup_steps = int(config.max_steps * config.warmup_ratio)
        self.scheduler_warmup_steps = warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self._linear_warmup_decay_factor(
                step, warmup_steps, config.max_steps
            ),
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.mixed_precision == "fp16"
        )
        self.autocast_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.initial_model_state_hash = tensor_mapping_hash(self.model.state_dict())
        self.initial_task_head_hashes = {
            task: tensor_mapping_hash(self.model.heads[task].state_dict())
            for task in self.model.task_names
        }
        self.control_state: dict[str, Any] = {
            "best_composite": None, "best_mean_pr_auc": None,
            "best_endpoints": {task: None for task in self.model.task_names},
            "evaluations_without_improvement": 0, "evaluation_count": 0,
            "stopped_early": False, "stop_reason": None,
            "selection_events": [],
            "validation_history": [],
        }
        self._loader_iterators = {task: iter(loader) for task, loader in self.train_loaders.items()}

    def _next_batch(self, task_name: str) -> Mapping[str, Any]:
        loader = self.train_loaders[task_name]
        stateful_sampler = getattr(loader, "sampler", None)
        if (
            getattr(stateful_sampler, "permutation", None)
            and stateful_sampler.cursor >= len(stateful_sampler.permutation)
        ):
            # Make epoch rollover explicit so uninterrupted and reconstructed
            # iterators consume the same task-local DataLoader generator state.
            self._loader_iterators[task_name] = iter(loader)
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
        batch_identity = batch.get("task_name", task_name)
        if isinstance(batch_identity, (list, tuple)):
            identity_matches = bool(batch_identity) and all(value == task_name for value in batch_identity)
        else:
            identity_matches = batch_identity == task_name
        if not identity_matches:
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
        autocast_enabled = self.config.mixed_precision != "no"
        with torch.autocast(
            device_type=self.device.type, dtype=self.autocast_dtype, enabled=autocast_enabled
        ):
            logits = self.model(**inputs, task_name=task_name)
            loss_output = self.loss_module(task_name, logits, labels)
        self.scaler.scale(loss_output.combined_loss).backward()
        self.scaler.unscale_(self.optimizer)
        self._reject_nonfinite_gradients(task_name)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm detected for task '{task_name}'.")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        self.sampler.record_batch(task_name, int(labels.numel()))
        record = self._loss_record("train", task_name, loss_output, labels.numel())
        record["gradient_norm_before_clipping"] = float(gradient_norm)
        record["gradient_clip_norm"] = self.config.gradient_clip_norm
        record["learning_rates"] = {
            group["name"]: float(group["lr"]) for group in self.optimizer.param_groups
        }
        molecule_ids = [str(value) for value in selected_batch.get("molecule_id", [])]
        record["molecule_ids"] = molecule_ids
        record["batch_hash"] = hashlib.sha256(
            json.dumps(molecule_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.history.append(record)
        return record

    @torch.no_grad()
    def evaluation_step(self, task_name: str, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate without changing model mode or any global training RNG state."""

        previous_mode = self.model.training
        rng_state = self._rng_state_dict()
        try:
            inputs, labels = self._prepare_batch(task_name, batch)
            self.model.eval()
            logits = self.model(**inputs, task_name=task_name)
            output = self.loss_module(task_name, logits, labels)
            record = self._loss_record("evaluation", task_name, output, labels.numel())
            record["logits"] = logits.detach().cpu()
            return record
        finally:
            self.model.train(previous_mode)
            self._load_rng_state_dict(rng_state)

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
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "sampler_state": self.sampler.state_dict(),
            "loader_states": self._loader_state_dict(),
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
            "initial_model_state_hash": self.initial_model_state_hash,
            "initial_task_head_hashes": self.initial_task_head_hashes,
            "control_state": self.control_state,
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
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.sampler.load_state_dict(checkpoint["sampler_state"])
        self.global_step = int(checkpoint["global_step"])
        if int(checkpoint["logical_pass"]) != self.sampler.logical_pass:
            raise ValueError("Checkpoint logical pass is inconsistent with sampler state.")
        self.history = list(checkpoint.get("history", []))
        self.initial_model_state_hash = checkpoint.get(
            "initial_model_state_hash", self.initial_model_state_hash
        )
        self.initial_task_head_hashes = dict(checkpoint.get(
            "initial_task_head_hashes", self.initial_task_head_hashes
        ))
        self.control_state = dict(checkpoint.get("control_state", self.control_state))
        self._restore_loader_positions(checkpoint.get("loader_states", {}))
        # Iterator construction/replay may consume RNG. The saved training RNG is
        # deliberately restored last, immediately before the resumed forward pass.
        self._load_rng_state_dict(checkpoint["rng_state"])

    def _loader_state_dict(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for task, loader in self.train_loaders.items():
            sampler = getattr(loader, "sampler", None)
            generator = getattr(loader, "generator", None)
            states[task] = {
                "metadata": dict(getattr(loader, "reproducibility_metadata", {})),
                "generator_state": generator.get_state() if generator is not None else None,
                "sampler_state": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
            }
        return states

    def _restore_loader_positions(self, states: Mapping[str, Any]) -> None:
        stateful_tasks: set[str] = set()
        for task, state in states.items():
            if task not in self.train_loaders or state.get("sampler_state") is None:
                continue
            loader = self.train_loaders[task]
            sampler = getattr(loader, "sampler", None)
            generator = getattr(loader, "generator", None)
            if not hasattr(sampler, "load_state_dict"):
                continue
            sampler.load_state_dict(state["sampler_state"])
            if generator is not None and state.get("generator_state") is not None:
                generator.set_state(state["generator_state"])
            stateful_tasks.add(task)
        self._loader_iterators = {task: iter(loader) for task, loader in self.train_loaders.items()}
        # DataLoader iterator construction consumes its explicit base-seed generator.
        # Restore the checkpointed state so loader metadata remains bit-identical.
        for task in stateful_tasks:
            loader = self.train_loaders[task]
            generator = getattr(loader, "generator", None)
            saved = states[task].get("generator_state")
            if generator is not None and saved is not None:
                generator.set_state(saved)
        for task, consumed in self.sampler.batch_counts.items():
            if task not in stateful_tasks:
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
            "initial_model_state_hash": self.initial_model_state_hash,
            "initial_task_head_hashes": self.initial_task_head_hashes,
            "loader_states": self._json_loader_metadata(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return destination

    def _json_loader_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for task, state in self._loader_state_dict().items():
            sampler_state = state.get("sampler_state") or {}
            metadata[task] = {
                **state["metadata"],
                "sampler_cursor": sampler_state.get("cursor"),
                "sampler_permutation": sampler_state.get("permutation"),
                "loader_generator_state_hash": (
                    hashlib.sha256(state["generator_state"].numpy().tobytes()).hexdigest()
                    if state.get("generator_state") is not None else None
                ),
                "sampler_generator_state_hash": (
                    hashlib.sha256(sampler_state["generator_state"].numpy().tobytes()).hexdigest()
                    if sampler_state.get("generator_state") is not None else None
                ),
            }
        return metadata

    @staticmethod
    def _linear_warmup_decay_factor(step: int, warmup_steps: int, total_steps: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )


__all__ = ["MultiTaskTrainer"]
