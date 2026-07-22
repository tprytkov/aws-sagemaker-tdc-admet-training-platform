"""Shared ChemBERTa encoder with endpoint-specific classification heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


DEFAULT_MULTITASK_ENDPOINTS = ("bbb_martins", "herg_karim", "ames")
MULTITASK_MODEL_CONFIG_NAME = "multitask_model_config.json"
MULTITASK_MODEL_STATE_NAME = "model_state.pt"
ENCODER_CONFIG_DIRECTORY = "encoder_config"
PoolingStrategy = Literal["masked_mean", "cls"]
HeadType = Literal["linear"]


@dataclass(frozen=True)
class MultiTaskChemBERTaConfig:
    """Serializable architecture configuration for the shared model."""

    model_name_or_path: str
    tasks: tuple[str, ...] = DEFAULT_MULTITASK_ENDPOINTS
    pooling: PoolingStrategy = "masked_mean"
    dropout: float = 0.15
    head_type: HeadType = "linear"
    head_output_size: int = 1
    model_revision: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string.")
        if not isinstance(self.tasks, tuple):
            object.__setattr__(self, "tasks", tuple(self.tasks))
        if not self.tasks:
            raise ValueError("tasks must contain at least one classification endpoint.")
        if len(self.tasks) != len(set(self.tasks)):
            raise ValueError("tasks must not contain duplicate task names.")
        unknown = sorted(set(self.tasks) - set(DEFAULT_MULTITASK_ENDPOINTS))
        if unknown:
            supported = ", ".join(DEFAULT_MULTITASK_ENDPOINTS)
            raise ValueError(
                f"Unknown classification task(s): {', '.join(unknown)}. Supported tasks: {supported}."
            )
        if self.pooling not in {"masked_mean", "cls"}:
            raise ValueError("pooling must be either 'masked_mean' or 'cls'.")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be greater than or equal to 0 and less than 1.")
        if self.head_type != "linear":
            raise ValueError("head_type must be 'linear'.")
        if self.head_output_size != 1:
            raise ValueError("head_output_size must be 1 for binary-classification logits.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiTaskChemBERTaConfig":
        data = dict(payload)
        if "tasks" in data:
            data["tasks"] = tuple(data["tasks"])
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MultiTaskChemBERTaConfig":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid multi-task model configuration JSON: {source}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Multi-task model configuration {source} must contain a JSON object.")
        return cls.from_dict(payload)


class MultiTaskChemBERTa(nn.Module):
    """One Hugging Face encoder shared by independent binary task heads."""

    def __init__(
        self,
        config: MultiTaskChemBERTaConfig,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.multitask_config = config
        self.encoder = encoder if encoder is not None else self._load_encoder(config)
        hidden_size = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("The encoder configuration must define a positive integer hidden_size.")

        self.dropout = nn.Dropout(float(config.dropout))
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Linear(hidden_size, config.head_output_size)
                for task_name in config.tasks
            }
        )

    @staticmethod
    def _load_encoder(config: MultiTaskChemBERTaConfig) -> nn.Module:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                revision=config.model_revision,
                local_files_only=config.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load the configured shared encoder '{config.model_name_or_path}'. "
                "The model was not replaced with a fallback checkpoint."
            ) from exc

    @property
    def task_names(self) -> tuple[str, ...]:
        return self.multitask_config.tasks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        task_name: str,
        **encoder_kwargs: Any,
    ) -> torch.Tensor:
        """Return one raw binary-classification logit per input row."""

        self._validate_task_name(task_name)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )
        hidden_state = getattr(outputs, "last_hidden_state", None)
        if hidden_state is None:
            raise ValueError("The configured encoder output does not provide last_hidden_state.")
        pooled = self.pool_hidden_state(hidden_state, attention_mask)
        return self.heads[task_name](self.dropout(pooled)).squeeze(-1)

    def pool_hidden_state(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool token representations according to the configured strategy."""

        if hidden_state.ndim != 3:
            raise ValueError("hidden_state must have shape [batch, sequence, hidden_size].")
        if self.multitask_config.pooling == "cls":
            if hidden_state.shape[1] == 0:
                raise ValueError("CLS pooling requires at least one sequence token.")
            return hidden_state[:, 0, :]

        if attention_mask is None:
            raise ValueError("attention_mask is required for masked_mean pooling.")
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(hidden_state.shape[:2]):
            raise ValueError("attention_mask must match hidden_state batch and sequence dimensions.")
        mask = attention_mask.unsqueeze(-1).to(device=hidden_state.device, dtype=hidden_state.dtype)
        token_counts = mask.sum(dim=1)
        if torch.any(token_counts == 0):
            raise ValueError("masked_mean pooling requires at least one unmasked token per row.")
        return (hidden_state * mask).sum(dim=1) / token_counts

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        """Return shared encoder parameters for a dedicated optimizer group."""

        return self.encoder.parameters()

    def task_head_parameters(self, task_name: str | None = None) -> Iterable[nn.Parameter]:
        """Return one task head or all head parameters for a dedicated optimizer group."""

        if task_name is None:
            return self.heads.parameters()
        self._validate_task_name(task_name)
        return self.heads[task_name].parameters()

    def parameter_groups(
        self,
        encoder_learning_rate: float,
        head_learning_rate: float,
    ) -> list[dict[str, Any]]:
        """Build explicit optimizer groups without constructing an optimizer."""

        if encoder_learning_rate <= 0 or head_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        return [
            {"name": "encoder", "params": self.encoder_parameters(), "lr": encoder_learning_rate},
            {"name": "task_heads", "params": self.task_head_parameters(), "lr": head_learning_rate},
        ]

    def save_model(self, output_dir: str | Path) -> dict[str, str]:
        """Save reconstruction metadata and one non-duplicated model state dictionary."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        config_path = self.multitask_config.save(destination / MULTITASK_MODEL_CONFIG_NAME)
        encoder_config_dir = destination / ENCODER_CONFIG_DIRECTORY
        encoder_config_dir.mkdir(parents=True, exist_ok=True)
        encoder_config = getattr(self.encoder, "config", None)
        if encoder_config is None or not hasattr(encoder_config, "save_pretrained"):
            raise ValueError("The encoder must expose a Hugging Face configuration with save_pretrained().")
        encoder_config.save_pretrained(encoder_config_dir)
        state_path = destination / MULTITASK_MODEL_STATE_NAME
        torch.save(self.state_dict(), state_path)
        return {
            "model_config": str(config_path),
            "encoder_config": str(encoder_config_dir / "config.json"),
            "model_state": str(state_path),
        }

    @classmethod
    def load_model(
        cls,
        input_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "MultiTaskChemBERTa":
        """Reconstruct a saved model without contacting a remote model registry."""

        source = Path(input_dir)
        config = MultiTaskChemBERTaConfig.load(source / MULTITASK_MODEL_CONFIG_NAME)
        encoder_config_path = source / ENCODER_CONFIG_DIRECTORY
        state_path = source / MULTITASK_MODEL_STATE_NAME
        if not (encoder_config_path / "config.json").is_file():
            raise FileNotFoundError(f"Missing saved encoder configuration: {encoder_config_path / 'config.json'}")
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing saved multi-task model state: {state_path}")
        encoder_config = AutoConfig.from_pretrained(encoder_config_path, local_files_only=True)
        encoder = AutoModel.from_config(encoder_config)
        model = cls(config=config, encoder=encoder)
        state = torch.load(state_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    def _validate_task_name(self, task_name: str) -> None:
        if task_name not in self.heads:
            available = ", ".join(self.task_names)
            raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}.")


__all__ = [
    "DEFAULT_MULTITASK_ENDPOINTS",
    "MultiTaskChemBERTa",
    "MultiTaskChemBERTaConfig",
]
