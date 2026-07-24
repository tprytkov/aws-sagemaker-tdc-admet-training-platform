"""Shared molecular encoder with endpoint-specific scalar regression heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


DEFAULT_REGRESSION_ENDPOINTS = (
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "ppbr_az",
    "vdss_lombardo",
)
REGRESSION_MODEL_CONFIG_NAME = "multitask_regression_model_config.json"
REGRESSION_MODEL_STATE_NAME = "regression_model_state.pt"
REGRESSION_ENCODER_CONFIG_DIRECTORY = "encoder_config"
PoolingStrategy = Literal["masked_mean", "cls"]


@dataclass(frozen=True)
class MultiTaskRegressionChemBERTaConfig:
    model_name_or_path: str
    tasks: tuple[str, ...] = DEFAULT_REGRESSION_ENDPOINTS
    pooling: PoolingStrategy = "masked_mean"
    dropout: float = 0.15
    head_output_size: int = 1
    model_revision: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string.")
        if not isinstance(self.tasks, tuple):
            object.__setattr__(self, "tasks", tuple(self.tasks))
        if not self.tasks or any(not isinstance(task, str) or not task for task in self.tasks):
            raise ValueError("tasks must contain non-empty regression endpoint names.")
        if len(self.tasks) != len(set(self.tasks)):
            raise ValueError("tasks must not contain duplicates.")
        if self.pooling not in {"masked_mean", "cls"}:
            raise ValueError("pooling must be masked_mean or cls.")
        if not isinstance(self.dropout, (int, float)) or not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.head_output_size != 1:
            raise ValueError("Regression heads must output exactly one scalar.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiTaskRegressionChemBERTaConfig:
        data = dict(payload)
        if "tasks" in data:
            data["tasks"] = tuple(data["tasks"])
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> MultiTaskRegressionChemBERTaConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Regression model configuration must be a JSON object.")
        return cls.from_dict(payload)


class MultiTaskRegressionChemBERTa(nn.Module):
    """One encoder and one unsquashed Linear(hidden_size, 1) head per endpoint."""

    def __init__(
        self,
        config: MultiTaskRegressionChemBERTaConfig,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.multitask_config = config
        self.encoder = encoder if encoder is not None else self._load_encoder(config)
        hidden_size = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("Encoder config must define a positive hidden_size.")
        self.dropout = nn.Dropout(float(config.dropout))
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, 1) for task in config.tasks}
        )

    @staticmethod
    def _load_encoder(config: MultiTaskRegressionChemBERTaConfig) -> nn.Module:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                revision=config.model_revision,
                local_files_only=config.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load regression encoder '{config.model_name_or_path}'; "
                "no fallback checkpoint was substituted."
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
        if task_name not in self.heads:
            raise ValueError(f"Unknown regression task '{task_name}'.")
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Encoder output does not provide last_hidden_state.")
        pooled = self.pool_hidden_state(hidden, attention_mask)
        return self.heads[task_name](self.dropout(pooled)).squeeze(-1)

    def pool_hidden_state(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_state.ndim != 3:
            raise ValueError("hidden_state must have shape [batch, sequence, hidden].")
        if self.multitask_config.pooling == "cls":
            if hidden_state.shape[1] == 0:
                raise ValueError("CLS pooling requires at least one token.")
            return hidden_state[:, 0, :]
        if attention_mask is None:
            raise ValueError("attention_mask is required for masked_mean pooling.")
        if tuple(attention_mask.shape) != tuple(hidden_state.shape[:2]):
            raise ValueError("attention_mask dimensions do not match hidden_state.")
        mask = attention_mask.unsqueeze(-1).to(hidden_state)
        counts = mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("Every row must contain an unmasked token.")
        return (hidden_state * mask).sum(dim=1) / counts

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.parameters()

    def task_head_parameters(self, task_name: str | None = None) -> Iterable[nn.Parameter]:
        if task_name is None:
            return self.heads.parameters()
        if task_name not in self.heads:
            raise ValueError(f"Unknown regression task '{task_name}'.")
        return self.heads[task_name].parameters()

    def parameter_groups(
        self, encoder_learning_rate: float, head_learning_rate: float
    ) -> list[dict[str, Any]]:
        if encoder_learning_rate <= 0 or head_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        return [
            {
                "name": "encoder",
                "params": self.encoder_parameters(),
                "lr": encoder_learning_rate,
            },
            {
                "name": "task_heads",
                "params": self.task_head_parameters(),
                "lr": head_learning_rate,
            },
        ]

    def save_model(self, output_dir: str | Path) -> dict[str, str]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        config_path = self.multitask_config.save(
            destination / REGRESSION_MODEL_CONFIG_NAME
        )
        encoder_dir = destination / REGRESSION_ENCODER_CONFIG_DIRECTORY
        encoder_config = getattr(self.encoder, "config", None)
        if encoder_config is None or not hasattr(encoder_config, "save_pretrained"):
            raise ValueError("Encoder must expose save_pretrained configuration.")
        encoder_config.save_pretrained(encoder_dir)
        state_path = destination / REGRESSION_MODEL_STATE_NAME
        torch.save(self.state_dict(), state_path)
        return {
            "model_config": str(config_path),
            "encoder_config": str(encoder_dir / "config.json"),
            "model_state": str(state_path),
        }

    @classmethod
    def load_model(
        cls,
        input_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> MultiTaskRegressionChemBERTa:
        source = Path(input_dir)
        config = MultiTaskRegressionChemBERTaConfig.load(
            source / REGRESSION_MODEL_CONFIG_NAME
        )
        encoder_dir = source / REGRESSION_ENCODER_CONFIG_DIRECTORY
        encoder_config = AutoConfig.from_pretrained(encoder_dir, local_files_only=True)
        encoder = AutoModel.from_config(encoder_config)
        model = cls(config, encoder=encoder)
        state = torch.load(
            source / REGRESSION_MODEL_STATE_NAME,
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        return model


__all__ = [
    "DEFAULT_REGRESSION_ENDPOINTS",
    "MultiTaskRegressionChemBERTa",
    "MultiTaskRegressionChemBERTaConfig",
]
