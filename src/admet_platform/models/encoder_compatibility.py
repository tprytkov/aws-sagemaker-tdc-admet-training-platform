"""Generic Hugging Face encoder compatibility preflight for multi-task ADMET."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModel, AutoTokenizer

from admet_platform.models.multitask_chemberta import (
    DEFAULT_MULTITASK_ENDPOINTS,
    MultiTaskChemBERTa,
    MultiTaskChemBERTaConfig,
)


REPRESENTATIVE_SMILES = ("CCO", "c1ccccc1", "CC(=O)O")


class EncoderCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot safely serve as the shared encoder."""


def verify_encoder_compatibility(
    *,
    model_name_or_path: str,
    revision: str | None = None,
    local_files_only: bool = False,
    output_json: str | Path,
) -> dict[str, Any]:
    """Load and exercise an encoder without substituting any fallback model."""

    output_path = Path(output_json)
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "failed",
        "model_name_or_path": model_name_or_path,
        "model_revision": revision,
        "local_files_only": local_files_only,
        "representative_smiles": list(REPRESENTATIVE_SMILES),
        "fallback_encoder_used": False,
        "errors": [],
    }
    try:
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string.")
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        loaded = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            output_loading_info=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise TypeError("AutoModel did not return model loading information.")
        encoder, loading_info = loaded
        if not isinstance(loading_info, Mapping):
            raise TypeError("AutoModel loading information must be a mapping.")

        diagnostics = classify_loading_information(encoder, loading_info)
        report["loading_information"] = diagnostics
        if diagnostics["compatibility_problems"]:
            raise ValueError("; ".join(diagnostics["compatibility_problems"]))

        architecture = _architecture_report(encoder)
        report["architecture"] = architecture
        encoded = tokenizer(
            list(REPRESENTATIVE_SMILES),
            padding=True,
            truncation=True,
            max_length=min(64, architecture["maximum_positions"]),
            return_tensors="pt",
        )
        model_inputs = {
            key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
        }
        if "input_ids" not in model_inputs or "attention_mask" not in model_inputs:
            raise ValueError("Tokenizer output must include input_ids and attention_mask tensors.")
        batch_size = len(REPRESENTATIVE_SMILES)
        encoder.eval()
        with torch.no_grad():
            encoder_output = encoder(**model_inputs)
        hidden_state = getattr(encoder_output, "last_hidden_state", None)
        if not isinstance(hidden_state, torch.Tensor) or hidden_state.ndim != 3:
            raise ValueError("Encoder output must expose a rank-3 last_hidden_state tensor.")
        if hidden_state.shape[0] != batch_size or hidden_state.shape[2] != architecture["hidden_size"]:
            raise ValueError("Encoder hidden-state shape is incompatible with its configuration.")
        if not torch.isfinite(hidden_state).all():
            raise ValueError("Encoder forward pass produced non-finite hidden states.")

        multitask_model = MultiTaskChemBERTa(
            MultiTaskChemBERTaConfig(
                model_name_or_path=model_name_or_path,
                model_revision=revision,
                tasks=DEFAULT_MULTITASK_ENDPOINTS,
                pooling="masked_mean",
                dropout=0.15,
                local_files_only=local_files_only,
            ),
            encoder=encoder,
        )
        multitask_model.eval()
        task_heads: dict[str, Any] = {}
        with torch.no_grad():
            for task in DEFAULT_MULTITASK_ENDPOINTS:
                logits = multitask_model(**model_inputs, task_name=task)
                if tuple(logits.shape) != (batch_size,):
                    raise ValueError(
                        f"Task '{task}' logits have shape {tuple(logits.shape)}; "
                        f"expected ({batch_size},)."
                    )
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Task '{task}' produced non-finite logits.")
                task_heads[task] = {
                    "logit_shape": list(logits.shape),
                    "finite_logits": True,
                }

        report.update(
            {
                "status": "compatible",
                "tokenizer": _tokenizer_report(tokenizer),
                "encoder_forward": {
                    "batch_size": batch_size,
                    "sequence_length": int(hidden_state.shape[1]),
                    "hidden_state_shape": list(hidden_state.shape),
                    "finite_hidden_states": True,
                },
                "task_heads": task_heads,
                "parameters": {
                    "encoder_total": _parameter_count(encoder.parameters()),
                    "encoder_trainable": _parameter_count(
                        parameter for parameter in encoder.parameters() if parameter.requires_grad
                    ),
                    "model_total": _parameter_count(multitask_model.parameters()),
                    "model_trainable": _parameter_count(
                        parameter
                        for parameter in multitask_model.parameters()
                        if parameter.requires_grad
                    ),
                },
            }
        )
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        _write_report(output_path, report)
        if isinstance(exc, EncoderCompatibilityError):
            raise
        raise EncoderCompatibilityError(str(exc)) from exc

    _write_report(output_path, report)
    return report


def classify_loading_information(
    encoder: torch.nn.Module, loading_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Separate expected discarded MLM-head keys from real encoder problems."""

    missing = sorted(str(key) for key in loading_info.get("missing_keys", []))
    unexpected = sorted(str(key) for key in loading_info.get("unexpected_keys", []))
    mismatched = [str(item) for item in loading_info.get("mismatched_keys", [])]
    loading_errors = [str(item) for item in loading_info.get("error_msgs", [])]
    expected_unused_mlm_head = [key for key in unexpected if _is_mlm_head_key(key)]
    unexpected_non_mlm = [key for key in unexpected if not _is_mlm_head_key(key)]
    expected_missing_unused_pooler = [key for key in missing if _is_pooler_key(key)]
    unexpected_missing_encoder = [key for key in missing if not _is_pooler_key(key)]
    encoder_state_keys = set(encoder.state_dict())
    loaded_encoder_keys = sorted(encoder_state_keys - set(missing))
    problems: list[str] = []
    if unexpected_missing_encoder:
        problems.append("Missing encoder weights: " + ", ".join(unexpected_missing_encoder))
    if unexpected_non_mlm:
        problems.append("Unexpected non-MLM checkpoint keys: " + ", ".join(unexpected_non_mlm))
    if mismatched:
        problems.append("Mismatched checkpoint tensors: " + ", ".join(mismatched))
    if loading_errors:
        problems.append("Checkpoint loading errors: " + ", ".join(loading_errors))
    if not loaded_encoder_keys:
        problems.append("No shared encoder weights were loaded.")
    return {
        "missing_encoder_keys": missing,
        "expected_missing_unused_pooler_keys": expected_missing_unused_pooler,
        "unexpected_missing_encoder_keys": unexpected_missing_encoder,
        "unexpected_checkpoint_keys": unexpected,
        "expected_unused_mlm_head_keys": expected_unused_mlm_head,
        "unexpected_non_mlm_keys": unexpected_non_mlm,
        "mismatched_keys": mismatched,
        "error_messages": loading_errors,
        "encoder_state_key_count": len(encoder_state_keys),
        "loaded_encoder_state_key_count": len(loaded_encoder_keys),
        "shared_encoder_weights_loaded": bool(loaded_encoder_keys) and not unexpected_missing_encoder,
        "compatibility_problems": problems,
    }


def _architecture_report(encoder: torch.nn.Module) -> dict[str, Any]:
    config = getattr(encoder, "config", None)
    if config is None:
        raise ValueError("Encoder must expose a Hugging Face configuration.")
    fields = {
        "hidden_size": getattr(config, "hidden_size", None),
        "layer_count": getattr(config, "num_hidden_layers", None),
        "attention_heads": getattr(config, "num_attention_heads", None),
        "vocabulary_size": getattr(config, "vocab_size", None),
        "maximum_positions": getattr(config, "max_position_embeddings", None),
    }
    invalid = [name for name, value in fields.items() if not isinstance(value, int) or value <= 0]
    if invalid:
        raise ValueError("Encoder configuration has invalid fields: " + ", ".join(invalid))
    return {"model_type": getattr(config, "model_type", None), **fields}


def _tokenizer_report(tokenizer: Any) -> dict[str, Any]:
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "class_name": type(tokenizer).__name__,
        "vocabulary_size": len(tokenizer),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_token_id", None)
            for name in ("pad", "unk", "cls", "sep", "mask", "bos", "eos")
        },
    }


def _is_mlm_head_key(key: str) -> bool:
    return key.startswith("lm_head.") or ".lm_head." in key


def _is_pooler_key(key: str) -> bool:
    return key.startswith("pooler.") or ".pooler." in key


def _parameter_count(parameters: Any) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EncoderCompatibilityError",
    "classify_loading_information",
    "verify_encoder_compatibility",
]
