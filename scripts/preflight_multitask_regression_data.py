"""Train/validation-only real-data loader and model preflight."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.multitask_regression import (  # noqa: E402
    build_regression_dataloaders,
    fit_training_transforms,
    load_multitask_regression_config,
    load_regression_training_datasets,
)
from admet_platform.models.multitask_regression_chemberta import (  # noqa: E402
    MultiTaskRegressionChemBERTa,
    MultiTaskRegressionChemBERTaConfig,
)
from admet_platform.training.task_sampler import RoundRobinTaskSampler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify real regression train/validation plumbing without test access."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = load_multitask_regression_config(args.config)
    datasets = load_regression_training_datasets(config, args.prepared_root)
    transforms = fit_training_transforms(datasets)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    loaders = build_regression_dataloaders(
        datasets,
        transforms,
        tokenizer,
        seed=42,
        train_batch_size=2,
        evaluation_batch_size=2,
        max_length=config.training.max_sequence_length,
        limit_samples_per_task=2,
        limit_validation_samples_per_task=2,
    )
    model = MultiTaskRegressionChemBERTa(
        MultiTaskRegressionChemBERTaConfig(
            model_name_or_path=args.checkpoint,
            tasks=tuple(config.tasks),
            pooling=config.training.pooling,  # type: ignore[arg-type]
            dropout=0.0,
            local_files_only=True,
        )
    )
    model.eval()
    endpoints = {}
    for task in model.task_names:
        batch = next(iter(loaders[task]["train"]))
        labels = batch["labels"].numpy()
        original = batch["target_original"].numpy()
        restored = transforms[task].inverse_values(labels)
        with torch.no_grad():
            prediction = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                task_name=task,
            )
        endpoints[task] = {
            "input_ids_shape": list(batch["input_ids"].shape),
            "attention_mask_shape": list(batch["attention_mask"].shape),
            "normalized_target_shape": list(batch["labels"].shape),
            "prediction_shape": list(prediction.shape),
            "normalized_targets_finite": bool(np.isfinite(labels).all()),
            "original_targets_finite": bool(np.isfinite(original).all()),
            "predictions_finite": bool(torch.isfinite(prediction).all()),
            "task_identity_matches": set(batch["task_name"]) == {task},
            "inverse_transform_matches_original": bool(
                np.allclose(restored, original, rtol=1e-6, atol=1e-6)
            ),
        }
    sampler = RoundRobinTaskSampler(model.task_names)
    sequence = [sampler.next_task() for _ in model.task_names]
    report = {
        "schema_version": "1.0.0",
        "loaded_splits": ["train", "validation"],
        "test_data_used": False,
        "offline": True,
        "task_names": list(model.task_names),
        "round_robin_first_pass": sequence,
        "all_endpoints_yielded_once": tuple(sequence) == model.task_names,
        "endpoints": endpoints,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
