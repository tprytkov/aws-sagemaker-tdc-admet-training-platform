"""Train the separate prepared-data multi-task regression model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.training.multitask_regression_run import (  # noqa: E402
    run_multitask_regression_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train normalized ChemBERTa regression heads from coordinated splits."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples-per-task", type=int)
    parser.add_argument("--limit-validation-samples-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"))
    parser.add_argument("--loss", choices=("huber", "mse"))
    parser.add_argument("--evaluation-interval-steps", type=int)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    args = parser.parse_args()
    result = run_multitask_regression_training(
        config_path=args.config,
        prepared_root=args.prepared_root,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        resume_from=args.resume_from,
        max_steps=args.max_steps,
        limit_samples_per_task=args.limit_samples_per_task,
        limit_validation_samples_per_task=args.limit_validation_samples_per_task,
        seed=args.seed,
        device=args.device,
        offline=args.offline,
        deterministic_algorithms=args.deterministic_algorithms,
        mixed_precision=args.mixed_precision,
        loss=args.loss,
        evaluation_interval_steps=args.evaluation_interval_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
