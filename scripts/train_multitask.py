"""Platform-independent prepared-data multi-task ChemBERTa entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.training.multitask_run import run_multitask_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one or more configured ChemBERTa task heads from prepared split CSVs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Base Hugging Face model/tokenizer name or local directory.")
    parser.add_argument("--resume-from", help="Trainer checkpoint.pt to resume.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Steps to execute in this invocation; defaults to training.max_steps.")
    parser.add_argument("--limit-samples-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--classical-baseline-json")
    parser.add_argument("--single-task-baseline-json")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--evaluation-interval-steps", type=int)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--early-stopping-patience-evaluations", type=int)
    parser.add_argument("--minimum-training-steps-before-stopping", type=int)
    args = parser.parse_args()
    result = run_multitask_training(
        config_path=args.config, prepared_root=args.prepared_root, output_dir=args.output_dir,
        checkpoint=args.checkpoint, resume_from=args.resume_from, max_steps=args.max_steps,
        limit_samples_per_task=args.limit_samples_per_task, seed=args.seed,
        device=args.device, offline=args.offline,
        deterministic_algorithms=args.deterministic_algorithms,
        classical_baseline_json=args.classical_baseline_json,
        single_task_baseline_json=args.single_task_baseline_json,
        mixed_precision=args.mixed_precision,
        evaluation_interval_steps=args.evaluation_interval_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience_evaluations=args.early_stopping_patience_evaluations,
        minimum_training_steps_before_stopping=args.minimum_training_steps_before_stopping,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
