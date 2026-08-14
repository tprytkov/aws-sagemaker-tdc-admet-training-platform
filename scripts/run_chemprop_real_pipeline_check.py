"""Run a bounded real train/validation execution check without locked-test access."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import load_chemprop_config  # noqa: E402
from admet_platform.chemprop.data import load_verified_development_data  # noqa: E402
from admet_platform.chemprop.pipeline_check import (  # noqa: E402
    balanced_binary_subset,
    endpoint_balanced_sparse_subset,
)
from admet_platform.chemprop.training import train_graph_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    output = Path(args.output_dir)
    regression_config = load_chemprop_config("configs/chemprop/multitask_admet_regression.yaml")
    bbb_config = load_chemprop_config("configs/chemprop/bbb_martins.yaml")
    regression = load_verified_development_data(regression_config)
    bbb = load_verified_development_data(bbb_config)
    endpoints = list(regression_config.tasks)

    regression_train = endpoint_balanced_sparse_subset(regression.train, endpoints, 64)
    regression_validation = endpoint_balanced_sparse_subset(
        regression.validation, endpoints, 24
    )
    bbb_train = balanced_binary_subset(bbb.train, 64)
    bbb_validation = balanced_binary_subset(bbb.validation, 24)
    regression_result = train_graph_model(
        regression_config,
        regression_train,
        regression_validation,
        output / "multitask_regression",
        seed=args.seed,
        smoke=True,
    )
    bbb_result = train_graph_model(
        bbb_config,
        bbb_train,
        bbb_validation,
        output / "bbb_martins",
        seed=args.seed,
        smoke=True,
    )
    _assert_finite_metrics(regression_result["validation_metrics"])
    _assert_finite_metrics(bbb_result["validation_metrics_calibrated_selected"])
    summary = {
        "protocol": "bounded_real_train_validation_pipeline_check",
        "seed": args.seed,
        "full_training": False,
        "locked_test_csv_opened": False,
        "multitask_regression": {
            "train_rows": len(regression_train),
            "validation_rows": len(regression_validation),
            "observed_labels": {
                split: {endpoint: int(frame[endpoint].notna().sum()) for endpoint in endpoints}
                for split, frame in (
                    ("train", regression_train), ("validation", regression_validation)
                )
            },
            "missing_labels_present": bool(regression_train[endpoints].isna().any().any()),
            "completed_without_nan_loss": True,
            "monitor": regression_result["monitor"],
            "checkpoint": regression_result["checkpoint"],
            "validation_prediction_files": regression_result["validation_prediction_files"],
        },
        "bbb": {
            "train_rows": len(bbb_train),
            "validation_rows": len(bbb_validation),
            "train_class_counts": _class_counts(bbb_train),
            "validation_class_counts": _class_counts(bbb_validation),
            "calibration_fit_split": bbb_result["calibrator"]["fit_split"],
            "threshold_selection_split": "validation",
            "thresholds_reported": [0.5, bbb_result["validation_selected_threshold"]],
            "checkpoint": bbb_result["checkpoint"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pipeline_check_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _class_counts(frame) -> dict[str, int]:
    counts = frame["target"].astype(int).value_counts()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def _assert_finite_metrics(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_metrics(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_metrics(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Pipeline check produced a non-finite validation metric.")


if __name__ == "__main__":
    main()
