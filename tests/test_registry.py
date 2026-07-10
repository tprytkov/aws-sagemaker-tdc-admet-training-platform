import json
import subprocess
import sys
from pathlib import Path

import pytest

from admet_platform.registry import build_model_registry_entry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_registry_entry_json_is_written_with_required_fields(tmp_path: Path) -> None:
    metrics_path = _write_metrics_json(tmp_path)
    output_path = tmp_path / "bbb_registry.json"

    entry = build_model_registry_entry(
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        metrics_json_path=metrics_path,
        artifact_uri="outputs/bbb_martins_baseline.joblib",
        output_json_path=output_path,
        validation_status="toy_sample_only",
    )

    written_entry = json.loads(output_path.read_text(encoding="utf-8"))
    required_fields = {
        "model_id",
        "endpoint_id",
        "tdc_name",
        "task_group",
        "task_type",
        "model_type",
        "base_model",
        "artifact_uri",
        "training_source",
        "metrics",
        "validation_status",
        "input_schema",
        "output_schema",
        "moloptima_enabled",
        "limitations",
        "created_by",
        "notes",
    }

    assert output_path.exists()
    assert entry == written_entry
    assert required_fields <= set(written_entry)


def test_registry_entry_copies_config_and_metrics_values(tmp_path: Path) -> None:
    metrics_path = _write_metrics_json(tmp_path)
    output_path = tmp_path / "bbb_registry.json"

    entry = build_model_registry_entry(
        config_path=CONFIG_DIR / "bbb_martins.yaml",
        metrics_json_path=metrics_path,
        artifact_uri="outputs/bbb_martins_baseline.joblib",
        output_json_path=output_path,
        validation_status="toy_sample_only",
    )

    assert entry["endpoint_id"] == "bbb_martins"
    assert entry["task_type"] == "binary_classification"
    assert entry["model_type"] == "tfidf_logistic_regression"
    assert entry["moloptima_enabled"] is False
    assert entry["validation_status"] == "toy_sample_only"


def test_mismatched_endpoint_id_raises_value_error(tmp_path: Path) -> None:
    metrics_path = _write_metrics_json(tmp_path, endpoint_id="wrong_endpoint")

    with pytest.raises(ValueError, match="endpoint_id"):
        build_model_registry_entry(
            config_path=CONFIG_DIR / "bbb_martins.yaml",
            metrics_json_path=metrics_path,
            artifact_uri="outputs/bbb_martins_baseline.joblib",
            output_json_path=tmp_path / "bbb_registry.json",
        )


def test_build_registry_entry_cli_works(tmp_path: Path) -> None:
    metrics_path = _write_metrics_json(tmp_path)
    output_path = tmp_path / "bbb_registry_cli.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_registry_entry.py"),
            "--config",
            str(CONFIG_DIR / "bbb_martins.yaml"),
            "--metrics-json",
            str(metrics_path),
            "--artifact-uri",
            "outputs/bbb_martins_baseline.joblib",
            "--output-json",
            str(output_path),
            "--validation-status",
            "toy_sample_only",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    assert "Wrote model registry entry" in result.stdout


def _write_metrics_json(
    tmp_path: Path,
    endpoint_id: str = "bbb_martins",
    task_type: str = "binary_classification",
) -> Path:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "endpoint_id": endpoint_id,
                "task_type": task_type,
                "model_type": "tfidf_logistic_regression",
                "n_train": 3,
                "n_test": 2,
                "metrics": {
                    "accuracy": 1.0,
                    "balanced_accuracy": 1.0,
                    "f1": 1.0,
                    "auroc": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return metrics_path
