from pathlib import Path

import pytest

from admet_platform.config import EndpointConfig, load_endpoint_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_all_endpoint_configs_load_successfully() -> None:
    config_paths = [
        CONFIG_DIR / "bbb_martins.yaml",
        CONFIG_DIR / "caco2_wang.yaml",
        CONFIG_DIR / "herg_karim.yaml",
    ]

    configs = [load_endpoint_config(path) for path in config_paths]

    assert all(isinstance(config, EndpointConfig) for config in configs)
    assert {config.endpoint_id for config in configs} == {
        "bbb_martins",
        "caco2_wang",
        "herg_karim",
    }


def test_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    config = _valid_config()
    del config["endpoint_id"]
    config_path = tmp_path / "missing_field.yaml"
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="missing required field"):
        load_endpoint_config(config_path)


def test_invalid_task_type_raises_value_error(tmp_path: Path) -> None:
    config = _valid_config()
    config["task_type"] = "multiclass"
    config_path = tmp_path / "invalid_task_type.yaml"
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="task_type"):
        load_endpoint_config(config_path)


def test_metric_names_must_be_non_empty(tmp_path: Path) -> None:
    config = _valid_config()
    config["metric_names"] = []
    config_path = tmp_path / "empty_metrics.yaml"
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="metric_names"):
        load_endpoint_config(config_path)


def _valid_config() -> dict[str, object]:
    return {
        "endpoint_id": "example_endpoint",
        "tdc_name": "Example_TDC",
        "task_group": "ADME",
        "task_type": "binary_classification",
        "target_column": "Y",
        "smiles_column": "Drug",
        "split_strategy": "scaffold",
        "metric_names": ["roc_auc"],
        "base_model": "seyonec/ChemBERTa-zinc-base-v1",
        "problem_description": "Example endpoint configuration.",
        "limitations": ["Public-safe example only."],
        "output_prediction_column": "example_prediction",
        "output_score_column": "example_score",
    }


def _write_config(path: Path, config: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
