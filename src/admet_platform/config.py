"""Endpoint configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only when PyYAML is unavailable.
    yaml = None  # type: ignore[assignment]


ALLOWED_TASK_TYPES = {"binary_classification", "regression"}

REQUIRED_FIELDS = (
    "endpoint_id",
    "tdc_name",
    "task_group",
    "task_type",
    "target_column",
    "smiles_column",
    "split_strategy",
    "metric_names",
    "base_model",
    "problem_description",
    "limitations",
    "output_prediction_column",
    "output_score_column",
)


@dataclass(frozen=True)
class EndpointConfig:
    """Validated metadata for one TDC ADMET endpoint."""

    endpoint_id: str
    tdc_name: str
    task_group: str
    task_type: str
    target_column: str
    smiles_column: str
    split_strategy: str
    metric_names: list[str]
    base_model: str
    problem_description: str
    limitations: list[str]
    output_prediction_column: str
    output_score_column: str


def load_endpoint_config(path: str | Path) -> EndpointConfig:
    """Load and validate an endpoint YAML config file."""

    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    raw_config = _load_yaml_mapping(raw_text, source=str(config_path))

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping.")

    return _validate_endpoint_config(raw_config, source=str(config_path))


def _load_yaml_mapping(raw_text: str, source: str) -> dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {source} must contain a YAML mapping.")
        return loaded

    parsed: dict[str, Any] = {}
    current_list_key: str | None = None

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"Config {source} has an unexpected list item on line {line_number}.")
            parsed[current_list_key].append(stripped[2:].strip())
            continue

        if ":" not in stripped:
            raise ValueError(f"Config {source} has invalid YAML syntax on line {line_number}.")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Config {source} has an empty key on line {line_number}.")

        if value:
            parsed[key] = value
            current_list_key = None
        else:
            parsed[key] = []
            current_list_key = key

    return parsed


def _validate_endpoint_config(raw_config: dict[str, Any], source: str) -> EndpointConfig:
    missing_fields = [field for field in REQUIRED_FIELDS if field not in raw_config]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Config {source} is missing required field(s): {missing}.")

    string_fields = [field for field in REQUIRED_FIELDS if field not in {"metric_names", "limitations"}]
    for field in string_fields:
        value = raw_config[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config {source} field '{field}' must be a non-empty string.")

    task_type = raw_config["task_type"]
    if task_type not in ALLOWED_TASK_TYPES:
        allowed = ", ".join(sorted(ALLOWED_TASK_TYPES))
        raise ValueError(f"Config {source} field 'task_type' must be one of: {allowed}.")

    metric_names = raw_config["metric_names"]
    if (
        not isinstance(metric_names, list)
        or not metric_names
        or not all(isinstance(metric, str) and metric.strip() for metric in metric_names)
    ):
        raise ValueError(f"Config {source} field 'metric_names' must be a non-empty list of strings.")

    limitations = raw_config["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(limitation, str) and limitation.strip() for limitation in limitations)
    ):
        raise ValueError(f"Config {source} field 'limitations' must be a non-empty list of strings.")

    return EndpointConfig(
        endpoint_id=raw_config["endpoint_id"],
        tdc_name=raw_config["tdc_name"],
        task_group=raw_config["task_group"],
        task_type=task_type,
        target_column=raw_config["target_column"],
        smiles_column=raw_config["smiles_column"],
        split_strategy=raw_config["split_strategy"],
        metric_names=metric_names,
        base_model=raw_config["base_model"],
        problem_description=raw_config["problem_description"],
        limitations=limitations,
        output_prediction_column=raw_config["output_prediction_column"],
        output_score_column=raw_config["output_score_column"],
    )
