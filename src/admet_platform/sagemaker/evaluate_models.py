"""SageMaker Processing-compatible model evaluation entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - requirements include PyYAML.
    yaml = None  # type: ignore[assignment]

from admet_platform.evaluation import ComparisonOptions, discover_run_dirs, evaluate_model_runs
from admet_platform.models.artifacts import write_json
from admet_platform.sagemaker.launch_training import sanitize_text


DEFAULT_RUNS_DIR = "/opt/ml/processing/input/runs"
DEFAULT_CONFIG_DIR = "/opt/ml/processing/input/config"
DEFAULT_OUTPUT_DIR = "/opt/ml/processing/output"


def run_processing_evaluation(
    *,
    runs_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    endpoint_id: str | None = None,
    near_tie_tolerance: float | None = None,
    primary_metric_override: str | None = None,
    include_development_runs: bool | None = None,
    registry_schema_version: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate candidate model runs and write SageMaker Processing outputs."""

    start = datetime.now(UTC)
    start_time = time.perf_counter()
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    if not runs_path.exists():
        raise ValueError(f"Candidate runs input directory does not exist: {runs_path}.")

    config = load_evaluation_config(config_path) if config_path else {}
    effective = build_effective_config(
        config,
        {
            "endpoint_id": endpoint_id,
            "near_tie_tolerance": near_tie_tolerance,
            "primary_metric_override": primary_metric_override,
            "include_development_runs": include_development_runs,
            "registry_schema_version": registry_schema_version,
        },
    )
    discovered = discover_run_dirs(runs_path)
    if effective.get("candidate_run_ids"):
        requested = set(effective["candidate_run_ids"])
        discovered = [path for path in discovered if _candidate_run_id(path) in requested]
        missing = sorted(requested - {_candidate_run_id(path) for path in discovered})
        if missing:
            raise ValueError(f"Requested candidate run ID(s) were not found: {', '.join(missing)}.")

    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="admet-evaluation-processing-") as temp_root:
        local_output = Path(temp_root) / "evaluation"
        result = evaluate_model_runs(
            discovered,
            local_output,
            options=ComparisonOptions(
                near_tie_tolerance=float(effective["near_tie_tolerance"]),
                primary_metric_override=effective.get("primary_metric_override"),
                include_development_runs=bool(effective["include_development_runs"]),
                registry_schema_version=str(effective["registry_schema_version"]),
            ),
            explicit_endpoint_id=effective.get("endpoint_id"),
        )
        _write_processing_layout(local_output, output_path)
        inventory = build_artifact_inventory(output_path)
        manifest = build_processing_manifest(
            processing_run_id=run_id or str(uuid.uuid4()),
            result=result,
            effective=effective,
            input_paths={"runs_dir": str(runs_path), "config_path": str(config_path) if config_path else None},
            output_paths=processing_output_paths(output_path),
            artifact_inventory=inventory,
            started_at=start,
            runtime_seconds=time.perf_counter() - start_time,
            status="completed",
            error=None,
        )
        write_json(output_path / "metadata" / "artifact_inventory.json", inventory)
        write_json(output_path / "metadata" / "evaluation_processing_manifest.json", manifest)
        final_inventory = build_artifact_inventory(output_path)
        manifest["generated_artifact_inventory"] = final_inventory
        write_json(output_path / "metadata" / "evaluation_processing_manifest.json", manifest)
        write_json(output_path / "metadata" / "artifact_inventory.json", final_inventory)
        return manifest


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Evaluation config file does not exist: {config_path}.")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load evaluation YAML configuration.")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config {config_path} must contain an object/mapping.")
    return payload


def build_effective_config(raw: dict[str, Any], cli_overrides: dict[str, Any]) -> dict[str, Any]:
    effective = {
        "endpoint_id": raw.get("endpoint_id"),
        "near_tie_tolerance": raw.get("near_tie_tolerance", 0.01),
        "primary_metric_override": raw.get("primary_metric_override"),
        "include_development_runs": raw.get("include_development_runs", False),
        "registry_schema_version": raw.get("registry_schema_version", "1.0.0"),
        "candidate_run_ids": raw.get("candidate_run_ids"),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            effective[key] = value
    return effective


def resolve_processing_paths(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Path | None]:
    config_path = Path(args.config) if args.config else _resolve_optional_config_dir(
        Path(args.config_dir or env.get("SM_EVALUATION_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    )
    return {
        "runs_dir": Path(args.runs_dir or env.get("SM_EVALUATION_RUNS_DIR", DEFAULT_RUNS_DIR)),
        "output_dir": Path(args.output_dir or env.get("SM_EVALUATION_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        "config_path": config_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ADMET model runs in a SageMaker Processing layout.")
    parser.add_argument("--runs-dir")
    parser.add_argument("--config-dir")
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--near-tie-tolerance", type=float)
    parser.add_argument("--primary-metric")
    parser.add_argument("--include-development-runs", action="store_true")
    parser.add_argument("--registry-schema-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir: Path | None = None
    try:
        paths = resolve_processing_paths(args, os.environ)
        output_dir = Path(paths["output_dir"])  # type: ignore[arg-type]
        run_processing_evaluation(
            runs_dir=paths["runs_dir"],  # type: ignore[arg-type]
            output_dir=output_dir,
            config_path=paths["config_path"],
            endpoint_id=args.endpoint_id,
            near_tie_tolerance=args.near_tie_tolerance,
            primary_metric_override=args.primary_metric,
            include_development_runs=True if args.include_development_runs else None,
            registry_schema_version=args.registry_schema_version,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - entry point must write failure manifest.
        if output_dir is not None:
            write_failed_manifest(output_dir, exc)
        print(f"SageMaker evaluation processing failed: {sanitize_text(str(exc))}", file=sys.stderr)
        return 1


def write_failed_manifest(output_dir: str | Path, exc: Exception) -> Path:
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "evaluation_processing_manifest.json"
    write_json(
        manifest_path,
        {
            "processing_run_id": str(uuid.uuid4()),
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": sanitize_text(str(exc))},
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return manifest_path


def build_processing_manifest(
    *,
    processing_run_id: str,
    result: Any,
    effective: dict[str, Any],
    input_paths: dict[str, str | None],
    output_paths: dict[str, str],
    artifact_inventory: dict[str, Any],
    started_at: datetime,
    runtime_seconds: float,
    status: str,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "processing_run_id": processing_run_id,
        "endpoint_id": result.endpoint_id,
        "task_type": result.task_type,
        "source_dataset": result.source_dataset,
        "discovered_candidate_run_ids": result.evaluated_run_ids,
        "eligible_run_ids": result.eligible_run_ids,
        "excluded_run_ids": result.excluded_runs,
        "recommendation_status": result.recommendation_status,
        "recommended_run_id": result.recommended_run_id,
        "selection_metric": result.comparison_metric,
        "near_tie_tolerance": effective["near_tie_tolerance"],
        "include_development_runs": effective["include_development_runs"],
        "input_paths": input_paths,
        "output_paths": output_paths,
        "generated_artifact_inventory": artifact_inventory,
        "package_versions": evaluation_package_versions(),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "status": status,
        "warnings": result.warnings,
        "development_mode": bool(effective["include_development_runs"]),
        "error": error,
    }


def build_artifact_inventory(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    artifacts = {
        "evaluation": [
            "evaluation_summary.json",
            "model_comparison.csv",
            "model_comparison.json",
            "recommended_model.json",
            "evaluation_warnings.json",
        ],
        "model_card": ["model_card.md"],
        "registry": ["registry_entry.json"],
        "metadata": ["evaluation_processing_manifest.json", "artifact_inventory.json"],
    }
    return {
        key: [
            {
                "path": str(output_path / key / name),
                "exists": (output_path / key / name).exists(),
                "bytes": (output_path / key / name).stat().st_size if (output_path / key / name).exists() else 0,
            }
            for name in names
        ]
        for key, names in artifacts.items()
    }


def evaluation_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in ("pandas", "numpy", "PyYAML"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def processing_output_paths(output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    return {
        "evaluation": str(output_path / "evaluation"),
        "model_card": str(output_path / "model_card"),
        "registry": str(output_path / "registry"),
        "metadata": str(output_path / "metadata"),
    }


def _write_processing_layout(local_output: Path, output_dir: Path) -> None:
    layout = {
        "evaluation_summary.json": output_dir / "evaluation" / "evaluation_summary.json",
        "model_comparison.csv": output_dir / "evaluation" / "model_comparison.csv",
        "model_comparison.json": output_dir / "evaluation" / "model_comparison.json",
        "recommended_model.json": output_dir / "evaluation" / "recommended_model.json",
        "evaluation_warnings.json": output_dir / "evaluation" / "evaluation_warnings.json",
        "model_card.md": output_dir / "model_card" / "model_card.md",
        "registry_entry.json": output_dir / "registry" / "registry_entry.json",
    }
    for source_name, destination in layout.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_output / source_name, destination)


def _resolve_optional_config_dir(config_dir: Path) -> Path | None:
    if not config_dir.exists():
        return None
    configs = sorted(
        path
        for path in config_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}
    )
    if not configs:
        return None
    if len(configs) > 1:
        names = ", ".join(path.name for path in configs)
        raise ValueError(f"Multiple evaluation config files found in {config_dir}: {names}.")
    return configs[0]


def _candidate_run_id(path: Path) -> str | None:
    metadata_path = path / "training_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get("run_id")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
