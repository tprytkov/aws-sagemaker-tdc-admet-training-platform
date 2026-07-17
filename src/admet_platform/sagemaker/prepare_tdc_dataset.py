"""SageMaker Processing-compatible TDC ADMET dataset preparation entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from admet_platform.benchmarks.local import summarize_prepared_dataset
from admet_platform.config import EndpointConfig, load_endpoint_config
from admet_platform.data.prepare import prepare_dataset_artifacts
from admet_platform.data.tdc_loader import (
    load_tdc_data,
    normalize_tdc_dataframe,
    normalize_tdc_raw_dataframe,
)
from admet_platform.models.artifacts import write_json
from admet_platform.sagemaker.launch_training import sanitize_text


PROCESSING_MODES = {"tdc_download", "supplied_csv"}
DEFAULT_CONFIG_DIR = "/opt/ml/processing/input/config"
DEFAULT_INPUT_DATA_DIR = "/opt/ml/processing/input/data"
DEFAULT_OUTPUT_DIR = "/opt/ml/processing/output"
REQUIRED_CSV_COLUMNS = {"molecule_id", "smiles", "target", "split"}
PACKAGE_NAMES = ("pandas", "rdkit", "PyYAML", "tdc")


def prepare_processing_dataset(
    *,
    endpoint_config_path: str | Path,
    processing_mode: str,
    output_dir: str | Path,
    input_data_dir: str | Path | None = None,
    development_row_limit: int | None = None,
    run_id: str | None = None,
    split_seed: int | None = None,
    tdc_split_loader: Callable[[EndpointConfig], dict[str, pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """Prepare ADMET splits into the SageMaker Processing output contract."""

    if processing_mode not in PROCESSING_MODES:
        allowed = ", ".join(sorted(PROCESSING_MODES))
        raise ValueError(f"Unsupported processing mode '{processing_mode}'. Expected one of: {allowed}.")
    start = datetime.now(UTC)
    start_time = time.perf_counter()
    config = load_endpoint_config(endpoint_config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    run_identifier = run_id or str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix="admet-processing-") as temp_root:
        temp_path = Path(temp_root)
        source_csv, source_row_count, input_paths = _resolve_source_csv(
            config=config,
            endpoint_config_path=endpoint_config_path,
            processing_mode=processing_mode,
            input_data_dir=input_data_dir,
            temp_path=temp_path,
            development_row_limit=development_row_limit,
            tdc_split_loader=tdc_split_loader,
        )
        flat_output = temp_path / "prepared_flat"
        prepare_dataset_artifacts(source_csv, endpoint_config_path, flat_output)
        dataset_summary = summarize_prepared_dataset(flat_output, config)
        if any(dataset_summary["cross_split_overlap_counts"].values()):
            warnings.append("canonical_smiles overlap detected across splits")
        if dataset_summary.get("warnings"):
            warnings.extend(dataset_summary["warnings"])

        manifest = _build_processing_manifest(
            run_id=run_identifier,
            config=config,
            processing_mode=processing_mode,
            source_row_count=source_row_count,
            dataset_summary=dataset_summary,
            split_seed=split_seed,
            input_paths=input_paths,
            output_dir=output_path,
            start=start,
            runtime_seconds=time.perf_counter() - start_time,
            status="completed",
            warnings=sorted(set(warnings)),
            development_mode=development_row_limit is not None,
            error=None,
        )
        _write_processing_layout(flat_output, output_path, manifest)
        return manifest


def resolve_processing_paths(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Path | str | int | None]:
    config_dir = Path(args.config_dir or env.get("SM_PROCESSING_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    input_data_dir = Path(args.input_data_dir or env.get("SM_PROCESSING_INPUT_DATA_DIR", DEFAULT_INPUT_DATA_DIR))
    output_dir = Path(args.output_dir or env.get("SM_PROCESSING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    endpoint_config = Path(args.endpoint_config) if args.endpoint_config else _resolve_endpoint_config(config_dir)
    mode = args.mode or env.get("SM_PROCESSING_MODE")
    row_limit = args.development_row_limit
    if row_limit is None and env.get("SM_PROCESSING_DEVELOPMENT_ROW_LIMIT"):
        row_limit = int(env["SM_PROCESSING_DEVELOPMENT_ROW_LIMIT"])
    return {
        "config_dir": config_dir,
        "input_data_dir": input_data_dir,
        "output_dir": output_dir,
        "endpoint_config": endpoint_config,
        "mode": mode,
        "development_row_limit": row_limit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TDC ADMET data in a SageMaker Processing layout.")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--input-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--endpoint-config", default=None)
    parser.add_argument("--mode", choices=sorted(PROCESSING_MODES), default=None)
    parser.add_argument("--development-row-limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir: Path | None = None
    try:
        paths = resolve_processing_paths(args, os.environ)
        output_dir = Path(paths["output_dir"])  # type: ignore[arg-type]
        mode = paths["mode"]
        if not mode:
            raise ValueError("Processing mode is required. Use --mode tdc_download or --mode supplied_csv.")
        prepare_processing_dataset(
            endpoint_config_path=paths["endpoint_config"],  # type: ignore[arg-type]
            processing_mode=str(mode),
            output_dir=output_dir,
            input_data_dir=paths["input_data_dir"],  # type: ignore[arg-type]
            development_row_limit=paths["development_row_limit"],  # type: ignore[arg-type]
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - entry point must write failure manifest.
        if output_dir is not None:
            write_failed_manifest(output_dir, exc)
        print(f"SageMaker Processing preparation failed: {sanitize_text(str(exc))}", file=sys.stderr)
        return 1


def write_failed_manifest(output_dir: str | Path, exc: Exception) -> Path:
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "processing_manifest.json"
    write_json(
        manifest_path,
        {
            "run_id": str(uuid.uuid4()),
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": sanitize_text(str(exc)),
            },
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return manifest_path


def _resolve_source_csv(
    *,
    config: EndpointConfig,
    endpoint_config_path: str | Path,
    processing_mode: str,
    input_data_dir: str | Path | None,
    temp_path: Path,
    development_row_limit: int | None,
    tdc_split_loader: Callable[[EndpointConfig], dict[str, pd.DataFrame]] | None,
) -> tuple[Path, int, dict[str, str]]:
    if processing_mode == "supplied_csv":
        if input_data_dir is None:
            raise ValueError("input_data_dir is required in supplied_csv mode.")
        csv_path = resolve_single_input_csv(input_data_dir)
        _validate_required_csv_columns(csv_path)
        source_rows = int(len(pd.read_csv(csv_path)))
        return csv_path, source_rows, {
            "endpoint_config": str(endpoint_config_path),
            "input_data_dir": str(input_data_dir),
            "source_csv": str(csv_path),
        }

    loader = tdc_split_loader or load_tdc_data
    loaded_data = loader(config)
    if isinstance(loaded_data, dict):  # Compatibility for existing injected test/supplied loaders.
        normalized = [
            normalize_tdc_dataframe(split_df, split_name, config)
            for split_name, split_df in loaded_data.items()
        ]
        source_df = pd.concat(normalized, ignore_index=True)
    else:
        source_df = normalize_tdc_raw_dataframe(loaded_data, config)
    if development_row_limit is not None:
        if "split" in source_df.columns:
            source_df = source_df.groupby("split", group_keys=False).head(development_row_limit).reset_index(drop=True)
        else:
            source_df = source_df.head(development_row_limit).reset_index(drop=True)
    source_csv = temp_path / "tdc_normalized_source.csv"
    source_df.to_csv(source_csv, index=False)
    return source_csv, int(len(source_df)), {
        "endpoint_config": str(endpoint_config_path),
        "tdc_name": config.tdc_name,
        "task_group": config.task_group,
    }


def resolve_single_input_csv(input_data_dir: str | Path) -> Path:
    input_path = Path(input_data_dir)
    if not input_path.exists():
        raise ValueError(f"Input-data directory does not exist: {input_path}.")
    csv_files = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    if not csv_files:
        raise ValueError(f"No CSV file found in supplied input-data directory: {input_path}.")
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise ValueError(f"Multiple CSV files found in supplied input-data directory {input_path}: {names}.")
    return csv_files[0]


def _resolve_endpoint_config(config_dir: Path) -> Path:
    if not config_dir.exists():
        raise ValueError(f"Input configuration directory does not exist: {config_dir}.")
    configs = sorted(
        path for path in config_dir.iterdir() if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not configs:
        raise ValueError(f"No endpoint YAML config found in input configuration directory: {config_dir}.")
    if len(configs) > 1:
        names = ", ".join(path.name for path in configs)
        raise ValueError(f"Multiple endpoint YAML configs found in input configuration directory {config_dir}: {names}.")
    return configs[0]


def _validate_required_csv_columns(csv_path: Path) -> None:
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = sorted(REQUIRED_CSV_COLUMNS - columns)
    if missing:
        raise ValueError(f"Input CSV {csv_path} is missing required column(s): {', '.join(missing)}.")


def _write_processing_layout(flat_output: Path, output_dir: Path, manifest: dict[str, Any]) -> None:
    layout = {
        "train.csv": output_dir / "train" / "train.csv",
        "valid.csv": output_dir / "validation" / "valid.csv",
        "test.csv": output_dir / "test" / "test.csv",
        "data_profile.json": output_dir / "metadata" / "data_profile.json",
        "split_metadata.json": output_dir / "metadata" / "split_metadata.json",
        "rejected_rows.csv": output_dir / "metadata" / "rejected_rows.csv",
        "problematic_molecules.csv": output_dir / "metadata" / "problematic_molecules.csv",
    }
    for source_name, destination in layout.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(flat_output / source_name, destination)
    write_json(output_dir / "metadata" / "processing_manifest.json", manifest)


def _build_processing_manifest(
    *,
    run_id: str,
    config: EndpointConfig,
    processing_mode: str,
    source_row_count: int,
    dataset_summary: dict[str, Any],
    split_seed: int | None,
    input_paths: dict[str, str],
    output_dir: Path,
    start: datetime,
    runtime_seconds: float,
    status: str,
    warnings: list[str],
    development_mode: bool,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "endpoint_id": config.endpoint_id,
        "task_type": config.task_type,
        "source_dataset": config.tdc_name,
        "processing_mode": processing_mode,
        "source_row_count": source_row_count,
        "accepted_row_count": dataset_summary["accepted_rows"],
        "rejected_row_count": dataset_summary["rejected_rows"],
        "split_counts": {
            "train": dataset_summary["train_rows"],
            "validation": dataset_summary["validation_rows"],
            "test": dataset_summary["test_rows"],
        },
        "target_statistics": _target_statistics(dataset_summary, config.task_type),
        "duplicate_canonical_smiles_count": dataset_summary["duplicate_canonical_smiles_count"],
        "cross_split_overlap_counts": dataset_summary["cross_split_overlap_counts"],
        "cross_split_overlap_examples": dataset_summary["cross_split_overlap_examples"],
        "split_seed": split_seed,
        "split_strategy": config.split_strategy,
        "input_paths": input_paths,
        "output_paths": {
            "train": str(output_dir / "train" / "train.csv"),
            "validation": str(output_dir / "validation" / "valid.csv"),
            "test": str(output_dir / "test" / "test.csv"),
            "metadata": str(output_dir / "metadata"),
        },
        "package_versions": package_versions(),
        "started_at": start.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "status": status,
        "warnings": warnings,
        "development_mode": development_mode,
        "error": error,
    }


def _target_statistics(dataset_summary: dict[str, Any], task_type: str) -> dict[str, Any]:
    if task_type == "binary_classification":
        return {
            "target_distribution": dataset_summary.get("target_distribution", {}),
            "minority_class_fraction": dataset_summary.get("minority_class_fraction"),
        }
    return {
        "target_mean": dataset_summary.get("target_mean"),
        "target_std": dataset_summary.get("target_std"),
        "target_min": dataset_summary.get("target_min"),
        "target_max": dataset_summary.get("target_max"),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
