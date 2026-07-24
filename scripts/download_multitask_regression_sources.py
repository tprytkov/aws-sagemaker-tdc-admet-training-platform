"""Acquire the frozen five-endpoint regression source set with PyTDC 0.3.9."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.config import EndpointConfig  # noqa: E402
from admet_platform.data.coordinated_multitask_regression import (  # noqa: E402
    load_regression_split_config,
)
from admet_platform.data.tdc_loader import (  # noqa: E402
    load_tdc_data,
    normalize_tdc_raw_dataframe,
)


VERIFIED_PYTDC_VERSION = "0.3.9"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download unsplit regression source rows without deduplicating them."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    version = importlib.metadata.version("PyTDC")
    if version != VERIFIED_PYTDC_VERSION:
        raise RuntimeError(
            f"Expected PyTDC {VERIFIED_PYTDC_VERSION}, found {version}."
        )
    config = load_regression_split_config(args.config)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    endpoints = {}
    for task, endpoint in config.tasks.items():
        acquisition_config = EndpointConfig(
            endpoint_id=endpoint.endpoint_id,
            tdc_name=endpoint.tdc_name,
            task_group="ADME",
            task_type="regression",
            target_column="target",
            smiles_column="smiles",
            split_strategy="coordinated_multitask_regression",
            metric_names=["mae", "rmse", "r2"],
            base_model="seyonec/ChemBERTa-zinc-base-v1",
            problem_description=endpoint.target_definition,
            limitations=["Public TDC data; endpoint provenance must be retained."],
            output_prediction_column=f"{endpoint.endpoint_id}_prediction",
            output_score_column=f"{endpoint.endpoint_id}_score",
        )
        raw = load_tdc_data(acquisition_config)
        normalized = normalize_tdc_raw_dataframe(raw, acquisition_config)
        path = output / endpoint.endpoint_id / "raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(path, index=False, lineterminator="\n")
        endpoints[task] = {
            "endpoint_id": endpoint.endpoint_id,
            "tdc_name": endpoint.tdc_name,
            "source_row_count": int(len(normalized)),
            "raw_file": str(path),
            "raw_file_sha256": _sha256(path),
            "duplicates_preserved": True,
        }
    manifest = {
        "schema_version": "1.0.0",
        "pytdc_version": version,
        "source_rows_unmodified": True,
        "deduplication_performed": False,
        "endpoints": endpoints,
    }
    manifest_path = output / "source_acquisition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
