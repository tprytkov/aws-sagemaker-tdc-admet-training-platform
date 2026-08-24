#!/usr/bin/env python
"""Run frozen MolOptima ChemBERTa classification inference for an ordered CSV batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.models.chemberta_production_inference import (  # noqa: E402
    PRODUCTION_ENDPOINT_ORDER,
    predict_chemberta_batch,
)


ENDPOINT_FIELDS = (
    "raw_logit",
    "raw_probability",
    "calibrated_probability",
    "predicted_class",
    "calibration_method",
    "evidence_status",
    "positive_class",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-path", required=True, type=Path)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_csv}")
    rows = _read_inputs(args.input_csv)
    predictions = predict_chemberta_batch(
        rows,
        bundle_path=args.bundle_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(_flatten(row) for row in predictions)


def _read_inputs(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"molecule_id", "source_smiles"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain molecule_id and source_smiles columns.")
        return [
            {"molecule_id": row.get("molecule_id"), "source_smiles": row.get("source_smiles")}
            for row in reader
        ]


def _fieldnames() -> list[str]:
    fields = [
        "molecule_id",
        "source_smiles",
        "canonical_smiles",
        "status",
        "error_code",
        "error_message",
        "model_family",
        "bundle_version",
        "checkpoint_sha256",
        "endpoint_order_json",
    ]
    for endpoint in PRODUCTION_ENDPOINT_ORDER:
        fields.extend(f"{endpoint}_{field}" for field in ENDPOINT_FIELDS)
    return fields


def _flatten(row: dict[str, object]) -> dict[str, Any]:
    flattened: dict[str, Any] = {
        field: row.get(field)
        for field in (
            "molecule_id",
            "source_smiles",
            "canonical_smiles",
            "status",
            "error_code",
            "error_message",
            "model_family",
            "bundle_version",
            "checkpoint_sha256",
        )
    }
    flattened["endpoint_order_json"] = json.dumps(
        row.get("endpoint_order"), separators=(",", ":")
    )
    endpoints = row.get("endpoints", {})
    if not isinstance(endpoints, dict):
        endpoints = {}
    for endpoint in PRODUCTION_ENDPOINT_ORDER:
        item = endpoints.get(endpoint, {})
        if not isinstance(item, dict):
            item = {}
        for field in ENDPOINT_FIELDS:
            flattened[f"{endpoint}_{field}"] = item.get(field)
    return flattened


if __name__ == "__main__":
    main()
