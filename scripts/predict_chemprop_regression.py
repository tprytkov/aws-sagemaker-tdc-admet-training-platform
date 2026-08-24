#!/usr/bin/env python
"""Run frozen five-seed Chemprop regression inference for an ordered CSV batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.production_inference import predict_regression_batch  # noqa: E402
from admet_platform.chemprop.production_manifest import (  # noqa: E402
    ENDPOINT_ORDER,
    PRODUCTION_SEEDS,
)


REPRESENTATION_SUFFIX = {
    "caco2_wang": "log10_papp_cm_per_s",
    "lipophilicity_astrazeneca": "log_ratio",
    "solubility_aqsoldb": "log_mol_per_l",
    "ppbr_az": "percent_bound",
    "vdss_lombardo": "l_per_kg",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_csv}")
    rows = _read_inputs(args.input_csv)
    predictions = predict_regression_batch(
        rows,
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        verify_runtime=True,
        num_workers=args.num_workers,
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
        "manifest_schema_version",
        "manifest_sha256",
        "release_status",
        "endpoint_order_json",
        "applicability_domain_status",
    ]
    for endpoint in ENDPOINT_ORDER:
        suffix = REPRESENTATION_SUFFIX[endpoint]
        fields.extend(f"{endpoint}_seed{seed}_{suffix}" for seed in PRODUCTION_SEEDS)
        if endpoint == "caco2_wang":
            fields.extend(
                [
                    "caco2_wang_ensemble_mean_log10_papp_cm_per_s",
                    "caco2_wang_seed_standard_deviation_log10_papp_cm_per_s",
                    "caco2_wang_physical_papp_cm_per_s_from_ensemble_log10",
                ]
            )
        else:
            fields.extend(
                [
                    f"{endpoint}_ensemble_mean_{suffix}",
                    f"{endpoint}_seed_standard_deviation_{suffix}",
                ]
            )
        fields.extend(
            [
                f"{endpoint}_seed_standard_deviation_ddof",
                f"{endpoint}_uncertainty_interpretation",
                f"{endpoint}_unit",
                f"{endpoint}_representation",
            ]
        )
    return fields


def _flatten(row: dict[str, object]) -> dict[str, Any]:
    flattened: dict[str, Any] = {
        "molecule_id": row.get("molecule_id"),
        "source_smiles": row.get("source_smiles"),
        "canonical_smiles": row.get("canonical_smiles"),
        "status": row.get("status"),
        "error_code": row.get("error_code"),
        "error_message": row.get("error_message"),
        "model_family": row.get("model_family"),
        "manifest_schema_version": row.get("manifest_schema_version"),
        "manifest_sha256": row.get("manifest_sha256"),
        "release_status": row.get("release_status"),
        "endpoint_order_json": json.dumps(row.get("endpoint_order"), separators=(",", ":")),
        "applicability_domain_status": row.get("applicability_domain", {}).get("status"),
    }
    endpoints = row.get("endpoints", {})
    for endpoint in ENDPOINT_ORDER:
        item = endpoints.get(endpoint, {})
        suffix = REPRESENTATION_SUFFIX[endpoint]
        per_seed = item.get("per_seed", {})
        for seed in PRODUCTION_SEEDS:
            flattened[f"{endpoint}_seed{seed}_{suffix}"] = per_seed.get(str(seed))
        if endpoint == "caco2_wang":
            flattened["caco2_wang_ensemble_mean_log10_papp_cm_per_s"] = item.get(
                "ensemble_mean_log10_papp_cm_per_s"
            )
            flattened["caco2_wang_seed_standard_deviation_log10_papp_cm_per_s"] = item.get(
                "seed_standard_deviation_log10_papp_cm_per_s"
            )
            flattened["caco2_wang_physical_papp_cm_per_s_from_ensemble_log10"] = item.get(
                "physical_papp_cm_per_s_from_ensemble_log10"
            )
        else:
            flattened[f"{endpoint}_ensemble_mean_{suffix}"] = item.get("ensemble_mean")
            flattened[f"{endpoint}_seed_standard_deviation_{suffix}"] = item.get(
                "seed_standard_deviation"
            )
        for key in (
            "seed_standard_deviation_ddof",
            "uncertainty_interpretation",
            "unit",
            "representation",
        ):
            flattened[f"{endpoint}_{key}"] = item.get(key)
    return flattened


if __name__ == "__main__":
    main()
