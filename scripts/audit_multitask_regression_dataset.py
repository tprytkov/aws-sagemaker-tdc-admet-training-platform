"""Audit coordinated regression data without opening locked test rows downstream."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.coordinated_multitask_regression import (  # noqa: E402
    build_coordinated_multitask_regression_splits,
    load_regression_split_config,
)
from admet_platform.data.multitask_regression import (  # noqa: E402
    fit_training_transforms,
    load_multitask_regression_config,
    load_regression_training_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify deterministic coordinated outputs and train-only transforms."
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--coordinated-root", required=True)
    args = parser.parse_args()

    source_config = load_regression_split_config(args.source_config)
    coordinated_root = Path(args.coordinated_root).resolve()
    manifest_path = coordinated_root / "coordinated_regression_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="regression-determinism-") as temporary:
        reproduction = build_coordinated_multitask_regression_splits(
            source_config,
            source_root=args.source_root,
            output_root=temporary,
            seed=42,
        )
        deterministic = (
            reproduction.manifest["output_file_sha256"]
            == manifest["output_file_sha256"]
            and _supporting_hashes(Path(temporary))
            == _supporting_hashes(coordinated_root)
        )
    if not deterministic:
        raise RuntimeError("Seed-42 coordinated regression reproduction differs.")

    training_config = load_multitask_regression_config(args.training_config)
    datasets = load_regression_training_datasets(
        training_config, prepared_root=coordinated_root
    )
    transforms = fit_training_transforms(datasets)
    inverse_checks = {}
    for task, splits in datasets.items():
        endpoint_checks = {}
        for split, frame in (("train", splits.train), ("validation", splits.validation)):
            original = frame["target_original"].to_numpy(dtype=np.float64)
            restored = transforms[task].inverse_values(
                transforms[task].transform_values(original)
            )
            endpoint_checks[split] = bool(
                np.allclose(restored, original, rtol=1e-10, atol=1e-10)
            )
        if not all(endpoint_checks.values()):
            raise RuntimeError(f"Inverse transform verification failed for {task}.")
        inverse_checks[task] = endpoint_checks

    transform_payload = {
        "schema_version": "1.0.0",
        "fit_split": "train",
        "validation_statistics_used": False,
        "test_statistics_used": False,
        "endpoints": {
            task: {
                **transforms[task].to_metadata(),
                "scientific_transform": training_config.tasks[task].target_transform,
                "target_definition": training_config.tasks[task].target_definition,
                "provenance_note": training_config.tasks[task].provenance_note,
                "inverse_transform_verified": inverse_checks[task],
            }
            for task in training_config.tasks
        },
    }
    _write_json(coordinated_root / "target_transforms.json", transform_payload)

    endpoint_rows = []
    for task, endpoint in manifest["endpoints"].items():
        transform = transforms[task]
        endpoint_rows.append(
            {
                "endpoint": task,
                "tdc_name": endpoint["tdc_name"],
                "source_rows": endpoint["source_row_count"],
                "valid_canonicalized_records": endpoint[
                    "valid_canonicalized_records"
                ],
                "unique_valid_canonical_structures": endpoint[
                    "unique_valid_canonical_structures"
                ] if "unique_valid_canonical_structures" in endpoint else (
                    endpoint["retained_rows"]
                    + endpoint["conflicting_label_duplicate_groups_quarantined"]
                ),
                "invalid_records": endpoint["invalid_records"],
                "exact_duplicate_groups": endpoint["exact_duplicate_groups"],
                "identical_label_duplicates_collapsed": endpoint[
                    "identical_label_duplicate_groups_collapsed"
                ],
                "conflicting_duplicate_groups_quarantined": endpoint[
                    "conflicting_label_duplicate_groups_quarantined"
                ],
                "retained_rows": endpoint["retained_rows"],
                "train": endpoint["splits"]["train"]["row_count"],
                "validation": endpoint["splits"]["validation"]["row_count"],
                "test": endpoint["splits"]["test"]["row_count"],
                "scientific_transform": transform.transform,
                "transformed_train_mean": transform.transformed_train_mean,
                "transformed_train_std": transform.transformed_train_std,
                "train_sha256": endpoint["splits"]["train"]["sha256"],
                "validation_sha256": endpoint["splits"]["validation"]["sha256"],
                "test_sha256": endpoint["splits"]["test"]["sha256"],
            }
        )

    report = {
        "schema_version": "1.0.0",
        "pytdc_version": manifest["pytdc_version"],
        "rdkit_version": manifest["rdkit_version"],
        "seed": manifest["seed"],
        "test_split_status": "locked_after_coordinated_generation",
        "downstream_loaded_splits": ["train", "validation"],
        "validation_statistics_used_for_transform_fit": False,
        "test_statistics_used_for_transform_fit": False,
        "inverse_transform_checks": inverse_checks,
        "deterministic_seed_42_reproduction": deterministic,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "endpoints": endpoint_rows,
        "global": {
            **manifest["global_summary"],
            "exact_smiles_leakage_count": manifest["leakage_audit"][
                "exact_smiles_cross_split_groups"
            ],
            "scaffold_leakage_count": manifest["leakage_audit"][
                "murcko_scaffold_cross_split_groups"
            ],
            "conflicting_duplicate_structure_cross_split_count": 0,
        },
    }
    _write_json(coordinated_root / "dataset_audit.json", report)
    (coordinated_root / "dataset_audit.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    _write_json(
        coordinated_root / "LOCKED_TEST_SPLITS.json",
        {
            "status": "locked",
            "allowed_use": "future final evaluation only",
            "checkpoint_selection_use_prohibited": True,
            "test_file_sha256": {
                row["endpoint"]: row["test_sha256"] for row in endpoint_rows
            },
        },
    )
    print(json.dumps(report, indent=2))


def _supporting_hashes(root: Path) -> dict[str, str]:
    names = (
        "invalid_source_records.csv",
        "quarantined_continuous_duplicates.csv",
        "deduplication_provenance.csv",
        "global_scaffold_assignments.csv",
    )
    return {name: _sha256(root / name) for name in names}


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Coordinated Multi-Task Regression Dataset Audit",
        "",
        f"- PyTDC: {report['pytdc_version']}",
        f"- RDKit: {report['rdkit_version']}",
        f"- Seed: {report['seed']}",
        f"- Deterministic reproduction: {report['deterministic_seed_42_reproduction']}",
        f"- Dataset manifest SHA-256: `{report['dataset_manifest_sha256']}`",
        f"- Test status: {report['test_split_status']}",
        "",
        "| Endpoint | Source | Retained | Train | Validation | Test | Transform | "
        "Train mean | Train std | Conflicts quarantined |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in report["endpoints"]:
        lines.append(
            f"| {row['endpoint']} | {row['source_rows']} | {row['retained_rows']} | "
            f"{row['train']} | {row['validation']} | {row['test']} | "
            f"{row['scientific_transform']} | {row['transformed_train_mean']:.10g} | "
            f"{row['transformed_train_std']:.10g} | "
            f"{row['conflicting_duplicate_groups_quarantined']} |"
        )
    global_values = report["global"]
    lines.extend(
        [
            "",
            "## Global checks",
            "",
            f"- Total retained endpoint records: "
            f"{global_values['total_retained_endpoint_records']}",
            f"- Unique canonical structures: "
            f"{global_values['unique_canonical_structures']}",
            f"- Unique scaffold groups: {global_values['unique_scaffold_groups']}",
            f"- Exact-SMILES leakage count: "
            f"{global_values['exact_smiles_leakage_count']}",
            f"- Scaffold leakage count: {global_values['scaffold_leakage_count']}",
            "- Within-split cross-endpoint overlap is allowed and reported in JSON.",
            "- Transform fitting opened train only; inverse checks opened train and validation.",
            "- Locked test target distributions and predictions were not inspected downstream.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
