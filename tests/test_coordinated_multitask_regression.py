import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from admet_platform.data.coordinated_multitask_regression import (
    build_coordinated_multitask_regression_splits,
    load_regression_split_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_coordinated_multitask_regression_splits.py"
SOURCE_AUDIT_CONFIG = PROJECT_ROOT / "configs" / "multitask_regression_source_audit.yaml"
TASKS = ("permeability", "solubility", "distribution")


def test_source_audit_config_records_verified_candidate_metadata() -> None:
    config = load_regression_split_config(SOURCE_AUDIT_CONFIG)

    assert config.random_seed == 42
    assert config.pytdc_version == "0.3.9"
    assert set(config.tasks) == {
        "caco2_wang",
        "lipophilicity_astrazeneca",
        "solubility_aqsoldb",
        "ppbr_az",
        "vdss_lombardo",
    }
    assert config.tasks["caco2_wang"].tdc_name == "Caco2_Wang"
    assert config.tasks["vdss_lombardo"].recommended_transform == "log10"


def test_regression_outputs_are_deterministic_continuous_and_leakage_safe(
    tmp_path: Path,
) -> None:
    config_path, source = _fixture(tmp_path)
    config = load_regression_split_config(config_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = build_coordinated_multitask_regression_splits(
        config, source_root=source, output_root=first
    )
    second_result = build_coordinated_multitask_regression_splits(
        config, source_root=source, output_root=second
    )

    for task in TASKS:
        for name in ("train.csv", "valid.csv", "test.csv"):
            assert (first / task / name).read_bytes() == (second / task / name).read_bytes()
    for name in (
        "invalid_source_records.csv",
        "quarantined_continuous_duplicates.csv",
        "deduplication_provenance.csv",
        "global_scaffold_assignments.csv",
        "coordinated_regression_split_manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    all_rows = pd.concat(
        [_endpoint_rows(first, task).assign(task_name=task) for task in TASKS],
        ignore_index=True,
    )
    assert all_rows["target"].dtype.kind == "f"
    assert all_rows["target"].equals(all_rows["target_original"])
    assert all_rows.groupby("canonical_smiles")["split"].nunique().max() == 1

    assignments = pd.read_csv(first / "global_scaffold_assignments.csv")
    assert assignments.groupby("scaffold_key")["split"].nunique().max() == 1
    assert "ACYCLIC::CS" in set(assignments["scaffold_key"])

    permeability_shared = _endpoint_rows(first, "permeability").query(
        "canonical_smiles == 'CS'"
    )
    solubility_shared = _endpoint_rows(first, "solubility").query(
        "canonical_smiles == 'CS'"
    )
    assert permeability_shared.iloc[0]["split"] == solubility_shared.iloc[0]["split"]

    aromatic_a = _endpoint_rows(first, "permeability").query(
        "canonical_smiles == 'Cc1ccccc1'"
    )
    aromatic_b = _endpoint_rows(first, "distribution").query(
        "canonical_smiles == 'Oc1ccccc1'"
    )
    assert aromatic_a.iloc[0]["split"] == aromatic_b.iloc[0]["split"]

    manifest = first_result.manifest
    assert manifest == second_result.manifest
    assert manifest["seed"] == 42
    assert manifest["target_transforms_fitted"] is False
    assert manifest["target_normalization_statistics_used"] == []
    assert manifest["leakage_audit"]["status"] == "passed"
    assert manifest["leakage_audit"]["exact_smiles_cross_split_groups"] == 0
    assert manifest["leakage_audit"]["murcko_scaffold_cross_split_groups"] == 0
    assert manifest["invalid_source_records"] == 1
    invalid = pd.read_csv(first / "invalid_source_records.csv")
    assert invalid.iloc[0]["reason"] == "missing_or_nonfinite_continuous_target"
    for task in TASKS:
        assert set(manifest["endpoints"][task]["splits"]) == {
            "train",
            "validation",
            "test",
        }


def test_duplicate_measurements_are_collapsed_or_quarantined_deterministically(
    tmp_path: Path,
) -> None:
    config_path, source = _fixture(tmp_path)
    config = load_regression_split_config(config_path)
    output = tmp_path / "output"

    build_coordinated_multitask_regression_splits(
        config, source_root=source, output_root=output
    )

    rows = _endpoint_rows(output, "permeability")
    assert rows["canonical_smiles"].eq("CCO").sum() == 1
    duplicate = pd.read_csv(output / "deduplication_provenance.csv").query(
        "canonical_smiles == 'CCO'"
    )
    assert duplicate.iloc[0]["kept_molecule_id"] == "permeability-0"
    assert duplicate.iloc[0]["removed_row_count"] == 1

    conflicts = pd.read_csv(output / "quarantined_continuous_duplicates.csv")
    conflict = conflicts.query(
        "endpoint_id == 'permeability' and canonical_smiles == 'CCl'"
    ).iloc[0]
    assert conflict["targets"] == "1.25|9.75"
    assert conflict["reason"] == "distinct_continuous_labels"
    assert not rows["canonical_smiles"].eq("CCl").any()


def test_cli_builds_regression_track_without_fitting_target_statistics(
    tmp_path: Path,
) -> None:
    config_path, source = _fixture(tmp_path)
    output = tmp_path / "cli-output"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--source-root",
            str(source),
            "--output-root",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(
        (output / "coordinated_regression_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["target_transforms_fitted"] is False
    assert manifest["leakage_audit"]["status"] == "passed"


def test_unsplit_raw_sources_preserve_duplicate_audit_and_lock_test_statistics(
    tmp_path: Path,
) -> None:
    config_path, source = _fixture(tmp_path)
    raw_root = tmp_path / "raw-source"
    for task in TASKS:
        frames = [
            pd.read_csv(source / task / filename)
            for filename in ("train.csv", "valid.csv", "test.csv")
        ]
        frame = pd.concat(frames, ignore_index=True).drop(columns="split")
        path = raw_root / task / "raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    raw_config_path = tmp_path / "raw-regression.yaml"
    raw_config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "source_files: {train: train.csv, validation: valid.csv, test: test.csv}",
            "source_files: {raw: raw.csv}",
        ),
        encoding="utf-8",
    )
    output = tmp_path / "raw-output"

    result = build_coordinated_multitask_regression_splits(
        load_regression_split_config(raw_config_path),
        source_root=raw_root,
        output_root=output,
    )

    assert set(result.manifest["input_file_sha256"]) == {
        f"{task}/raw" for task in TASKS
    }
    permeability = result.manifest["endpoints"]["permeability"]
    assert permeability["source_row_count"] == 14
    assert permeability["exact_duplicate_groups"] == 2
    assert permeability["splits"]["test"]["target_statistics_locked"] is True
    assert set(permeability["splits"]["test"]) == {
        "row_count",
        "sha256",
        "target_statistics_locked",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    task_rows = {
        "permeability": [
            ("CCO", -5.2),
            ("CCCO", -4.8),
            ("CCCCO", -4.1),
            ("CCCCCO", -3.9),
            ("CCN", -6.0),
            ("CCCN", -5.7),
            ("CCCCN", -5.1),
            ("CCCCCN", -4.9),
            ("CS", -5.5),
            ("Cc1ccccc1", -4.4),
        ],
        "solubility": [
            ("COC", -2.1),
            ("CCOC", -2.5),
            ("CCCOC", -3.0),
            ("CCCCOC", -3.4),
            ("CNC", -1.2),
            ("CCNC", -1.5),
            ("CCCNC", -1.9),
            ("CCCCNC", -2.2),
            ("CS", -0.8),
            ("Nc1ccccc1", -2.8),
        ],
        "distribution": [
            ("CC(=O)O", 0.2),
            ("CCC(=O)O", 0.4),
            ("CCCC(=O)O", 0.7),
            ("CCCCC(=O)O", 1.1),
            ("CC(=O)N", 0.3),
            ("CCC(=O)N", 0.6),
            ("CCCC(=O)N", 0.9),
            ("CCCCC(=O)N", 1.4),
            ("CBr", 0.1),
            ("Oc1ccccc1", 0.8),
        ],
    }
    for task, rows in task_rows.items():
        split_rows: dict[str, list[tuple[str, str, float]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for index, (smiles, target) in enumerate(rows):
            split = ("train", "validation", "test")[index % 3]
            split_rows[split].append((f"{task}-{index}", smiles, target))
        if task == "permeability":
            split_rows["validation"].append(("permeability-identical", "CCO", -5.2))
            split_rows["validation"].append(("permeability-missing", "CF", ""))
            split_rows["train"].append(("permeability-conflict-a", "CCl", 1.25))
            split_rows["test"].append(("permeability-conflict-b", "CCl", 9.75))
        for split, values in split_rows.items():
            filename = "valid.csv" if split == "validation" else f"{split}.csv"
            _write_split(source / task / filename, split, values)

    config_path = tmp_path / "regression.yaml"
    config_path.write_text(
        """schema_version: "1.0.0"
run_name: synthetic-regression
split_track: coordinated_multitask_regression
pytdc_version: "0.3.9"
random_seed: 42
source_root: source
output_root: coordinated
split_fractions: {train: 0.8, validation: 0.1, test: 0.1}
source_files: {train: train.csv, validation: valid.csv, test: test.csv}
duplicate_policy:
  identical_continuous_labels: keep_first_deterministically
  distinct_continuous_labels: quarantine_entire_endpoint_structure_group
tasks:
  permeability:
    endpoint_id: permeability
    tdc_name: Synthetic_Permeability
    task_group: ADME
    task_type: regression
    units: log cm/s
    target_definition: Synthetic permeability
    recommended_transform: identity
  solubility:
    endpoint_id: solubility
    tdc_name: Synthetic_Solubility
    task_group: ADME
    task_type: regression
    units: log mol/L
    target_definition: Synthetic solubility
    recommended_transform: identity
  distribution:
    endpoint_id: distribution
    tdc_name: Synthetic_Distribution
    task_group: ADME
    task_type: regression
    units: L/kg
    target_definition: Synthetic distribution
    recommended_transform: log10
""",
        encoding="utf-8",
    )
    return config_path, source


def _write_split(
    path: Path, split: str, rows: list[tuple[str, str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "canonical_smiles": smiles,
                "target": target,
                "split": split,
            }
            for molecule_id, smiles, target in rows
        ]
    ).to_csv(path, index=False)


def _endpoint_rows(output: Path, task: str) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(output / task / name)
            for name in ("train.csv", "valid.csv", "test.csv")
        ],
        ignore_index=True,
    )
