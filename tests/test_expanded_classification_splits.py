from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from admet_platform.data.expanded_classification_splits import (
    EXPECTED_ENDPOINTS,
    build_expanded_classification_splits,
)
from admet_platform.data.multitask import load_multitask_config
from admet_platform.data.scaffolds import safe_murcko_scaffold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPANDED_CONFIG = PROJECT_ROOT / "configs" / "multitask_classification_expanded.yaml"
ORIGINAL_CONFIG = PROJECT_ROOT / "configs" / "multitask_classification.yaml"


def test_expanded_config_loads_exactly_ten_endpoints_and_preserves_original() -> None:
    original = load_multitask_config(ORIGINAL_CONFIG)
    expanded = load_multitask_config(EXPANDED_CONFIG)

    assert {task: endpoint.tdc_name for task, endpoint in expanded.tasks.items()} == (
        EXPECTED_ENDPOINTS
    )
    assert set(original.tasks) == {"bbb_martins", "herg_karim", "ames"}
    assert original.prepared_root == (
        PROJECT_ROOT / "outputs" / "local" / "multitask" / "coordinated"
    )


def test_expanded_split_is_deterministic_coordinated_and_fully_reported(
    tmp_path: Path,
) -> None:
    sources = _fixture_sources(tmp_path)
    config = load_multitask_config(EXPANDED_CONFIG)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = build_expanded_classification_splits(
        config, sources, first, seed=42
    )
    second_result = build_expanded_classification_splits(
        config, sources, second, seed=42
    )

    for task in EXPECTED_ENDPOINTS:
        for filename in ("train.csv", "valid.csv", "test.csv"):
            assert (first / task / filename).read_bytes() == (
                second / task / filename
            ).read_bytes()
    assert first_result.manifest == second_result.manifest

    pgp_dedup = pd.read_csv(first / "deduplication_provenance.csv").query(
        "endpoint_id == 'pgp_broccatelli'"
    )
    assert int(pgp_dedup["removed_row_count"].sum()) == 6
    assert first_result.manifest["endpoints"]["pgp_broccatelli"][
        "duplicate_rows_removed"
    ] == 6

    all_rows = _all_output_rows(first)
    assert all_rows.groupby("canonical_smiles")["split"].nunique().max() == 1
    all_rows["scaffold_key"] = all_rows["canonical_smiles"].map(_scaffold_key)
    assert all_rows.groupby("scaffold_key")["split"].nunique().max() == 1

    shared = all_rows.query("canonical_smiles == 'CCO'")
    assert shared["task_name"].nunique() == 10
    assert shared["split"].nunique() == 1
    assert set(shared["target"]) == {0, 1}

    aromatic = all_rows[
        all_rows["canonical_smiles"].isin({"Cc1ccccc1", "Oc1ccccc1"})
    ]
    assert aromatic["split"].nunique() == 1
    assert aromatic["task_name"].nunique() == 2

    manifest = first_result.manifest
    assert manifest["exact_molecule_overlap_count_between_splits"] == 0
    assert manifest["scaffold_overlap_count_between_splits"] == 0
    assert manifest["cross_endpoint_shared_molecule_count"] > 0
    assert manifest["cross_endpoint_shared_scaffold_count"] > 0
    assert len(manifest["split_manifest_id"]) == 64
    assert manifest["endpoints"]["bbb_martins"]["old_split_assignment_removed"] is True
    assert manifest["endpoints"]["herg_karim"]["old_split_assignment_removed"] is True
    assert manifest["endpoints"]["ames"]["old_split_assignment_removed"] is False

    for endpoint in manifest["endpoints"].values():
        for summary in endpoint["splits"].values():
            assert summary["row_count"] > 0
            assert set(summary["class_counts"]) == {"0", "1"}
            assert all(summary["class_counts"][label] > 0 for label in ("0", "1"))
            assert sum(summary["class_fractions"].values()) == pytest.approx(1.0)
            assert summary["unique_canonical_molecules"] == summary["row_count"]
            assert len(summary["output_csv_sha256"]) == 64

    assert first_result.audit.summary["leakage_safe_for_training"] is True
    assert first_result.audit.summary["counts"]["blocking_violations"] == 0
    assert (first / "endpoint_split_summary.csv").is_file()
    assert (first / "coordinated_split_report.md").is_file()


def test_within_endpoint_conflict_is_quarantined_but_cross_endpoint_labels_are_allowed(
    tmp_path: Path,
) -> None:
    sources = _fixture_sources(tmp_path)
    hia_path = tmp_path / "sources" / "hia_hou.csv"
    hia = pd.read_csv(hia_path)
    hia.loc[len(hia)] = {
        "molecule_id": "hia-conflict",
        "smiles": "C",
        "target": 1,
    }
    hia.to_csv(hia_path, index=False)

    output = tmp_path / "conflict-output"
    result = build_expanded_classification_splits(
        load_multitask_config(EXPANDED_CONFIG),
        sources,
        output,
        seed=42,
    )

    conflicts = pd.read_csv(output / "quarantined_conflicts.csv")
    conflict = conflicts.query(
        "endpoint_id == 'hia_hou' and canonical_smiles == 'C'"
    )
    assert len(conflict) == 1
    assert result.manifest["endpoints"]["hia_hou"]["conflicting_label_groups"] == 1
    hia_output = _task_output_rows(output, "hia_hou")
    assert "C" not in set(hia_output["canonical_smiles"])

    cross_endpoint = _all_output_rows(output).query("canonical_smiles == 'CCO'")
    assert set(cross_endpoint["target"]) == {0, 1}
    assert cross_endpoint["task_name"].nunique() == 10


def test_invalid_source_molecule_fails_before_outputs_are_written(tmp_path: Path) -> None:
    sources = _fixture_sources(tmp_path)
    hia_path = tmp_path / "sources" / "hia_hou.csv"
    hia = pd.read_csv(hia_path)
    hia.loc[len(hia)] = {
        "molecule_id": "invalid",
        "smiles": "not-a-smiles",
        "target": 0,
    }
    hia.to_csv(hia_path, index=False)
    output = tmp_path / "invalid-output"

    with pytest.raises(ValueError, match="invalid source molecule"):
        build_expanded_classification_splits(
            load_multitask_config(EXPANDED_CONFIG),
            sources,
            output,
            seed=42,
        )
    assert not output.exists()


def _fixture_sources(tmp_path: Path) -> Path:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    entries = []
    for task_index, task_name in enumerate(EXPECTED_ENDPOINTS):
        rows = _base_rows(task_name, task_index)
        if task_name == "pgp_broccatelli":
            rows.extend(
                {
                    **rows[index],
                    "molecule_id": f"pgp-duplicate-{index}",
                }
                for index in range(6)
            )
        if task_name in {"bbb_martins", "herg_karim"}:
            paths = []
            for split_index, split in enumerate(("train", "validation", "test")):
                split_rows = rows[split_index::3]
                frame = pd.DataFrame(split_rows)
                frame["canonical_smiles"] = frame["smiles"]
                frame["split"] = split
                filename = f"{task_name}-{split}.csv"
                frame.to_csv(source_root / filename, index=False)
                paths.append(f"sources/{filename}")
            entries.append((task_name, "concatenate_prepared_splits", paths))
        else:
            filename = f"{task_name}.csv"
            pd.DataFrame(rows).to_csv(source_root / filename, index=False)
            entries.append(
                (task_name, "normalized_unsplit_csv", [f"sources/{filename}"])
            )

    lines = ['schema_version: "1.0.0"', "endpoints:"]
    for task_name, method, paths in entries:
        lines.extend(
            [
                f"  {task_name}:",
                f"    method: {method}",
                "    paths:",
                *(f"      - {path}" for path in paths),
            ]
        )
    source_config = tmp_path / "sources.yaml"
    source_config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source_config


def _base_rows(task_name: str, task_index: int) -> list[dict[str, object]]:
    class_zero = ["C", "CC", "CCC", "CCO", "CCCO", "CCCCO"]
    class_one = ["N", "CN", "CCN", "CCCN", "CCCCN", "CCCCCN"]
    if task_name == "pgp_broccatelli":
        class_zero.remove("CCO")
        class_one.append("CCO")
    rows = [
        {
            "molecule_id": f"{task_name}-zero-{index}",
            "smiles": smiles,
            "target": 0,
        }
        for index, smiles in enumerate(class_zero)
    ]
    rows.extend(
        {
            "molecule_id": f"{task_name}-one-{index}",
            "smiles": smiles,
            "target": 1,
        }
        for index, smiles in enumerate(class_one)
    )
    if task_index == 0:
        rows.append(
            {"molecule_id": "hia-aromatic", "smiles": "Cc1ccccc1", "target": 1}
        )
    if task_index == 1:
        rows.append(
            {"molecule_id": "pgp-aromatic", "smiles": "Oc1ccccc1", "target": 0}
        )
    return rows


def _task_output_rows(output: Path, task: str) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(output / task / filename)
            for filename in ("train.csv", "valid.csv", "test.csv")
        ],
        ignore_index=True,
    )


def _all_output_rows(output: Path) -> pd.DataFrame:
    frames = []
    for task in EXPECTED_ENDPOINTS:
        frame = _task_output_rows(output, task)
        frame["task_name"] = task
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _scaffold_key(canonical: str) -> str:
    scaffold = safe_murcko_scaffold(Chem.MolFromSmiles(canonical)).scaffold
    return scaffold or f"ACYCLIC::{canonical}"
