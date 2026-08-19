from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from rdkit import Chem

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.gmc_mpnn.data import DevelopmentLeakageReport
from scripts import audit_gmc_mpnn_disconnected_fragments as audit


def test_connected_molecule_detection() -> None:
    result = audit.analyze_molecule_fragments("CCO")

    assert result.fragment_count == 1
    assert result.candidate_parent.canonical_smiles == "CCO"
    assert result.candidate_parent.heavy_atom_count == 3
    assert result.removed_fragments == ()
    assert result.total_heavy_atom_count == 3
    assert result.total_formal_charge == 0


def test_sodium_salt_detection_and_removed_fragment_recording() -> None:
    result = audit.analyze_molecule_fragments("CC(=O)[O-].[Na+]")

    assert result.fragment_count == 2
    assert result.candidate_parent.canonical_smiles == "CC(=O)[O-]"
    assert result.candidate_parent.heavy_atom_count == 4
    assert result.candidate_parent.formal_charge == -1
    assert result.removed_fragments == (audit.FragmentRecord("[Na+]", 1, 1),)
    assert result.removed_fragment_categories == ("simple_inorganic_or_monatomic_counterion",)
    assert result.total_heavy_atom_count == 5
    assert result.total_formal_charge == 0


def test_chloride_counterion_is_detected_and_categorized() -> None:
    result = audit.analyze_molecule_fragments("C[NH3+].[Cl-]")

    assert result.fragment_count == 2
    assert result.candidate_parent.canonical_smiles == "C[NH3+]"
    assert result.removed_fragments[0].canonical_smiles == "[Cl-]"
    assert result.removed_fragment_categories[0] == ("simple_inorganic_or_monatomic_counterion")


def test_multi_organic_fragment_case_records_organic_coformer() -> None:
    result = audit.analyze_molecule_fragments("CCc1ccccc1.O=C(O)C=CC(=O)O")

    assert result.fragment_count == 2
    assert result.candidate_parent.heavy_atom_count == 8
    assert result.removed_fragments[0].heavy_atom_count == 8
    assert result.removed_fragment_categories == ("organic_coformer_or_counterion",)


def test_largest_heavy_atom_parent_selection_is_deterministic() -> None:
    first = audit.analyze_molecule_fragments("[Na+].CCCO")
    second = audit.analyze_molecule_fragments("CCCO.[Na+]")

    assert first.candidate_parent == second.candidate_parent
    assert first.candidate_parent.canonical_smiles == "CCCO"
    assert first.candidate_parent.heavy_atom_count == 4


def test_equal_size_parent_tie_break_uses_canonical_smiles_order() -> None:
    result = audit.analyze_molecule_fragments("CN.CC")

    assert [fragment.canonical_smiles for fragment in result.fragments] == ["CC", "CN"]
    assert result.candidate_parent.canonical_smiles == "CC"
    assert result.removed_fragments[0].canonical_smiles == "CN"


def test_fragment_categorization_is_small_and_transparent() -> None:
    phosphate = audit.analyze_molecule_fragments("O=P(O)(O)O").candidate_parent
    fumarate = audit.analyze_molecule_fragments("O=C(O)C=CC(=O)O").candidate_parent
    sulfur_ring = audit.analyze_molecule_fragments("S1SSSSSS1").candidate_parent

    assert audit.categorize_removed_fragment(phosphate) == "small_inorganic_fragment"
    assert audit.categorize_removed_fragment(fumarate) == "organic_coformer_or_counterion"
    assert audit.categorize_removed_fragment(sulfur_ring) == "other"


def test_collision_conflict_parent_overlap_and_scaffold_overlap_analysis() -> None:
    train = _frame(
        "train",
        [
            ("t-sodium", "CCO.[Na+]", 0),
            ("t-potassium", "CCO.[K+]", 1),
            ("t-toluene", "Cc1ccccc1.[Na+]", 1),
        ],
    )
    validation = _frame(
        "validation",
        [
            ("v-chloride", "CCN.[Cl-]", 0),
            ("v-bromide", "CCN.[Br-]", 0),
            ("v-lithium", "CCO.[Li+]", 0),
            ("v-ethylbenzene", "CCc1ccccc1.[Cl-]", 1),
        ],
    )

    result = audit.analyze_development_frames(train, validation)

    train_collisions = result.statistics["splits"]["train"]["parent_collision_statistics"]
    validation_collisions = result.statistics["splits"]["validation"]["parent_collision_statistics"]
    assert train_collisions["duplicate_candidate_parent_smiles"] == ["CCO"]
    assert train_collisions["parent_collision_group_count"] == 1
    assert train_collisions["maximum_collision_group_size"] == 2
    assert train_collisions["candidate_parents_with_multiple_original_structures"] == ["CCO"]
    assert train_collisions["label_conflict_parent_smiles"] == ["CCO"]
    assert validation_collisions["duplicate_candidate_parent_smiles"] == ["CCN"]
    assert validation_collisions["parent_collision_group_count"] == 1
    assert validation_collisions["label_conflict_group_count"] == 0

    assert result.statistics["candidate_parent_train_validation_overlap_smiles"] == ["CCO"]
    overlap = result.train_validation_parent_overlap.iloc[0]
    assert overlap["has_label_conflict"]
    assert json.loads(overlap["train_identities"])[0]["molecule_id"] == "t-potassium"
    assert json.loads(overlap["validation_identities"])[0]["molecule_id"] == "v-lithium"

    scaffold_rows = result.train_validation_parent_scaffold_overlap
    assert "c1ccccc1" in scaffold_rows["candidate_parent_scaffold"].tolist()
    aromatic_overlap = scaffold_rows.loc[
        scaffold_rows["candidate_parent_scaffold"] == "c1ccccc1"
    ].iloc[0]
    assert json.loads(aromatic_overlap["train_parent_smiles"]) == ["Cc1ccccc1"]
    assert json.loads(aromatic_overlap["validation_parent_smiles"]) == ["CCc1ccccc1"]

    conflicts = result.statistics["label_conflict_groups"]
    ethanol_conflict = next(item for item in conflicts if item["candidate_parent_smiles"] == "CCO")
    assert {identity["molecule_id"] for identity in ethanol_conflict["identities"]} == {
        "t-sodium",
        "t-potassium",
        "v-lithium",
    }
    assert {identity["target"] for identity in ethanol_conflict["identities"]} == {0, 1}


def test_no_parent_or_scaffold_overlap_case() -> None:
    result = audit.analyze_development_frames(
        _frame("train", [("t", "CCO", 0)]),
        _frame("validation", [("v", "c1ccccc1", 1)]),
    )

    assert result.train_validation_parent_overlap.empty
    assert result.train_validation_parent_scaffold_overlap.empty
    assert result.statistics["candidate_parent_train_validation_overlap_count"] == 0
    assert result.statistics["candidate_parent_scaffold_overlap_count"] == 0


def test_identity_statistics_and_source_rows_are_not_modified() -> None:
    train = _frame(
        "train",
        [("one", "CCO.[Na+]", 0), ("two", "CCO.[K+]", 0)],
    )
    validation = _frame("validation", [("three", "CCN", 1)])
    original_train = train.copy(deep=True)
    original_validation = validation.copy(deep=True)

    result = audit.analyze_development_frames(train, validation)

    assert_frame_equal(train, original_train)
    assert_frame_equal(validation, original_validation)
    train_identity = result.statistics["splits"]["train"]
    assert train_identity["source_row_count"] == 2
    assert train_identity["unique_original_canonical_smiles_count"] == 2
    assert train_identity["unique_candidate_parent_smiles_count"] == 1
    assert train_identity["structures_collapsed_by_parent_selection"] == 1
    assert result.statistics["combined_identity_statistics"]["source_row_count"] == 3
    assert result.statistics["total_structures_affected_by_candidate_parent_selection"] == 2


def test_empty_output_csvs_are_created_with_headers(tmp_path: Path) -> None:
    analysis = audit.analyze_development_frames(
        _frame("train", [("t", "CCO", 0)]),
        _frame("validation", [("v", "c1ccccc1", 1)]),
    )
    output = tmp_path / "audit"

    audit._write_audit_outputs(output, {"synthetic": True}, analysis, overwrite=False)

    assert {path.name for path in output.iterdir()} == set(audit.OUTPUT_FILENAMES)
    assert pd.read_csv(output / "disconnected_molecules.csv").columns.tolist() == list(
        audit.DISCONNECTED_COLUMNS
    )
    assert pd.read_csv(output / "parent_collision_groups.csv").columns.tolist() == list(
        audit.COLLISION_COLUMNS
    )
    assert pd.read_csv(output / "train_validation_parent_overlap.csv").columns.tolist() == list(
        audit.PARENT_OVERLAP_COLUMNS
    )
    assert pd.read_csv(
        output / "train_validation_parent_scaffold_overlap.csv"
    ).columns.tolist() == list(audit.SCAFFOLD_OVERLAP_COLUMNS)


def test_run_writes_provenance_without_modifying_sources_or_accessing_test(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    train = _frame("train", [("t", "CCO.[Na+]", 0)])
    validation = _frame("validation", [("v", "CCN", 1)])
    original_train = train.copy(deep=True)
    original_validation = validation.copy(deep=True)
    development = SimpleNamespace(
        train=train,
        validation=validation,
        leakage=DevelopmentLeakageReport((), ()),
    )
    output = tmp_path / "output"

    summary = audit.run_disconnected_fragment_audit(
        tmp_path / "config.yaml",
        output_dir=output,
        config_loader=lambda _: config,
        development_loader=lambda _: development,
    )

    assert summary["dataset"] == "BBB_Martins"
    assert summary["dataset_version"] == "synthetic-audit-fixture"
    assert summary["loaded_splits"] == ["train", "validation"]
    assert summary["train_row_count"] == 1
    assert summary["validation_row_count"] == 1
    assert summary["train_disconnected_count"] == 1
    assert summary["validation_disconnected_count"] == 0
    assert summary["test_artifact_accessed"] is False
    assert summary["policy_approved"] is False
    assert_frame_equal(train, original_train)
    assert_frame_equal(validation, original_validation)


@pytest.mark.parametrize("operation", ("resolve", "stat", "exists", "read", "pandas"))
def test_test_artifact_access_is_blocked(tmp_path: Path, operation: str) -> None:
    config = _config(tmp_path)
    locked = config.prepared_root / config.split_files["test"]

    with audit._deny_test_artifact_access(config) as guard:
        with pytest.raises(audit.LockedTestAccessError):
            if operation == "resolve":
                locked.resolve()
            elif operation == "stat":
                locked.stat()
            elif operation == "exists":
                locked.exists()
            elif operation == "read":
                locked.read_bytes()
            else:
                pd.read_csv(locked)
        assert guard.test_artifact_accessed is True


def test_run_installs_test_access_guard_around_development_loader(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def forbidden_loader(_: ChempropExperimentConfig):
        locked = config.prepared_root / config.split_files["test"]
        locked.read_text(encoding="utf-8")
        raise AssertionError("unreachable")

    with pytest.raises(audit.LockedTestAccessError):
        audit.run_disconnected_fragment_audit(
            tmp_path / "config.yaml",
            output_dir=tmp_path / "output",
            config_loader=lambda _: config,
            development_loader=forbidden_loader,
        )


def _frame(split: str, rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "molecule_id": molecule_id,
                "canonical_smiles": _canonical(smiles),
                "target": target,
                "split": split,
            }
            for molecule_id, smiles, target in rows
        ]
    )


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _config(root: Path) -> ChempropExperimentConfig:
    prepared = root / "prepared"
    prepared.mkdir()
    (prepared / "locked.csv").write_text("must-not-be-read", encoding="utf-8")
    return ChempropExperimentConfig(
        source_path=root / "config.yaml",
        raw={},
        endpoint="bbb_martins",
        endpoint_id="bbb_martins",
        dataset="BBB_Martins",
        dataset_version="synthetic-audit-fixture",
        task_type="binary_classification",
        primary_metric="auroc",
        prepared_root=prepared,
        split_manifest=root / "unused-manifest.json",
        split_files={"train": "train.csv", "validation": "valid.csv", "test": "locked.csv"},
        split_hash_keys={"train": "train", "validation": "validation", "test": "test"},
        tasks={},
        model={},
        training={},
    )
