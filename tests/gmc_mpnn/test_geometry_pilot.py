from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, rdBase

from admet_platform.chemprop.config import ChempropExperimentConfig
from admet_platform.gmc_mpnn.geometry import GeometryConfig, GeometryError, GeometryResult
from admet_platform.gmc_mpnn.ggl import GGLConfig, GGLResult
from admet_platform.gmc_mpnn.standardization import standardize_for_gmc_geometry
from scripts import pilot_gmc_mpnn_geometry as pilot


APTAZAPINE_SMILES = "CN1CCN2c3ccccc3Cn3cccc3C2C1.O=C(O)/C=C\\C(=O)O"
CEFPODOXIME_SMILES = "COCC1=C(C(=O)[O-])N2C(=O)[C@@H](NC(=O)/C(=N\\OC)c3csc(N)n3)[C@H]2SC1.[Na+]"


def test_deterministic_selection_is_order_independent_and_stable() -> None:
    frame = _training_frame(40)
    first = pilot.select_training_pilot(frame, 25)
    reordered = pilot.select_training_pilot(
        frame.sample(frac=1.0, random_state=123).reset_index(drop=True), 25
    )

    assert len(first) == len(reordered) == 25
    assert first["molecule_id"].tolist() == reordered["molecule_id"].tolist()
    assert first["canonical_smiles"].tolist() == reordered["canonical_smiles"].tolist()
    assert first["selection_hash"].tolist() == reordered["selection_hash"].tolist()
    assert first["pilot_index"].tolist() == list(range(1, 26))
    assert first["selection_hash"].is_unique
    assert all(len(value) == 64 for value in first["selection_hash"])


def test_selection_hash_uses_target_and_explicit_seed() -> None:
    baseline = pilot.stable_selection_hash("id", "CCO", 0)

    assert baseline == pilot.stable_selection_hash("id", "CCO", 0)
    assert baseline != pilot.stable_selection_hash("id", "CCO", 1)
    assert baseline != pilot.stable_selection_hash("id", "CCO", 0, selection_seed=37)
    assert pilot.stable_selection_hash("id", "CCO.[Na+]", 1) == (
        "4d6e092bc87707e990c26e74718d6660a34530540b6888b5bea0c3d290c62c14"
    )


def test_standardization_occurs_after_selection_without_changing_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, train_count=1)
    training = pd.DataFrame(
        [
            _source_row("connected", "CCO", 0),
            _source_row("sodium", "CCO.[Na+]", 1),
            _source_row("chloride", "CCN.[Cl-]", 0),
            _source_row("other", "CCCC", 1),
        ]
    )
    _replace_training(config, training)
    expected = pilot.select_training_pilot(training, 2)
    standardization_calls: list[tuple[str, str]] = []

    def tracking_standardization(molecule_id: object, smiles: str):
        standardization_calls.append((str(molecule_id), smiles))
        return standardize_for_gmc_geometry(molecule_id, smiles)

    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)
    output = tmp_path / "pilot-output"
    pilot.run_geometry_pilot(
        tmp_path / "config.yaml",
        pilot_size=2,
        output_dir=output,
        standardization_function=tracking_standardization,
        geometry_function=lambda smiles, config: _geometry_result(smiles, config),
        ggl_function=_ggl_result,
    )

    manifest = json.loads((output / "pilot_selection.json").read_text(encoding="utf-8"))
    expected_identities = [
        (str(row.molecule_id), str(row.canonical_smiles), str(row.selection_hash))
        for row in expected.itertuples(index=False)
    ]
    manifest_identities = [
        (row["molecule_id"], row["canonical_smiles"], row["selection_hash"])
        for row in manifest["molecules"]
    ]
    assert manifest_identities == expected_identities
    assert standardization_calls == [(row[0], row[1]) for row in expected_identities]


def test_pilot_loads_and_processes_training_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, train_count=8)
    output = tmp_path / "pilot-output"
    geometry_calls: list[str] = []

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        geometry_calls.append(smiles)
        return _geometry_result(smiles, config)

    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)
    summary = pilot.run_geometry_pilot(
        tmp_path / "config.yaml",
        pilot_size=5,
        output_dir=output,
        geometry_function=geometry_function,
        ggl_function=_ggl_result,
    )

    selected = json.loads((output / "pilot_selection.json").read_text(encoding="utf-8"))
    assert geometry_calls == [row["canonical_smiles"] for row in selected["molecules"]]
    assert not any(smiles.startswith("N") for smiles in geometry_calls)
    assert summary["loaded_split"] == "train"
    assert summary["split"] == "train"
    assert summary["validation_artifact_accessed"] is False
    assert summary["test_artifact_accessed"] is False


@pytest.mark.parametrize(
    ("molecule_id", "source_smiles", "removed_smiles"),
    (
        ("aptazapine", APTAZAPINE_SMILES, "O=C(O)/C=C\\C(=O)O"),
        ("cefpodoxime", CEFPODOXIME_SMILES, "[Na+]"),
    ),
)
def test_disconnected_pilot_rows_use_parent_geometry_and_reach_ggl(
    molecule_id: str, source_smiles: str, removed_smiles: str
) -> None:
    training = pd.DataFrame([_source_row(molecule_id, source_smiles, 1)])
    selected = pilot.select_training_pilot(training, 1)
    geometry_calls: list[str] = []

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        geometry_calls.append(smiles)
        return _geometry_result(smiles, config)

    records = pilot._process_selected_rows(
        selected,
        geometry_config=GeometryConfig(),
        ggl_config=GGLConfig(),
        standardization_function=standardize_for_gmc_geometry,
        geometry_function=geometry_function,
        ggl_function=_ggl_result,
    )

    record = records[0]
    expected = standardize_for_gmc_geometry(molecule_id, _canonical(source_smiles))
    assert record["status"] == "success"
    assert record["failure_category"] == ""
    assert record["standardization_action"] == "parent_selected"
    assert record["canonical_smiles"] == _canonical(source_smiles)
    assert record["geometry_smiles"] == expected.geometry_canonical_smiles
    assert geometry_calls == [expected.geometry_canonical_smiles]
    assert json.loads(record["removed_fragment_smiles"]) == [_canonical(removed_smiles)]
    assert record["parent_heavy_atom_count"] == expected.parent_heavy_atom_count
    assert record["parent_formal_charge"] == expected.parent_formal_charge


def test_connected_pilot_row_uses_original_geometry_smiles() -> None:
    training = pd.DataFrame([_source_row("connected", "C[C@H](O)F", 0)])
    selected = pilot.select_training_pilot(training, 1)
    geometry_calls: list[str] = []

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        geometry_calls.append(smiles)
        return _geometry_result(smiles, config)

    record = pilot._process_selected_rows(
        selected,
        geometry_config=GeometryConfig(),
        ggl_config=GGLConfig(),
        standardization_function=standardize_for_gmc_geometry,
        geometry_function=geometry_function,
        ggl_function=_ggl_result,
    )[0]

    assert record["standardization_action"] == "unchanged"
    assert record["canonical_smiles"] == record["geometry_smiles"] == _canonical("C[C@H](O)F")
    assert geometry_calls == [record["canonical_smiles"]]


def test_policy_exclusions_skip_geometry_and_ggl_and_have_separate_summary(
    tmp_path: Path,
) -> None:
    training = pd.DataFrame(
        [
            _source_row("eqvalan", "CCO.[Na+]", 0),
            _source_row("sultamicillin", "CCN.[Cl-]", 1),
            _source_row("ordinary", "CCO", 1),
        ]
    )
    selected = pilot.select_training_pilot(training, 3)
    geometry_calls: list[str] = []
    ggl_calls = 0

    def geometry_function(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        geometry_calls.append(smiles)
        return _geometry_result(smiles, config)

    def ggl_function(*args: object, **kwargs: object) -> GGLResult:
        nonlocal ggl_calls
        ggl_calls += 1
        return _ggl_result(*args, **kwargs)  # type: ignore[arg-type]

    records = pilot._process_selected_rows(
        selected,
        geometry_config=GeometryConfig(),
        ggl_config=GGLConfig(),
        standardization_function=standardize_for_gmc_geometry,
        geometry_function=geometry_function,
        ggl_function=ggl_function,
    )
    by_id = {record["molecule_id"]: record for record in records}
    config = _fixture_config(tmp_path, train_count=1)
    summary = pilot._build_summary(
        config=config,
        requested_pilot_size=3,
        records=records,
        total_elapsed_seconds=1.0,
        geometry_config=GeometryConfig(),
        ggl_config=GGLConfig(),
        access_guard=pilot.ArtifactAccessGuard(
            validation_path=tmp_path / "valid.csv",
            test_path=tmp_path / "locked.csv",
        ),
    )

    assert geometry_calls == ["CCO"]
    assert ggl_calls == 1
    for molecule_id in ("eqvalan", "sultamicillin"):
        assert by_id[molecule_id]["status"] == "excluded"
        assert by_id[molecule_id]["failure_category"] == "excluded_by_policy"
        assert by_id[molecule_id]["standardization_action"] == "excluded_by_policy"
        assert by_id[molecule_id]["geometry_smiles"] == ""
        assert by_id[molecule_id]["exclusion_reason"]
    assert summary["successful_molecule_count"] == 1
    assert summary["failed_molecule_count"] == 0
    assert summary["policy_excluded_molecule_count"] == 2
    assert summary["counts_by_failure_category"]["unexpected_error"] == 0
    assert summary["counts_by_failure_category"]["excluded_by_policy"] == 2
    assert summary["counts_by_standardization_action"] == {
        "unchanged": 1,
        "parent_selected": 0,
        "excluded_by_policy": 2,
    }


def test_nontraining_artifact_guard_blocks_inspection(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, train_count=2)
    validation = config.prepared_root / config.split_files["validation"]
    locked = config.prepared_root / config.split_files["test"]

    with pilot._deny_nontraining_artifact_access(config) as guard:
        with pytest.raises(pilot.NonTrainingArtifactAccessError):
            validation.read_bytes()
        with pytest.raises(pilot.NonTrainingArtifactAccessError):
            locked.resolve()
        with pytest.raises(pilot.NonTrainingArtifactAccessError):
            locked.exists()
        with pytest.raises(pilot.NonTrainingArtifactAccessError):
            pd.read_csv(locked)

    assert guard.validation_artifact_accessed is True
    assert guard.test_artifact_accessed is True


def test_every_selected_row_and_success_schema_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, train_count=6)
    output = tmp_path / "pilot-output"
    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)

    summary = pilot.run_geometry_pilot(
        tmp_path / "config.yaml",
        pilot_size=4,
        output_dir=output,
        geometry_function=lambda smiles, config: _geometry_result(smiles, config),
        ggl_function=_ggl_result,
    )
    status = pd.read_csv(output / "molecule_status.csv")

    assert status.columns.tolist() == list(pilot.STATUS_COLUMNS)
    assert len(status) == 4
    assert status["status"].tolist() == ["success"] * 4
    assert status["source_split"].tolist() == ["train"] * 4
    assert status["raw_ggl_rows"].tolist() == [2] * 4
    assert status["raw_ggl_columns"].tolist() == [6] * 4
    assert status["all_ggl_values_finite"].all()
    assert summary["successful_molecule_count"] == 4
    assert summary["failed_molecule_count"] == 0
    assert summary["policy_excluded_molecule_count"] == 0
    assert summary["counts_by_standardization_action"]["unchanged"] == 4
    assert all(value == 0 for value in summary["counts_by_failure_category"].values())


def test_controlled_failure_remains_in_status_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, train_count=5)
    output = tmp_path / "pilot-output"
    call_count = 0

    def controlled_geometry(smiles: str, *, config: GeometryConfig) -> GeometryResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise GeometryError("embedding_failed", "synthetic controlled failure")
        return _geometry_result(smiles, config)

    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)
    summary = pilot.run_geometry_pilot(
        tmp_path / "config.yaml",
        pilot_size=5,
        output_dir=output,
        geometry_function=controlled_geometry,
        ggl_function=_ggl_result,
    )
    status = pd.read_csv(output / "molecule_status.csv", keep_default_na=False)

    assert len(status) == 5
    failed = status.loc[status["status"] == "failed"]
    assert len(failed) == 1
    assert failed.iloc[0]["failure_category"] == "embedding_failed"
    assert "synthetic controlled failure" in failed.iloc[0]["failure_message"]
    assert summary["successful_molecule_count"] == 4
    assert summary["failed_molecule_count"] == 1
    assert summary["counts_by_failure_category"]["embedding_failed"] == 1
    assert summary["success_fraction"] == pytest.approx(0.8)


def test_disconnected_fragment_has_explicit_failure_category() -> None:
    error = GeometryError("disconnected_fragment", "synthetic salt")

    assert "disconnected_fragment" in pilot.FAILURE_CATEGORIES
    assert pilot._failure_category(error) == "disconnected_fragment"


def test_unrecognized_failure_remains_unexpected_error() -> None:
    assert pilot._failure_category(RuntimeError("synthetic surprise")) == "unexpected_error"


def test_output_files_and_summary_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fixture_config(tmp_path, train_count=3)
    output = tmp_path / "pilot-output"
    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)

    pilot.run_geometry_pilot(
        tmp_path / "config.yaml",
        pilot_size=3,
        output_dir=output,
        geometry_function=lambda smiles, config: _geometry_result(smiles, config),
        ggl_function=_ggl_result,
    )

    assert {path.name for path in output.iterdir()} == {
        "molecule_status.csv",
        "pilot_summary.json",
        "pilot_selection.json",
    }
    summary = json.loads((output / "pilot_summary.json").read_text(encoding="utf-8"))
    required = {
        "dataset",
        "loaded_split",
        "requested_pilot_size",
        "selected_pilot_size",
        "successful_molecule_count",
        "failed_molecule_count",
        "policy_excluded_molecule_count",
        "success_fraction",
        "counts_by_failure_category",
        "counts_by_standardization_action",
        "parent_standardized_molecule_count",
        "standardization_version",
        "counts_by_optimization_method",
        "total_elapsed_seconds",
        "mean_seconds_per_selected_molecule",
        "median_seconds_per_selected_molecule",
        "mean_geometry_seconds_among_successes",
        "mean_ggl_seconds_among_successes",
        "minimum_heavy_atom_count",
        "maximum_heavy_atom_count",
        "finite_ggl_molecule_count",
        "geometry_settings",
        "ggl_settings",
        "rdkit_version",
        "git_commit",
        "test_artifact_accessed",
    }
    assert required <= set(summary)
    assert summary["geometry_settings"]["num_conformers"] == 20
    assert summary["geometry_settings"]["etkdg_version"] == "ETKDGv3"
    assert summary["ggl_settings"]["feature_order"] == [
        "minimum",
        "maximum",
        "sum",
        "mean",
        "median",
        "population_standard_deviation",
    ]
    assert summary["ggl_settings"]["scaled"] is False


def test_existing_output_requires_explicit_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, train_count=2)
    output = tmp_path / "pilot-output"
    output.mkdir()
    unrelated = output / "keep-me.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)

    with pytest.raises(FileExistsError, match="--overwrite"):
        pilot.run_geometry_pilot(tmp_path / "config.yaml", output_dir=output)

    pilot.run_geometry_pilot(
        tmp_path / "config.yaml",
        pilot_size=2,
        output_dir=output,
        overwrite=True,
        geometry_function=lambda smiles, config: _geometry_result(smiles, config),
        ggl_function=_ggl_result,
    )
    assert unrelated.read_text(encoding="utf-8") == "preserve"


def test_nontraining_alias_is_rejected_before_data_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, train_count=2)
    config.split_files["train"] = config.split_files["test"]
    monkeypatch.setattr(pilot, "load_chemprop_config", lambda _: config)
    touched = False

    def forbidden_loader(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal touched
        touched = True
        raise AssertionError("loader was called")

    monkeypatch.setattr(pilot, "load_bbb_development_split", forbidden_loader)
    with pytest.raises(pilot.NonTrainingArtifactAccessError, match="distinct"):
        pilot.run_geometry_pilot(tmp_path / "config.yaml", output_dir=tmp_path / "pilot-output")
    assert touched is False


def test_parser_defaults_and_help_contract() -> None:
    args = pilot.build_parser().parse_args([])

    assert args.pilot_size == 25
    assert args.overwrite is False
    assert args.config == pilot.DEFAULT_CONFIG
    assert args.output_dir == pilot.DEFAULT_OUTPUT_DIR


def _training_frame(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "molecule_id": f"train-{index:03d}",
                "canonical_smiles": "C" * (index + 1),
                "target": index % 2,
                "split": "train",
            }
            for index in range(count)
        ]
    )


def _source_row(molecule_id: str, smiles: str, target: int) -> dict[str, object]:
    canonical = _canonical(smiles)
    return {
        "molecule_id": molecule_id,
        "smiles": canonical,
        "canonical_smiles": canonical,
        "target": target,
        "split": "train",
    }


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _replace_training(config: ChempropExperimentConfig, training: pd.DataFrame) -> None:
    training.to_csv(config.prepared_root / config.split_files["train"], index=False)


def _fixture_config(root: Path, *, train_count: int) -> ChempropExperimentConfig:
    prepared = root / "prepared"
    prepared.mkdir()
    training = _training_frame(train_count)
    training.insert(2, "smiles", training["canonical_smiles"])
    training.to_csv(prepared / "train.csv", index=False)
    # Non-training synthetic sentinels must never be passed to geometry or opened by the pilot.
    pd.DataFrame(
        [
            {
                "molecule_id": "validation-sentinel",
                "smiles": "N",
                "canonical_smiles": "N",
                "target": 1,
                "split": "validation",
            }
        ]
    ).to_csv(prepared / "valid.csv", index=False)
    (prepared / "locked.csv").write_text("must-not-be-read", encoding="utf-8")
    return ChempropExperimentConfig(
        source_path=root / "config.yaml",
        raw={},
        endpoint="bbb_martins",
        endpoint_id="bbb_martins",
        dataset="BBB_Martins",
        dataset_version="synthetic-pilot-fixture",
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


def _geometry_result(smiles: str, config: GeometryConfig) -> GeometryResult:
    return GeometryResult(
        input_smiles=smiles,
        canonical_isomeric_smiles=smiles,
        seed=config.seed,
        embedding_seed=12345,
        etkdg_version="ETKDGv3",
        requested_conformer_count=config.num_conformers,
        generated_conformer_count=3,
        optimization_method="MMFF94s",
        selected_conformer_id=1,
        selected_energy=-2.5,
        convergence_status="converged",
        heavy_atom_count=2,
        heavy_atom_rdkit_indices=(0, 1),
        heavy_atom_atomic_numbers=(6, 6),
        heavy_atom_formal_charges=(0, 0),
        coordinates=np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        rdkit_version=rdBase.rdkitVersion,
        geometry_status="success_mmff94s",
        geometry_fingerprint=f"geometry-{smiles}",
        config=config,
    )


def _ggl_result(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    geometry_fingerprint: str,
    config: GGLConfig,
) -> GGLResult:
    assert coordinates.shape == (2, 3)
    assert atomic_numbers.tolist() == [6, 6]
    return GGLResult(
        features=np.ones((2, 6), dtype=np.float64),
        feature_names=(
            "minimum",
            "maximum",
            "sum",
            "mean",
            "median",
            "population_standard_deviation",
        ),
        geometry_fingerprint=geometry_fingerprint,
        ggl_fingerprint=f"ggl-{geometry_fingerprint}",
        ggl_status="success",
        config=config,
        preprocessing_version="test",
        element_radius_mapping_version="test",
    )
