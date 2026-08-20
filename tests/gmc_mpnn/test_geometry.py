from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from admet_platform.gmc_mpnn import geometry
from admet_platform.gmc_mpnn.geometry import (
    GeometryConfig,
    GeometryError,
    OptimizationRecord,
    generate_deterministic_geometry,
)


SYNTHETIC_FIXTURES = (
    ("aromatic", "c1ccncc1", 6),
    ("stereochemical", "C[C@H](O)F", 4),
    ("charged", "C[NH2+]C", 3),
    ("amide", "CNC(C)=O", 5),
    ("flexible_aliphatic", "CCCCCC", 6),
)
COORDINATE_REPEATABILITY_ATOL = 1e-8
ENERGY_REPEATABILITY_ATOL = 1e-10
APALCILLIN_SMILES = (
    "CC1(C)S[C@@H]2C(NC(=O)[C@H](NC(=O)c3c[nH]c4cccnc4c3=O)c3ccccc3)C(=O)N2[C@H]1C(=O)O"
)


def test_etkdgv3_is_available_with_resolved_defaults() -> None:
    assert callable(getattr(AllChem, "ETKDGv3", None))
    config = GeometryConfig()
    assert config.seed == 13
    assert config.num_conformers == 20
    assert config.prune_rms_threshold == 0.5
    assert config.optimization_max_iterations == 1_000
    assert config.num_threads == 1


@pytest.mark.parametrize(("fixture_name", "smiles", "heavy_count"), SYNTHETIC_FIXTURES)
def test_five_synthetic_geometries_are_identity_safe_and_same_seed_deterministic(
    fixture_name: str, smiles: str, heavy_count: int
) -> None:
    first = generate_deterministic_geometry(smiles)
    second = generate_deterministic_geometry(smiles)

    assert fixture_name
    assert first.input_smiles == second.input_smiles == smiles
    assert first.canonical_isomeric_smiles == second.canonical_isomeric_smiles
    assert first.seed == second.seed == 13
    assert first.embedding_seed == second.embedding_seed
    assert first.etkdg_version == second.etkdg_version == "ETKDGv3"
    assert first.requested_conformer_count == second.requested_conformer_count == 20
    assert 1 <= first.generated_conformer_count <= 20
    assert first.generated_conformer_count == second.generated_conformer_count
    assert first.optimization_method == second.optimization_method
    assert first.optimization_method in {"MMFF94s", "MMFF94s_retry_2000", "UFF"}
    assert first.convergence_status == second.convergence_status == "converged"
    assert first.geometry_status == second.geometry_status
    assert first.heavy_atom_count == second.heavy_atom_count == heavy_count
    assert (
        first.heavy_atom_rdkit_indices
        == second.heavy_atom_rdkit_indices
        == tuple(range(heavy_count))
    )
    assert first.heavy_atom_atomic_numbers == second.heavy_atom_atomic_numbers
    assert first.heavy_atom_formal_charges == second.heavy_atom_formal_charges
    assert first.coordinates.shape == second.coordinates.shape == (heavy_count, 3)
    assert np.isfinite(first.coordinates).all()
    np.testing.assert_allclose(
        first.coordinates,
        second.coordinates,
        rtol=0.0,
        atol=COORDINATE_REPEATABILITY_ATOL,
    )
    assert first.selected_energy == pytest.approx(
        second.selected_energy, rel=0.0, abs=ENERGY_REPEATABILITY_ATOL
    )
    assert first.selected_conformer_id == second.selected_conformer_id
    assert first.geometry_fingerprint == second.geometry_fingerprint
    assert len(first.geometry_fingerprint) == 64
    assert first.rdkit_version == second.rdkit_version == rdBase.rdkitVersion

    canonical_molecule = Chem.MolFromSmiles(first.canonical_isomeric_smiles)
    assert canonical_molecule is not None
    expected_heavy_atoms = [
        atom for atom in canonical_molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    assert first.heavy_atom_atomic_numbers == tuple(
        atom.GetAtomicNum() for atom in expected_heavy_atoms
    )
    assert first.heavy_atom_formal_charges == tuple(
        atom.GetFormalCharge() for atom in expected_heavy_atoms
    )


def test_changing_seed_preserves_identity_and_shape() -> None:
    first = generate_deterministic_geometry("CCCCCC", config=GeometryConfig(seed=13))
    changed = generate_deterministic_geometry("CCCCCC", config=GeometryConfig(seed=37))

    assert first.canonical_isomeric_smiles == changed.canonical_isomeric_smiles
    assert first.heavy_atom_rdkit_indices == changed.heavy_atom_rdkit_indices
    assert first.heavy_atom_atomic_numbers == changed.heavy_atom_atomic_numbers
    assert first.heavy_atom_formal_charges == changed.heavy_atom_formal_charges
    assert first.coordinates.shape == changed.coordinates.shape == (6, 3)
    assert first.embedding_seed != changed.embedding_seed
    assert first.geometry_fingerprint != changed.geometry_fingerprint


def test_stereochemical_fixture_preserves_canonical_isomeric_identity() -> None:
    smiles = "C[C@H](O)F"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    expected = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

    result = generate_deterministic_geometry(smiles)

    assert result.canonical_isomeric_smiles == expected
    assert "@" in result.canonical_isomeric_smiles


def test_apalcillin_mmff_aromaticity_mutation_preserves_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_properties = geometry.AllChem.MMFFGetMoleculeProperties
    original_verify = geometry._verify_heavy_atom_identity
    aromaticity_changes: list[tuple[int, ...]] = []
    verified_topologies: list[geometry.HeavyAtomTopology] = []

    def tracking_properties(molecule: Chem.Mol, *args: object, **kwargs: object):
        before = tuple(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        properties = original_properties(molecule, *args, **kwargs)
        after = tuple(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        aromaticity_changes.append(
            tuple(
                index for index, values in enumerate(zip(before, after)) if values[0] != values[1]
            )
        )
        return properties

    def tracking_verify(
        molecule: Chem.Mol,
        expected: tuple[geometry.AtomIdentity, ...],
        expected_topology: geometry.HeavyAtomTopology,
    ) -> None:
        original_verify(molecule, expected, expected_topology)
        assert geometry._capture_heavy_atom_topology(molecule) == expected_topology
        verified_topologies.append(expected_topology)

    monkeypatch.setattr(geometry.AllChem, "MMFFGetMoleculeProperties", tracking_properties)
    monkeypatch.setattr(geometry, "_verify_heavy_atom_identity", tracking_verify)

    canonical = geometry._canonical_molecule(APALCILLIN_SMILES)
    expected_atomic_numbers = tuple(atom.GetAtomicNum() for atom in canonical.GetAtoms())
    expected_formal_charges = tuple(atom.GetFormalCharge() for atom in canonical.GetAtoms())
    result = generate_deterministic_geometry(APALCILLIN_SMILES)

    assert result.optimization_method == "MMFF94s"
    assert result.heavy_atom_rdkit_indices == tuple(range(result.heavy_atom_count))
    assert result.heavy_atom_atomic_numbers == expected_atomic_numbers
    assert result.heavy_atom_formal_charges == expected_formal_charges
    assert len(verified_topologies) == 2
    assert verified_topologies[0] == verified_topologies[1]
    assert aromaticity_changes and aromaticity_changes[0]
    assert np.isfinite(result.coordinates).all()
    assert math.isfinite(result.selected_energy)


def test_selection_excludes_lower_energy_unconverged_conformer() -> None:
    selected = geometry._select_lowest_energy_converged(
        (
            OptimizationRecord(0, converged=False, energy=-100.0, status_code=1),
            OptimizationRecord(2, converged=True, energy=5.0, status_code=0),
            OptimizationRecord(1, converged=True, energy=5.0, status_code=0),
            OptimizationRecord(3, converged=True, energy=7.0, status_code=0),
        )
    )

    assert selected.conformer_id == 1
    assert selected.energy == 5.0


def test_no_converged_conformer_is_an_explicit_failure() -> None:
    with pytest.raises(GeometryError, match="optimization_failed") as exc_info:
        geometry._select_lowest_energy_converged(
            (OptimizationRecord(0, converged=False, energy=-1.0, status_code=1),)
        )
    assert exc_info.value.status == "optimization_failed"


def test_ordinary_mmff_success_uses_only_normal_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def successful_mmff(molecule: Chem.Mol, **kwargs: object):
        calls.append(int(kwargs["maxIters"]))
        return [(0, float(conformer.GetId())) for conformer in molecule.GetConformers()]

    monkeypatch.setattr(geometry.AllChem, "MMFFOptimizeMoleculeConfs", successful_mmff)

    result = generate_deterministic_geometry("CCO")

    assert calls == [1_000]
    assert result.optimization_method == "MMFF94s"
    assert result.geometry_status == "success_mmff94s"


def test_zero_mmff_convergence_retries_same_embedding_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimization_calls: list[int] = []
    coordinate_snapshots: list[np.ndarray] = []
    embed_calls = 0
    original_embed = geometry.AllChem.EmbedMultipleConfs

    def tracking_embed(*args: object, **kwargs: object):
        nonlocal embed_calls
        embed_calls += 1
        return original_embed(*args, **kwargs)

    def retrying_mmff(molecule: Chem.Mol, **kwargs: object):
        max_iterations = int(kwargs["maxIters"])
        optimization_calls.append(max_iterations)
        coordinate_snapshots.append(
            np.asarray(
                [
                    [
                        (
                            conformer.GetAtomPosition(atom_index).x,
                            conformer.GetAtomPosition(atom_index).y,
                            conformer.GetAtomPosition(atom_index).z,
                        )
                        for atom_index in range(molecule.GetNumAtoms())
                    ]
                    for conformer in molecule.GetConformers()
                ],
                dtype=np.float64,
            )
        )
        if max_iterations == 1_000:
            for conformer in molecule.GetConformers():
                point = conformer.GetAtomPosition(0)
                conformer.SetAtomPosition(0, (point.x + 100.0, point.y, point.z))
            return [(1, -100.0) for _ in molecule.GetConformers()]
        energies = [5.0, 1.0, 1.0]
        return [(0, energies[index]) for index, _conformer in enumerate(molecule.GetConformers())]

    monkeypatch.setattr(geometry.AllChem, "EmbedMultipleConfs", tracking_embed)
    monkeypatch.setattr(geometry.AllChem, "MMFFOptimizeMoleculeConfs", retrying_mmff)

    result = generate_deterministic_geometry(
        "CCCC", config=GeometryConfig(num_conformers=3, prune_rms_threshold=0.0)
    )

    assert embed_calls == 1
    assert optimization_calls == [1_000, 2_000]
    np.testing.assert_array_equal(coordinate_snapshots[0], coordinate_snapshots[1])
    assert result.generated_conformer_count == 3
    assert result.selected_conformer_id == 1
    assert result.selected_energy == 1.0
    assert result.optimization_method == "MMFF94s_retry_2000"
    assert result.geometry_status == "success_mmff94s_retry_2000"
    assert len(result.geometry_fingerprint) == 64


def test_failed_mmff_retry_preserves_optimization_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def unconverged_mmff(molecule: Chem.Mol, **kwargs: object):
        calls.append(int(kwargs["maxIters"]))
        return [(1, -1.0) for _ in molecule.GetConformers()]

    monkeypatch.setattr(geometry.AllChem, "MMFFOptimizeMoleculeConfs", unconverged_mmff)

    with pytest.raises(GeometryError, match="optimization_failed") as exc_info:
        generate_deterministic_geometry("CCO")

    assert calls == [1_000, 2_000]
    assert exc_info.value.status == "optimization_failed"


def test_no_conformer_failure_does_not_attempt_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimization_called = False

    def unexpected_mmff(*args: object, **kwargs: object):
        nonlocal optimization_called
        optimization_called = True
        raise AssertionError("optimization must not run without conformers")

    monkeypatch.setattr(geometry.AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])
    monkeypatch.setattr(geometry.AllChem, "MMFFOptimizeMoleculeConfs", unexpected_mmff)

    with pytest.raises(GeometryError, match="no_conformers_generated") as exc_info:
        generate_deterministic_geometry("CCO")

    assert exc_info.value.status == "no_conformers_generated"
    assert optimization_called is False


def test_uff_is_used_only_when_mmff_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry.AllChem, "MMFFHasAllMoleculeParams", lambda molecule: False)

    result = generate_deterministic_geometry("CCO")

    assert result.optimization_method == "UFF"
    assert result.geometry_status == "success_uff_fallback"
    assert result.convergence_status == "converged"


def test_missing_mmff_and_uff_parameters_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry.AllChem, "MMFFHasAllMoleculeParams", lambda molecule: False)
    monkeypatch.setattr(geometry.AllChem, "UFFHasAllMoleculeParams", lambda molecule: False)

    with pytest.raises(GeometryError, match="uff_unavailable") as exc_info:
        generate_deterministic_geometry("CCO")
    assert exc_info.value.status == "uff_unavailable"


def test_invalid_smiles_fails_explicitly() -> None:
    with pytest.raises(GeometryError, match="invalid_smiles") as exc_info:
        generate_deterministic_geometry("not-a-smiles")
    assert exc_info.value.status == "invalid_smiles"


def test_missing_etkdgv3_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry.AllChem, "ETKDGv3", None)

    with pytest.raises(GeometryError, match="etkdgv3_unavailable") as exc_info:
        generate_deterministic_geometry("CCO")
    assert exc_info.value.status == "etkdgv3_unavailable"


def test_no_generated_conformers_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry.AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])

    with pytest.raises(GeometryError, match="no_conformers_generated") as exc_info:
        generate_deterministic_geometry("CCO")
    assert exc_info.value.status == "no_conformers_generated"


@pytest.mark.parametrize(
    "mutation",
    (
        "original_index",
        "atomic_number",
        "charge",
        "isotope",
        "adjacency",
        "missing_atom",
        "added_atom",
    ),
)
def test_true_atom_or_topology_mismatch_fails_explicitly(mutation: str) -> None:
    molecule = Chem.MolFromSmiles("CO")
    assert molecule is not None
    expected = geometry._mark_and_capture_heavy_atom_identity(molecule)
    expected_topology = geometry._capture_heavy_atom_topology(molecule)
    molecule_h = Chem.AddHs(molecule)
    if mutation == "original_index":
        molecule_h.GetAtomWithIdx(0).SetIntProp(geometry.ORIGINAL_INDEX_PROPERTY, 1)
    elif mutation == "atomic_number":
        molecule_h.GetAtomWithIdx(0).SetAtomicNum(7)
    elif mutation == "charge":
        molecule_h.GetAtomWithIdx(0).SetFormalCharge(1)
    elif mutation == "isotope":
        molecule_h.GetAtomWithIdx(0).SetIsotope(13)
    elif mutation == "adjacency":
        editable = Chem.RWMol(molecule_h)
        editable.RemoveBond(0, 1)
        molecule_h = editable.GetMol()
    elif mutation == "missing_atom":
        editable = Chem.RWMol(molecule_h)
        editable.RemoveAtom(1)
        molecule_h = editable.GetMol()
    else:
        editable = Chem.RWMol(molecule_h)
        editable.AddAtom(Chem.Atom(6))
        molecule_h = editable.GetMol()

    with pytest.raises(GeometryError, match="atom_alignment_failed") as exc_info:
        geometry._verify_heavy_atom_identity(molecule_h, expected, expected_topology)
    assert exc_info.value.status == "atom_alignment_failed"


def test_fingerprint_rounding_policy_is_documented_and_stable() -> None:
    assert geometry.GEOMETRY_FINGERPRINT_DECIMALS == 8
    assert geometry.GEOMETRY_PREPROCESSING_VERSION == "rdkit-etkdgv3-mmff94s-retry2000-v2"
    assert geometry.EMBEDDING_SEED_VERSION == "rdkit-etkdgv3-mmff94s-v1"
    assert math.isfinite(float(GeometryConfig().prune_rms_threshold))
