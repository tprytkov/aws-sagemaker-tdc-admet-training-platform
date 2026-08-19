from __future__ import annotations

import pytest
from rdkit import Chem

from admet_platform.gmc_mpnn import standardization
from admet_platform.gmc_mpnn.standardization import (
    EXCLUDED_BY_POLICY,
    GMC_STANDARDIZATION_VERSION,
    GMCStandardizationError,
    PARENT_SELECTED,
    UNCHANGED,
    standardize_for_gmc_geometry,
)


APTAZAPINE_DRUG = "CN1CCN2c3ccccc3Cn3cccc3C2C1"
FUMARATE = "O=C(O)/C=C\\C(=O)O"
APTAZAPINE_SMILES = f"{APTAZAPINE_DRUG}.{FUMARATE}"
CEFPODOXIME_DRUG = "COCC1=C(C(=O)[O-])N2C(=O)[C@@H](NC(=O)/C(=N\\OC)c3csc(N)n3)[C@H]2SC1"
CEFPODOXIME_SMILES = f"{CEFPODOXIME_DRUG}.[Na+]"


def test_connected_molecule_remains_unchanged() -> None:
    result = standardize_for_gmc_geometry("connected", "C[C@H](O)F")

    assert result.action == UNCHANGED
    assert result.geometry_canonical_smiles == "C[C@H](O)F"
    assert result.fragment_count == 1
    assert result.removed_fragment_smiles == ()


def test_source_smiles_remains_separate_from_geometry_smiles() -> None:
    source = _canonical("CC(=O)[O-].[Na+]")
    result = standardize_for_gmc_geometry("salt", source)

    assert result.source_canonical_smiles == source
    assert result.geometry_canonical_smiles == "CC(=O)[O-]"
    assert result.source_canonical_smiles != result.geometry_canonical_smiles


def test_sodium_salt_selects_organic_parent() -> None:
    result = standardize_for_gmc_geometry("sodium", "CC(=O)[O-].[Na+]")

    assert result.action == PARENT_SELECTED
    assert result.geometry_canonical_smiles == "CC(=O)[O-]"
    assert result.removed_fragment_smiles == ("[Na+]",)


def test_chloride_counterion_selects_organic_parent() -> None:
    result = standardize_for_gmc_geometry("chloride", "C[NH3+].[Cl-]")

    assert result.action == PARENT_SELECTED
    assert result.geometry_canonical_smiles == "C[NH3+]"
    assert result.removed_fragment_smiles == ("[Cl-]",)


def test_aptazapine_selects_drug_and_removes_fumarate() -> None:
    result = standardize_for_gmc_geometry("aptazapine", APTAZAPINE_SMILES)

    assert result.action == PARENT_SELECTED
    assert result.geometry_canonical_smiles == _canonical(APTAZAPINE_DRUG)
    assert result.removed_fragment_smiles == (_canonical(FUMARATE),)


def test_cefpodoxime_selects_drug_and_removes_sodium() -> None:
    result = standardize_for_gmc_geometry("cefpodoxime", CEFPODOXIME_SMILES)

    assert result.action == PARENT_SELECTED
    assert result.geometry_canonical_smiles == _canonical(CEFPODOXIME_DRUG)
    assert result.removed_fragment_smiles == ("[Na+]",)


def test_multi_organic_coformer_selects_largest_fragment() -> None:
    large = "CCCc1ccccc1"
    coformer = "O=C(O)CC(=O)O"
    result = standardize_for_gmc_geometry("coformer", f"{coformer}.{large}")

    assert result.geometry_canonical_smiles == _canonical(large)
    assert result.removed_fragment_smiles == (_canonical(coformer),)


def test_equal_heavy_atom_tie_uses_canonical_smiles_order() -> None:
    result = standardize_for_gmc_geometry("tie", "CN.CC")

    assert result.geometry_canonical_smiles == "CC"
    assert result.removed_fragment_smiles == ("CN",)


def test_removed_fragment_ordering_is_deterministic() -> None:
    first = standardize_for_gmc_geometry("ions", "CCO.[Na+].[Cl-].[K+]")
    second = standardize_for_gmc_geometry("ions", "[K+].[Cl-].CCO.[Na+]")

    assert first.removed_fragment_smiles == ("[Cl-]", "[K+]", "[Na+]")
    assert first.removed_fragment_smiles == second.removed_fragment_smiles
    assert first.removed_fragment_heavy_atom_counts == second.removed_fragment_heavy_atom_counts
    assert first.removed_fragment_formal_charges == second.removed_fragment_formal_charges


def test_fragment_heavy_atom_counts_are_recorded() -> None:
    result = standardize_for_gmc_geometry("counts", "CC(=O)[O-].[Na+]")

    assert result.fragment_count == 2
    assert result.source_heavy_atom_count == 5
    assert result.parent_heavy_atom_count == 4
    assert result.removed_fragment_heavy_atom_counts == (1,)


def test_fragment_formal_charges_are_recorded() -> None:
    result = standardize_for_gmc_geometry("charges", "CC(=O)[O-].[Na+]")

    assert result.source_formal_charge == 0
    assert result.parent_formal_charge == -1
    assert result.removed_fragment_formal_charges == (1,)


def test_source_identity_is_not_mutated() -> None:
    source = _canonical("CCO.[Na+]")
    result = standardize_for_gmc_geometry(" source-id ", source)

    assert source == _canonical("CCO.[Na+]")
    assert result.molecule_id == "source-id"
    assert result.source_canonical_smiles == source


def test_eqvalan_is_excluded_by_frozen_policy() -> None:
    result = standardize_for_gmc_geometry(" eqvalan ", "CCO.[Na+]")

    assert result.action == EXCLUDED_BY_POLICY
    assert result.geometry_canonical_smiles is None
    assert result.exclusion_reason == "ambiguous_multi_active_mixture"


def test_sultamicillin_is_excluded_by_frozen_policy() -> None:
    result = standardize_for_gmc_geometry("sultamicillin", "CCN.[Cl-]")

    assert result.action == EXCLUDED_BY_POLICY
    assert result.geometry_canonical_smiles is None
    assert result.exclusion_reason == ("chemically_inconsistent_multicomponent_representation")


@pytest.mark.parametrize("molecule_id", ("eqvalan", "sultamicillin"))
def test_frozen_exclusions_never_invoke_largest_fragment_fallback(
    molecule_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fallback(*args: object, **kwargs: object):
        raise AssertionError("parent selection was invoked")

    monkeypatch.setattr(standardization, "_select_parent_fragment", forbidden_fallback)

    result = standardize_for_gmc_geometry(molecule_id, "CCO.[Na+]")

    assert result.action == EXCLUDED_BY_POLICY


def test_unknown_molecule_id_uses_generic_parent_rule() -> None:
    result = standardize_for_gmc_geometry("eqvalan-like", "CCO.[Na+]")

    assert result.action == PARENT_SELECTED
    assert result.geometry_canonical_smiles == "CCO"


def test_repeated_execution_is_identical() -> None:
    source = _canonical(APTAZAPINE_SMILES)

    assert standardize_for_gmc_geometry("aptazapine", source) == (
        standardize_for_gmc_geometry("aptazapine", source)
    )


def test_standardization_version_is_frozen() -> None:
    result = standardize_for_gmc_geometry("connected", "CCO")

    assert GMC_STANDARDIZATION_VERSION == "parent-fragment-v1"
    assert result.standardization_version == GMC_STANDARDIZATION_VERSION


@pytest.mark.parametrize("smiles", ("", "not-a-smiles"))
def test_invalid_smiles_fails_explicitly(smiles: str) -> None:
    with pytest.raises(GMCStandardizationError, match="invalid_smiles") as exc_info:
        standardize_for_gmc_geometry("invalid", smiles)

    assert exc_info.value.status == "invalid_smiles"


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
