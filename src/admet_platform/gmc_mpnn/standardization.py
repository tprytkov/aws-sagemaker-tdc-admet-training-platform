"""GMC-MPNN-specific parent-fragment policy with source provenance preserved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from rdkit import Chem


GMC_STANDARDIZATION_VERSION: Final = "parent-fragment-v1"
StandardizationAction = Literal["unchanged", "parent_selected", "excluded_by_policy"]

UNCHANGED: Final = "unchanged"
PARENT_SELECTED: Final = "parent_selected"
EXCLUDED_BY_POLICY: Final = "excluded_by_policy"
FROZEN_EXCLUSION_REASONS: Final = {
    "eqvalan": "ambiguous_multi_active_mixture",
    "sultamicillin": "chemically_inconsistent_multicomponent_representation",
}


class GMCStandardizationError(ValueError):
    """A parent-fragment standardization failure with a stable status."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass(frozen=True)
class StandardizationResult:
    """Source identity and the separate representation approved for geometry."""

    molecule_id: str
    source_canonical_smiles: str
    geometry_canonical_smiles: str | None
    action: StandardizationAction
    fragment_count: int
    source_heavy_atom_count: int
    source_formal_charge: int
    parent_heavy_atom_count: int | None
    parent_formal_charge: int | None
    removed_fragment_smiles: tuple[str, ...]
    removed_fragment_heavy_atom_counts: tuple[int, ...]
    removed_fragment_formal_charges: tuple[int, ...]
    exclusion_reason: str
    standardization_version: str = GMC_STANDARDIZATION_VERSION


@dataclass(frozen=True)
class _Fragment:
    canonical_smiles: str
    heavy_atom_count: int
    formal_charge: int


def standardize_for_gmc_geometry(
    molecule_id: object,
    source_canonical_smiles: str,
) -> StandardizationResult:
    """Apply the frozen Phase-1 geometry representation policy to one source row.

    The source canonical SMILES is never replaced. Connected structures use that
    exact string for geometry. Ordinary disconnected structures derive a parent
    representation. Frozen molecule IDs are excluded before parent selection.
    """

    normalized_molecule_id = str(molecule_id).strip()
    molecule = _parse_source_smiles(source_canonical_smiles)
    fragments = _fragment_records(molecule)
    source_heavy_atom_count = sum(atom.GetAtomicNum() != 1 for atom in molecule.GetAtoms())
    source_formal_charge = int(Chem.GetFormalCharge(molecule))

    exclusion_reason = FROZEN_EXCLUSION_REASONS.get(normalized_molecule_id)
    if exclusion_reason is not None:
        return StandardizationResult(
            molecule_id=normalized_molecule_id,
            source_canonical_smiles=source_canonical_smiles,
            geometry_canonical_smiles=None,
            action=EXCLUDED_BY_POLICY,
            fragment_count=len(fragments),
            source_heavy_atom_count=source_heavy_atom_count,
            source_formal_charge=source_formal_charge,
            parent_heavy_atom_count=None,
            parent_formal_charge=None,
            removed_fragment_smiles=(),
            removed_fragment_heavy_atom_counts=(),
            removed_fragment_formal_charges=(),
            exclusion_reason=exclusion_reason,
        )

    if len(fragments) == 1:
        return StandardizationResult(
            molecule_id=normalized_molecule_id,
            source_canonical_smiles=source_canonical_smiles,
            geometry_canonical_smiles=source_canonical_smiles,
            action=UNCHANGED,
            fragment_count=1,
            source_heavy_atom_count=source_heavy_atom_count,
            source_formal_charge=source_formal_charge,
            parent_heavy_atom_count=source_heavy_atom_count,
            parent_formal_charge=source_formal_charge,
            removed_fragment_smiles=(),
            removed_fragment_heavy_atom_counts=(),
            removed_fragment_formal_charges=(),
            exclusion_reason="",
        )

    parent, removed = _select_parent_fragment(fragments)
    return StandardizationResult(
        molecule_id=normalized_molecule_id,
        source_canonical_smiles=source_canonical_smiles,
        geometry_canonical_smiles=parent.canonical_smiles,
        action=PARENT_SELECTED,
        fragment_count=len(fragments),
        source_heavy_atom_count=source_heavy_atom_count,
        source_formal_charge=source_formal_charge,
        parent_heavy_atom_count=parent.heavy_atom_count,
        parent_formal_charge=parent.formal_charge,
        removed_fragment_smiles=tuple(fragment.canonical_smiles for fragment in removed),
        removed_fragment_heavy_atom_counts=tuple(fragment.heavy_atom_count for fragment in removed),
        removed_fragment_formal_charges=tuple(fragment.formal_charge for fragment in removed),
        exclusion_reason="",
    )


def _parse_source_smiles(source_canonical_smiles: str) -> Chem.Mol:
    if not isinstance(source_canonical_smiles, str) or not source_canonical_smiles.strip():
        raise GMCStandardizationError(
            "invalid_smiles", "Source canonical SMILES must be a non-empty string."
        )
    molecule = Chem.MolFromSmiles(source_canonical_smiles)
    if molecule is None:
        raise GMCStandardizationError(
            "invalid_smiles", f"RDKit could not parse {source_canonical_smiles!r}."
        )
    return molecule


def _fragment_records(molecule: Chem.Mol) -> tuple[_Fragment, ...]:
    fragments = (
        _Fragment(
            canonical_smiles=Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True),
            heavy_atom_count=sum(atom.GetAtomicNum() != 1 for atom in fragment.GetAtoms()),
            formal_charge=int(Chem.GetFormalCharge(fragment)),
        )
        for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    )
    return tuple(
        sorted(
            fragments,
            key=lambda fragment: (
                fragment.canonical_smiles,
                fragment.heavy_atom_count,
                fragment.formal_charge,
            ),
        )
    )


def _select_parent_fragment(
    fragments: tuple[_Fragment, ...],
) -> tuple[_Fragment, tuple[_Fragment, ...]]:
    parent_index = min(
        range(len(fragments)),
        key=lambda index: (
            -fragments[index].heavy_atom_count,
            fragments[index].canonical_smiles,
        ),
    )
    parent = fragments[parent_index]
    removed = tuple(fragment for index, fragment in enumerate(fragments) if index != parent_index)
    return parent, removed


__all__ = [
    "EXCLUDED_BY_POLICY",
    "FROZEN_EXCLUSION_REASONS",
    "GMC_STANDARDIZATION_VERSION",
    "GMCStandardizationError",
    "PARENT_SELECTED",
    "StandardizationAction",
    "StandardizationResult",
    "UNCHANGED",
    "standardize_for_gmc_geometry",
]
