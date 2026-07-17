"""Safe, non-chiral Murcko scaffold assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class ScaffoldResult:
    """A scaffold string and whether stereo removal was required for assignment."""

    scaffold: str
    used_stereo_fallback: bool = False


def safe_murcko_scaffold(molecule_or_smiles: Chem.Mol | str) -> ScaffoldResult:
    """Generate a non-chiral Murcko scaffold with a narrow bad-stereo fallback.

    The copied molecule is used only to calculate the split group. The caller's
    molecule, original SMILES, and normally generated canonical SMILES are not changed.
    """

    molecule = (
        Chem.MolFromSmiles(molecule_or_smiles)
        if isinstance(molecule_or_smiles, str)
        else molecule_or_smiles
    )
    if molecule is None:
        raise ValueError("SMILES could not be parsed for scaffold assignment.")
    try:
        return ScaffoldResult(
            MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        )
    except RuntimeError as exc:
        if "bad bond stereo" not in str(exc).lower():
            raise
        scaffold_molecule = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(scaffold_molecule)
        return ScaffoldResult(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=scaffold_molecule, includeChirality=False
            ),
            used_stereo_fallback=True,
        )


__all__ = ["ScaffoldResult", "safe_murcko_scaffold"]
