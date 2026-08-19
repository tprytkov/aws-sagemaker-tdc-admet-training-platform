"""Deterministic, label-free RDKit geometry for GMC-MPNN Phase 1.

This is a project-owned implementation of the geometry contract documented in
``docs/gmc_mpnn_geometry_ggl_plan.md``. It does not reproduce the authors'
undocumented OMEGA-first conformer workflow. It intentionally stops at one
selected heavy-atom coordinate matrix and has no dataset or model integration.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Final, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


GEOMETRY_PREPROCESSING_VERSION: Final = "rdkit-etkdgv3-mmff94s-v1"
GEOMETRY_FINGERPRINT_DECIMALS: Final = 8
ORIGINAL_INDEX_PROPERTY: Final = "_GMCOriginalHeavyAtomIndex"


class GeometryError(RuntimeError):
    """A deterministic geometry failure with a stable machine-readable status."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass(frozen=True)
class GeometryConfig:
    """Fully resolved settings for deterministic conformer generation."""

    seed: int = 13
    num_conformers: int = 20
    prune_rms_threshold: float = 0.5
    embed_max_iterations: int = 1_000
    optimization_max_iterations: int = 1_000
    num_threads: int = 1
    enforce_chirality: bool = True
    use_random_coordinates: bool = False
    use_basic_knowledge: bool = True
    use_experimental_torsions: bool = True
    use_symmetry_for_pruning: bool = True
    only_heavy_atoms_for_rms: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("Geometry seed must be in the non-negative signed 32-bit range.")
        if self.num_conformers < 1:
            raise ValueError("At least one conformer must be requested.")
        if self.prune_rms_threshold < 0:
            raise ValueError("Conformer pruning threshold must be non-negative.")
        if self.embed_max_iterations < 1 or self.optimization_max_iterations < 1:
            raise ValueError("Embedding and optimization iteration limits must be positive.")
        if self.num_threads != 1:
            raise ValueError("Deterministic Phase-1 geometry requires num_threads=1.")


@dataclass(frozen=True)
class AtomIdentity:
    """Identity fields that must survive hydrogen addition and geometry operations."""

    original_index: int
    atomic_number: int
    formal_charge: int
    aromatic: bool
    isotope: int


@dataclass(frozen=True)
class OptimizationRecord:
    """One force-field result associated with an RDKit conformer ID."""

    conformer_id: int
    converged: bool
    energy: float
    status_code: int


@dataclass(frozen=True)
class GeometryResult:
    """One selected conformer and its identity-bearing provenance."""

    input_smiles: str
    canonical_isomeric_smiles: str
    seed: int
    embedding_seed: int
    etkdg_version: str
    requested_conformer_count: int
    generated_conformer_count: int
    optimization_method: str
    selected_conformer_id: int
    selected_energy: float
    convergence_status: str
    heavy_atom_count: int
    heavy_atom_rdkit_indices: tuple[int, ...]
    heavy_atom_atomic_numbers: tuple[int, ...]
    heavy_atom_formal_charges: tuple[int, ...]
    coordinates: np.ndarray
    rdkit_version: str
    geometry_status: str
    geometry_fingerprint: str
    config: GeometryConfig


def generate_deterministic_geometry(
    smiles: str, *, config: GeometryConfig | None = None
) -> GeometryResult:
    """Generate one lowest-energy converged conformer in canonical RDKit atom order.

    The caller-supplied seed is mixed with canonical isomeric SMILES so molecule
    execution order cannot affect embedding. MMFF94s is used only when every atom
    is parameterized. UFF is the recorded fallback only when MMFF94s is unavailable.
    Unconverged conformers are never eligible for selection.
    """

    resolved = config or GeometryConfig()
    molecule = _canonical_molecule(smiles)
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise GeometryError(
            "disconnected_fragment",
            "Phase-1 geometry does not assign arbitrary relative coordinates to fragments.",
        )

    expected_identity = _mark_and_capture_heavy_atom_identity(molecule)
    try:
        molecule_h = Chem.AddHs(molecule, addCoords=False)
    except Exception as exc:  # pragma: no cover - defensive RDKit boundary
        raise GeometryError("hydrogen_addition_failed", str(exc)) from exc
    _verify_heavy_atom_identity(molecule_h, expected_identity)

    embedding_seed = _derive_embedding_seed(canonical_smiles, resolved.seed)
    parameters = _etkdgv3_parameters(resolved, embedding_seed)
    try:
        conformer_ids = tuple(
            int(value)
            for value in AllChem.EmbedMultipleConfs(
                molecule_h,
                numConfs=resolved.num_conformers,
                params=parameters,
            )
        )
    except Exception as exc:
        raise GeometryError("embedding_failed", str(exc)) from exc
    if not conformer_ids:
        raise GeometryError("no_conformers_generated", "ETKDGv3 generated no conformers.")

    method, records = _optimize_conformers(molecule_h, conformer_ids, resolved)
    selected = _select_lowest_energy_converged(records)
    _verify_heavy_atom_identity(molecule_h, expected_identity)
    coordinates = _extract_heavy_atom_coordinates(
        molecule_h, selected.conformer_id, expected_identity
    )
    if coordinates.shape != (len(expected_identity), 3):
        raise GeometryError("atom_alignment_failed", "Heavy-atom coordinate shape changed.")
    if not np.isfinite(coordinates).all():
        raise GeometryError("nonfinite_coordinates", "Selected coordinates contain NaN or Inf.")

    geometry_status = "success_mmff94s" if method == "MMFF94s" else "success_uff_fallback"
    fingerprint = _geometry_fingerprint(
        canonical_smiles=canonical_smiles,
        config=resolved,
        embedding_seed=embedding_seed,
        optimization_method=method,
        identities=expected_identity,
        coordinates=coordinates,
    )
    return GeometryResult(
        input_smiles=smiles,
        canonical_isomeric_smiles=canonical_smiles,
        seed=resolved.seed,
        embedding_seed=embedding_seed,
        etkdg_version="ETKDGv3",
        requested_conformer_count=resolved.num_conformers,
        generated_conformer_count=len(conformer_ids),
        optimization_method=method,
        selected_conformer_id=selected.conformer_id,
        selected_energy=selected.energy,
        convergence_status="converged",
        heavy_atom_count=len(expected_identity),
        heavy_atom_rdkit_indices=tuple(atom.original_index for atom in expected_identity),
        heavy_atom_atomic_numbers=tuple(atom.atomic_number for atom in expected_identity),
        heavy_atom_formal_charges=tuple(atom.formal_charge for atom in expected_identity),
        coordinates=coordinates,
        rdkit_version=rdBase.rdkitVersion,
        geometry_status=geometry_status,
        geometry_fingerprint=fingerprint,
        config=resolved,
    )


def _canonical_molecule(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise GeometryError("invalid_smiles", "SMILES must be a non-empty string.")
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise GeometryError("invalid_smiles", f"RDKit could not parse {smiles!r}.")
    canonical = Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # pragma: no cover - defensive canonical round-trip boundary
        raise GeometryError("invalid_smiles", "Canonical isomeric SMILES could not be reparsed.")
    return molecule


def _mark_and_capture_heavy_atom_identity(molecule: Chem.Mol) -> tuple[AtomIdentity, ...]:
    identities: list[AtomIdentity] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        original_index = atom.GetIdx()
        atom.SetIntProp(ORIGINAL_INDEX_PROPERTY, original_index)
        identities.append(_atom_identity(atom, original_index))
    if not identities:
        raise GeometryError("no_heavy_atoms", "Molecule has no heavy atoms.")
    return tuple(identities)


def _atom_identity(atom: Chem.Atom, original_index: int) -> AtomIdentity:
    return AtomIdentity(
        original_index=original_index,
        atomic_number=atom.GetAtomicNum(),
        formal_charge=atom.GetFormalCharge(),
        aromatic=atom.GetIsAromatic(),
        isotope=atom.GetIsotope(),
    )


def _verify_heavy_atom_identity(
    molecule: Chem.Mol, expected: Sequence[AtomIdentity]
) -> None:
    observed: list[AtomIdentity] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if not atom.HasProp(ORIGINAL_INDEX_PROPERTY):
            raise GeometryError(
                "atom_alignment_failed", "A heavy atom lost its original-index marker."
            )
        original_index = atom.GetIntProp(ORIGINAL_INDEX_PROPERTY)
        if atom.GetIdx() != original_index:
            raise GeometryError(
                "atom_alignment_failed",
                "RDKit heavy-atom index no longer matches its original canonical index.",
            )
        observed.append(_atom_identity(atom, original_index))
    if tuple(observed) != tuple(expected):
        raise GeometryError(
            "atom_alignment_failed",
            "Atomic number, charge, aromaticity, isotope, or original order changed.",
        )


def _derive_embedding_seed(canonical_smiles: str, base_seed: int) -> int:
    payload = (
        f"{GEOMETRY_PREPROCESSING_VERSION}\n{base_seed}\n{canonical_smiles}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _etkdgv3_parameters(config: GeometryConfig, embedding_seed: int):
    factory = getattr(AllChem, "ETKDGv3", None)
    if factory is None:
        raise GeometryError("etkdgv3_unavailable", "This RDKit build does not provide ETKDGv3.")
    parameters = factory()
    parameters.randomSeed = embedding_seed
    parameters.numThreads = config.num_threads
    parameters.pruneRmsThresh = config.prune_rms_threshold
    parameters.maxIterations = config.embed_max_iterations
    parameters.enforceChirality = config.enforce_chirality
    parameters.useRandomCoords = config.use_random_coordinates
    parameters.useBasicKnowledge = config.use_basic_knowledge
    parameters.useExpTorsionAnglePrefs = config.use_experimental_torsions
    parameters.useSymmetryForPruning = config.use_symmetry_for_pruning
    parameters.onlyHeavyAtomsForRMS = config.only_heavy_atoms_for_rms
    parameters.clearConfs = True
    return parameters


def _optimize_conformers(
    molecule_h: Chem.Mol,
    conformer_ids: Sequence[int],
    config: GeometryConfig,
) -> tuple[str, tuple[OptimizationRecord, ...]]:
    try:
        mmff_available = bool(AllChem.MMFFHasAllMoleculeParams(molecule_h))
    except Exception as exc:  # pragma: no cover - defensive RDKit boundary
        raise GeometryError("mmff_unavailable", str(exc)) from exc

    if mmff_available:
        properties = AllChem.MMFFGetMoleculeProperties(molecule_h, mmffVariant="MMFF94s")
        if properties is None:
            raise GeometryError(
                "mmff_unavailable", "MMFF94s parameter check passed but properties are unavailable."
            )
        try:
            raw_results = AllChem.MMFFOptimizeMoleculeConfs(
                molecule_h,
                numThreads=config.num_threads,
                maxIters=config.optimization_max_iterations,
                mmffVariant="MMFF94s",
            )
        except Exception as exc:
            raise GeometryError("optimization_failed", f"MMFF94s failed: {exc}") from exc
        method = "MMFF94s"
    else:
        try:
            uff_available = bool(AllChem.UFFHasAllMoleculeParams(molecule_h))
        except Exception as exc:  # pragma: no cover - defensive RDKit boundary
            raise GeometryError("uff_unavailable", str(exc)) from exc
        if not uff_available:
            raise GeometryError(
                "uff_unavailable", "Neither MMFF94s nor UFF covers every atom in the molecule."
            )
        try:
            raw_results = AllChem.UFFOptimizeMoleculeConfs(
                molecule_h,
                numThreads=config.num_threads,
                maxIters=config.optimization_max_iterations,
            )
        except Exception as exc:
            raise GeometryError("optimization_failed", f"UFF failed: {exc}") from exc
        method = "UFF"

    if len(raw_results) != len(conformer_ids):
        raise GeometryError(
            "optimization_failed", "Force-field results do not match generated conformers."
        )
    records = tuple(
        OptimizationRecord(
            conformer_id=int(conformer_id),
            converged=int(status_code) == 0 and math.isfinite(float(energy)),
            energy=float(energy),
            status_code=int(status_code),
        )
        for conformer_id, (status_code, energy) in zip(conformer_ids, raw_results, strict=True)
    )
    return method, records


def _select_lowest_energy_converged(
    records: Sequence[OptimizationRecord],
) -> OptimizationRecord:
    eligible = [record for record in records if record.converged and math.isfinite(record.energy)]
    if not eligible:
        raise GeometryError(
            "optimization_failed", "No generated conformer converged to a finite energy."
        )
    return min(eligible, key=lambda record: (record.energy, record.conformer_id))


def _extract_heavy_atom_coordinates(
    molecule_h: Chem.Mol,
    conformer_id: int,
    expected: Sequence[AtomIdentity],
) -> np.ndarray:
    conformer = molecule_h.GetConformer(conformer_id)
    coordinates: list[tuple[float, float, float]] = []
    observed_indices: list[int] = []
    for atom in molecule_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if not atom.HasProp(ORIGINAL_INDEX_PROPERTY):
            raise GeometryError("atom_alignment_failed", "Heavy atom has no original index.")
        observed_indices.append(atom.GetIntProp(ORIGINAL_INDEX_PROPERTY))
        point = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append((float(point.x), float(point.y), float(point.z)))
    expected_indices = [atom.original_index for atom in expected]
    if observed_indices != expected_indices:
        raise GeometryError("atom_alignment_failed", "Heavy-atom extraction order changed.")
    return np.asarray(coordinates, dtype=np.float64)


def _geometry_fingerprint(
    *,
    canonical_smiles: str,
    config: GeometryConfig,
    embedding_seed: int,
    optimization_method: str,
    identities: Sequence[AtomIdentity],
    coordinates: np.ndarray,
) -> str:
    """Hash selected coordinates rounded to 8 decimals with resolved settings.

    Rounding prevents insignificant text/serialization noise from changing the
    provenance fingerprint. The fingerprint identifies a preprocessing result,
    not molecular identity.
    """

    payload = {
        "canonical_isomeric_smiles": canonical_smiles,
        "config": asdict(config),
        "coordinates_angstrom": np.round(
            coordinates, decimals=GEOMETRY_FINGERPRINT_DECIMALS
        ).tolist(),
        "embedding_seed": embedding_seed,
        "formal_charges": [atom.formal_charge for atom in identities],
        "heavy_atom_atomic_numbers": [atom.atomic_number for atom in identities],
        "optimization_method": optimization_method,
        "preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GEOMETRY_FINGERPRINT_DECIMALS",
    "GEOMETRY_PREPROCESSING_VERSION",
    "AtomIdentity",
    "GeometryConfig",
    "GeometryError",
    "GeometryResult",
    "OptimizationRecord",
    "generate_deterministic_geometry",
]
