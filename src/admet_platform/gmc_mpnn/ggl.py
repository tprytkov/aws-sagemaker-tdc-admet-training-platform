"""RDKit-order implementation of the released six-column GMC GGL calculation.

This clean-room implementation follows the formulas and literal mask behavior in
MathIntelligence/GMC-MPNN-BBBP at revision
ae080431950832be43e51f7cd9b0f7d4203e2267, specifically the public
``utils/ggl_ligand.py`` algorithm and ``utils/ligand_SYBYL_atom_types.csv``
radius values. No upstream source code is copied or vendored. Fine-grained SYBYL
subtypes are omitted because every released subtype of an element has one common
radius; RDKit atomic number is the project identity authority.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

import numpy as np


GGL_PREPROCESSING_VERSION: Final = "released-pooled-six-feature-v1"
ELEMENT_RADIUS_MAPPING_VERSION: Final = (
    "gmc-mpnn-ae080431950832be43e51f7cd9b0f7d4203e2267-element-radii-v1"
)
GGL_FINGERPRINT_DECIMALS: Final = 12
DEFAULT_CUTOFF_ANGSTROM: Final = 12.0
GGL_FEATURE_NAMES: Final = (
    "minimum",
    "maximum",
    "sum",
    "mean",
    "median",
    "population_standard_deviation",
)
KernelType = Literal["exponential_kernel", "lorentz_kernel"]


# Project-owned atomic-number mapping of the exact released numeric radii. Hydrogen
# is retained here for provenance but rejected by the heavy-atom GGL API because the
# released parser filters exact type H before forming its distance matrix.
_ELEMENT_RADII_ANGSTROM = {
    1: 1.20,   # H (filtered)
    4: 1.53,   # Be
    5: 0.85,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    12: 1.73,  # Mg
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    23: 1.34,  # V
    26: 1.26,  # Fe
    27: 2.00,  # Co
    29: 1.28,  # Cu
    30: 1.39,  # Zn
    33: 1.85,  # As
    34: 1.90,  # Se
    35: 1.85,  # Br
    44: 2.05,  # Ru
    45: 2.00,  # Rh
    51: 2.06,  # Sb
    52: 1.40,  # Te
    53: 1.98,  # I
    75: 2.05,  # Re
    76: 2.00,  # Os
    77: 2.00,  # Ir
    78: 1.75,  # Pt
    80: 1.50,  # Hg
}
ELEMENT_RADII_ANGSTROM: Final[Mapping[int, float]] = MappingProxyType(
    _ELEMENT_RADII_ANGSTROM
)


@dataclass(frozen=True)
class KernelParameters:
    """One row in the released 1,600-member kernel grid."""

    kernel_type: KernelType
    tau: float
    kappa: float


KERNEL_PARAMETER_GRID: Final = tuple(
    KernelParameters(kernel_type=kernel_type, tau=tau_index / 2, kappa=power_index / 2)
    for kernel_type in ("exponential_kernel", "lorentz_kernel")
    for tau_index in range(1, 21)
    for power_index in range(1, 41)
)


class GGLPreprocessingError(ValueError):
    """A GGL validation failure with a stable machine-readable status."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass(frozen=True)
class GGLConfig:
    """Resolved released-kernel settings for one raw feature matrix."""

    kernel_type: KernelType = "exponential_kernel"
    tau: float = 0.5
    kappa: float = 17.0
    cutoff_angstrom: float = DEFAULT_CUTOFF_ANGSTROM

    def __post_init__(self) -> None:
        parameters = KernelParameters(self.kernel_type, float(self.tau), float(self.kappa))
        if parameters not in KERNEL_PARAMETER_GRID:
            raise ValueError("Kernel type, tau, and kappa must be in the released 1,600-row grid.")
        if not np.isfinite(self.cutoff_angstrom) or self.cutoff_angstrom <= 0:
            raise ValueError("GGL cutoff must be a positive finite Angstrom value.")


@dataclass(frozen=True)
class GGLMatrices:
    """Auditable intermediate matrices for literal-reference tests."""

    distances: np.ndarray
    pairwise_radii: np.ndarray
    row_summed_radius_thresholds: np.ndarray
    kernel_weights: np.ndarray
    retained_mask: np.ndarray


@dataclass(frozen=True)
class GGLResult:
    """One raw, unscaled six-feature matrix and its provenance fingerprint."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    geometry_fingerprint: str
    ggl_fingerprint: str | None
    ggl_status: str
    config: GGLConfig
    preprocessing_version: str
    element_radius_mapping_version: str


def kernel_parameters_from_index(index: int) -> KernelParameters:
    """Return the released user-facing one-based kernel-grid row."""

    if not 1 <= index <= len(KERNEL_PARAMETER_GRID):
        raise ValueError("Kernel index must be between 1 and 1600 inclusive.")
    return KERNEL_PARAMETER_GRID[index - 1]


def build_ggl_matrices(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    config: GGLConfig | None = None,
) -> GGLMatrices:
    """Build distances, radii, weights, and the literal released retained mask."""

    resolved = config or GGLConfig()
    coords, numbers = _validated_inputs(coordinates, atomic_numbers)
    radii = np.asarray([ELEMENT_RADII_ANGSTROM[int(number)] for number in numbers])
    differences = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    pairwise_radii = radii[:, np.newaxis] + radii[np.newaxis, :]

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scaled_power = (distances / (resolved.tau * pairwise_radii)) ** resolved.kappa
        if resolved.kernel_type == "exponential_kernel":
            raw_weights = np.exp(-scaled_power)
        else:
            raw_weights = 1.0 / (1.0 + scaled_power)

    # This intentionally preserves the upstream implementation's unusual row-summed
    # radius threshold. It is not changed to a pairwise covalent-radius comparison.
    row_thresholds = pairwise_radii.sum(axis=1)
    row_sum_mask = distances >= row_thresholds[:, np.newaxis]
    cutoff_mask = distances > resolved.cutoff_angstrom
    diagonal_mask = np.eye(len(numbers), dtype=bool)
    finite_mask = np.isfinite(raw_weights)
    retained_mask = ~(row_sum_mask | cutoff_mask | diagonal_mask) & finite_mask
    kernel_weights = np.where(retained_mask, raw_weights, np.nan)
    return GGLMatrices(
        distances=distances,
        pairwise_radii=pairwise_radii,
        row_summed_radius_thresholds=row_thresholds,
        kernel_weights=kernel_weights,
        retained_mask=retained_mask,
    )


def compute_ggl_features(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    geometry_fingerprint: str,
    config: GGLConfig | None = None,
    require_finite: bool = True,
) -> GGLResult:
    """Compute raw `[n_heavy_atoms, 6]` source-faithful GGL features.

    Empty neighborhoods retain the upstream numerical behavior: sum is zero and
    the other five statistics are NaN. By default that condition is then reported
    as ``nonfinite_features`` instead of silently reaching later model code.
    ``require_finite=False`` is available only for explicit reference/failure tests.
    """

    if not isinstance(geometry_fingerprint, str) or not geometry_fingerprint:
        raise GGLPreprocessingError(
            "invalid_geometry_fingerprint", "A non-empty geometry fingerprint is required."
        )
    resolved = config or GGLConfig()
    matrices = build_ggl_matrices(coordinates, atomic_numbers, config=resolved)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_min = np.nanmin(matrices.kernel_weights, axis=1)
        row_max = np.nanmax(matrices.kernel_weights, axis=1)
        row_sum = np.nansum(matrices.kernel_weights, axis=1)
        row_mean = np.nanmean(matrices.kernel_weights, axis=1)
        row_median = np.nanmedian(matrices.kernel_weights, axis=1)
        row_std = np.nanstd(matrices.kernel_weights, axis=1, ddof=0)
    features = np.column_stack((row_min, row_max, row_sum, row_mean, row_median, row_std))
    finite = bool(np.isfinite(features).all())
    if require_finite and not finite:
        empty_rows = np.flatnonzero(~matrices.retained_mask.any(axis=1)).tolist()
        raise GGLPreprocessingError(
            "nonfinite_features",
            f"Final GGL matrix contains NaN/Inf; empty-neighborhood rows: {empty_rows}.",
        )
    fingerprint = (
        _ggl_fingerprint(
            geometry_fingerprint=geometry_fingerprint,
            config=resolved,
            features=features,
        )
        if finite
        else None
    )
    return GGLResult(
        features=features,
        feature_names=GGL_FEATURE_NAMES,
        geometry_fingerprint=geometry_fingerprint,
        ggl_fingerprint=fingerprint,
        ggl_status="success" if finite else "nonfinite_features",
        config=resolved,
        preprocessing_version=GGL_PREPROCESSING_VERSION,
        element_radius_mapping_version=ELEMENT_RADIUS_MAPPING_VERSION,
    )


def _validated_inputs(
    coordinates: np.ndarray, atomic_numbers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        coords = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GGLPreprocessingError(
            "invalid_coordinate_shape", "Coordinates must be a numeric [n_atoms, 3] array."
        ) from exc
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 1:
        raise GGLPreprocessingError(
            "invalid_coordinate_shape", "Coordinates must have shape [n_atoms, 3]."
        )
    if not np.isfinite(coords).all():
        raise GGLPreprocessingError("nonfinite_coordinates", "Coordinates contain NaN or Inf.")

    raw_numbers = np.asarray(atomic_numbers)
    if raw_numbers.ndim != 1 or raw_numbers.shape[0] != coords.shape[0]:
        raise GGLPreprocessingError(
            "atom_count_mismatch", "Atomic numbers must be one-dimensional and match coordinates."
        )
    try:
        numeric_numbers = raw_numbers.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise GGLPreprocessingError(
            "invalid_atomic_number", "Atomic numbers must be finite integers."
        ) from exc
    if not np.isfinite(numeric_numbers).all() or not np.equal(
        numeric_numbers, np.floor(numeric_numbers)
    ).all():
        raise GGLPreprocessingError(
            "invalid_atomic_number", "Atomic numbers must be finite integers."
        )
    numbers = numeric_numbers.astype(np.int64)
    if (numbers == 1).any():
        raise GGLPreprocessingError(
            "hydrogen_not_allowed", "GGL input must contain heavy atoms only."
        )
    unsupported = sorted({int(number) for number in numbers if number not in ELEMENT_RADII_ANGSTROM})
    if unsupported:
        raise GGLPreprocessingError(
            "unsupported_element", f"No released radius exists for atomic numbers {unsupported}."
        )
    return coords, numbers


def _ggl_fingerprint(
    *, geometry_fingerprint: str, config: GGLConfig, features: np.ndarray
) -> str:
    payload = {
        "config": asdict(config),
        "element_radius_mapping_version": ELEMENT_RADIUS_MAPPING_VERSION,
        "feature_names": GGL_FEATURE_NAMES,
        "features": np.round(features, decimals=GGL_FINGERPRINT_DECIMALS).tolist(),
        "geometry_fingerprint": geometry_fingerprint,
        "preprocessing_version": GGL_PREPROCESSING_VERSION,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_CUTOFF_ANGSTROM",
    "ELEMENT_RADII_ANGSTROM",
    "ELEMENT_RADIUS_MAPPING_VERSION",
    "GGL_FEATURE_NAMES",
    "GGL_FINGERPRINT_DECIMALS",
    "GGL_PREPROCESSING_VERSION",
    "KERNEL_PARAMETER_GRID",
    "GGLConfig",
    "GGLMatrices",
    "GGLPreprocessingError",
    "GGLResult",
    "KernelParameters",
    "build_ggl_matrices",
    "compute_ggl_features",
    "kernel_parameters_from_index",
]
