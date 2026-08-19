from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

from admet_platform.gmc_mpnn.geometry import GeometryResult, generate_deterministic_geometry
from admet_platform.gmc_mpnn.ggl import (
    ELEMENT_RADII_ANGSTROM,
    GGL_FEATURE_NAMES,
    KERNEL_PARAMETER_GRID,
    GGLConfig,
    GGLPreprocessingError,
    KernelParameters,
    build_ggl_matrices,
    compute_ggl_features,
    kernel_parameters_from_index,
)


SYNTHETIC_FIXTURES = (
    ("aromatic", "c1ccncc1", 6),
    ("stereochemical", "C[C@H](O)F", 4),
    ("charged", "C[NH2+]C", 3),
    ("amide", "CNC(C)=O", 5),
    ("flexible_aliphatic", "CCCCCC", 6),
)
GGL_REPEATABILITY_ATOL = 1e-12


@lru_cache(maxsize=None)
def _cached_geometry(smiles: str) -> GeometryResult:
    return generate_deterministic_geometry(smiles)


def test_released_element_radius_mapping_is_exact() -> None:
    expected = {
        1: 1.20,
        4: 1.53,
        5: 0.85,
        6: 1.70,
        7: 1.55,
        8: 1.52,
        9: 1.47,
        12: 1.73,
        14: 2.10,
        15: 1.80,
        16: 1.80,
        17: 1.75,
        23: 1.34,
        26: 1.26,
        27: 2.00,
        29: 1.28,
        30: 1.39,
        33: 1.85,
        34: 1.90,
        35: 1.85,
        44: 2.05,
        45: 2.00,
        51: 2.06,
        52: 1.40,
        53: 1.98,
        75: 2.05,
        76: 2.00,
        77: 2.00,
        78: 1.75,
        80: 1.50,
    }
    assert dict(ELEMENT_RADII_ANGSTROM) == expected


def test_released_kernel_grid_has_exact_order_and_size() -> None:
    assert len(KERNEL_PARAMETER_GRID) == 1_600
    assert kernel_parameters_from_index(1) == KernelParameters("exponential_kernel", 0.5, 0.5)
    assert kernel_parameters_from_index(40) == KernelParameters("exponential_kernel", 0.5, 20.0)
    assert kernel_parameters_from_index(41) == KernelParameters("exponential_kernel", 1.0, 0.5)
    assert kernel_parameters_from_index(800) == KernelParameters("exponential_kernel", 10.0, 20.0)
    assert kernel_parameters_from_index(801) == KernelParameters("lorentz_kernel", 0.5, 0.5)
    assert kernel_parameters_from_index(1_600) == KernelParameters("lorentz_kernel", 10.0, 20.0)


@pytest.mark.parametrize("kernel_type", ["exponential_kernel", "lorentz_kernel"])
def test_two_carbon_literal_reference_calculation(kernel_type: str) -> None:
    coordinates = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atomic_numbers = np.asarray([6, 6])
    config = GGLConfig(kernel_type=kernel_type, tau=1.0, kappa=2.0)  # type: ignore[arg-type]

    matrices = build_ggl_matrices(coordinates, atomic_numbers, config=config)
    ratio_squared = (2.0 / (1.0 * (1.70 + 1.70))) ** 2.0
    expected_weight = (
        math.exp(-ratio_squared)
        if kernel_type == "exponential_kernel"
        else 1.0 / (1.0 + ratio_squared)
    )
    expected_features = np.asarray(
        [
            [expected_weight, expected_weight, expected_weight, expected_weight, expected_weight, 0.0],
            [expected_weight, expected_weight, expected_weight, expected_weight, expected_weight, 0.0],
        ]
    )

    np.testing.assert_allclose(matrices.distances, [[0.0, 2.0], [2.0, 0.0]], atol=0.0)
    np.testing.assert_allclose(matrices.pairwise_radii, np.full((2, 2), 3.4), atol=0.0)
    np.testing.assert_allclose(matrices.row_summed_radius_thresholds, [6.8, 6.8], atol=0.0)
    np.testing.assert_array_equal(matrices.retained_mask, [[False, True], [True, False]])
    result = compute_ggl_features(
        coordinates,
        atomic_numbers,
        geometry_fingerprint="literal-two-carbon-geometry",
        config=config,
    )
    np.testing.assert_allclose(result.features, expected_features, rtol=1e-15, atol=1e-15)


def test_unusual_row_summed_radius_mask_and_empty_behavior_are_preserved() -> None:
    coordinates = np.asarray([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    atomic_numbers = np.asarray([6, 6])
    config = GGLConfig(kernel_type="exponential_kernel", tau=1.0, kappa=2.0)

    matrices = build_ggl_matrices(coordinates, atomic_numbers, config=config)

    assert matrices.distances[0, 1] == 7.0
    assert matrices.row_summed_radius_thresholds[0] == pytest.approx(6.8)
    assert not matrices.retained_mask.any()
    raw = compute_ggl_features(
        coordinates,
        atomic_numbers,
        geometry_fingerprint="row-sum-mask-reference",
        config=config,
        require_finite=False,
    )
    np.testing.assert_array_equal(raw.features[:, 2], [0.0, 0.0])
    assert np.isnan(raw.features[:, [0, 1, 3, 4, 5]]).all()
    assert raw.ggl_status == "nonfinite_features"
    assert raw.ggl_fingerprint is None
    with pytest.raises(GGLPreprocessingError, match="nonfinite_features"):
        compute_ggl_features(
            coordinates,
            atomic_numbers,
            geometry_fingerprint="row-sum-mask-reference",
            config=config,
        )


@pytest.mark.parametrize(("fixture_name", "smiles", "heavy_count"), SYNTHETIC_FIXTURES)
def test_five_fixture_ggl_is_finite_shaped_and_same_seed_deterministic(
    fixture_name: str, smiles: str, heavy_count: int
) -> None:
    first_geometry = generate_deterministic_geometry(smiles)
    second_geometry = generate_deterministic_geometry(smiles)
    first = compute_ggl_features(
        first_geometry.coordinates,
        np.asarray(first_geometry.heavy_atom_atomic_numbers),
        geometry_fingerprint=first_geometry.geometry_fingerprint,
    )
    second = compute_ggl_features(
        second_geometry.coordinates,
        np.asarray(second_geometry.heavy_atom_atomic_numbers),
        geometry_fingerprint=second_geometry.geometry_fingerprint,
    )

    assert fixture_name
    assert first.features.shape == second.features.shape == (heavy_count, 6)
    assert first.feature_names == second.feature_names == GGL_FEATURE_NAMES
    assert np.isfinite(first.features).all()
    np.testing.assert_allclose(
        first.features, second.features, rtol=1e-12, atol=GGL_REPEATABILITY_ATOL
    )
    assert first.ggl_fingerprint == second.ggl_fingerprint
    assert first.ggl_fingerprint is not None
    assert len(first.ggl_fingerprint) == 64


@pytest.mark.parametrize(("fixture_name", "smiles", "heavy_count"), SYNTHETIC_FIXTURES)
def test_translation_and_rotation_invariance(
    fixture_name: str, smiles: str, heavy_count: int
) -> None:
    geometry = _cached_geometry(smiles)
    numbers = np.asarray(geometry.heavy_atom_atomic_numbers)
    reference = compute_ggl_features(
        geometry.coordinates,
        numbers,
        geometry_fingerprint=geometry.geometry_fingerprint,
    )
    translated_coordinates = geometry.coordinates + np.asarray([13.25, -7.5, 2.75])
    angle = math.radians(37.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    # Apply the rotation component-wise. This avoids invoking a BLAS/OpenMP matrix
    # multiplication runtime in the Windows test environment after RDKit has loaded.
    rotated_coordinates = np.column_stack(
        (
            cosine * geometry.coordinates[:, 0] - sine * geometry.coordinates[:, 1],
            sine * geometry.coordinates[:, 0] + cosine * geometry.coordinates[:, 1],
            geometry.coordinates[:, 2],
        )
    )
    translated = compute_ggl_features(
        translated_coordinates,
        numbers,
        geometry_fingerprint=geometry.geometry_fingerprint,
    )
    rotated = compute_ggl_features(
        rotated_coordinates,
        numbers,
        geometry_fingerprint=geometry.geometry_fingerprint,
    )

    assert fixture_name and reference.features.shape == (heavy_count, 6)
    np.testing.assert_allclose(
        translated.features, reference.features, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(rotated.features, reference.features, rtol=1e-12, atol=1e-12)
    # Geometry fingerprints intentionally identify preprocessing coordinates, so test-only
    # transformed matrices reuse the original only to isolate GGL invariance.
    assert translated.ggl_fingerprint == reference.ggl_fingerprint
    assert rotated.ggl_fingerprint == reference.ggl_fingerprint


def test_ggl_fingerprint_depends_on_geometry_provenance() -> None:
    geometry = _cached_geometry("CCO")
    numbers = np.asarray(geometry.heavy_atom_atomic_numbers)
    first = compute_ggl_features(
        geometry.coordinates,
        numbers,
        geometry_fingerprint=geometry.geometry_fingerprint,
    )
    changed = compute_ggl_features(
        geometry.coordinates,
        numbers,
        geometry_fingerprint="different-geometry-provenance",
    )

    np.testing.assert_array_equal(first.features, changed.features)
    assert first.ggl_fingerprint != changed.ggl_fingerprint


@pytest.mark.parametrize(
    ("coordinates", "atomic_numbers", "status"),
    [
        (np.zeros((2, 2)), np.asarray([6, 6]), "invalid_coordinate_shape"),
        (np.asarray([[0.0, 0.0, np.nan]]), np.asarray([6]), "nonfinite_coordinates"),
        (np.zeros((2, 3)), np.asarray([6]), "atom_count_mismatch"),
        (np.zeros((1, 3)), np.asarray([1]), "hydrogen_not_allowed"),
        (np.zeros((1, 3)), np.asarray([10]), "unsupported_element"),
    ],
)
def test_invalid_inputs_fail_with_explicit_status(
    coordinates: np.ndarray, atomic_numbers: np.ndarray, status: str
) -> None:
    with pytest.raises(GGLPreprocessingError, match=status) as exc_info:
        compute_ggl_features(
            coordinates,
            atomic_numbers,
            geometry_fingerprint="synthetic-invalid-input",
        )
    assert exc_info.value.status == status


def test_single_heavy_atom_reports_nonfinite_final_features() -> None:
    with pytest.raises(GGLPreprocessingError, match="nonfinite_features") as exc_info:
        compute_ggl_features(
            np.zeros((1, 3)),
            np.asarray([6]),
            geometry_fingerprint="single-carbon",
        )
    assert exc_info.value.status == "nonfinite_features"


def test_raw_features_are_unscaled_and_in_exact_order() -> None:
    assert GGL_FEATURE_NAMES == (
        "minimum",
        "maximum",
        "sum",
        "mean",
        "median",
        "population_standard_deviation",
    )
    assert not any("scale" in name or "normal" in name for name in GGL_FEATURE_NAMES)
