from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_platform.data.regression_transforms import (
    FittedRegressionTransform,
    fit_regression_target_transform,
)


def test_normalization_is_fitted_from_train_only_and_preserves_originals() -> None:
    train = np.array([1.0, 2.0, 4.0, 8.0])
    validation = pd.DataFrame({"target_original": [1000.0, 2000.0]})
    fitted = fit_regression_target_transform(
        train,
        endpoint_id="vdss_lombardo",
        units="L/kg",
        transform="log10",
    )

    normalized_train = fitted.transform_values(train)
    transformed_validation = fitted.transform_frame(validation)

    assert normalized_train.mean() == pytest.approx(0.0, abs=1e-12)
    assert normalized_train.std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert fitted.train_row_count == len(train)
    assert fitted.fit_split == "train"
    assert transformed_validation["target_original"].tolist() == [1000.0, 2000.0]
    assert "target_normalized" in transformed_validation
    assert fitted.transformed_train_mean == pytest.approx(np.log10(train).mean())


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("identity", [-3.0, -1.0, 2.0, 5.0]),
        ("log10", [0.01, 0.1, 1.0, 100.0]),
        ("log1p", [0.0, 1.0, 4.0, 20.0]),
        ("logit_percent", [5.0, 25.0, 75.0, 99.0]),
    ],
)
def test_transform_inverse_round_trip(kind: str, values: list[float]) -> None:
    fitted = fit_regression_target_transform(
        values,
        endpoint_id="endpoint",
        units="original units",
        transform=kind,
    )

    restored = fitted.inverse_values(fitted.transform_values(values))

    assert restored == pytest.approx(values)


def test_metadata_round_trip_contains_only_train_fit_statistics(tmp_path: Path) -> None:
    fitted = fit_regression_target_transform(
        [10.0, 20.0, 30.0],
        endpoint_id="ppbr_az",
        units="percent bound",
    )
    path = tmp_path / "ppbr_transform.json"

    fitted.save(path)
    restored = FittedRegressionTransform.load(path)

    assert restored == fitted
    assert restored.to_metadata()["fit_split"] == "train"
    assert "validation" not in path.read_text(encoding="utf-8")
    assert "test" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("log10", [0.0, 1.0]),
        ("log1p", [-1.0, 1.0]),
        ("logit_percent", [0.0, 50.0]),
        ("logit_percent", [50.0, 100.0]),
    ],
)
def test_transform_domain_errors_fail_clearly(kind: str, values: list[float]) -> None:
    with pytest.raises(ValueError):
        fit_regression_target_transform(
            values,
            endpoint_id="endpoint",
            units="units",
            transform=kind,
        )


def test_nonfinite_and_constant_training_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_regression_target_transform(
            [1.0, np.nan],
            endpoint_id="endpoint",
            units="units",
        )
    with pytest.raises(ValueError, match="non-zero"):
        fit_regression_target_transform(
            [2.0, 2.0],
            endpoint_id="endpoint",
            units="units",
        )
