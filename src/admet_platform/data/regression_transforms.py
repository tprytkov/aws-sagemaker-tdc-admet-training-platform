"""Train-only target transformations for continuous ADMET endpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SUPPORTED_TRANSFORMS = ("identity", "log10", "log1p", "logit_percent")


@dataclass(frozen=True)
class FittedRegressionTransform:
    """A reversible endpoint transform and z-score fitted on training targets."""

    endpoint_id: str
    units: str
    transform: str
    transformed_train_mean: float
    transformed_train_std: float
    train_row_count: int
    fit_split: str = "train"
    schema_version: str = "1.0.0"

    def transform_values(self, values: Iterable[float]) -> np.ndarray:
        """Transform values to the normalized training-target scale."""

        transformed = _forward(_finite_array(values), self.transform)
        return (transformed - self.transformed_train_mean) / self.transformed_train_std

    def inverse_values(self, values: Iterable[float]) -> np.ndarray:
        """Return normalized predictions to the endpoint's original units."""

        normalized = _finite_array(values)
        transformed = (
            normalized * self.transformed_train_std + self.transformed_train_mean
        )
        return _inverse(transformed, self.transform)

    def transform_frame(
        self,
        frame: pd.DataFrame,
        *,
        source_column: str = "target_original",
        output_column: str = "target_normalized",
    ) -> pd.DataFrame:
        """Add normalized targets while preserving the original target column."""

        if source_column not in frame:
            raise ValueError(f"Missing regression target column '{source_column}'.")
        result = frame.copy()
        result[output_column] = self.transform_values(result[source_column])
        return result

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe reproducibility metadata."""

        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write transformation metadata without target observations."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_metadata(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> FittedRegressionTransform:
        """Load and validate saved transformation metadata."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fitted = cls(**raw)
        fitted._validate()
        return fitted

    def _validate(self) -> None:
        if self.transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"Unsupported regression transform '{self.transform}'.")
        if self.fit_split != "train":
            raise ValueError("Regression transforms must be fitted on the train split.")
        if self.train_row_count < 2:
            raise ValueError("At least two training targets are required.")
        if (
            not np.isfinite(self.transformed_train_mean)
            or not np.isfinite(self.transformed_train_std)
            or self.transformed_train_std <= 0
        ):
            raise ValueError("Saved regression normalization statistics are invalid.")


def fit_regression_target_transform(
    train_values: Iterable[float],
    *,
    endpoint_id: str,
    units: str,
    transform: str = "identity",
) -> FittedRegressionTransform:
    """Fit a reversible transform and normalization from training values only."""

    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unsupported regression transform '{transform}'; "
            f"choose one of {', '.join(SUPPORTED_TRANSFORMS)}."
        )
    if not endpoint_id.strip() or not units.strip():
        raise ValueError("endpoint_id and units must be non-empty.")
    train = _finite_array(train_values)
    if train.size < 2:
        raise ValueError("At least two training targets are required.")
    transformed = _forward(train, transform)
    mean = float(np.mean(transformed))
    std = float(np.std(transformed, ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training targets must have non-zero transformed variance.")
    return FittedRegressionTransform(
        endpoint_id=endpoint_id.strip(),
        units=units.strip(),
        transform=transform,
        transformed_train_mean=mean,
        transformed_train_std=std,
        train_row_count=int(train.size),
    )


def _finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Regression targets must be one-dimensional.")
    if not np.isfinite(array).all():
        raise ValueError("Regression targets must all be finite.")
    return array


def _forward(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.copy()
    if transform == "log10":
        if np.any(values <= 0):
            raise ValueError("log10 regression transforms require positive targets.")
        return np.log10(values)
    if transform == "log1p":
        if np.any(values <= -1):
            raise ValueError("log1p regression transforms require targets greater than -1.")
        return np.log1p(values)
    if transform == "logit_percent":
        if np.any((values <= 0) | (values >= 100)):
            raise ValueError("logit_percent requires targets strictly between 0 and 100.")
        proportion = values / 100.0
        return np.log(proportion / (1.0 - proportion))
    raise ValueError(f"Unsupported regression transform '{transform}'.")


def _inverse(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values.copy()
    if transform == "log10":
        return np.power(10.0, values)
    if transform == "log1p":
        return np.expm1(values)
    if transform == "logit_percent":
        return 100.0 / (1.0 + np.exp(-values))
    raise ValueError(f"Unsupported regression transform '{transform}'.")


__all__ = [
    "FittedRegressionTransform",
    "SUPPORTED_TRANSFORMS",
    "fit_regression_target_transform",
]
