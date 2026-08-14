"""Versioned endpoint-unit conversions that do not alter prepared labels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


CACO2_LABEL_CONTRACT_VERSION = "caco2_wang_log10_cm_per_s_v1"
CACO2_MODEL_OUTPUT_UNIT = "log10(Papp [cm/s])"


def caco2_log10_to_cm_per_s(values: ArrayLike) -> NDArray[np.float64]:
    """Convert stored/model Caco-2 labels to physical Papp in cm/s."""

    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Caco-2 logarithmic values must be finite.")
    return np.power(10.0, data)


def caco2_cm_per_s_to_log10(values: ArrayLike) -> NDArray[np.float64]:
    """Convert strictly positive physical Papp in cm/s to the stored label representation."""

    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all() or np.any(data <= 0):
        raise ValueError("Physical Caco-2 Papp values must be finite and strictly positive.")
    return np.log10(data)


__all__ = [
    "CACO2_LABEL_CONTRACT_VERSION",
    "CACO2_MODEL_OUTPUT_UNIT",
    "caco2_cm_per_s_to_log10",
    "caco2_log10_to_cm_per_s",
]
