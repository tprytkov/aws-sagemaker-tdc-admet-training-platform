"""Thin wrapper for the package-based SageMaker Processing entry point."""

from __future__ import annotations

import sys
from pathlib import Path


ENTRYPOINT_DIR = Path(__file__).resolve().parent
for candidate in (ENTRYPOINT_DIR / "src", ENTRYPOINT_DIR.parent / "src"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from admet_platform.sagemaker.prepare_tdc_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
