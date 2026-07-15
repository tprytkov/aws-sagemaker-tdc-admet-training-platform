"""CLI wrapper for SageMaker evaluation Processing launch planning."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.sagemaker.launch_evaluation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
