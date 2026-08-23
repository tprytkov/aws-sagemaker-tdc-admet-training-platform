"""Freeze the fixed Section 8.3 GMC-MPNN BBB production qualification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from admet_platform.gmc_mpnn.qualification import (  # noqa: E402
    run_production_qualification,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and freeze fixed GMC-MPNN BBB production inference qualification."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "configs" / "gmc_mpnn_bbb_qualification_input.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_production_qualification(
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        input_path=args.input_csv,
        output_dir=args.output_dir,
        git_commit=args.git_commit,
        num_workers=args.num_workers,
    )
    counts = summary["counts"]
    print(
        f"Qualification passed and was frozen at {args.output_dir} "
        f"({counts['successful_records']} success, {counts['failed_records']} expected failure)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
