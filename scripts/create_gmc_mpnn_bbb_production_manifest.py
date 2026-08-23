"""Create and freeze the authoritative GMC-MPNN BBB production manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.gmc_mpnn.production_manifest import (  # noqa: E402
    PRODUCTION_SEEDS,
    ProductionManifestConfig,
    build_production_manifest,
    write_production_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the validated five-seed GMC-MPNN BBB production manifest."
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--training-preprocessing-dir", type=Path, required=True)
    parser.add_argument("--validation-preprocessing-dir", type=Path, required=True)
    parser.add_argument("--scaler-dir", type=Path, required=True)
    parser.add_argument("--validation-evaluation-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--environment-file", type=Path, default=ROOT / "environment-gmc-mpnn.yml")
    for seed in PRODUCTION_SEEDS:
        parser.add_argument(f"--checkpoint-seed{seed}", type=Path, required=True)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    git_commit = args.git_commit or _git_commit()
    payload = build_production_manifest(
        ProductionManifestConfig(
            artifact_root=args.artifact_root,
            checkpoint_paths=tuple(
                (seed, getattr(args, f"checkpoint_seed{seed}")) for seed in PRODUCTION_SEEDS
            ),
            training_preprocessing_dir=args.training_preprocessing_dir,
            validation_preprocessing_dir=args.validation_preprocessing_dir,
            scaler_dir=args.scaler_dir,
            validation_evaluation_dir=args.validation_evaluation_dir,
            calibration_dir=args.calibration_dir,
            environment_file=args.environment_file,
            git_commit=git_commit,
        )
    )
    write_production_manifest(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to record the release Git commit.") from exc


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
