"""Create the frozen five-seed Chemprop regression production manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.production_manifest import (  # noqa: E402
    PRODUCTION_SEEDS,
    create_regression_production_manifest,
    current_git_commit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate immutable five-seed artifacts and create a production manifest."
    )
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--config", default="configs/chemprop/multitask_admet_regression.yaml")
    parser.add_argument("--run-root", default="outputs/gpu/pilot")
    parser.add_argument(
        "--ensemble-dir",
        default="outputs/gpu/pilot/multitask_regression_ensemble_validation",
    )
    parser.add_argument(
        "--ensemble-summary",
        help="Frozen ensemble_validation_summary.json path.",
    )
    parser.add_argument("--ensemble-checksums")
    parser.add_argument("--requirements", default="requirements-chemprop.txt")
    parser.add_argument("--environment", default="environment-chemprop-gpu.yml")
    parser.add_argument(
        "--output",
        default="outputs/gpu/release/chemprop_regression/production_manifest.json",
    )
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    run_root = Path(args.run_root)
    ensemble_dir = Path(args.ensemble_dir)
    ensemble_summary = (
        Path(args.ensemble_summary)
        if args.ensemble_summary
        else ensemble_dir / "ensemble_validation_summary.json"
    )
    ensemble_checksums = (
        Path(args.ensemble_checksums) if args.ensemble_checksums else ensemble_dir / "SHA256SUMS"
    )
    runs = {seed: run_root / f"multitask_regression_seed{seed}" for seed in PRODUCTION_SEEDS}
    commit = args.git_commit or current_git_commit(artifact_root)
    manifest = create_regression_production_manifest(
        artifact_root=artifact_root,
        config_path=args.config,
        run_directories=runs,
        ensemble_validation_directory=ensemble_dir,
        ensemble_summary_path=ensemble_summary,
        ensemble_checksums_path=ensemble_checksums,
        requirements_path=args.requirements,
        environment_path=args.environment,
        output_path=args.output,
        git_commit=commit,
    )
    print(
        json.dumps(
            {
                "release_status": manifest["release_status"],
                "manifest": args.output,
                "manifest_sha256": args.output + ".sha256",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
