"""Command-line interface for public-safe model registry entry generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.registry import build_model_registry_entry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a public-safe ADMET model registry JSON entry.")
    parser.add_argument("--config", required=True, help="Path to the endpoint YAML config.")
    parser.add_argument("--metrics-json", required=True, help="Path to the training metrics JSON.")
    parser.add_argument("--artifact-uri", required=True, help="Model artifact URI or local ignored path.")
    parser.add_argument("--output-json", required=True, help="Path for the registry JSON entry.")
    parser.add_argument(
        "--validation-status",
        default="experimental",
        help="Validation status to store in the registry entry.",
    )
    args = parser.parse_args()

    build_model_registry_entry(
        config_path=args.config,
        metrics_json_path=args.metrics_json,
        artifact_uri=args.artifact_uri,
        output_json_path=args.output_json,
        validation_status=args.validation_status,
    )

    print(f"Wrote model registry entry: {args.output_json}")


if __name__ == "__main__":
    main()
