"""Command-line interface for local ADMET dataset preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.prepare import prepare_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local ADMET CSV dataset.")
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV.")
    parser.add_argument("--config", required=True, help="Path to the endpoint YAML config.")
    parser.add_argument("--output-csv", required=True, help="Path for the cleaned output CSV.")
    parser.add_argument("--summary-json", required=True, help="Path for the summary JSON.")
    args = parser.parse_args()

    prepare_dataset(
        input_csv=args.input_csv,
        config_path=args.config,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
    )

    print(f"Wrote cleaned CSV: {args.output_csv}")
    print(f"Wrote summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
