"""Command-line interface for optional public TDC dataset download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.tdc_loader import download_and_prepare_tdc_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize a public TDC ADMET dataset.")
    parser.add_argument("--config", required=True, help="Path to the endpoint YAML config.")
    parser.add_argument("--output-csv", required=True, help="Path for the normalized output CSV.")
    parser.add_argument("--summary-json", required=True, help="Path for the summary JSON.")
    args = parser.parse_args()

    download_and_prepare_tdc_dataset(
        config_path=args.config,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
    )

    print(f"Wrote normalized TDC CSV: {args.output_csv}")
    print(f"Wrote TDC summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
