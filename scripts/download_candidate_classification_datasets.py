"""Download and audit unsplit candidate classification datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.config import load_endpoint_config  # noqa: E402
from admet_platform.data.classification_acquisition import (  # noqa: E402
    acquire_and_audit_binary_tdc_dataset,
)


DEFAULT_CONFIGS = (
    "hia_hou.yaml",
    "pgp_broccatelli.yaml",
    "cyp1a2_veith.yaml",
    "cyp2c19_veith.yaml",
    "cyp2c9_veith.yaml",
    "cyp2d6_veith.yaml",
    "cyp3a4_veith.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and audit unsplit candidate binary-classification TDC datasets."
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Endpoint config path; repeat as needed. Defaults to all seven candidate configs.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Ignored local directory that will receive one subdirectory per endpoint.",
    )
    args = parser.parse_args()

    config_paths = (
        [Path(path) for path in args.configs]
        if args.configs
        else [PROJECT_ROOT / "configs" / name for name in DEFAULT_CONFIGS]
    )
    output_root = Path(args.output_root)
    audits = []
    for config_path in config_paths:
        config = load_endpoint_config(config_path)
        audit = acquire_and_audit_binary_tdc_dataset(
            config_path,
            output_root / config.endpoint_id,
        )
        audits.append(audit)
        print(
            f"{config.endpoint_id}: {audit['raw_row_count']} rows; "
            f"sha256={audit['normalized_csv_sha256']}"
        )

    summary_path = output_root / "acquisition_summary.json"
    summary_path.write_text(
        json.dumps({"split_status": "unsplit", "endpoints": audits}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote acquisition summary: {summary_path}")


if __name__ == "__main__":
    main()
