"""Command-line interface for local RDKit ADMET featurization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.features import (  # noqa: E402
    DEFAULT_MORGAN_BITS,
    DEFAULT_MORGAN_RADIUS,
    FeatureConfig,
    featurize_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Featurize a prepared ADMET CSV with RDKit.")
    parser.add_argument("--input-csv", required=True, help="Path to a prepared CSV with canonical_smiles.")
    parser.add_argument("--output-csv", required=True, help="Path for the feature CSV.")
    parser.add_argument(
        "--feature-type",
        choices=["descriptors", "morgan"],
        required=True,
        help="Feature representation to compute.",
    )
    parser.add_argument("--metadata-json", help="Optional path for feature metadata JSON.")
    parser.add_argument("--rejected-csv", help="Optional path for rejected rows CSV.")
    parser.add_argument("--morgan-radius", type=int, default=DEFAULT_MORGAN_RADIUS)
    parser.add_argument("--morgan-bits", type=int, default=DEFAULT_MORGAN_BITS)
    args = parser.parse_args()

    metadata = featurize_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        rejected_csv=args.rejected_csv,
        config=FeatureConfig(
            feature_type=args.feature_type,
            morgan_radius=args.morgan_radius,
            morgan_bits=args.morgan_bits,
        ),
    )

    print(f"Wrote feature CSV: {args.output_csv}")
    if args.metadata_json:
        print(f"Wrote feature metadata JSON: {args.metadata_json}")
    print(
        "Rows: "
        f"accepted={metadata['accepted_row_count']}, rejected={metadata['rejected_row_count']}, "
        f"features={metadata['n_features']}"
    )


if __name__ == "__main__":
    main()
