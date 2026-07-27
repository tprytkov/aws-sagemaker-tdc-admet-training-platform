"""Build the ten-endpoint globally coordinated classification split track."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.expanded_classification_splits import (  # noqa: E402
    build_expanded_classification_splits,
)
from admet_platform.data.multitask import load_multitask_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe coordinated splits for ten binary endpoints."
    )
    parser.add_argument("--config", required=True, help="Expanded multi-task YAML config.")
    parser.add_argument("--sources", required=True, help="Unsplit source provenance YAML.")
    parser.add_argument("--output-root", help="New, empty coordinated output root.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = build_expanded_classification_splits(
            load_multitask_config(args.config),
            args.sources,
            args.output_root,
            seed=args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Expanded coordinated split build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Expanded coordinated output: {result.output_root}")
    print(f"Split manifest ID: {result.manifest['split_manifest_id']}")
    print(json.dumps(result.manifest["global_split_rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
