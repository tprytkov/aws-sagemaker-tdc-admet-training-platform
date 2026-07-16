"""Audit prepared multi-task endpoint splits before shared-model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admet_platform.data.multitask import load_endpoint_datasets, load_multitask_config  # noqa: E402
from admet_platform.data.multitask_audit import (  # noqa: E402
    MultiTaskAuditError,
    audit_multitask_splits,
    require_leakage_safe,
    write_audit_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit exact-molecule and Murcko-scaffold leakage across ADMET tasks."
    )
    parser.add_argument("--config", required=True, help="Multi-task YAML configuration path.")
    parser.add_argument("--prepared-root", help="Optional prepared dataset root override.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV audit artifacts.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write violations but return success. Training must not use this as a safety pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_multitask_config(args.config)
        datasets = load_endpoint_datasets(config, prepared_root=args.prepared_root)
        result = audit_multitask_splits(config, datasets)
        paths = write_audit_outputs(result, args.output_dir)
        if not args.report_only:
            require_leakage_safe(result)
        print(f"Audit status: {result.summary['status']}")
        print(f"Audit summary: {paths['summary']}")
        return 0
    except (FileNotFoundError, ValueError, MultiTaskAuditError) as exc:
        print(f"Multi-task split audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

