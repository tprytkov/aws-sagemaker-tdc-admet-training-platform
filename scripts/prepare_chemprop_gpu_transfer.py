"""Prepare or verify the seed-13 GPU transfer staging tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from admet_platform.chemprop.transfer import (  # noqa: E402
    prepare_gpu_transfer,
    verify_transfer_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination")
    parser.add_argument("--verify-root")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.destination and not args.verify_root and not args.manifest:
        print(json.dumps(prepare_gpu_transfer(ROOT, args.destination), indent=2))
        return
    if args.verify_root and args.manifest and not args.destination:
        verify_transfer_manifest(args.verify_root, args.manifest)
        print("TRANSFER_MANIFEST_VERIFIED")
        return
    parser.error("use either --destination or both --verify-root and --manifest")


if __name__ == "__main__":
    main()
