"""Run the generic shared-encoder compatibility preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admet_platform.models.encoder_compatibility import (  # noqa: E402
    EncoderCompatibilityError,
    verify_encoder_compatibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a Hugging Face encoder and the three ADMET task heads."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--local-files-only",
        "--offline",
        action="store_true",
        dest="local_files_only",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        report = verify_encoder_compatibility(
            model_name_or_path=args.model_name_or_path,
            revision=args.revision,
            local_files_only=args.local_files_only,
            output_json=args.output_json,
        )
    except EncoderCompatibilityError as exc:
        parser.exit(1, f"Encoder compatibility preflight failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
