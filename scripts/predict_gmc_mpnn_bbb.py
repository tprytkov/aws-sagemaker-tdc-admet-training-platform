"""Small CSV entry point for GMC-MPNN BBB production batch inference."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.gmc_mpnn.inference import (  # noqa: E402
    OUTPUT_FIELDS,
    GMCInferenceInput,
    GMCProductionPredictor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen five-seed GMC-MPNN BBB production adapter on a CSV batch."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--molecule-id-column", default="molecule_id")
    parser.add_argument("--smiles-column", default="source_smiles")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_csv.exists():
        raise FileExistsError(f"Output CSV already exists: {args.output_csv}")
    inputs = _read_inputs(args.input_csv, args.molecule_id_column, args.smiles_column)
    predictor = GMCProductionPredictor(
        args.manifest,
        artifact_root=args.artifact_root,
        verify_runtime=True,
        num_workers=args.num_workers,
    )
    results = predictor.predict_batch(inputs)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    successes = sum(result["status"] == "success" for result in results)
    print(
        f"Wrote {len(results)} ordered predictions to {args.output_csv} "
        f"({successes} success, {len(results) - successes} failed)."
    )
    return 0


def _read_inputs(
    path: Path, molecule_id_column: str, smiles_column: str
) -> list[GMCInferenceInput]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Input CSV has no header.")
            missing = {molecule_id_column, smiles_column}.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
            return [
                GMCInferenceInput(
                    molecule_id=row.get(molecule_id_column, ""),
                    source_smiles=row.get(smiles_column, ""),
                )
                for row in reader
            ]
    except OSError as exc:
        raise ValueError(f"Cannot read input CSV: {path}") from exc


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
