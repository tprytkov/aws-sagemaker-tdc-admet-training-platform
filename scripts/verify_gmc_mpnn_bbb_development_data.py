"""Verify GMC-MPNN BBB development data without accessing the locked test split."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_platform.chemprop.config import (  # noqa: E402
    ChempropExperimentConfig,
    load_chemprop_config,
)
from admet_platform.gmc_mpnn.data import load_bbb_development_data  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "chemprop" / "bbb_martins.yaml"
IDENTITY_COLUMNS = ("molecule_id", "canonical_smiles", "target")


class LockedTestAccessError(RuntimeError):
    """Raised before a locked test artifact can be resolved, opened, or hashed."""


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Verify the leakage-controlled BBB_Martins train/validation artifacts for "
            "GMC-MPNN without accessing the locked test split."
        ),
        epilog=(
            "Intended ECHO use after pulling feature/gmc-mpnn-bbb: "
            "cd ~/chemprop-admet-training && "
            "python scripts/verify_gmc_mpnn_bbb_development_data.py"
        ),
    )


def verify_development_data(config_path: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable verification report for train/validation only."""

    _reject_test_named_config(config_path)
    config = load_chemprop_config(config_path)
    _validate_development_paths(config)
    with _deny_test_artifact_access(config):
        development = load_bbb_development_data(config)
        chemprop_validation = _read_chemprop_validation_membership(config)

    membership = _compare_validation_membership(
        development.validation, chemprop_validation
    )
    train = development.provenance.splits["train"]
    validation = development.provenance.splits["validation"]
    return {
        "dataset_name": config.dataset,
        "split_manifest_identity": development.provenance.split_manifest_id,
        "train_path": str(train.source_path),
        "validation_path": str(validation.source_path),
        "expected_train_hash": train.expected_sha256,
        "observed_train_hash": train.observed_sha256,
        "expected_validation_hash": validation.expected_sha256,
        "observed_validation_hash": validation.observed_sha256,
        "train_row_count": train.row_count,
        "validation_row_count": validation.row_count,
        "train_negative_count": train.label_counts["0"],
        "train_positive_count": train.label_counts["1"],
        "validation_negative_count": validation.label_counts["0"],
        "validation_positive_count": validation.label_counts["1"],
        "exact_canonical_smiles_overlap_count": (
            development.leakage.exact_canonical_smiles_count
        ),
        "murcko_scaffold_overlap_count": development.leakage.murcko_scaffold_count,
        "canonical_smiles_verification_passed": True,
        "chemprop_validation_membership": membership,
        "loaded_splits": ["train", "validation"],
        "test_artifact_accessed": False,
    }


def _read_chemprop_validation_membership(
    config: ChempropExperimentConfig,
) -> pd.DataFrame:
    """Read only the validation artifact consumed by the Chemprop BBB workflow."""

    validation_path = config.prepared_root / config.split_files["validation"]
    return pd.read_csv(validation_path, usecols=list(IDENTITY_COLUMNS))


def _compare_validation_membership(
    gmc_validation: pd.DataFrame, chemprop_validation: pd.DataFrame
) -> dict[str, int | bool]:
    gmc_keys = _composite_identity_counts(gmc_validation, "GMC")
    chemprop_keys = _composite_identity_counts(chemprop_validation, "Chemprop")
    missing = chemprop_keys - gmc_keys
    extra = gmc_keys - chemprop_keys
    return {
        "gmc_validation_row_count": int(sum(gmc_keys.values())),
        "chemprop_validation_row_count": int(sum(chemprop_keys.values())),
        "exact_composite_key_match": gmc_keys == chemprop_keys,
        "missing_from_gmc_count": int(sum(missing.values())),
        "extra_in_gmc_count": int(sum(extra.values())),
    }


def _composite_identity_counts(
    frame: pd.DataFrame, source: str
) -> Counter[tuple[str, str, int]]:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} validation membership is missing columns: {missing}")
    targets = pd.to_numeric(frame["target"], errors="coerce")
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise ValueError(f"{source} validation membership has invalid binary targets.")
    keys = zip(
        frame["molecule_id"].astype(str),
        frame["canonical_smiles"].astype(str),
        targets.astype(int),
    )
    return Counter(keys)


def _validate_development_paths(config: ChempropExperimentConfig) -> None:
    test_name = config.split_files.get("test")
    if not test_name:
        raise LockedTestAccessError("BBB configuration must identify a locked test artifact.")
    for split in ("train", "validation"):
        if config.split_files.get(split) == test_name:
            raise LockedTestAccessError(
                f"Configured {split} data aliases the locked test artifact; refusing to continue."
            )


def _reject_test_named_config(config_path: str | Path) -> None:
    name = Path(config_path).name.lower()
    if name in {"test.csv", "locked.csv"}:
        raise LockedTestAccessError("A test artifact cannot be used as configuration input.")


@contextmanager
def _deny_test_artifact_access(config: ChempropExperimentConfig) -> Iterator[None]:
    """Block common resolution/read paths for the configured locked test artifact."""

    blocked = config.prepared_root / config.split_files["test"]
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_path_open = Path.open
    original_resolve = Path.resolve
    original_builtin_open = builtins.open
    original_read_csv = pd.read_csv

    def reject(candidate: object) -> None:
        try:
            candidate_path = Path(os.fspath(candidate))  # type: ignore[arg-type]
        except TypeError:
            return
        if candidate_path == blocked:
            raise LockedTestAccessError(
                "Locked BBB test artifact access is prohibited in the development verifier."
            )

    def guarded_read_bytes(path: Path) -> bytes:
        reject(path)
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        reject(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        reject(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        reject(path)
        return original_resolve(path, *args, **kwargs)

    def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any):
        reject(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_read_csv(source: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        reject(source)
        return original_read_csv(source, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Path, "read_bytes", guarded_read_bytes))
        stack.enter_context(patch.object(Path, "read_text", guarded_read_text))
        stack.enter_context(patch.object(Path, "open", guarded_path_open))
        stack.enter_context(patch.object(Path, "resolve", guarded_resolve))
        stack.enter_context(patch.object(builtins, "open", guarded_builtin_open))
        stack.enter_context(patch.object(pd, "read_csv", guarded_read_csv))
        yield


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Chemprop BBB configuration (default: configs/chemprop/bbb_martins.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Without this argument, JSON is printed to stdout.",
    )
    args = parser.parse_args(argv)
    report = verify_development_data(args.config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["chemprop_validation_membership"]["exact_composite_key_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
