"""Create a train/validation-only, SHA-256-verifiable GPU transfer staging tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from admet_platform.chemprop.config import load_chemprop_config
from admet_platform.chemprop.data import load_verified_development_data


PILOT_FILES = (
    (".gitignore", "code_or_configuration"),
    ("environment-chemprop-gpu.yml", "environment_definition"),
    ("requirements-chemprop.txt", "environment_definition"),
    ("configs/chemprop/base.yaml", "code_or_configuration"),
    ("configs/chemprop/multitask_admet_regression.yaml", "code_or_configuration"),
    ("configs/chemprop/bbb_martins.yaml", "code_or_configuration"),
    ("docs/chemprop_gpu_seed13.md", "documentation_prerequisite"),
    ("docs/chemprop_phase1.md", "documentation_prerequisite"),
    ("scripts/run_chemprop_smoke.py", "code_or_configuration"),
    ("scripts/run_chemprop_experiment.py", "code_or_configuration"),
    ("scripts/prepare_chemprop_gpu_transfer.py", "code_or_configuration"),
    ("src/admet_platform/__init__.py", "code_or_configuration"),
    ("src/admet_platform/config.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/__init__.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/applicability.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/baselines.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/calibration.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/comparison.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/config.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/data.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/export.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/losses.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/metrics.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/smoke.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/training.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/transfer.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/uncertainty.py", "code_or_configuration"),
    ("src/admet_platform/chemprop/units.py", "code_or_configuration"),
    ("src/admet_platform/data/__init__.py", "code_or_configuration"),
    ("src/admet_platform/data/multitask.py", "code_or_configuration"),
    ("src/admet_platform/data/scaffolds.py", "code_or_configuration"),
)


def prepare_gpu_transfer(repository: str | Path, destination: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    output = Path(destination).resolve()
    regression = load_chemprop_config(root / "configs/chemprop/multitask_admet_regression.yaml")
    bbb = load_chemprop_config(root / "configs/chemprop/bbb_martins.yaml")
    verified_regression = load_verified_development_data(regression)
    verified_bbb = load_verified_development_data(bbb)
    copied: list[dict[str, Any]] = []
    for relative, category in PILOT_FILES:
        _copy(root, output, Path(relative), copied, category=category)
    _copy(root, output, regression.split_manifest.relative_to(root), copied, "split_manifest")
    _copy(root, output, bbb.split_manifest.relative_to(root), copied, "split_manifest")
    for endpoint in regression.tasks:
        for split in ("train", "validation"):
            relative = (
                regression.prepared_root.relative_to(root)
                / endpoint
                / regression.split_files[split]
            )
            _copy(root, output, relative, copied, "prepared_development_data")
    for split in ("train", "validation"):
        relative = bbb.prepared_root.relative_to(root) / bbb.split_files[split]
        _copy(root, output, relative, copied, "prepared_development_data")
    manifest = {
        "schema_version": "1.0.0",
        "purpose": "chemprop_gpu_seed13_train_validation_pilot",
        "locked_test_files_included": False,
        "pretrained_model_weights_included": False,
        "credentials_included": False,
        "outputs_or_caches_included": False,
        "file_count": len(copied),
        "files": sorted(copied, key=lambda item: item["path"]),
        "source_split_verification": {
            "regression": {
                "manifest_sha256": verified_regression.split_manifest_sha256,
                "verified_train_validation_hashes": verified_regression.actual_hashes,
                "locked_test_expected_hashes_not_accessed": {
                    key: value for key, value in verified_regression.expected_hashes.items()
                    if key.endswith("/test")
                },
            },
            "bbb": {
                "manifest_sha256": verified_bbb.split_manifest_sha256,
                "verified_train_validation_hashes": verified_bbb.actual_hashes,
                "locked_test_expected_hash_not_accessed": verified_bbb.expected_hashes["test"],
            },
        },
        "exclusions": [
            "**/test.csv", "outputs/**", "**/__pycache__/**", "*.ckpt", "*.pt", "*.pth",
            ".env", "*.pem", "*.key", "credentials*", ".git/**", "docs/*.docx",
        ],
    }
    manifest_path = output / "transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path)}


def verify_transfer_manifest(root: str | Path, manifest_path: str | Path) -> None:
    base = Path(root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = manifest["files"]
    if manifest.get("file_count") != len(entries):
        raise ValueError("Transfer manifest file_count does not match its entries.")
    relative_paths = [item["path"] for item in entries]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Transfer manifest contains duplicate paths.")
    for item in manifest["files"]:
        relative = _portable_relative_path(item["path"])
        path = base / Path(*relative.parts)
        if not path.is_file():
            raise ValueError(f"Transfer manifest entry is missing: {item['path']}")
        if _sha256(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Transfer file hash or size mismatch: {item['path']}")
        _reject_forbidden_file(relative)
    staged = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.resolve() != manifest_file
    }
    if staged != set(relative_paths):
        missing = sorted(set(relative_paths) - staged)
        unexpected = sorted(staged - set(relative_paths))
        raise ValueError(
            f"Transfer staging inventory mismatch; missing={missing}, unexpected={unexpected}"
        )


def _portable_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Transfer path is not portable POSIX form: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"Transfer path is not a safe repository-relative path: {value!r}")
    return relative


def _reject_forbidden_file(relative: PurePosixPath) -> None:
    name = relative.name.lower()
    suffix = relative.suffix.lower()
    if name == "test.csv":
        raise ValueError("Transfer staging contains a locked-test CSV.")
    if suffix in {".ckpt", ".pt", ".pth", ".bin", ".safetensors"}:
        raise ValueError(f"Transfer staging contains model weights: {relative}")
    if name == ".env" or suffix in {".pem", ".key"} or name.startswith("credentials"):
        raise ValueError(f"Transfer staging contains a credential or secret file: {relative}")
    if "__pycache__" in relative.parts or relative.parts[0] in {".git"}:
        raise ValueError(f"Transfer staging contains a cache or repository metadata: {relative}")


def _copy(root: Path, output: Path, relative: Path, copied: list, category: str) -> None:
    source = (root / relative).resolve()
    portable = _portable_relative_path(relative.as_posix())
    _reject_forbidden_file(portable)
    if not source.is_file() or not source.is_relative_to(root):
        raise ValueError(f"Transfer source is missing or outside the repository: {relative}")
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append({
        "path": relative.as_posix(), "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size, "category": category,
    })


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["PILOT_FILES", "prepare_gpu_transfer", "verify_transfer_manifest"]
