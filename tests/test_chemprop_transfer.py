import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from admet_platform.chemprop.transfer import (
    PILOT_FILES,
    prepare_gpu_transfer,
    verify_transfer_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_transfer_package_has_no_locked_test_and_verifies(tmp_path: Path) -> None:
    staging = tmp_path / "transfer"
    result = prepare_gpu_transfer(ROOT, staging)
    assert result["locked_test_files_included"] is False
    assert all(not item["path"].endswith("/test.csv") for item in result["files"])
    verify_transfer_manifest(staging, result["manifest_path"])
    first = staging / result["files"][0]["path"]
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        verify_transfer_manifest(staging, result["manifest_path"])


def test_transfer_manifest_is_complete_portable_and_includes_required_pilot_files(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "transfer"
    result = prepare_gpu_transfer(ROOT, staging)
    entries = result["files"]
    paths = [entry["path"] for entry in entries]
    required = {path for path, _category in PILOT_FILES}
    required_data = {
        "outputs/local/classification_expansion/coordinated/coordinated_split_manifest.json",
        "outputs/local/classification_expansion/coordinated/bbb_martins/train.csv",
        "outputs/local/classification_expansion/coordinated/bbb_martins/valid.csv",
        "outputs/local/multitask_regression/coordinated/coordinated_regression_split_manifest.json",
        *{
            f"outputs/local/multitask_regression/coordinated/{endpoint}/{filename}"
            for endpoint in (
                "caco2_wang",
                "lipophilicity_astrazeneca",
                "solubility_aqsoldb",
                "ppbr_az",
                "vdss_lombardo",
            )
            for filename in ("train.csv", "valid.csv")
        },
    }

    assert "docs/chemprop_phase1.md" in paths
    document = next(entry for entry in entries if entry["path"] == "docs/chemprop_phase1.md")
    source = ROOT / document["path"]
    assert document["category"] == "documentation_prerequisite"
    assert document["size_bytes"] == source.stat().st_size
    assert document["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (required | required_data).issubset(paths)
    assert len(paths) == len(set(paths)) == result["file_count"]
    assert all("\\" not in path and not Path(path).is_absolute() for path in paths)

    staged = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != "transfer_manifest.json"
    }
    assert staged == set(paths)
    for entry in entries:
        path = staging / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert not list(staging.rglob("test.csv"))


def test_staged_verification_command_creates_no_cache_files(tmp_path: Path) -> None:
    staging = tmp_path / "transfer"
    prepare_gpu_transfer(ROOT, staging)
    completed = subprocess.run(
        [
            sys.executable,
            str(staging / "scripts/prepare_chemprop_gpu_transfer.py"),
            "--verify-root",
            str(staging),
            "--manifest",
            str(staging / "transfer_manifest.json"),
        ],
        cwd=staging,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "TRANSFER_MANIFEST_VERIFIED" in completed.stdout
    assert not list(staging.rglob("__pycache__"))
    assert not list(staging.rglob("*.pyc"))


def test_transfer_verification_rejects_unmanifested_duplicate_and_nonportable_files(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "transfer"
    result = prepare_gpu_transfer(ROOT, staging)
    manifest_path = Path(result["manifest_path"])

    unexpected = staging / "unexpected.txt"
    unexpected.write_text("not manifested", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_transfer_manifest(staging, manifest_path)
    unexpected.unlink()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append(dict(manifest["files"][0]))
    manifest["file_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate paths"):
        verify_transfer_manifest(staging, manifest_path)

    manifest["files"].pop()
    manifest["file_count"] -= 1
    nested = next(entry for entry in manifest["files"] if "/" in entry["path"])
    nested["path"] = nested["path"].replace("/", "\\")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="portable POSIX form"):
        verify_transfer_manifest(staging, manifest_path)
