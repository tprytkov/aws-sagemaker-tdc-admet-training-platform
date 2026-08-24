from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import admet_platform.chemprop.production_inference as inference
from admet_platform.chemprop.production_inference import (
    ChempropRegressionPredictor,
    ProductionInferenceContractError,
)
from admet_platform.chemprop.production_manifest import (
    ENDPOINT_ORDER,
    PRODUCTION_SEEDS,
    RELEASE_STATUS,
    SCHEMA_VERSION,
    UNCERTAINTY_INTERPRETATION,
)


class FakeBackend:
    instances: list[FakeBackend] = []

    def __init__(self) -> None:
        self.smiles: list[str] = []
        self.seen_checkpoints: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def prepare(self, canonical_smiles: list[str], *, num_workers: int) -> list[int]:
        assert num_workers == 0
        self.smiles = list(canonical_smiles)
        return []

    def predict_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ) -> np.ndarray:
        seed = int(checkpoint_path.parent.parent.name.removeprefix("seed"))
        position = PRODUCTION_SEEDS.index(seed)
        self.seen_checkpoints.append(checkpoint_path.as_posix())
        # Simulate the checkpoint's embedded UnscaleTransform exactly once. If the adapter
        # unscaled again, every assertion against DESIRED_UNSCALED below would fail.
        desired = DESIRED_UNSCALED[position]
        standardized = (desired - scaler_mean) / scaler_scale
        checkpoint_output = standardized * scaler_scale + scaler_mean
        return np.vstack([checkpoint_output + index * 0.01 for index in range(len(self.smiles))])

    def close(self) -> None:
        self.closed = True


DESIRED_UNSCALED = np.asarray(
    [
        [0.0, 1.0, -4.0, -10.0, 0.0],
        [1.0, 2.0, -3.0, 0.0, 1.0],
        [2.0, 3.0, -2.0, 50.0, 2.0],
        [3.0, 4.0, -1.0, 100.0, 3.0],
        [4.0, 5.0, 0.0, 150.0, 4.0],
    ],
    dtype=float,
)


@pytest.fixture(autouse=True)
def _clear_fake_instances() -> None:
    FakeBackend.instances.clear()


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    hashes: dict[int, str] = {}
    checkpoints: list[dict[str, Any]] = []
    scaler_sha = "b" * 64
    for seed in PRODUCTION_SEEDS:
        path = tmp_path / "models" / f"seed{seed}" / "checkpoints" / "best.ckpt"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"synthetic-checkpoint-{seed}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[seed] = digest
        checkpoints.append(
            {
                "seed": seed,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": digest,
                "target_scaler_sha256": scaler_sha,
            }
        )
    monkeypatch.setattr(inference, "CHECKPOINT_SHA256", hashes)
    scaler = {
        endpoint: {
            "mean": float(index + 10),
            "scale": float(index + 2),
            "train_label_count": 100 + index,
            "scientific_transform": "log10" if endpoint == "vdss_lombardo" else "identity",
        }
        for index, endpoint in enumerate(ENDPOINT_ORDER)
    }
    endpoint_units = {
        "caco2_wang": ("log10(Papp [cm/s])", "cm/s", "Papp expressed in cm/s"),
        "lipophilicity_astrazeneca": ("log-ratio", "log-ratio", "log-ratio"),
        "solubility_aqsoldb": ("log mol/L", "log mol/L", "log mol/L"),
        "ppbr_az": ("percent bound", "percent bound", "percent bound"),
        "vdss_lombardo": ("log10(L/kg)", "L/kg", "volume in L/kg"),
    }
    endpoints: dict[str, Any] = {}
    for index, endpoint in enumerate(ENDPOINT_ORDER):
        internal, output, representation = endpoint_units[endpoint]
        transform = "log10" if endpoint == "vdss_lombardo" else "identity"
        inverse = "10**y" if endpoint == "vdss_lombardo" else "identity"
        validation_space = "user_facing_output_space"
        if endpoint == "caco2_wang":
            inverse = "identity_in_stored_label_space; physical_conversion=10**y"
            validation_space = "stored_log10_papp_space_not_physical_cm_per_s"
        endpoints[endpoint] = {
            "order_index": index,
            "scientific_transform": transform,
            "inverse_transform_for_user_output": inverse,
            "internal_model_output_unit": internal,
            "user_facing_unit": output,
            "user_facing_output_representation": representation,
            "clipping": "none",
            "training_only_scaler": scaler[endpoint],
            "validation_metrics": {"space": validation_space},
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_status": RELEASE_STATUS,
        "model": {
            "provider": "moloptima_internal_chemprop",
            "family": "chemprop_dmpnn",
            "chemprop_version": "2.3.1",
            "task_type": "regression",
            "checkpoint_contents": "five_regression_outputs_no_classification_outputs",
        },
        "production_ensemble": {
            "seeds": list(PRODUCTION_SEEDS),
            "checkpoint_selection": "best.ckpt_per_seed",
            "rule": "unweighted_arithmetic_mean_of_exactly_all_five_seed_predictions",
            "missing_seed_policy": "fail_closed",
            "seed_weighting": "none",
            "calibration": "none",
            "disagreement": {
                "statistic": "sample_standard_deviation_across_seed_predictions",
                "ddof": 1,
                "seed_count": 5,
                "interpretation": UNCERTAINTY_INTERPRETATION,
            },
        },
        "checkpoints": checkpoints,
        "endpoint_order": list(ENDPOINT_ORDER),
        "endpoints": endpoints,
        "data": {
            "preprocessing": {
                "canonical_isomeric_smiles": True,
                "preserve_disconnected_fragments": True,
                "preserve_charges": True,
                "preserve_stereochemistry": True,
                "largest_fragment_selection": False,
                "neutralization": False,
            },
            "locked_test": {"status": "not_accessed", "used_for_manifest": False},
        },
        "target_scaler": {
            "sha256": scaler_sha,
            "fit_split": "train",
            "validation_statistics_used": False,
            "test_statistics_used": False,
            "per_endpoint": scaler,
        },
        "environment": {
            "expected_pinned": {
                "package_versions": {
                    "chemprop": "2.3.1",
                    "torch": "2.6.0+cu124",
                    "rdkit": "2026.3.5",
                }
            }
        },
    }
    manifest_path = tmp_path / "release" / "production_manifest.json"
    _write_manifest(manifest_path, manifest)
    return manifest_path, tmp_path


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The production manifest writer canonicalizes all JSON mappings alphabetically.
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n", encoding="utf-8"
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _predictor(release: tuple[Path, Path]) -> ChempropRegressionPredictor:
    manifest, root = release
    return ChempropRegressionPredictor(
        manifest,
        artifact_root=root,
        verify_runtime=False,
        _backend_factory=FakeBackend,
    )


def test_manifest_loads_and_exact_five_seed_checkpoint_hashes_are_enforced(
    release: tuple[Path, Path],
) -> None:
    predictor = _predictor(release)
    assert predictor.manifest["schema_version"] == "1.0.0"
    assert predictor.manifest_sha256 == hashlib.sha256(release[0].read_bytes()).hexdigest()
    manifest_path, root = release
    manifest = _load_manifest(manifest_path)
    manifest["checkpoints"].pop()
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ProductionInferenceContractError, match="exactly five"):
        ChempropRegressionPredictor(manifest_path, artifact_root=root, verify_runtime=False)


def test_wrong_checkpoint_hash_fails_closed(release: tuple[Path, Path]) -> None:
    manifest_path, root = release
    checkpoint = root / _load_manifest(manifest_path)["checkpoints"][0]["path"]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ProductionInferenceContractError, match="SHA-256 mismatch"):
        ChempropRegressionPredictor(manifest_path, artifact_root=root, verify_runtime=False)


def test_invalid_smiles_order_duplicates_and_output_schema(release: tuple[Path, Path]) -> None:
    outputs = _predictor(release).predict(
        [
            {"molecule_id": "ethanol-a", "source_smiles": "CCO"},
            {"molecule_id": "bad", "source_smiles": "C("},
            {"molecule_id": "benzene", "source_smiles": "c1ccccc1"},
            {"molecule_id": "ethanol-b", "source_smiles": "CCO"},
        ]
    )
    assert [row["molecule_id"] for row in outputs] == [
        "ethanol-a",
        "bad",
        "benzene",
        "ethanol-b",
    ]
    assert outputs[1]["status"] == "failed"
    assert outputs[1]["error_code"] == "invalid_smiles"
    assert outputs[1]["endpoints"] == {}
    assert outputs[0]["canonical_smiles"] == outputs[3]["canonical_smiles"] == "CCO"
    assert outputs[0]["status"] == outputs[2]["status"] == outputs[3]["status"] == "success"
    assert list(outputs[0]["endpoints"]) == list(ENDPOINT_ORDER)
    assert outputs[0]["model_family"] == "chemprop_dmpnn"
    assert outputs[0]["manifest_sha256"] == hashlib.sha256(release[0].read_bytes()).hexdigest()
    assert outputs[1]["manifest_sha256"] == outputs[0]["manifest_sha256"]
    assert outputs[0]["applicability_domain"]["status"] == (
        "unavailable_frozen_training_reference_not_packaged"
    )
    backend = FakeBackend.instances[0]
    assert backend.smiles == ["CCO", "c1ccccc1", "CCO"]
    assert len(backend.seen_checkpoints) == 5
    assert backend.closed


def test_scaler_and_scientific_inverse_occur_exactly_once(release: tuple[Path, Path]) -> None:
    output = _predictor(release).predict([{"molecule_id": "x", "source_smiles": "CCO"}])[0]
    endpoints = output["endpoints"]
    caco = endpoints["caco2_wang"]
    assert list(caco["per_seed"].values()) == pytest.approx([0, 1, 2, 3, 4])
    assert caco["ensemble_mean_log10_papp_cm_per_s"] == pytest.approx(2.0)
    assert caco["physical_papp_cm_per_s_from_ensemble_log10"] == pytest.approx(100.0)
    vdss = endpoints["vdss_lombardo"]
    assert list(vdss["per_seed"].values()) == pytest.approx([1, 10, 100, 1000, 10000])
    # Frozen validation convention: inverse each seed, then aggregate in L/kg.
    assert vdss["ensemble_mean"] == pytest.approx(2222.2)
    assert vdss["ensemble_mean"] != pytest.approx(10 ** np.mean([0, 1, 2, 3, 4]))
    ppbr = endpoints["ppbr_az"]
    assert list(ppbr["per_seed"].values()) == pytest.approx([-10, 0, 50, 100, 150])


def test_five_seed_mean_and_sample_standard_deviation(release: tuple[Path, Path]) -> None:
    output = _predictor(release).predict([{"molecule_id": "x", "source_smiles": "CCO"}])[0]
    lipo = output["endpoints"]["lipophilicity_astrazeneca"]
    assert lipo["ensemble_mean"] == pytest.approx(3.0)
    assert lipo["seed_standard_deviation"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert lipo["seed_standard_deviation_ddof"] == 1


def test_repeated_fake_predictions_are_deterministic(release: tuple[Path, Path]) -> None:
    inputs = [{"molecule_id": "x", "source_smiles": "C[C@H](O)Cl"}]
    assert _predictor(release).predict(inputs) == _predictor(release).predict(inputs)


def test_inference_does_not_open_development_artifacts(
    release: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = release
    forbidden = [
        root / "train.csv",
        root / "valid.csv",
        root / "test.csv",
        root / "ensemble-validation.csv",
        root / "training.log",
    ]
    for path in forbidden:
        path.write_text("must not be read", encoding="utf-8")
    opened: list[Path] = []
    original_open = Path.open

    def recording_open(self: Path, *args: object, **kwargs: object):
        opened.append(self.resolve())
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    _predictor(release).predict([{"molecule_id": "x", "source_smiles": "CCO"}])
    assert not set(forbidden) & set(opened)


def test_backend_missing_seed_prediction_fails_overall(release: tuple[Path, Path]) -> None:
    class FailingBackend(FakeBackend):
        def predict_checkpoint(self, checkpoint_path: Path, **kwargs: Any) -> np.ndarray:
            if "seed137" in checkpoint_path.as_posix():
                raise RuntimeError("model unavailable")
            return super().predict_checkpoint(checkpoint_path, **kwargs)

    manifest, root = release
    predictor = ChempropRegressionPredictor(
        manifest,
        artifact_root=root,
        verify_runtime=False,
        _backend_factory=FailingBackend,
    )
    with pytest.raises(ProductionInferenceContractError, match="whole ensemble"):
        predictor.predict([{"molecule_id": "x", "source_smiles": "CCO"}])


def test_single_molecule_preprocessing_failure_does_not_fail_valid_rows(
    release: tuple[Path, Path],
) -> None:
    class PartialBackend(FakeBackend):
        def prepare(self, canonical_smiles: list[str], *, num_workers: int) -> list[int]:
            super().prepare([canonical_smiles[0], canonical_smiles[2]], num_workers=num_workers)
            return [1]

    manifest, root = release
    predictor = ChempropRegressionPredictor(
        manifest,
        artifact_root=root,
        verify_runtime=False,
        _backend_factory=PartialBackend,
    )
    outputs = predictor.predict(
        [
            {"molecule_id": "a", "source_smiles": "CCO"},
            {"molecule_id": "b", "source_smiles": "CCN"},
            {"molecule_id": "c", "source_smiles": "CCC"},
        ]
    )
    assert [row["status"] for row in outputs] == ["success", "failed", "success"]
    assert outputs[1]["error_code"] == "molecule_preprocessing_failed"


def test_real_backend_loads_synthetic_chemprop_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("chemprop")
    from admet_platform.chemprop.production_inference import _ChempropBackend
    from admet_platform.chemprop.smoke import run_synthetic_smoke

    result = run_synthetic_smoke("multitask_regression", tmp_path / "synthetic", seed=13)
    scaler = result["target_scaler"]["per_endpoint"]
    mean = np.asarray([scaler[endpoint]["mean"] for endpoint in ENDPOINT_ORDER], dtype=float)
    scale = np.asarray([scaler[endpoint]["scale"] for endpoint in ENDPOINT_ORDER], dtype=float)
    backend = _ChempropBackend()
    try:
        assert backend.prepare(["CCO", "c1ccccc1"], num_workers=0) == []
        values = backend.predict_checkpoint(
            Path(result["checkpoint"]), scaler_mean=mean, scaler_scale=scale
        )
    finally:
        backend.close()
    assert values.shape == (2, len(ENDPOINT_ORDER))
    assert np.isfinite(values).all()


def test_cli_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("existing\n", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "predict_chemprop_regression.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(tmp_path / "missing-manifest.json"),
            "--input-csv",
            str(tmp_path / "missing-input.csv"),
            "--output-csv",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing to overwrite existing output" in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"
