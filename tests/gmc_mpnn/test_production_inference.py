from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
from rdkit import Chem

from admet_platform.gmc_mpnn import inference
from admet_platform.gmc_mpnn.model import MODEL_INTERFACE_VERSION
from admet_platform.gmc_mpnn.production_manifest import (
    MANIFEST_VERSION,
    PRODUCTION_SEEDS,
)
from scripts import predict_gmc_mpnn_bbb as inference_cli


class _Scaler:
    portable_scaler_sha256 = "e" * 64

    def __init__(self) -> None:
        self.transform_calls = 0

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.transform_calls += 1
        return np.asarray(values, dtype=np.float64) + 1.0

    def fit(self, values: np.ndarray) -> None:
        raise AssertionError("Production inference must never fit a scaler.")


class _Executor:
    def __init__(
        self,
        probabilities: Mapping[int, Mapping[int, float]],
        errors: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        self.probabilities = probabilities
        self.errors = errors or {}
        self.calls: list[tuple[inference.PreparedProductionMolecule, ...]] = []

    def predict(
        self, prepared: Sequence[inference.PreparedProductionMolecule]
    ) -> inference.ModelExecutionResult:
        self.calls.append(tuple(prepared))
        return inference.ModelExecutionResult(
            probabilities=self.probabilities,
            errors=self.errors,
        )


def test_batch_order_aggregation_threshold_boundary_and_stable_schema(tmp_path: Path) -> None:
    manifest = _manifest_fixture(tmp_path)
    probabilities = {
        0: dict(zip(PRODUCTION_SEEDS, (0.4, 0.45, 0.5, 0.55, 0.6), strict=True)),
        1: {seed: 0.2 for seed in PRODUCTION_SEEDS},
        2: {seed: 0.8 for seed in PRODUCTION_SEEDS},
    }
    executor = _Executor(probabilities)
    predictor, calls = _predictor(tmp_path, manifest, executor)
    inputs = [
        {"molecule_id": "first", "source_smiles": "CC"},
        {"molecule_id": "second", "source_smiles": "CCC"},
        {"molecule_id": "third", "source_smiles": "O"},
    ]

    results = predictor.predict_batch(inputs)

    assert [result["molecule_id"] for result in results] == ["first", "second", "third"]
    assert all(tuple(result) == inference.OUTPUT_FIELDS for result in results)
    assert results[0]["ensemble_probability"] == pytest.approx(0.5)
    assert results[0]["ensemble_standard_deviation"] == pytest.approx(
        np.std([0.4, 0.45, 0.5, 0.55, 0.6], ddof=0)
    )
    assert results[0]["prediction"] == "BBB+"
    assert results[1]["prediction"] == "BBB-"
    assert results[2]["prediction"] == "BBB+"
    assert all(result["threshold"] == 0.5 for result in results)
    assert all(result["status"] == "success" for result in results)
    assert calls == ["first", "second", "third"]
    assert len(executor.calls) == 1
    assert [item.molecule_id for item in executor.calls[0]] == calls


def test_invalid_smiles_is_structured_and_does_not_call_geometry() -> None:
    scaler = _Scaler()

    with pytest.raises(inference.MoleculePreprocessingError, match="RDKit could not parse") as exc:
        inference.preprocess_production_molecule(
            inference.GMCInferenceInput("bad", "not-a-smiles"), 0, scaler
        )

    assert exc.value.code == "invalid_smiles"
    assert scaler.transform_calls == 0


def test_preprocessing_failure_and_model_failure_are_isolated(tmp_path: Path) -> None:
    manifest = _manifest_fixture(tmp_path)
    executor = _Executor(
        probabilities={2: {seed: 0.7 for seed in PRODUCTION_SEEDS}},
        errors={1: ("model_inference_failed", "synthetic seed failure")},
    )

    def preprocess(
        item: inference.GMCInferenceInput, index: int, scaler: _Scaler
    ) -> inference.PreparedProductionMolecule:
        if item.molecule_id == "preprocess-bad":
            raise inference.MoleculePreprocessingError(
                "geometry_no_conformers_generated",
                "no conformer",
                parent_status="unchanged",
            )
        return _prepared(item, index)

    predictor, _ = _predictor(tmp_path, manifest, executor, preprocess=preprocess)
    results = predictor.predict_batch(
        [
            {"molecule_id": "preprocess-bad", "source_smiles": "CC"},
            {"molecule_id": "model-bad", "source_smiles": "CCC"},
            {"molecule_id": "good", "source_smiles": "O"},
        ]
    )

    assert results[0]["error_code"] == "geometry_no_conformers_generated"
    assert results[0]["preprocessing_status"] == "failed"
    assert results[1]["error_code"] == "model_inference_failed"
    assert results[1]["preprocessing_status"] == "success"
    assert results[2]["status"] == "success"
    assert results[2]["ensemble_probability"] == pytest.approx(0.7)


def test_manifest_is_validated_before_scaler_or_preprocessing(tmp_path: Path) -> None:
    events: list[str] = []

    def manifest_loader(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("manifest")
        raise RuntimeError("invalid manifest")

    def scaler_loader(path: Path) -> _Scaler:
        events.append("scaler")
        return _Scaler()

    with pytest.raises(RuntimeError, match="invalid manifest"):
        inference.GMCProductionPredictor(
            tmp_path / "manifest.json",
            artifact_root=tmp_path,
            manifest_loader=manifest_loader,
            scaler_loader=scaler_loader,
            verify_runtime=False,
        )

    assert events == ["manifest"]


def test_missing_seed_probability_fails_closed_without_ensemble(tmp_path: Path) -> None:
    manifest = _manifest_fixture(tmp_path)
    executor = _Executor(
        {0: {seed: 0.6 for seed in PRODUCTION_SEEDS if seed != PRODUCTION_SEEDS[-1]}}
    )
    predictor, _ = _predictor(tmp_path, manifest, executor)

    result = predictor.predict_one({"molecule_id": "mol", "source_smiles": "CC"})

    assert result["status"] == "failed"
    assert result["error_code"] == "model_inference_failed"
    assert result["ensemble_probability"] is None
    assert result["prediction"] is None


def test_preprocessor_uses_frozen_transform_without_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = SimpleNamespace(
        coordinates=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        heavy_atom_atomic_numbers=(6, 6),
        geometry_fingerprint="geometry",
        canonical_isomeric_smiles="CC",
    )
    ggl = SimpleNamespace(
        features=np.zeros((2, 6), dtype=np.float64),
        feature_names=inference.GGL_FEATURE_NAMES,
    )
    monkeypatch.setattr(inference, "generate_deterministic_geometry", lambda smiles: geometry)
    monkeypatch.setattr(inference, "compute_ggl_features", lambda *args, **kwargs: ggl)
    scaler = _Scaler()

    prepared = inference.preprocess_production_molecule(
        inference.GMCInferenceInput("mol", "CC"), 0, scaler
    )

    assert scaler.transform_calls == 1
    np.testing.assert_array_equal(prepared.scaled_ggl_features, np.ones((2, 6)))


def test_repeated_batch_is_deterministic_and_preprocesses_once_per_call(tmp_path: Path) -> None:
    manifest = _manifest_fixture(tmp_path)
    probabilities = {0: {seed: 0.1 + seed / 1000 for seed in PRODUCTION_SEEDS}}
    executor = _Executor(probabilities)
    predictor, calls = _predictor(tmp_path, manifest, executor)
    inputs = [{"molecule_id": "same", "source_smiles": "CC"}]

    first = predictor.predict_batch(inputs)
    second = predictor.predict_batch(inputs)

    assert first == second
    assert calls == ["same", "same"]
    assert len(executor.calls) == 2


def test_sequential_executor_reuses_one_dataset_and_loader_for_all_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest_fixture(tmp_path)
    state: dict[str, Any] = {"dataset_calls": 0, "loader_calls": 0, "checkpoints": []}

    class Data:
        @staticmethod
        def MoleculeDatapoint(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        @staticmethod
        def MoleculeDataset(datapoints: list[Any], *, featurizer: Any) -> list[Any]:
            state["dataset_calls"] += 1
            return datapoints

        @staticmethod
        def build_dataloader(dataset: list[Any], **kwargs: Any) -> list[Any]:
            state["loader_calls"] += 1
            return dataset

    class Trainer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def predict(self, model: Any, *, dataloaders: list[Any], ckpt_path: str) -> list[Any]:
            match = re.search(r"seed(\d+)\.ckpt", ckpt_path)
            assert match is not None
            seed = int(match.group(1))
            state["checkpoints"].append(seed)
            return [np.full((len(dataloaders), 1), seed / 1000, dtype=np.float64)]

    lightning = SimpleNamespace(
        Trainer=Trainer,
        seed_everything=lambda seed, workers: None,
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        use_deterministic_algorithms=lambda enabled: None,
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
    )
    runtime = inference.InferenceRuntime(
        chemprop=SimpleNamespace(data=Data), lightning=lightning, torch=torch
    )
    bundle = SimpleNamespace(
        model=object(),
        featurizer=SimpleNamespace(atom_fdim=78, bond_fdim=14),
    )
    monkeypatch.setattr(inference, "build_gmc_mpnn_model", lambda **kwargs: bundle)
    executor = inference.SequentialChempropExecutor(
        manifest=manifest,
        artifact_root=tmp_path,
        runtime=runtime,
    )
    items = [
        _prepared(inference.GMCInferenceInput("a", "CC"), 0),
        _prepared(inference.GMCInferenceInput("b", "CC"), 1),
    ]

    result = executor.predict(items)

    assert result.errors == {}
    assert state["dataset_calls"] == 1
    assert state["loader_calls"] == 1
    assert state["checkpoints"] == list(PRODUCTION_SEEDS)
    assert result.probabilities[0] == {seed: seed / 1000 for seed in PRODUCTION_SEEDS}
    assert result.probabilities[1] == result.probabilities[0]


def test_sequential_executor_isolates_one_molecule_after_batch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest_fixture(tmp_path)
    state = {"dataset_calls": 0, "loader_calls": 0}

    class Data:
        @staticmethod
        def MoleculeDatapoint(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        @staticmethod
        def MoleculeDataset(datapoints: list[Any], *, featurizer: Any) -> list[Any]:
            state["dataset_calls"] += 1
            return datapoints

        @staticmethod
        def build_dataloader(dataset: list[Any], **kwargs: Any) -> list[Any]:
            state["loader_calls"] += 1
            return dataset

    class Trainer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def predict(self, model: Any, *, dataloaders: list[Any], ckpt_path: str) -> list[Any]:
            if len(dataloaders) > 1:
                raise RuntimeError("synthetic batch failure")
            if dataloaders[0]["name"] == "bad":
                raise RuntimeError("synthetic molecule failure")
            seed = int(re.search(r"seed(\d+)\.ckpt", ckpt_path).group(1))  # type: ignore[union-attr]
            return [np.asarray([[seed / 1000]], dtype=np.float64)]

    runtime = inference.InferenceRuntime(
        chemprop=SimpleNamespace(data=Data),
        lightning=SimpleNamespace(
            Trainer=Trainer,
            seed_everything=lambda seed, workers: None,
        ),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            use_deterministic_algorithms=lambda enabled: None,
            backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
        ),
    )
    bundle = SimpleNamespace(
        model=object(),
        featurizer=SimpleNamespace(atom_fdim=78, bond_fdim=14),
    )
    monkeypatch.setattr(inference, "build_gmc_mpnn_model", lambda **kwargs: bundle)
    executor = inference.SequentialChempropExecutor(
        manifest=manifest,
        artifact_root=tmp_path,
        runtime=runtime,
    )

    result = executor.predict(
        [
            _prepared(inference.GMCInferenceInput("bad", "CC"), 0),
            _prepared(inference.GMCInferenceInput("good", "CC"), 1),
        ]
    )

    assert result.errors[0][0] == "model_inference_failed"
    assert 0 not in result.probabilities
    assert result.probabilities[1] == {seed: seed / 1000 for seed in PRODUCTION_SEEDS}
    assert state == {"dataset_calls": 3, "loader_calls": 3}


def test_csv_smoke_cli_preserves_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "smoke.csv"
    output_path = tmp_path / "predictions.csv"
    input_path.write_text(
        "molecule_id,source_smiles\nsecond,CCC\nfirst,CC\n",
        encoding="utf-8",
    )

    class Predictor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def predict_batch(
            self, inputs: Sequence[inference.GMCInferenceInput]
        ) -> list[dict[str, Any]]:
            rows = []
            for item in inputs:
                row = dict.fromkeys(inference.OUTPUT_FIELDS)
                row.update(
                    {
                        "molecule_id": item.molecule_id,
                        "source_smiles": item.source_smiles,
                        "threshold": 0.5,
                        "status": "success",
                    }
                )
                rows.append(row)
            return rows

    monkeypatch.setattr(inference_cli, "GMCProductionPredictor", Predictor)

    assert (
        inference_cli.main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--artifact-root",
                str(tmp_path),
                "--input-csv",
                str(input_path),
                "--output-csv",
                str(output_path),
            ]
        )
        == 0
    )
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("second,CCC,")
    assert lines[2].startswith("first,CC,")


def _predictor(
    tmp_path: Path,
    manifest: dict[str, Any],
    executor: _Executor,
    *,
    preprocess: Any | None = None,
) -> tuple[inference.GMCProductionPredictor, list[str]]:
    calls: list[str] = []

    def default_preprocess(
        item: inference.GMCInferenceInput, index: int, scaler: _Scaler
    ) -> inference.PreparedProductionMolecule:
        calls.append(str(item.molecule_id))
        return _prepared(item, index)

    predictor = inference.GMCProductionPredictor(
        tmp_path / "manifest.json",
        artifact_root=tmp_path,
        verify_runtime=False,
        manifest_loader=lambda *args, **kwargs: manifest,
        scaler_loader=lambda path: _Scaler(),
        preprocess_function=preprocess or default_preprocess,
        model_executor=executor,
    )
    return predictor, calls


def _manifest_fixture(tmp_path: Path) -> dict[str, Any]:
    scaler_dir = tmp_path / "scaler"
    scaler_dir.mkdir(exist_ok=True)
    scaler_artifacts = {}
    for key, name in (
        ("json", "scaler.json"),
        ("npz", "scaler.npz"),
        ("fit_summary", "fit_summary.json"),
    ):
        path = scaler_dir / name
        path.write_bytes(key.encode())
        scaler_artifacts[key] = {"path": path.relative_to(tmp_path).as_posix()}
    checkpoints = []
    for seed in PRODUCTION_SEEDS:
        path = tmp_path / f"seed{seed}.ckpt"
        path.write_bytes(str(seed).encode())
        checkpoints.append({"seed": seed, "path": path.name})
    return {
        "manifest_version": MANIFEST_VERSION,
        "model_interface_version": MODEL_INTERFACE_VERSION,
        "model_family": "GMC-MPNN",
        "ensemble": {"seeds": list(PRODUCTION_SEEDS)},
        "production_decision": {"threshold": 0.5},
        "scaler": {
            "portable_sha256": "e" * 64,
            "artifacts": scaler_artifacts,
        },
        "checkpoints": checkpoints,
    }


def _prepared(
    item: inference.GMCInferenceInput, index: int
) -> inference.PreparedProductionMolecule:
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None
    return inference.PreparedProductionMolecule(
        input_index=index,
        molecule_id=str(item.molecule_id),
        source_smiles=str(item.source_smiles),
        canonical_smiles=Chem.MolToSmiles(molecule),
        parent_standardization_status="unchanged",
        geometry_smiles="CC",
        molecule=molecule,
        geometry=SimpleNamespace(geometry_fingerprint="geometry"),
        scaled_ggl_features=np.zeros((2, 6), dtype=np.float64),
    )
