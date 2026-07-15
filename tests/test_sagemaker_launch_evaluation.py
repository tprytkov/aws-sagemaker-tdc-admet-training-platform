import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from admet_platform.sagemaker import launch_evaluation as launcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sagemaker_evaluation.yaml"


def test_evaluation_yaml_loading_and_configuration_precedence() -> None:
    raw = launcher.load_evaluation_yaml(CONFIG_PATH)
    effective = launcher.build_effective_config(raw, {"endpoint_id": "herg_karim", "near_tie_tolerance": 0.2})

    assert raw["evaluation"]["endpoint_id"] == "bbb_martins"
    assert effective["endpoint_id"] == "herg_karim"
    assert effective["near_tie_tolerance"] == 0.2


def test_required_fields_and_s3_uri_validation() -> None:
    effective = _effective()
    effective["candidate_runs_s3_uri"] = "not-s3"
    with pytest.raises(ValueError, match="s3://bucket/key"):
        launcher.validate_effective_config(effective)

    effective = _effective()
    effective["image_uri"] = None
    with pytest.raises(ValueError, match="Processing image URI"):
        launcher.validate_effective_config(effective)


def test_job_name_generation_and_processing_io_construction() -> None:
    effective = _effective()
    name = launcher.generate_job_name(
        effective["job_name_prefix"],
        effective["endpoint_id"],
        "evaluation",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="XYZ",
    )
    inputs = launcher.build_processing_inputs(effective, dry_run=True)
    outputs = launcher.build_processing_outputs(effective, dry_run=True)

    assert name == "admet-evaluation-bbb-martins-evaluation-20260102-030405-xyz"
    assert inputs[0]["destination"] == "/opt/ml/processing/input/runs"
    assert {output["output_name"] for output in outputs} == {"evaluation", "model-card", "registry", "metadata"}


def test_processor_args_source_package_and_dry_run_no_aws(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    effective = _effective()
    args = launcher.build_processor_args(effective=effective, source_dir=Path("source"))
    source = launcher.validate_source_package_inputs("sagemaker", "evaluate_models.py", None, project_root=PROJECT_ROOT)

    assert args["entrypoint"] == ["python", "evaluate_models.py"]
    assert "sagemaker/evaluate_models.py" in source["includes"]
    assert "src/admet_platform" in source["includes"]
    assert "sagemaker/evaluation_requirements.txt" in source["includes"]

    def fail(*args, **kwargs):
        raise AssertionError("dry run must not create AWS sessions or processors")

    monkeypatch.setattr(launcher, "create_sagemaker_session", fail)
    monkeypatch.setattr(launcher, "create_processor", fail)
    result = launcher.run_evaluation_launch(
        CONFIG_PATH,
        dry_run=True,
        launch_plan_output=tmp_path / "evaluation_launch_plan.json",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="dry",
    )

    assert result.status == "dry_run"
    assert result.manifest_path.exists()


def test_dry_run_launch_plan_schema(tmp_path: Path) -> None:
    result = launcher.run_evaluation_launch(
        CONFIG_PATH,
        dry_run=True,
        launch_plan_output=tmp_path / "evaluation_launch_plan.json",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="dry",
    )
    plan = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert {
        "status",
        "job_name",
        "endpoint_id",
        "role_arn",
        "image_uri",
        "inputs",
        "outputs",
        "container_arguments",
        "evaluation_settings",
        "source_package",
        "warnings",
    } <= set(plan)
    assert plan["role_arn"] == "arn:aws:iam::************:role/SageMakerExecutionRole"
    assert "C:\\Users" not in json.dumps(plan)


def test_mocked_submission_failure_wait_no_wait_and_cli_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    waits: list[bool] = []

    class FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def run(self, inputs, outputs, arguments, job_name, wait):
            waits.append(wait)

    monkeypatch.setattr(launcher, "build_processing_inputs", lambda config, dry_run: ["input"])
    monkeypatch.setattr(launcher, "build_processing_outputs", lambda config, dry_run: ["output"])

    for wait in (True, False):
        launcher.run_evaluation_launch(
            CONFIG_PATH,
            dry_run=False,
            launch_plan_output=tmp_path / f"submitted_{wait}.json",
            wait=wait,
            sagemaker_session_factory=lambda region: {"region": region},
            processor_class=FakeProcessor,
        )
    assert waits == [True, False]

    class FailingProcessor:
        def __init__(self, **kwargs):
            pass

        def run(self, inputs, outputs, arguments, job_name, wait):
            raise RuntimeError("boom aws_secret_access_key=abc")

    with pytest.raises(launcher.SageMakerEvaluationSubmissionError, match="REDACTED"):
        launcher.run_evaluation_launch(
            CONFIG_PATH,
            dry_run=False,
            launch_plan_output=tmp_path / "failed.json",
            sagemaker_session_factory=lambda region: {"region": region},
            processor_class=FailingProcessor,
        )

    plan_path = tmp_path / "cli_plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "launch_evaluation.py"),
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--launch-plan-output",
            str(plan_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run:" in result.stdout
    assert plan_path.exists()


def _effective() -> dict:
    return launcher.build_effective_config(launcher.load_evaluation_yaml(CONFIG_PATH), {})
