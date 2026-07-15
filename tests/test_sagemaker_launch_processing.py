import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from admet_platform.sagemaker import launch_processing as launcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sagemaker_processing.yaml"


def test_processing_yaml_loading() -> None:
    config = launcher.load_processing_yaml(CONFIG_PATH)

    assert config["processing_mode"] == "supplied_csv"
    assert config["endpoint_config"] == "configs/bbb_martins.yaml"


def test_configuration_precedence() -> None:
    effective = launcher.build_effective_config(
        launcher.load_processing_yaml(CONFIG_PATH),
        {"region": "us-east-1", "processing_mode": "tdc_download", "development_row_limit": 5},
    )

    assert effective["region"] == "us-east-1"
    assert effective["processing_mode"] == "tdc_download"
    assert effective["development_row_limit"] == 5


def test_required_launcher_fields() -> None:
    raw = launcher.load_processing_yaml(CONFIG_PATH)
    raw["image_uri"] = None
    effective = launcher.build_effective_config(raw, {})
    endpoint = launcher.load_endpoint_config(PROJECT_ROOT / "configs" / "bbb_martins.yaml")

    with pytest.raises(ValueError, match="processing image URI"):
        launcher.validate_effective_config(effective, endpoint)


def test_s3_uri_validation_and_supplied_csv_requirement() -> None:
    effective = _effective()
    effective["source_csv_s3_uri"] = "not-s3"
    endpoint = launcher.load_endpoint_config(PROJECT_ROOT / "configs" / "bbb_martins.yaml")
    with pytest.raises(ValueError, match="s3://bucket/key"):
        launcher.validate_effective_config(effective, endpoint)

    effective = _effective()
    effective["source_csv_s3_uri"] = None
    with pytest.raises(ValueError, match="source_csv_s3_uri"):
        launcher.validate_effective_config(effective, endpoint)


def test_processing_job_name_generation() -> None:
    name = launcher.generate_processing_job_name(
        "ADMET Processing",
        "bbb_martins",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="XYZ",
    )

    assert len(name) <= launcher.JOB_NAME_MAX_LENGTH
    assert name == "admet-processing-bbb-martins-processing-20260102-030405-xyz"


def test_processor_argument_construction() -> None:
    effective = _effective()
    effective["kms_key_arn"] = "arn:aws:kms:us-west-2:123456789012:key/example"
    effective["vpc_subnets"] = ["subnet-1"]
    effective["vpc_security_group_ids"] = ["sg-1"]

    args = launcher.build_processor_args(effective=effective, source_dir=Path("source"))

    assert args["image_uri"] == effective["image_uri"]
    assert args["entrypoint"] == ["python", "prepare_tdc_dataset.py"]
    assert args["output_kms_key"] == effective["kms_key_arn"]
    assert args["subnets"] == ["subnet-1"]
    assert args["security_group_ids"] == ["sg-1"]


def test_dry_run_makes_no_aws_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("dry-run should not create AWS sessions or processors")

    monkeypatch.setattr(launcher, "create_sagemaker_session", fail)
    monkeypatch.setattr(launcher, "create_processor", fail)

    result = launcher.run_processing_launch(
        CONFIG_PATH,
        dry_run=True,
        launch_plan_output=tmp_path / "processing_launch_plan.json",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="dry",
    )

    assert result.status == "dry_run"
    assert result.manifest_path.exists()


def test_dry_run_launch_plan_schema(tmp_path: Path) -> None:
    result = launcher.run_processing_launch(
        CONFIG_PATH,
        dry_run=True,
        launch_plan_output=tmp_path / "processing_launch_plan.json",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="dry",
    )
    plan = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert {
        "status",
        "job_name",
        "endpoint_id",
        "task_type",
        "processing_mode",
        "role_arn",
        "image_uri",
        "inputs",
        "outputs",
        "container_arguments",
        "source_package",
        "warnings",
    } <= set(plan)
    assert plan["role_arn"] == "arn:aws:iam::************:role/SageMakerExecutionRole"
    assert "C:\\Users" not in json.dumps(plan)


def test_source_package_completeness() -> None:
    description = launcher.validate_source_package_inputs(
        "sagemaker",
        "prepare_tdc_dataset.py",
        "configs/bbb_martins.yaml",
        project_root=PROJECT_ROOT,
    )

    assert "sagemaker/prepare_tdc_dataset.py" in description["includes"]
    assert "src/admet_platform" in description["includes"]
    assert "sagemaker/processing_requirements.txt" in description["includes"]
    assert "configs/bbb_martins.yaml" in description["includes"]


def test_mocked_successful_submission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeProcessor:
        def __init__(self, **kwargs):
            calls["processor_args"] = kwargs

        def run(self, inputs, outputs, arguments, job_name, wait):
            calls["inputs"] = inputs
            calls["outputs"] = outputs
            calls["arguments"] = arguments
            calls["job_name"] = job_name
            calls["wait"] = wait

    monkeypatch.setattr(launcher, "build_processing_inputs", lambda config, dry_run: ["input"])
    monkeypatch.setattr(launcher, "build_processing_outputs", lambda config, dry_run: ["output"])

    result = launcher.run_processing_launch(
        CONFIG_PATH,
        dry_run=False,
        launch_plan_output=tmp_path / "submitted.json",
        wait=False,
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="submit",
        sagemaker_session_factory=lambda region: {"region": region},
        processor_class=FakeProcessor,
    )

    assert result.status == "submitted"
    assert calls["wait"] is False
    assert calls["arguments"] == ["--mode", "supplied_csv", "--endpoint-config", "configs/bbb_martins.yaml"]


def test_mocked_failed_submission_and_nonzero_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingProcessor:
        def __init__(self, **kwargs):
            pass

        def run(self, inputs, outputs, arguments, job_name, wait):
            raise RuntimeError("boom aws_secret_access_key=abc")

    monkeypatch.setattr(launcher, "build_processing_inputs", lambda config, dry_run: ["input"])
    monkeypatch.setattr(launcher, "build_processing_outputs", lambda config, dry_run: ["output"])

    with pytest.raises(launcher.SageMakerProcessingSubmissionError, match="REDACTED"):
        launcher.run_processing_launch(
            CONFIG_PATH,
            dry_run=False,
            launch_plan_output=tmp_path / "failed.json",
            sagemaker_session_factory=lambda region: {"region": region},
            processor_class=FailingProcessor,
        )

    with pytest.raises(SystemExit) as exc_info:
        launcher.main(["--config", str(CONFIG_PATH), "--launch-plan-output", str(tmp_path / "cli_failed.json")])
    assert exc_info.value.code == 1


def test_wait_and_no_wait_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    waits: list[bool] = []

    class FakeProcessor:
        def __init__(self, **kwargs):
            pass

        def run(self, inputs, outputs, arguments, job_name, wait):
            waits.append(wait)

    monkeypatch.setattr(launcher, "build_processing_inputs", lambda config, dry_run: ["input"])
    monkeypatch.setattr(launcher, "build_processing_outputs", lambda config, dry_run: ["output"])

    for wait in (True, False):
        launcher.run_processing_launch(
            CONFIG_PATH,
            dry_run=False,
            launch_plan_output=tmp_path / f"submitted_{wait}.json",
            wait=wait,
            sagemaker_session_factory=lambda region: {"region": region},
            processor_class=FakeProcessor,
        )

    assert waits == [True, False]


def test_cli_smoke_execution_in_dry_run_mode(tmp_path: Path) -> None:
    plan_path = tmp_path / "processing_launch_plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "launch_processing.py"),
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--launch-plan-output",
            str(plan_path),
            "--job-name-prefix",
            "processing-test",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run:" in result.stdout
    assert plan_path.exists()


def _effective() -> dict:
    return launcher.build_effective_config(launcher.load_processing_yaml(CONFIG_PATH), {})
