import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from admet_platform.sagemaker import launch_training as launcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sagemaker_training.yaml"


def test_yaml_loading() -> None:
    config = launcher.load_training_yaml(CONFIG_PATH)

    assert config["model_type"] == "chemberta"
    assert config["endpoint_config"] == "configs/bbb_martins.yaml"


def test_configuration_precedence() -> None:
    raw = launcher.load_training_yaml(CONFIG_PATH)
    effective = launcher.build_effective_config(
        raw,
        {
            "region": "us-east-1",
            "instance_type": "ml.m5.xlarge",
            "hyperparameters": {"epochs": 1},
        },
    )

    assert effective["region"] == "us-east-1"
    assert effective["instance_type"] == "ml.m5.xlarge"
    assert effective["hyperparameters"]["epochs"] == 1
    assert effective["model_type"] == "chemberta"


def test_required_field_validation() -> None:
    raw = launcher.load_training_yaml(CONFIG_PATH)
    raw["aws"]["role_arn"] = None
    effective = launcher.build_effective_config(raw, {})
    endpoint = launcher.load_endpoint_config(PROJECT_ROOT / "configs" / "bbb_martins.yaml")

    with pytest.raises(ValueError, match="execution role"):
        launcher.validate_effective_config(effective, endpoint)


def test_managed_image_configuration() -> None:
    strategy = launcher.resolve_image_strategy(
        {
            "strategy": "managed",
            "transformers_version": "4.37",
            "pytorch_version": "2.1",
            "py_version": "py310",
        }
    )

    assert strategy["strategy"] == "managed"
    assert strategy["transformers_version"] == "4.37"


def test_custom_image_configuration() -> None:
    strategy = launcher.resolve_image_strategy(
        {"strategy": "custom", "image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/admet:latest"}
    )

    assert strategy == {
        "strategy": "custom",
        "image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/admet:latest",
    }


def test_conflicting_image_rejection() -> None:
    with pytest.raises(ValueError, match="cannot also specify image_uri"):
        launcher.resolve_image_strategy(
            {
                "strategy": "managed",
                "image_uri": "example",
                "transformers_version": "4.37",
                "pytorch_version": "2.1",
                "py_version": "py310",
            }
        )


def test_missing_image_field_rejection() -> None:
    with pytest.raises(ValueError, match="missing"):
        launcher.resolve_image_strategy({"strategy": "managed", "transformers_version": "4.37"})


def test_s3_uri_validation() -> None:
    launcher.validate_s3_uri("s3://example-bucket/path/file.csv", "train_s3_uri")
    with pytest.raises(ValueError, match="s3://bucket/key"):
        launcher.validate_s3_uri("https://example-bucket/path/file.csv", "train_s3_uri")


def test_train_validation_test_channel_creation() -> None:
    effective = _effective()
    channels = launcher.build_training_inputs(effective, dry_run=True)

    assert set(channels) == {"train", "validation", "test"}
    assert channels["validation"]["content_type"] == "text/csv"
    assert channels["validation"]["input_mode"] == "File"


def test_hyperparameter_conversion_and_null_omission() -> None:
    effective = _effective()
    effective["hyperparameters"]["development_row_limit"] = None
    hyperparameters = launcher.build_hyperparameters(effective)

    assert hyperparameters["endpoint-config"] == "configs/bbb_martins.yaml"
    assert hyperparameters["local-files-only"] == "false"
    assert "development-row-limit" not in hyperparameters


def test_job_name_sanitization_and_length() -> None:
    name = launcher.generate_job_name(
        "Project_Name",
        "endpoint_with_really_long_identifier_" * 3,
        "chemberta",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="ABC_123",
    )

    assert len(name) <= launcher.JOB_NAME_MAX_LENGTH
    assert name.startswith("project-name-")
    assert name.endswith("abc-123")


def test_tag_kms_and_vpc_handling() -> None:
    effective = _effective()
    effective["kms_key_arn"] = "arn:aws:kms:us-west-2:123456789012:key/example"
    effective["enable_network_isolation"] = True
    effective["vpc_subnets"] = ["subnet-123"]
    effective["vpc_security_group_ids"] = ["sg-123"]

    args = launcher.build_estimator_args(
        effective=effective,
        image_strategy=launcher.resolve_image_strategy(effective["image"]),
        hyperparameters=launcher.build_hyperparameters(effective),
        source_dir=Path("packaged-source"),
    )

    assert args["tags"] == effective["tags"]
    assert args["output_kms_key"] == effective["kms_key_arn"]
    assert args["enable_network_isolation"] is True
    assert args["subnets"] == ["subnet-123"]
    assert args["security_group_ids"] == ["sg-123"]


def test_dry_run_makes_no_sagemaker_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("dry-run should not create SageMaker objects")

    monkeypatch.setattr(launcher, "create_sagemaker_session", fail)
    monkeypatch.setattr(launcher, "create_huggingface_estimator", fail)
    result = launcher.run_launch(
        CONFIG_PATH,
        dry_run=True,
        launch_plan_output=tmp_path / "launch_plan.json",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="test",
    )

    assert result.status == "dry_run"
    assert result.manifest_path.exists()


def test_dry_run_launch_plan_schema_and_redaction(tmp_path: Path) -> None:
    result = launcher.run_launch(
        CONFIG_PATH,
        dry_run=True,
        launch_plan_output=tmp_path / "launch_plan.json",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="test",
    )
    plan = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert {
        "status",
        "job_name",
        "role_arn",
        "endpoint_id",
        "task_type",
        "image_strategy",
        "s3_channels",
        "hyperparameters",
        "source_packaging",
        "warnings",
    } <= set(plan)
    assert plan["role_arn"] == "arn:aws:iam::************:role/SageMakerExecutionRole"
    assert "C:\\Users" not in json.dumps(plan)


def test_credential_and_secret_redaction() -> None:
    assert launcher.sanitize_text("aws_secret_access_key=abc hf_token=def") == "[REDACTED] [REDACTED]"


def test_estimator_argument_construction_for_managed_and_custom_images() -> None:
    effective = _effective()
    managed_args = launcher.build_estimator_args(
        effective=effective,
        image_strategy=launcher.resolve_image_strategy(effective["image"]),
        hyperparameters=launcher.build_hyperparameters(effective),
        source_dir=Path("packaged-source"),
    )
    assert managed_args["transformers_version"] == "4.37"
    assert "image_uri" not in managed_args

    effective["image"] = {"strategy": "custom", "image_uri": "example.dkr.ecr.us-west-2.amazonaws.com/admet:1"}
    custom_args = launcher.build_estimator_args(
        effective=effective,
        image_strategy=launcher.resolve_image_strategy(effective["image"]),
        hyperparameters=launcher.build_hyperparameters(effective),
        source_dir=Path("packaged-source"),
    )
    assert custom_args["image_uri"] == "example.dkr.ecr.us-west-2.amazonaws.com/admet:1"


def test_source_package_completeness_validation() -> None:
    description = launcher.validate_source_package_inputs(
        "sagemaker",
        "train_chemberta.py",
        "configs/bbb_martins.yaml",
        project_root=PROJECT_ROOT,
    )

    assert description["entry_point_at_package_root"] is True
    assert "sagemaker/train_chemberta.py" in description["includes"]
    assert "src/admet_platform" in description["includes"]
    assert "sagemaker/requirements.txt" in description["includes"]


def test_mocked_successful_fit_submission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeEstimator:
        def __init__(self, **kwargs):
            calls["estimator_args"] = kwargs

        def fit(self, channels, job_name, wait):
            calls["channels"] = channels
            calls["job_name"] = job_name
            calls["wait"] = wait

    monkeypatch.setattr(launcher, "build_training_inputs", lambda config, dry_run: {"train": "x", "validation": "y", "test": "z"})
    result = launcher.run_launch(
        CONFIG_PATH,
        dry_run=False,
        launch_plan_output=tmp_path / "submitted.json",
        wait=False,
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        suffix="submit",
        sagemaker_session_factory=lambda region: {"region": region},
        estimator_class=FakeEstimator,
    )

    assert result.status == "submitted"
    assert calls["wait"] is False
    assert calls["job_name"] == result.job_name
    assert calls["channels"] == {"train": "x", "validation": "y", "test": "z"}


def test_mocked_submission_failure_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingEstimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, channels, job_name, wait):
            raise RuntimeError("boom aws_secret_access_key=abc")

    monkeypatch.setattr(launcher, "build_training_inputs", lambda config, dry_run: {"train": "x", "validation": "y", "test": "z"})
    with pytest.raises(launcher.SageMakerSubmissionError, match="REDACTED"):
        launcher.run_launch(
            CONFIG_PATH,
            dry_run=False,
            launch_plan_output=tmp_path / "failed.json",
            sagemaker_session_factory=lambda region: {"region": region},
            estimator_class=FailingEstimator,
        )

    with pytest.raises(SystemExit) as exc_info:
        launcher.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--launch-plan-output",
                str(tmp_path / "cli_failed.json"),
            ]
        )
    assert exc_info.value.code == 1


def test_wait_and_no_wait_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    waits: list[bool] = []

    class FakeEstimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, channels, job_name, wait):
            waits.append(wait)

    monkeypatch.setattr(launcher, "build_training_inputs", lambda config, dry_run: {"train": "x", "validation": "y", "test": "z"})
    for wait in (True, False):
        launcher.run_launch(
            CONFIG_PATH,
            dry_run=False,
            launch_plan_output=tmp_path / f"submitted_{wait}.json",
            wait=wait,
            sagemaker_session_factory=lambda region: {"region": region},
            estimator_class=FakeEstimator,
        )

    assert waits == [True, False]


def test_cli_smoke_execution_in_dry_run_mode(tmp_path: Path) -> None:
    plan_path = tmp_path / "launch_plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "launch_training.py"),
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--launch-plan-output",
            str(plan_path),
            "--job-name-prefix",
            "admet-test",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run:" in result.stdout
    assert plan_path.exists()


def _effective() -> dict:
    return launcher.build_effective_config(launcher.load_training_yaml(CONFIG_PATH), {})
