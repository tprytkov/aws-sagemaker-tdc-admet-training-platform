import json
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "docker" / "evaluation" / "Dockerfile"
WORKFLOW = PROJECT_ROOT / "scripts" / "build_evaluation_image.ps1"
REAL_BBB = PROJECT_ROOT / "outputs" / "local" / "full_benchmarks" / "bbb_martins"
REAL_OUTPUT = PROJECT_ROOT / "outputs" / "local" / "docker_evaluation_full_bbb" / "opt" / "ml" / "processing" / "output"


def test_evaluation_dockerfile_presence_and_entrypoint() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim-bookworm" in text
    assert "COPY sagemaker/evaluate_models.py ./evaluate_models.py" in text
    assert "COPY sagemaker/evaluation_requirements.txt ./evaluation_requirements.txt" in text
    assert "COPY src/admet_platform ./src/admet_platform" in text
    assert "COPY configs ./configs" in text
    assert "PYTHONPATH=/opt/program/src" in text
    assert 'ENTRYPOINT ["python", "/opt/program/evaluate_models.py"]' in text


def test_evaluation_dockerfile_uses_minimal_evaluation_requirements() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "processing_requirements.txt" not in text
    assert "evaluation_requirements.txt" in text
    assert "pip install --no-cache-dir -r evaluation_requirements.txt" in text


def test_evaluation_workflow_local_only_no_ecr_push_or_sagemaker() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    forbidden = ["docker push", "get-login-password", "get-authorization-token", "aws ecr", "sagemaker"]
    lowered = text.lower()
    for item in forbidden:
        assert item not in lowered


def test_evaluation_workflow_builds_immutable_style_default_tag() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '[string]$ImageTag = "evaluation-v1"' in text
    assert '$LocalImageName = "admet-platform-evaluation:$ImageTag"' in text
    assert '"build",' in text
    assert '"-f", "docker/evaluation/Dockerfile"' in text
    assert '"-t", $LocalImageName' in text
    assert "Assert-ImageTag -Tag $ImageTag" in text


def test_evaluation_workflow_smoke_fixture_generation_and_mount() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function New-EvaluationSmokeFixtures" in text
    assert "function New-SmokeRun" in text
    assert "outputs/local/docker_evaluation_smoke/opt/ml/processing" in text
    assert "input/runs" in text
    assert "input/config" in text
    assert "evaluation.yaml" in text
    assert "descriptors" in text
    assert "morgan" in text
    assert "/opt/ml/processing/input/runs" in text
    assert "/opt/ml/processing/input/config" in text
    assert "/opt/ml/processing/output" in text
    assert "[System.Text.UTF8Encoding]::new($false)" in text


def test_evaluation_workflow_verifies_expected_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for artifact in [
        "output/evaluation/evaluation_summary.json",
        "output/evaluation/model_comparison.csv",
        "output/evaluation/model_comparison.json",
        "output/evaluation/recommended_model.json",
        "output/evaluation/evaluation_warnings.json",
        "output/model_card/model_card.md",
        "output/registry/registry_entry.json",
        "output/metadata/evaluation_processing_manifest.json",
        "output/metadata/artifact_inventory.json",
    ]:
        assert artifact in text
    assert "expected 'completed'" in text


def test_evaluation_workflow_supports_build_only_and_smoke_only_modes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "[switch]$BuildOnly" in text
    assert "[switch]$SmokeOnly" in text
    assert "if (!$SmokeOnly)" in text
    assert "if ($BuildOnly)" in text
    assert "Build-only mode complete" in text


def test_evaluation_workflow_supports_real_artifact_mode_and_isolated_output() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "[string]$RealEvaluationInput" in text
    assert "function Invoke-RealEvaluationTest" in text
    assert "outputs/local/docker_evaluation_full_bbb/opt/ml/processing" in text
    assert "outputs/local/docker_evaluation_smoke/opt/ml/processing" in text
    assert "Running local real-artifact evaluation container test" in text
    assert "Running local synthetic evaluation container smoke test" in text
    assert "$RealEvaluationInput" in text
    assert "docker_evaluation_full_bbb" in text
    assert "docker_evaluation_smoke" in text


def test_evaluation_workflow_real_artifact_validation_paths() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function Assert-RealEvaluationInput" in text
    assert "descriptors" in text
    assert "morgan" in text
    for artifact in [
        "metrics.json",
        "training_metadata.json",
        "feature_metadata.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "model.joblib",
    ]:
        assert artifact in text
    assert "Missing required real evaluation artifact" in text
    assert "Invalid JSON in required real evaluation artifact" in text


def test_evaluation_workflow_real_mode_selection_metrics_and_counts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function Assert-RealEvaluationResult" in text
    assert 'recommended_run_id -ne "morgan"' in text
    assert "Morgan validation ROC-AUC" in text
    assert "Morgan test ROC-AUC" in text
    assert "Morgan test PR-AUC" in text
    assert "training_row_count" in text
    assert "validation_row_count" in text
    assert "test_row_count" in text
    assert "Assert-TextContainsApprox" in text
    assert "Assert-ApproxEqual" in text
    assert "0.813789" not in text
    assert "0.848812" not in text
    assert "0.947853" not in text


def test_evaluation_workflow_captures_native_command_results() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function Invoke-NativeCommand" in text
    assert "ExitCode =" in text
    assert "Stdout =" in text
    assert "Stderr =" in text
    assert "function Assert-CommandSucceeded" in text
    assert "Docker evaluation smoke test" in text
    assert "Docker real-artifact evaluation test" in text


def test_evaluation_workflow_powershell_parse() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "& { $parseErrors=$null; "
                "[System.Management.Automation.PSParser]::Tokenize("
                "(Get-Content -LiteralPath 'scripts/build_evaluation_image.ps1' -Raw), "
                "[ref]$parseErrors) > $null; "
                "if ($parseErrors.Count -gt 0) { $parseErrors | Format-List *; exit 1 } }"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not REAL_BBB.exists(), reason="local full BBB benchmark artifacts are not available")
def test_real_bbb_fixture_metrics_match_expected_values() -> None:
    morgan = json.loads((REAL_BBB / "morgan" / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((REAL_BBB / "morgan" / "training_metadata.json").read_text(encoding="utf-8"))

    assert morgan["validation"]["roc_auc"] == pytest.approx(0.8137890884896872)
    assert morgan["test"]["roc_auc"] == pytest.approx(0.8488117573483427)
    assert morgan["test"]["pr_auc"] == pytest.approx(0.9478530588970069)
    assert metadata["training_row_count"] == 1421
    assert metadata["validation_row_count"] == 203
    assert metadata["test_row_count"] == 406


@pytest.mark.skipif(not REAL_OUTPUT.exists(), reason="local real-artifact Docker evaluation output is not available")
def test_real_artifact_docker_output_recommends_morgan_and_reports_full_metrics() -> None:
    recommended = json.loads((REAL_OUTPUT / "evaluation" / "recommended_model.json").read_text(encoding="utf-8"))
    summary = json.loads((REAL_OUTPUT / "evaluation" / "evaluation_summary.json").read_text(encoding="utf-8"))
    model_card = (REAL_OUTPUT / "model_card" / "model_card.md").read_text(encoding="utf-8")

    assert recommended["recommended_run_id"] == "morgan"
    assert summary["validation_summary"]["morgan"]["roc_auc"] == pytest.approx(0.8137890884896872)
    assert summary["test_summary"]["morgan"]["roc_auc"] == pytest.approx(0.8488117573483427)
    assert summary["test_summary"]["morgan"]["pr_auc"] == pytest.approx(0.9478530588970069)
    assert summary["dataset_and_split_provenance"]["train_rows"] == 1421
    assert summary["dataset_and_split_provenance"]["validation_rows"] == 203
    assert summary["dataset_and_split_provenance"]["test_rows"] == 406
    assert "0.813789" in model_card
    assert "0.848811" in model_card or "0.848812" in model_card
    assert "0.947853" in model_card
