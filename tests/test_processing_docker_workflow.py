from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "docker" / "processing" / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
WORKFLOW = PROJECT_ROOT / "scripts" / "build_push_processing_image.ps1"


def test_processing_dockerfile_presence_and_entrypoint() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim-bookworm" in text
    assert "COPY sagemaker/prepare_tdc_dataset.py ./prepare_tdc_dataset.py" in text
    assert "COPY sagemaker/processing_requirements.txt ./processing_requirements.txt" in text
    assert "COPY src/admet_platform ./src/admet_platform" in text
    assert "COPY configs ./configs" in text
    assert "PYTHONPATH=/opt/program/src" in text
    assert 'ENTRYPOINT ["python", "/opt/program/prepare_tdc_dataset.py"]' in text


def test_docker_context_exclusions() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    required = [
        ".git/",
        "outputs/",
        "infra/terraform/.terraform/",
        "*.tfstate",
        "*.tfvars",
        "*.tfplan",
        "tests/",
        "data/",
        ".venv/",
        ".aws/",
        "credentials.csv",
        "*.safetensors",
    ]

    for pattern in required:
        assert pattern in text


def test_powershell_ecr_uri_parsing_and_immutable_tag_detection() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function Get-EcrRegistry" in text
    assert "function Get-EcrRepositoryName" in text
    assert r"[0-9]{12}\.dkr\.ecr\." in text
    assert "function Test-EcrImageTagExists" in text
    assert "describe-images" in text
    assert "ImageNotFoundException" in text
    assert "ECR image tag already exists" in text
    assert "Verification-only mode complete" in text
    assert "Assert-ImageTag -Tag $ImageTag" in text


def test_powershell_docker_and_aws_command_construction() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "[string]$AwsCliPath" in text
    assert "function Resolve-AwsCliPath" in text
    assert "Get-Command aws" in text
    assert "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe" in text
    assert "function Invoke-NativeCommand" in text
    assert "ExitCode =" in text
    assert "Stdout =" in text
    assert "Stderr =" in text
    assert "docker version" in text
    assert "docker build" in text
    assert "-f docker/processing/Dockerfile" in text
    assert "docker run --rm" in text
    assert '"ecr", "get-login-password"' in text
    assert '"login",' in text
    assert '"--username", "AWS"' in text
    assert '"--password-stdin"' in text
    assert 'Invoke-DockerCli -Arguments @("push", $remoteImage)' in text
    assert 'Invoke-DockerCli -Arguments @("--config", $login.DockerConfigDir, "push", $remoteImage)' in text
    assert 'Assert-CommandSucceeded -Result $pushResult -Operation "Docker push"' in text
    assert "describe-image-scan-findings" in text


def test_powershell_ecr_error_handling_paths() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "ExpiredToken|SSO.*expired|UnauthorizedSSOToken|Token has expired" in text
    assert "aws sso login --profile $AwsProfile" in text
    assert "AccessDenied|AccessDeniedException|not authorized|UnauthorizedOperation" in text
    assert "ECR immutable tag check" in text
    assert "ECR describe pushed image" in text
    assert "function Assert-CommandSucceeded" in text
    assert "Docker push" in text


def test_powershell_normal_login_and_temp_config_fallback_cleanup() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function Invoke-EcrDockerLogin" in text
    assert "get-login-password" in text
    assert "return [pscustomobject]@{ Mode = \"standard\"; DockerConfigDir = $null }" in text
    assert "Standard Docker login failed; using temporary Docker config fallback." in text
    assert "function New-TemporaryDockerConfigLogin" in text
    assert "get-authorization-token" in text
    assert "admet-docker-config-" in text
    assert "config.json" in text
    assert '"--config", $login.DockerConfigDir, "push", $remoteImage' in text
    assert "finally" in text
    assert "Remove-Item -Recurse -Force -LiteralPath $login.DockerConfigDir" in text


def test_powershell_null_scan_status_tolerated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function Get-EcrScanStatus" in text
    assert "return $null" in text
    assert "$scanStatus = Get-EcrScanStatus" in text


def test_powershell_secret_redaction_and_push_manifest_schema() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "function ConvertTo-RedactedText" in text
    assert "function New-ImageManifest" in text
    assert "function Write-ImageManifest" in text
    assert "************" in text
    assert "$manifestPath = \"outputs/local/processing_image_push_manifest.json\"" in text
    for field in [
        "status",
        "aws_profile",
        "region",
        "repository_uri",
        "repository_name",
        "image_tag",
        "pushed_image_uri",
        "ecr_image_tag",
        "image_digest",
        "image_pushed_at",
        "image_size_bytes",
        "local_image_size_bytes",
        "scan_status",
        "auth_mode",
        "smoke_output_root",
        "created_at",
    ]:
        assert f"{field} =" in text


def test_powershell_verify_only_existing_image_records_metadata_without_push() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "if ($VerifyOnly)" in text
    assert "without building, pushing, or overwriting" in text
    assert "$tagExists = Test-EcrImageTagExists -RepositoryName $repositoryName -Tag $ImageTag" in text
    assert 'Status "verified_existing"' in text
    assert "Get-EcrImageDetails -RepositoryName $repositoryName -Tag $ImageTag" in text
    assert "Get-EcrScanStatus -RepositoryName $repositoryName -Tag $ImageTag" in text
    assert 'AuthMode "not_required"' in text
    assert "Existing immutable tag was verified" in text
    assert "exit 0" in text


def test_powershell_verify_only_available_tag_records_manifest_without_push() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'Status "available"' in text
    assert "Tag is available and image was not pushed" in text
    assert "ImageNotFoundException" in text
    assert "return $false" in text


def test_powershell_push_mode_existing_immutable_tag_stops_before_authentication() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    tag_check = text.index("Checking immutable tag availability in ECR")
    existing_tag_stop = text.index("ECR image tag already exists and will not be overwritten")
    auth = text.index("Authenticating Docker to ECR")

    assert tag_check < existing_tag_stop < auth
