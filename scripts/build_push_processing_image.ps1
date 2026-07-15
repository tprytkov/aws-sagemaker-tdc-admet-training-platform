param(
    [string]$AwsProfile = "admet-platform",
    [string]$Region = "us-west-2",
    [Parameter(Mandatory = $true)]
    [string]$RepositoryUri,
    [string]$ImageTag = "processing-v1",
    [string]$LocalImageName = "admet-platform-processing:local",
    [string]$AwsCliPath,
    [switch]$BuildOnly,
    [switch]$SmokeOnly,
    [switch]$Push,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

function Assert-ImageTag {
    param([string]$Tag)
    if ([string]::IsNullOrWhiteSpace($Tag) -or $Tag -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$') {
        throw "Invalid image tag '$Tag'. Use Docker/ECR-compatible tag characters."
    }
}

function Get-EcrRegistry {
    param([string]$Uri)
    if ($Uri -notmatch '^(?<registry>[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?)/(?<repository>[A-Za-z0-9._/-]+)$') {
        throw "RepositoryUri must look like '<account>.dkr.ecr.<region>.amazonaws.com/<repository>'."
    }
    return $Matches.registry
}

function Get-EcrRepositoryName {
    param([string]$Uri)
    if ($Uri -notmatch '^[^/]+/(?<repository>[A-Za-z0-9._/-]+)$') {
        throw "RepositoryUri is missing a repository name."
    }
    return $Matches.repository
}

function ConvertTo-RedactedText {
    param([string]$Text)
    $redacted = $Text -replace '([0-9]{12})(?=\.dkr\.ecr\.)', '************'
    $redacted = $redacted -replace '(arn:aws:iam::)[0-9]{12}(:)', '$1************$2'
    return $redacted
}

function Resolve-AwsCliPath {
    param([string]$ExplicitPath)
    if (![string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (!(Test-Path -LiteralPath $ExplicitPath)) {
            throw "AWS CLI path was provided but does not exist: $ExplicitPath"
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }
    $command = Get-Command aws -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $fallbacks = @(
        "C:\Program Files\Amazon\AWSCLIV2\aws.exe",
        "C:\Program Files (x86)\Amazon\AWSCLIV2\aws.exe",
        "$env:USERPROFILE\AppData\Local\Programs\Amazon\AWSCLIV2\aws.exe"
    )
    foreach ($path in $fallbacks) {
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }
    throw "AWS CLI executable was not found. Install AWS CLI v2, add aws.exe to PATH, or pass -AwsCliPath."
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$StandardInput
    )
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("admet-native-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    $stdoutPath = Join-Path $tempRoot "stdout.txt"
    $stderrPath = Join-Path $tempRoot "stderr.txt"
    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $FilePath
        foreach ($argument in $Arguments) {
            [void]$psi.ArgumentList.Add($argument)
        }
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.RedirectStandardInput = $null -ne $StandardInput
        $psi.UseShellExecute = $false
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        [void]$process.Start()
        if ($null -ne $StandardInput) {
            $process.StandardInput.Write($StandardInput)
            $process.StandardInput.Close()
        }
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = $stdout
            Stderr = $stderr
        }
    }
    finally {
        Remove-Item -Recurse -Force $tempRoot -ErrorAction SilentlyContinue
    }
}

function Invoke-AwsCli {
    param([string[]]$Arguments)
    return Invoke-NativeCommand -FilePath $script:ResolvedAwsCliPath -Arguments $Arguments
}

function Invoke-DockerCli {
    param(
        [string[]]$Arguments,
        [string]$StandardInput
    )
    return Invoke-NativeCommand -FilePath "docker" -Arguments $Arguments -StandardInput $StandardInput
}

function Get-AwsErrorText {
    param($Result)
    return (($Result.Stderr + "`n" + $Result.Stdout).Trim())
}

function Assert-AwsCommandSucceeded {
    param(
        $Result,
        [string]$Operation
    )
    if ($Result.ExitCode -eq 0) {
        return
    }
    $message = Get-AwsErrorText -Result $Result
    if ($message -match "ExpiredToken|SSO.*expired|UnauthorizedSSOToken|Token has expired") {
        throw "$Operation failed because the AWS SSO/session token is expired. Run 'aws sso login --profile $AwsProfile' and retry."
    }
    if ($message -match "AccessDenied|AccessDeniedException|not authorized|UnauthorizedOperation") {
        throw "$Operation failed because the AWS identity lacks required ECR permissions: $(ConvertTo-RedactedText $message)"
    }
    throw "$Operation failed: $(ConvertTo-RedactedText $message)"
}

function Test-EcrImageTagExists {
    param(
        [string]$RepositoryName,
        [string]$Tag
    )
    $result = Invoke-AwsCli -Arguments @(
        "ecr", "describe-images",
        "--profile", $AwsProfile,
        "--region", $Region,
        "--repository-name", $RepositoryName,
        "--image-ids", "imageTag=$Tag",
        "--output", "json"
    )
    if ($result.ExitCode -eq 0) {
        return $true
    }
    $message = Get-AwsErrorText -Result $result
    if ($message -match "ImageNotFoundException") {
        return $false
    }
    Assert-AwsCommandSucceeded -Result $result -Operation "ECR immutable tag check"
}

function Invoke-SmokeTest {
    param([string]$ImageName)
    $smokeRoot = Join-Path (Get-Location) "outputs/local/docker_processing_smoke/opt/ml/processing"
    if (Test-Path $smokeRoot) {
        Remove-Item -Recurse -Force $smokeRoot
    }
    New-Item -ItemType Directory -Force `
        -Path (Join-Path $smokeRoot "input/config"), `
              (Join-Path $smokeRoot "input/data"), `
              (Join-Path $smokeRoot "output") | Out-Null
    Copy-Item -LiteralPath "configs/bbb_martins.yaml" -Destination (Join-Path $smokeRoot "input/config/bbb_martins.yaml")
    Copy-Item -LiteralPath "data/sample/bbb_martins_sample.csv" -Destination (Join-Path $smokeRoot "input/data/bbb_martins_sample.csv")

    $containerMount = "${smokeRoot}:/opt/ml/processing"
    docker run --rm `
        -v $containerMount `
        $ImageName `
        --mode supplied_csv `
        --endpoint-config /opt/ml/processing/input/config/bbb_martins.yaml `
        --input-data-dir /opt/ml/processing/input/data `
        --output-dir /opt/ml/processing/output

    $required = @(
        "output/train/train.csv",
        "output/validation/valid.csv",
        "output/test/test.csv",
        "output/metadata/data_profile.json",
        "output/metadata/split_metadata.json",
        "output/metadata/rejected_rows.csv",
        "output/metadata/processing_manifest.json"
    )
    foreach ($relative in $required) {
        $path = Join-Path $smokeRoot $relative
        if (!(Test-Path $path)) {
            throw "Smoke test missing expected artifact: $path"
        }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $smokeRoot "output/metadata/processing_manifest.json") -Raw | ConvertFrom-Json
    if ($manifest.status -ne "completed") {
        throw "Smoke test manifest status was '$($manifest.status)', expected 'completed'."
    }
    return $smokeRoot
}

function Invoke-EcrDockerLogin {
    param([string]$Registry)
    $passwordResult = Invoke-AwsCli -Arguments @(
        "ecr", "get-login-password",
        "--profile", $AwsProfile,
        "--region", $Region
    )
    Assert-AwsCommandSucceeded -Result $passwordResult -Operation "ECR get-login-password"
    $loginResult = Invoke-DockerCli -Arguments @(
        "login",
        "--username", "AWS",
        "--password-stdin", $Registry
    ) -StandardInput $passwordResult.Stdout
    if ($loginResult.ExitCode -eq 0) {
        return [pscustomobject]@{ Mode = "standard"; DockerConfigDir = $null }
    }
    Write-Host "Standard Docker login failed; using temporary Docker config fallback."
    return New-TemporaryDockerConfigLogin -Registry $Registry
}

function New-TemporaryDockerConfigLogin {
    param([string]$Registry)
    $authResult = Invoke-AwsCli -Arguments @(
        "ecr", "get-authorization-token",
        "--profile", $AwsProfile,
        "--region", $Region,
        "--output", "json"
    )
    Assert-AwsCommandSucceeded -Result $authResult -Operation "ECR get-authorization-token"
    $payload = $authResult.Stdout | ConvertFrom-Json
    $authorizationToken = $payload.authorizationData[0].authorizationToken
    if ([string]::IsNullOrWhiteSpace($authorizationToken)) {
        throw "ECR get-authorization-token returned no authorization token."
    }
    $tempDockerConfig = Join-Path ([System.IO.Path]::GetTempPath()) ("admet-docker-config-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempDockerConfig | Out-Null
    $dockerConfig = @{
        auths = @{
            $Registry = @{
                auth = $authorizationToken
            }
        }
    }
    $dockerConfig | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $tempDockerConfig "config.json") -Encoding UTF8
    return [pscustomobject]@{ Mode = "temporary_config"; DockerConfigDir = $tempDockerConfig }
}

function Get-EcrImageDetails {
    param(
        [string]$RepositoryName,
        [string]$Tag
    )
    $result = Invoke-AwsCli -Arguments @(
        "ecr", "describe-images",
        "--profile", $AwsProfile,
        "--region", $Region,
        "--repository-name", $RepositoryName,
        "--image-ids", "imageTag=$Tag",
        "--output", "json"
    )
    Assert-AwsCommandSucceeded -Result $result -Operation "ECR describe pushed image"
    return ($result.Stdout | ConvertFrom-Json).imageDetails[0]
}

function Get-EcrScanStatus {
    param(
        [string]$RepositoryName,
        [string]$Tag
    )
    $result = Invoke-AwsCli -Arguments @(
        "ecr", "describe-image-scan-findings",
        "--profile", $AwsProfile,
        "--region", $Region,
        "--repository-name", $RepositoryName,
        "--image-id", "imageTag=$Tag",
        "--output", "json"
    )
    if ($result.ExitCode -ne 0) {
        return $null
    }
    $payload = $result.Stdout | ConvertFrom-Json
    if ($null -eq $payload.imageScanStatus) {
        return $null
    }
    return $payload.imageScanStatus.status
}

function New-ImageManifest {
    param(
        [string]$Status,
        $ImageDescription,
        [string]$ScanStatus,
        [string]$AuthMode,
        [string]$SmokeOutputRoot,
        [Nullable[Int64]]$LocalImageSizeBytes
    )
    $tag = $null
    $digest = $null
    $pushedAt = $null
    $sizeBytes = $null
    if ($null -ne $ImageDescription) {
        $tag = $ImageDescription.imageTags[0]
        $digest = $ImageDescription.imageDigest
        $pushedAt = $ImageDescription.imagePushedAt
        $sizeBytes = $ImageDescription.imageSizeInBytes
    }
    return [ordered]@{
        status = $Status
        aws_profile = $AwsProfile
        region = $Region
        repository_uri = ConvertTo-RedactedText $RepositoryUri
        repository_name = $repositoryName
        image_tag = $ImageTag
        pushed_image_uri = ConvertTo-RedactedText $remoteImage
        ecr_image_tag = $tag
        image_digest = $digest
        image_pushed_at = $pushedAt
        image_size_bytes = $sizeBytes
        local_image_size_bytes = $LocalImageSizeBytes
        scan_status = $ScanStatus
        auth_mode = $AuthMode
        smoke_output_root = $SmokeOutputRoot
        created_at = (Get-Date).ToUniversalTime().ToString("o")
    }
}

function Write-ImageManifest {
    param($Manifest)
    New-Item -ItemType Directory -Force -Path (Split-Path $manifestPath) | Out-Null
    $Manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "Wrote sanitized image manifest: $manifestPath"
}

function Assert-CommandSucceeded {
    param(
        $Result,
        [string]$Operation
    )
    if ($Result.ExitCode -eq 0) {
        return
    }
    $message = (($Result.Stderr + "`n" + $Result.Stdout).Trim())
    throw "$Operation failed: $(ConvertTo-RedactedText $message)"
}

Assert-ImageTag -Tag $ImageTag
$script:ResolvedAwsCliPath = $null
$registry = Get-EcrRegistry -Uri $RepositoryUri
$repositoryName = Get-EcrRepositoryName -Uri $RepositoryUri
$remoteImage = "${RepositoryUri}:${ImageTag}"
$manifestPath = "outputs/local/processing_image_push_manifest.json"

if ($VerifyOnly) {
    $script:ResolvedAwsCliPath = Resolve-AwsCliPath -ExplicitPath $AwsCliPath
    Write-Host "Verification-only mode: checking ECR image tag without building, pushing, or overwriting."
    $tagExists = Test-EcrImageTagExists -RepositoryName $repositoryName -Tag $ImageTag
    if ($tagExists) {
        $imageDescription = Get-EcrImageDetails -RepositoryName $repositoryName -Tag $ImageTag
        $scanStatus = Get-EcrScanStatus -RepositoryName $repositoryName -Tag $ImageTag
        $manifest = New-ImageManifest `
            -Status "verified_existing" `
            -ImageDescription $imageDescription `
            -ScanStatus $scanStatus `
            -AuthMode "not_required" `
            -SmokeOutputRoot $null `
            -LocalImageSizeBytes $null
        Write-ImageManifest -Manifest $manifest
        Write-Host "Verification-only mode complete. Existing immutable tag was verified: $(ConvertTo-RedactedText $remoteImage)"
        exit 0
    }
    $manifest = New-ImageManifest `
        -Status "available" `
        -ImageDescription $null `
        -ScanStatus $null `
        -AuthMode "not_required" `
        -SmokeOutputRoot $null `
        -LocalImageSizeBytes $null
    Write-ImageManifest -Manifest $manifest
    Write-Host "Verification-only mode complete. Tag is available and image was not pushed: $(ConvertTo-RedactedText $remoteImage)"
    exit 0
}

Write-Host "Checking Docker availability..."
docker version | Out-Null

if (!$SmokeOnly) {
    Write-Host "Building local image $LocalImageName..."
    docker build `
        -f docker/processing/Dockerfile `
        -t $LocalImageName `
        .
}

Write-Host "Running local Processing container smoke test..."
$smokeOutputRoot = Invoke-SmokeTest -ImageName $LocalImageName

$imageInspect = docker image inspect $LocalImageName | ConvertFrom-Json
$localImageSizeBytes = [int64]$imageInspect[0].Size

if ($BuildOnly -and !$Push) {
    Write-Host "Build-only mode complete. Smoke output: $smokeOutputRoot"
    exit 0
}

if (!$Push -and !$VerifyOnly) {
    Write-Host "Push mode was not requested. Smoke output: $smokeOutputRoot"
    exit 0
}

$script:ResolvedAwsCliPath = Resolve-AwsCliPath -ExplicitPath $AwsCliPath
Write-Host "Checking immutable tag availability in ECR..."
if (Test-EcrImageTagExists -RepositoryName $repositoryName -Tag $ImageTag) {
    throw "ECR image tag already exists and will not be overwritten: $(ConvertTo-RedactedText $remoteImage)"
}

$login = $null
try {
    Write-Host "Authenticating Docker to ECR..."
    $login = Invoke-EcrDockerLogin -Registry $registry

    Write-Host "Tagging and pushing image..."
    docker tag $LocalImageName $remoteImage
    if ($login.Mode -eq "temporary_config") {
        $pushResult = Invoke-DockerCli -Arguments @("--config", $login.DockerConfigDir, "push", $remoteImage)
    }
    else {
        $pushResult = Invoke-DockerCli -Arguments @("push", $remoteImage)
    }
    Assert-CommandSucceeded -Result $pushResult -Operation "Docker push"

    $imageDescription = Get-EcrImageDetails -RepositoryName $repositoryName -Tag $ImageTag
    $scanStatus = Get-EcrScanStatus -RepositoryName $repositoryName -Tag $ImageTag

    $pushManifest = New-ImageManifest `
        -Status "pushed" `
        -ImageDescription $imageDescription `
        -ScanStatus $scanStatus `
        -AuthMode $login.Mode `
        -SmokeOutputRoot $smokeOutputRoot `
        -LocalImageSizeBytes $localImageSizeBytes
    Write-ImageManifest -Manifest $pushManifest
}
finally {
    if ($null -ne $login -and $login.Mode -eq "temporary_config" -and (Test-Path -LiteralPath $login.DockerConfigDir)) {
        Remove-Item -Recurse -Force -LiteralPath $login.DockerConfigDir -ErrorAction SilentlyContinue
    }
}
