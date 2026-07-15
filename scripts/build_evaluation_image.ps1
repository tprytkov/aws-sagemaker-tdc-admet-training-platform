param(
    [string]$ImageTag = "evaluation-v1",
    [string]$LocalImageName,
    [string]$RealEvaluationInput,
    [switch]$BuildOnly,
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"

function Assert-ImageTag {
    param([string]$Tag)
    if ([string]::IsNullOrWhiteSpace($Tag) -or $Tag -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$') {
        throw "Invalid image tag '$Tag'. Use Docker-compatible tag characters."
    }
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("admet-evaluation-native-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    $stdoutPath = Join-Path $tempRoot "stdout.txt"
    $stderrPath = Join-Path $tempRoot "stderr.txt"
    try {
        $argumentLine = ($Arguments | ForEach-Object { ConvertTo-CommandLineArgument -Value $_ }) -join " "
        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $argumentLine `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue
            Stderr = Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue
        }
    }
    finally {
        Remove-Item -Recurse -Force -LiteralPath $tempRoot -ErrorAction SilentlyContinue
    }
}

function ConvertTo-CommandLineArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Assert-CommandSucceeded {
    param(
        $Result,
        [string]$Operation
    )
    if ($Result.ExitCode -ne 0) {
        $message = (($Result.Stderr + "`n" + $Result.Stdout).Trim())
        throw "$Operation failed: $message"
    }
}

function Invoke-DockerCli {
    param([string[]]$Arguments)
    return Invoke-NativeCommand -FilePath "docker" -Arguments $Arguments
}

function Write-TextFile {
    param(
        [string]$Path,
        [string]$Content
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $Path) | Out-Null
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath (Split-Path $Path)).Path + [System.IO.Path]::DirectorySeparatorChar + (Split-Path $Path -Leaf), $Content, $encoding)
}

function New-EvaluationSmokeFixtures {
    param([string]$SmokeRoot)
    if (Test-Path -LiteralPath $SmokeRoot) {
        Remove-Item -Recurse -Force -LiteralPath $SmokeRoot
    }
    $runsRoot = Join-Path $SmokeRoot "input/runs"
    $configRoot = Join-Path $SmokeRoot "input/config"
    New-Item -ItemType Directory -Force -Path $runsRoot, $configRoot, (Join-Path $SmokeRoot "output") | Out-Null
    Write-TextFile -Path (Join-Path $configRoot "evaluation.yaml") -Content @"
endpoint_id: bbb_martins
near_tie_tolerance: 0.01
include_development_runs: false
registry_schema_version: 1.0.0
"@
    New-SmokeRun -RunsRoot $runsRoot -RunId "descriptors" -FeatureType "descriptors" -ValidationRoc "0.70" -TestRoc "0.72"
    New-SmokeRun -RunsRoot $runsRoot -RunId "morgan" -FeatureType "morgan" -ValidationRoc "0.80" -TestRoc "0.77"
}

function New-SmokeRun {
    param(
        [string]$RunsRoot,
        [string]$RunId,
        [string]$FeatureType,
        [string]$ValidationRoc,
        [string]$TestRoc
    )
    $runDir = Join-Path $RunsRoot $RunId
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    Write-TextFile -Path (Join-Path $runDir "metrics.json") -Content @"
{
  "endpoint_id": "bbb_martins",
  "task_type": "binary_classification",
  "feature_type": "$FeatureType",
  "model_type": "${FeatureType}_model",
  "validation": {
    "roc_auc": $ValidationRoc,
    "pr_auc": 0.70,
    "balanced_accuracy": 0.60,
    "f1": 0.50,
    "matthews_correlation_coefficient": 0.40
  },
  "test": {
    "roc_auc": $TestRoc,
    "pr_auc": 0.68,
    "balanced_accuracy": 0.61,
    "f1": 0.51,
    "matthews_correlation_coefficient": 0.41
  },
  "warnings": []
}
"@
    Write-TextFile -Path (Join-Path $runDir "training_metadata.json") -Content @"
{
  "run_id": "$RunId",
  "endpoint_id": "bbb_martins",
  "task_type": "binary_classification",
  "source_dataset": "BBB_Martins",
  "feature_type": "$FeatureType",
  "model_type": "${FeatureType}_model",
  "training_row_count": 10,
  "validation_row_count": 3,
  "test_row_count": 3,
  "feature_count": 10,
  "development_row_limit": null,
  "package_versions": {"pandas": "smoke"},
  "warnings": []
}
"@
    Write-TextFile -Path (Join-Path $runDir "feature_metadata.json") -Content @"
{"feature_type": "$FeatureType", "n_features": 10}
"@
    Write-TextFile -Path (Join-Path $runDir "model.joblib") -Content "synthetic smoke model artifact"
    $predictionCsv = @"
observed_target,predicted_class,predicted_probability
0,0,0.2
1,1,0.8
1,0,0.4
"@
    Write-TextFile -Path (Join-Path $runDir "predictions_validation.csv") -Content $predictionCsv
    Write-TextFile -Path (Join-Path $runDir "predictions_test.csv") -Content $predictionCsv
}

function Invoke-SmokeTest {
    param([string]$ImageName)
    $smokeRoot = Join-Path (Get-Location) "outputs/local/docker_evaluation_smoke/opt/ml/processing"
    New-EvaluationSmokeFixtures -SmokeRoot $smokeRoot
    $containerMount = "${smokeRoot}:/opt/ml/processing"
    $result = Invoke-DockerCli -Arguments @(
        "run", "--rm",
        "-v", $containerMount,
        $ImageName,
        "--runs-dir", "/opt/ml/processing/input/runs",
        "--config-dir", "/opt/ml/processing/input/config",
        "--output-dir", "/opt/ml/processing/output"
    )
    Assert-CommandSucceeded -Result $result -Operation "Docker evaluation smoke test"
    $required = @(
        "output/evaluation/evaluation_summary.json",
        "output/evaluation/model_comparison.csv",
        "output/evaluation/model_comparison.json",
        "output/evaluation/recommended_model.json",
        "output/evaluation/evaluation_warnings.json",
        "output/model_card/model_card.md",
        "output/registry/registry_entry.json",
        "output/metadata/evaluation_processing_manifest.json",
        "output/metadata/artifact_inventory.json"
    )
    foreach ($relative in $required) {
        $path = Join-Path $smokeRoot $relative
        if (!(Test-Path -LiteralPath $path)) {
            throw "Smoke test missing expected artifact: $path"
        }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $smokeRoot "output/metadata/evaluation_processing_manifest.json") -Raw | ConvertFrom-Json
    if ($manifest.status -ne "completed") {
        throw "Smoke test manifest status was '$($manifest.status)', expected 'completed'."
    }
    return $smokeRoot
}

function Assert-RunArtifactJson {
    param(
        [string]$Path,
        [string]$Label
    )
    if (!(Test-Path -LiteralPath $Path)) {
        throw "Missing required real evaluation artifact for ${Label}: $Path"
    }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Invalid JSON in required real evaluation artifact for ${Label}: $Path"
    }
}

function Assert-RealEvaluationInput {
    param([string]$RunsRoot)
    foreach ($runId in @("descriptors", "morgan")) {
        $runDir = Join-Path $RunsRoot $runId
        if (!(Test-Path -LiteralPath $runDir)) {
            throw "Missing required real evaluation run directory: $runDir"
        }
        [void](Assert-RunArtifactJson -Path (Join-Path $runDir "metrics.json") -Label $runId)
        [void](Assert-RunArtifactJson -Path (Join-Path $runDir "training_metadata.json") -Label $runId)
        [void](Assert-RunArtifactJson -Path (Join-Path $runDir "feature_metadata.json") -Label $runId)
        foreach ($artifact in @("predictions_validation.csv", "predictions_test.csv", "model.joblib")) {
            $path = Join-Path $runDir $artifact
            if (!(Test-Path -LiteralPath $path)) {
                throw "Missing required real evaluation artifact for ${runId}: $path"
            }
        }
    }
}

function Assert-ApproxEqual {
    param(
        [double]$Actual,
        [double]$Expected,
        [string]$Label,
        [double]$Tolerance = 0.000001
    )
    if ([Math]::Abs($Actual - $Expected) -gt $Tolerance) {
        throw "$Label was $Actual, expected approximately $Expected."
    }
}

function Assert-TextContainsApprox {
    param(
        [string]$Text,
        [double]$Value,
        [string]$Label
    )
    $rounded = [Math]::Round($Value, 6).ToString("0.######", [System.Globalization.CultureInfo]::InvariantCulture)
    if (!$Text.Contains($rounded)) {
        throw "$Label value $rounded was not found in the generated model card."
    }
}

function Assert-EvaluationOutputs {
    param([string]$ProcessingRoot)
    $required = @(
        "output/evaluation/evaluation_summary.json",
        "output/evaluation/model_comparison.csv",
        "output/evaluation/model_comparison.json",
        "output/evaluation/recommended_model.json",
        "output/evaluation/evaluation_warnings.json",
        "output/model_card/model_card.md",
        "output/registry/registry_entry.json",
        "output/metadata/evaluation_processing_manifest.json",
        "output/metadata/artifact_inventory.json"
    )
    foreach ($relative in $required) {
        $path = Join-Path $ProcessingRoot $relative
        if (!(Test-Path -LiteralPath $path)) {
            throw "Evaluation container missing expected artifact: $path"
        }
    }
}

function Invoke-RealEvaluationTest {
    param(
        [string]$ImageName,
        [string]$InputPath
    )
    $realInput = (Resolve-Path -LiteralPath $InputPath).Path
    Assert-RealEvaluationInput -RunsRoot $realInput

    $processingRoot = Join-Path (Get-Location) "outputs/local/docker_evaluation_full_bbb/opt/ml/processing"
    if (Test-Path -LiteralPath $processingRoot) {
        Remove-Item -Recurse -Force -LiteralPath $processingRoot
    }
    $configRoot = Join-Path $processingRoot "input/config"
    New-Item -ItemType Directory -Force -Path $configRoot, (Join-Path $processingRoot "output") | Out-Null
    Write-TextFile -Path (Join-Path $configRoot "evaluation.yaml") -Content @"
endpoint_id: bbb_martins
near_tie_tolerance: 0.01
include_development_runs: false
registry_schema_version: 1.0.0
"@

    $processingMount = "${processingRoot}:/opt/ml/processing"
    $runsMount = "${realInput}:/opt/ml/processing/input/runs:ro"
    $result = Invoke-DockerCli -Arguments @(
        "run", "--rm",
        "-v", $processingMount,
        "-v", $runsMount,
        $ImageName,
        "--runs-dir", "/opt/ml/processing/input/runs",
        "--config-dir", "/opt/ml/processing/input/config",
        "--output-dir", "/opt/ml/processing/output"
    )
    Assert-CommandSucceeded -Result $result -Operation "Docker real-artifact evaluation test"
    Assert-EvaluationOutputs -ProcessingRoot $processingRoot
    Assert-RealEvaluationResult -ProcessingRoot $processingRoot -InputRoot $realInput
    return $processingRoot
}

function Assert-RealEvaluationResult {
    param(
        [string]$ProcessingRoot,
        [string]$InputRoot
    )
    $recommended = Get-Content -LiteralPath (Join-Path $ProcessingRoot "output/evaluation/recommended_model.json") -Raw | ConvertFrom-Json
    if ($recommended.recommended_run_id -ne "morgan") {
        throw "Expected Morgan to be recommended from validation ROC-AUC, got '$($recommended.recommended_run_id)'."
    }
    $morganMetrics = Get-Content -LiteralPath (Join-Path $InputRoot "morgan/metrics.json") -Raw | ConvertFrom-Json
    $morganMetadata = Get-Content -LiteralPath (Join-Path $InputRoot "morgan/training_metadata.json") -Raw | ConvertFrom-Json
    $summary = Get-Content -LiteralPath (Join-Path $ProcessingRoot "output/evaluation/evaluation_summary.json") -Raw | ConvertFrom-Json
    Assert-ApproxEqual -Actual ([double]$summary.validation_summary.morgan.roc_auc) -Expected ([double]$morganMetrics.validation.roc_auc) -Label "Morgan validation ROC-AUC"
    Assert-ApproxEqual -Actual ([double]$summary.test_summary.morgan.roc_auc) -Expected ([double]$morganMetrics.test.roc_auc) -Label "Morgan test ROC-AUC"
    Assert-ApproxEqual -Actual ([double]$summary.test_summary.morgan.pr_auc) -Expected ([double]$morganMetrics.test.pr_auc) -Label "Morgan test PR-AUC"
    if ($null -ne $summary.dataset_and_split_provenance.train_rows) {
        if ([int]$summary.dataset_and_split_provenance.train_rows -ne [int]$morganMetadata.training_row_count) {
            throw "Unexpected train count in evaluation summary."
        }
        if ([int]$summary.dataset_and_split_provenance.validation_rows -ne [int]$morganMetadata.validation_row_count) {
            throw "Unexpected validation count in evaluation summary."
        }
        if ([int]$summary.dataset_and_split_provenance.test_rows -ne [int]$morganMetadata.test_row_count) {
            throw "Unexpected test count in evaluation summary."
        }
    }
    $modelCard = Get-Content -LiteralPath (Join-Path $ProcessingRoot "output/model_card/model_card.md") -Raw
    Assert-TextContainsApprox -Text $modelCard -Value ([double]$morganMetrics.validation.roc_auc) -Label "Morgan validation ROC-AUC"
    Assert-TextContainsApprox -Text $modelCard -Value ([double]$morganMetrics.test.roc_auc) -Label "Morgan test ROC-AUC"
    Assert-TextContainsApprox -Text $modelCard -Value ([double]$morganMetrics.test.pr_auc) -Label "Morgan test PR-AUC"
    foreach ($count in @($morganMetadata.training_row_count, $morganMetadata.validation_row_count, $morganMetadata.test_row_count)) {
        if (!$modelCard.Contains([string]$count) -and !((Get-Content -LiteralPath (Join-Path $ProcessingRoot "output/evaluation/evaluation_summary.json") -Raw).Contains([string]$count))) {
            throw "Expected split count $count was not found in model card or evaluation summary."
        }
    }
}

Assert-ImageTag -Tag $ImageTag
if ([string]::IsNullOrWhiteSpace($LocalImageName)) {
    $LocalImageName = "admet-platform-evaluation:$ImageTag"
}

Write-Host "Checking Docker availability..."
Assert-CommandSucceeded -Result (Invoke-DockerCli -Arguments @("version")) -Operation "Docker availability check"

if (!$SmokeOnly) {
    Write-Host "Building local evaluation image $LocalImageName..."
    Assert-CommandSucceeded -Result (Invoke-DockerCli -Arguments @(
        "build",
        "-f", "docker/evaluation/Dockerfile",
        "-t", $LocalImageName,
        "."
    )) -Operation "Docker evaluation image build"
}

if ($BuildOnly) {
    Write-Host "Build-only mode complete: $LocalImageName"
    exit 0
}

if ([string]::IsNullOrWhiteSpace($RealEvaluationInput)) {
    Write-Host "Running local synthetic evaluation container smoke test..."
    $outputRoot = Invoke-SmokeTest -ImageName $LocalImageName
}
else {
    Write-Host "Running local real-artifact evaluation container test..."
    $outputRoot = Invoke-RealEvaluationTest -ImageName $LocalImageName -InputPath $RealEvaluationInput
}
$imageInspect = Invoke-DockerCli -Arguments @("image", "inspect", $LocalImageName)
Assert-CommandSucceeded -Result $imageInspect -Operation "Docker image inspect"
$image = ($imageInspect.Stdout | ConvertFrom-Json)[0]
Write-Host "Evaluation output: $outputRoot"
Write-Host "Local image: $LocalImageName"
Write-Host "Local image size bytes: $($image.Size)"
