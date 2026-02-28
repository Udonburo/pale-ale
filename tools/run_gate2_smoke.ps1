$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

$dataDir = Join-Path $repoRoot "data\gate2"
$runA = Join-Path $repoRoot "runs\gate2_smoke_A"
$runB = Join-Path $repoRoot "runs\gate2_smoke_B"
$runKink = Join-Path $repoRoot "runs\gate2_smoke_kink"

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$genScript = Join-Path $repoRoot "tools\gen_gate2_input.py"
$smoothInput = Join-Path $dataDir "smooth_4x24.json"
$kinkInput = Join-Path $dataDir "kink_4x24.json"

Write-Host "[1/5] Generating synthetic inputs..."
python $genScript --mode smooth --n-samples 4 --n-steps 24 --seed 7 --noise 0.02 --run-id gate2_smooth_4x24 --out $smoothInput
python $genScript --mode kink --n-samples 4 --n-steps 24 --seed 7 --noise 0.02 --kink-at 12 --run-id gate2_kink_4x24 --out $kinkInput

Write-Host "[2/5] Building CLI..."
cargo build -p pale-ale-cli --release

$cliPath = Join-Path $repoRoot "target\release\pale-ale.exe"
if (-not (Test-Path $cliPath)) {
    $cliPath = Join-Path $repoRoot "target\release\pale-ale"
}
if (-not (Test-Path $cliPath)) {
    throw "Could not locate built CLI binary in target/release."
}

function Invoke-Gate2Run {
    param(
        [string]$InputPath,
        [string]$OutDir,
        [string]$EncoderId
    )

    if (Test-Path $OutDir) {
        Remove-Item -Recurse -Force $OutDir
    }

    & $cliPath gate2 run `
      --input $InputPath `
      --out $OutDir `
      --dataset-revision-id synthetic_smoke_v1 `
      --dataset-hash-blake3 0000000000000000000000000000000000000000000000000000000000000000 `
      --spec-hash-raw-blake3 0000000000000000000000000000000000000000000000000000000000000000 `
      --spec-hash-blake3 0000000000000000000000000000000000000000000000000000000000000000 `
      --unitization-id synthetic_fixed_24 `
      --rotor-encoder-id $EncoderId `
      --rotor-encoder-preproc-id identity_v1 `
      --vec8-postproc-id identity_v1 `
      --evaluation-mode-id unsupervised_v1
}

Write-Host "[3/5] Running Gate2 smoke A/B/C..."
Invoke-Gate2Run -InputPath $smoothInput -OutDir $runA -EncoderId synthetic_smooth_v1
Invoke-Gate2Run -InputPath $smoothInput -OutDir $runB -EncoderId synthetic_smooth_v1
Invoke-Gate2Run -InputPath $kinkInput -OutDir $runKink -EncoderId synthetic_kink_v1

Write-Host "[4/5] Determinism check (A vs B)..."
$artifactNames = @("manifest.json", "summary.csv", "samples.csv")
$determinismPass = $true
foreach ($name in $artifactNames) {
    $aPath = Join-Path $runA $name
    $bPath = Join-Path $runB $name
    if (-not (Test-Path $aPath) -or -not (Test-Path $bPath)) {
        throw "Missing artifact for determinism check: $name"
    }

    $aHash = (Get-FileHash -Algorithm SHA256 $aPath).Hash.ToLowerInvariant()
    $bHash = (Get-FileHash -Algorithm SHA256 $bPath).Hash.ToLowerInvariant()
    $same = $aHash -eq $bHash
    Write-Host ("  {0}  A={1}  B={2}  match={3}" -f $name, $aHash, $bHash, $same)
    if (-not $same) {
        $determinismPass = $false
    }
}

if ($determinismPass) {
    Write-Host "Determinism: PASS"
}
else {
    Write-Host "Determinism: FAIL"
    throw "Determinism check failed."
}

Write-Host "[5/5] Sensitivity check (smooth vs kink summary rows)..."
$smoothSummary = @(Import-Csv (Join-Path $runA "summary.csv"))
$kinkSummary = @(Import-Csv (Join-Path $runKink "summary.csv"))
if ($smoothSummary.Count -ne 1 -or $kinkSummary.Count -ne 1) {
    throw "Expected summary.csv to contain exactly one row."
}

$smoothRow = $smoothSummary[0]
$kinkRow = $kinkSummary[0]
$columns = $smoothRow.PSObject.Properties.Name
$comparison = foreach ($col in $columns) {
    $smoothValue = [string]$smoothRow.$col
    $kinkValue = [string]$kinkRow.$col
    [PSCustomObject]@{
        column    = $col
        smooth    = $smoothValue
        kink      = $kinkValue
        different = ($smoothValue -ne $kinkValue)
    }
}
$comparison | Format-Table -AutoSize

Write-Host "Gate2 smoke completed successfully."
