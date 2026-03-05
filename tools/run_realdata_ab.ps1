Param(
    [string]$InputJsonl = "data/realdata/halueval_spans_train.jsonl",
    [int]$N0 = 100,
    [int]$N1 = 100,
    [string]$UnitizationId = "sentence_split_v2_min4",
    [switch]$E1Only,
    [switch]$E2Only
)

$ErrorActionPreference = "Stop"

if ($N0 -lt 0 -or $N1 -lt 0) {
    throw "-N0 and -N1 must be >= 0"
}
if (($N0 + $N1) -le 0) {
    throw "At least one of -N0 or -N1 must be > 0"
}
if ($E1Only -and $E2Only) {
    throw "-E1Only and -E2Only cannot both be set"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

$dataDir = Join-Path $repoRoot "data\realdata"
$attestDir = Join-Path $repoRoot "attestations\realdata_ab"
$dateStamp = Get-Date -Format "yyyy-MM-dd"

$labelCountsReportPath = Join-Path $attestDir ("{0}_halueval_label_counts.txt" -f $dateStamp)
if ($E2Only) {
    $reportPath = Join-Path $attestDir ("{0}_halueval_balanced_n0{1}_n1{2}_E2_report.txt" -f $dateStamp, $N0, $N1)
}
elseif ($E1Only) {
    $reportPath = Join-Path $attestDir ("{0}_halueval_balanced_n0{1}_n1{2}_E1_report.txt" -f $dateStamp, $N0, $N1)
}
else {
    $reportPath = Join-Path $attestDir ("{0}_halueval_balanced_n0{1}_n1{2}_ALL_report.txt" -f $dateStamp, $N0, $N1)
}

$runE0Gate2 = Join-Path $repoRoot "runs\realdata_E0_gate2"
$runE0Gate3 = Join-Path $repoRoot "runs\realdata_E0_gate3"
$runE1Gate2 = Join-Path $repoRoot "runs\realdata_E1_gate2"
$runE1Gate3 = Join-Path $repoRoot "runs\realdata_E1_gate3"
$runE2Gate2 = Join-Path $repoRoot "runs\realdata_E2_gate2"
$runE2Gate3 = Join-Path $repoRoot "runs\realdata_E2_gate3"

$inputE0 = Join-Path $dataDir "out_E0_chunk.json"
$inputE1 = Join-Path $dataDir "out_E1_proj.json"
$inputE2 = Join-Path $dataDir "out_E2_snap.json"
$metaPath = Join-Path $dataDir "ab_meta.json"
$anchorStatsCsv = Join-Path $dataDir "e2_anchor_stats.csv"
$provenancePath = Join-Path $dataDir "hf_halueval_provenance.json"

$sumE0Gate2 = Join-Path $runE0Gate2 "summary.csv"
$sumE0Gate3 = Join-Path $runE0Gate3 "summary.csv"
$sumE1Gate2 = Join-Path $runE1Gate2 "summary.csv"
$sumE1Gate3 = Join-Path $runE1Gate3 "summary.csv"
$sumE2Gate2 = Join-Path $runE2Gate2 "summary.csv"
$sumE2Gate3 = Join-Path $runE2Gate3 "summary.csv"

$samplesE0Gate2 = Join-Path $runE0Gate2 "samples.csv"
$samplesE0Gate3 = Join-Path $runE0Gate3 "samples.csv"
$samplesE1Gate2 = Join-Path $runE1Gate2 "samples.csv"
$samplesE1Gate3 = Join-Path $runE1Gate3 "samples.csv"
$samplesE2Gate2 = Join-Path $runE2Gate2 "samples.csv"
$samplesE2Gate3 = Join-Path $runE2Gate3 "samples.csv"

$zero64 = "0000000000000000000000000000000000000000000000000000000000000000"
$unitizationId = $UnitizationId
$rotorEncoderId = "minilm_all_MiniLM_L6_v2_384d_v1"
$rotorPreprocId = "sentence_units_embed_raw_v1"
$evaluationModeId = "unsupervised_v1"
$datasetRevisionId = "halueval-spans_train_main"
$vec8E0 = "chunk_sequential_48x8_v1"
$vec8E1 = "gaussian_proj_d8_seed7_v1"
$vec8E2 = "e8_softsnap_chunk48_k3_beta12_v1"

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $attestDir | Out-Null

function Remove-DirIfExists {
    param([string]$PathValue)
    if (Test-Path $PathValue) {
        Remove-Item -Recurse -Force $PathValue
    }
}

function Ensure-PythonModule {
    param(
        [string]$ModuleName,
        [string]$PipName
    )
    python -c "import $ModuleName" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("Installing Python dependency: {0}" -f $PipName)
        python -m pip install $PipName
    }
}

function Resolve-CliPath {
    $candidateExe = Join-Path $repoRoot "target\release\pale-ale.exe"
    if (Test-Path $candidateExe) {
        return $candidateExe
    }
    $candidate = Join-Path $repoRoot "target\release\pale-ale"
    if (Test-Path $candidate) {
        return $candidate
    }
    throw "Could not locate built CLI binary in target/release."
}

function Invoke-GateRun {
    param(
        [ValidateSet("gate2", "gate3")]
        [string]$Gate,
        [string]$InputPath,
        [string]$OutDir,
        [string]$Vec8PostprocId
    )
    Remove-DirIfExists -PathValue $OutDir
    & $cliPath $Gate run `
      --input $InputPath `
      --out $OutDir `
      --dataset-revision-id $datasetRevisionId `
      --dataset-hash-blake3 $zero64 `
      --spec-hash-raw-blake3 $zero64 `
      --spec-hash-blake3 $zero64 `
      --unitization-id $unitizationId `
      --rotor-encoder-id $rotorEncoderId `
      --rotor-encoder-preproc-id $rotorPreprocId `
      --vec8-postproc-id $Vec8PostprocId `
      --evaluation-mode-id $evaluationModeId
}

function Write-Utf8NoBom {
    param(
        [string]$PathValue,
        [string]$Content
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($PathValue, $Content, $utf8NoBom)
}

function Get-FirstSummaryRow {
    param([string]$PathValue)
    $rows = @(Import-Csv $PathValue)
    if ($rows.Count -eq 0) {
        throw "summary.csv has no rows: $PathValue"
    }
    return $rows[0]
}

Write-Host "[0/6] Label-count check..."
if (-not (Test-Path $InputJsonl)) {
    throw "Input JSONL not found: $InputJsonl"
}
try {
    $commitSha = (git rev-parse HEAD).Trim()
}
catch {
    $commitSha = "UNKNOWN"
}

$labelCountCommand = "python tools/realdata_label_utils.py `"$InputJsonl`""
$labelCountsJson = & python tools/realdata_label_utils.py $InputJsonl
$labelCounts = $labelCountsJson | ConvertFrom-Json
$provenanceSha = if (Test-Path $provenancePath) {
    (Get-FileHash -Algorithm SHA256 $provenancePath).Hash.ToLowerInvariant()
}
else {
    "NOT_FOUND"
}

$labelLines = @()
$labelLines += "label_count_check_date=$dateStamp"
$labelLines += "git_commit_sha=$commitSha"
$labelLines += "input_jsonl=$InputJsonl"
$labelLines += "label_count_command=$labelCountCommand"
$labelLines += "total_rows=$($labelCounts.total_rows)"
$labelLines += "count0=$($labelCounts.count0)"
$labelLines += "count1=$($labelCounts.count1)"
$labelLines += "countnull=$($labelCounts.countnull)"
$labelLines += "hf_provenance_path=$provenancePath"
$labelLines += "hf_provenance_sha256=$provenanceSha"
$labelText = ($labelLines -join "`n") + "`n"
Write-Utf8NoBom -PathValue $labelCountsReportPath -Content $labelText
Write-Host "Label-count report written: $labelCountsReportPath"

Write-Host "[1/6] Checking Python dependency (sentence-transformers)..."
Ensure-PythonModule -ModuleName "sentence_transformers" -PipName "sentence-transformers"

Write-Host "[2/6] Building CLI..."
cargo build -p pale-ale-cli --release
$cliPath = Resolve-CliPath

Write-Host "[3/6] Generating balanced E0/E1/E2 Gate2RunInputV1..."
python tools/gen_gate2_realdata_ab.py `
  --input-jsonl $InputJsonl `
  --balanced 1 `
  --n0 $N0 `
  --n1 $N1 `
  --unitization-id $unitizationId

if (-not (Test-Path $metaPath)) {
    throw "Expected metadata output is missing: $metaPath"
}
if (-not (Test-Path $inputE0) -or -not (Test-Path $inputE1) -or -not (Test-Path $inputE2)) {
    throw "Expected generated inputs are missing under data/realdata."
}

$runE0 = (-not $E1Only) -and (-not $E2Only)
$runE1 = $E1Only -or ((-not $E1Only) -and (-not $E2Only))
$runE2 = $E2Only -or ((-not $E1Only) -and (-not $E2Only))

Write-Host "[4/6] Running Gate2/Gate3..."
if ($runE0) {
    Invoke-GateRun -Gate "gate2" -InputPath $inputE0 -OutDir $runE0Gate2 -Vec8PostprocId $vec8E0
    Invoke-GateRun -Gate "gate3" -InputPath $inputE0 -OutDir $runE0Gate3 -Vec8PostprocId $vec8E0
}
if ($runE1) {
    Invoke-GateRun -Gate "gate2" -InputPath $inputE1 -OutDir $runE1Gate2 -Vec8PostprocId $vec8E1
    Invoke-GateRun -Gate "gate3" -InputPath $inputE1 -OutDir $runE1Gate3 -Vec8PostprocId $vec8E1
}
if ($runE2) {
    Invoke-GateRun -Gate "gate2" -InputPath $inputE2 -OutDir $runE2Gate2 -Vec8PostprocId $vec8E2
    Invoke-GateRun -Gate "gate3" -InputPath $inputE2 -OutDir $runE2Gate3 -Vec8PostprocId $vec8E2
}

if ($E2Only) {
    $focusId = "E2"
    $focusGate2Summary = $sumE2Gate2
    $focusGate3Summary = $sumE2Gate3
    $focusGate2Samples = $samplesE2Gate2
    $focusGate3Samples = $samplesE2Gate3
}
elseif ($E1Only) {
    $focusId = "E1"
    $focusGate2Summary = $sumE1Gate2
    $focusGate3Summary = $sumE1Gate3
    $focusGate2Samples = $samplesE1Gate2
    $focusGate3Samples = $samplesE1Gate3
}
else {
    $focusId = "E2"
    $focusGate2Summary = $sumE2Gate2
    $focusGate3Summary = $sumE2Gate3
    $focusGate2Samples = $samplesE2Gate2
    $focusGate3Samples = $samplesE2Gate3
}

Write-Host "[5/6] Computing label-wise effect stats..."
$gate2EffectsJson = & python tools/gate2_label_effects.py `
  --csv $focusGate2Samples `
  --label-col sample_label `
  --metrics "h1b_closure_error,h2_loop_mean,h3_ratio_total_product"

$gate3EffectsJson = & python tools/gate2_label_effects.py `
  --csv $focusGate3Samples `
  --label-col sample_label `
  --metrics "l3_kappa_mean,l3_kappa_max,l4_tau_mean,l4_tau_max,l4_tau_p90"

$anchorEffectsJson = ""
if ($focusId -eq "E2" -and (Test-Path $anchorStatsCsv)) {
    $anchorEffectsJson = & python tools/gate2_label_effects.py `
      --csv $anchorStatsCsv `
      --label-col label `
      --metrics "anchor_mean,anchor_p90,anchor_max"
}

$meta = Get-Content -Raw $metaPath | ConvertFrom-Json
$summaryGate2 = Get-FirstSummaryRow -PathValue $focusGate2Summary
$summaryGate3 = Get-FirstSummaryRow -PathValue $focusGate3Summary

$missingReasonRows = @(Import-Csv $focusGate3Samples | Where-Object {
        $_.missing_reason -and $_.missing_reason.Trim() -ne ""
    })
$missingReasonCounts = @()
if ($missingReasonRows.Count -gt 0) {
    $missingReasonCounts = @(
        $missingReasonRows |
        Group-Object -Property missing_reason |
        Sort-Object -Property Name |
        ForEach-Object {
            [PSCustomObject]@{
                reason = [string]$_.Name
                count  = [int]$_.Count
            }
        }
    )
}

$e2VerifyPath = ""
$e2VerifyShaFromFile = "NOT_FOUND"
if ($meta.vec8_methods -and $meta.vec8_methods.E2 -and $meta.vec8_methods.E2.root_verify_path) {
    $e2VerifyPath = [string]$meta.vec8_methods.E2.root_verify_path
    if ([System.IO.Path]::IsPathRooted($e2VerifyPath)) {
        $e2VerifyAbs = $e2VerifyPath
    }
    else {
        $e2VerifyAbs = Join-Path $repoRoot ($e2VerifyPath -replace "/", "\")
    }
    if (Test-Path $e2VerifyAbs) {
        $e2VerifyShaFromFile = (Get-FileHash -Algorithm SHA256 $e2VerifyAbs).Hash.ToLowerInvariant()
    }
}

Write-Host "[6/6] Writing report..."
$reportLines = @()
$reportLines += "realdata_balanced_report_date=$dateStamp"
$reportLines += "git_commit_sha=$commitSha"
$reportLines += "focus_method=$focusId"
$reportLines += "input_jsonl=$InputJsonl"
$reportLines += "n0_requested=$N0"
$reportLines += "n1_requested=$N1"
$reportLines += "n0_selected=$($meta.selection.n0_selected)"
$reportLines += "n1_selected=$($meta.selection.n1_selected)"
$reportLines += "label_count_0=$($meta.label_count_0)"
$reportLines += "label_count_1=$($meta.label_count_1)"
$reportLines += "label_count_null=$($meta.label_count_null)"
$reportLines += "unitization_id=$($meta.unitization_id)"
$reportLines += "unit_count_min=$($meta.unit_count_stats.min)"
$reportLines += "unit_count_max=$($meta.unit_count_stats.max)"
$reportLines += "unit_count_mean=$($meta.unit_count_stats.mean)"
$reportLines += "gate3_n_samples_valid=$($summaryGate3.n_samples_valid)"
$reportLines += "gate3_n_samples_missing=$($summaryGate3.n_samples_missing)"
$reportLines += "artifact_path_gate2=$([System.IO.Path]::GetDirectoryName($focusGate2Summary))"
$reportLines += "artifact_path_gate3=$([System.IO.Path]::GetDirectoryName($focusGate3Summary))"
$reportLines += "label_count_report_path=$labelCountsReportPath"
$reportLines += "hf_provenance_path=$provenancePath"
$reportLines += "hf_provenance_sha256=$provenanceSha"

if ($focusId -eq "E1") {
    $reportLines += "vec8_postproc_id=$vec8E1"
    $reportLines += "vec8_postproc_matrix_hash_sha256=$($meta.vec8_methods.E1.vec8_postproc_matrix_hash_sha256)"
}
elseif ($focusId -eq "E2") {
    $reportLines += "vec8_postproc_id=$vec8E2"
    $reportLines += "e2_k=$($meta.vec8_methods.E2.k)"
    $reportLines += "e2_beta=$($meta.vec8_methods.E2.beta)"
    $reportLines += "e2_roots_hash_sha256=$($meta.vec8_methods.E2.roots_hash_sha256)"
    $reportLines += "e2_root_verify_path=$e2VerifyPath"
    $reportLines += "e2_root_verify_sha256_meta=$($meta.vec8_methods.E2.root_verify_sha256)"
    $reportLines += "e2_root_verify_sha256_file=$e2VerifyShaFromFile"
    $reportLines += "e2_anchor_stats_csv=$anchorStatsCsv"
}

$reportLines += ""
$reportLines += "===== Gate3 Missing Reasons ====="
if ($missingReasonCounts.Count -eq 0) {
    $reportLines += "(none)"
}
else {
    foreach ($item in $missingReasonCounts) {
        $reportLines += ("{0},{1}" -f $item.reason, $item.count)
    }
}

$reportLines += ""
$reportLines += "===== Label-wise Effect Stats (Gate2) ====="
$reportLines += $gate2EffectsJson
$reportLines += ""
$reportLines += "===== Label-wise Effect Stats (Gate3) ====="
$reportLines += $gate3EffectsJson
if ($anchorEffectsJson -ne "") {
    $reportLines += ""
    $reportLines += "===== Label-wise Effect Stats (E2 Anchor) ====="
    $reportLines += $anchorEffectsJson
}

$reportLines += ""
$reportLines += "===== Focus Gate2 summary.csv ====="
$reportLines += (Get-Content -Raw $focusGate2Summary).TrimEnd("`r", "`n")
$reportLines += ""
$reportLines += "===== Focus Gate3 summary.csv ====="
$reportLines += (Get-Content -Raw $focusGate3Summary).TrimEnd("`r", "`n")
$reportLines += ""
$reportLines += "===== Focus Gate2 samples.csv ====="
$reportLines += (Get-Content -Raw $focusGate2Samples).TrimEnd("`r", "`n")
$reportLines += ""
$reportLines += "===== Focus Gate3 samples.csv ====="
$reportLines += (Get-Content -Raw $focusGate3Samples).TrimEnd("`r", "`n")
if ($anchorEffectsJson -ne "") {
    $reportLines += ""
    $reportLines += "===== E2 Anchor Stats CSV ====="
    $reportLines += (Get-Content -Raw $anchorStatsCsv).TrimEnd("`r", "`n")
}
$reportLines += ""
$reportLines += "===== ab_meta.json ====="
$reportLines += (Get-Content -Raw $metaPath).TrimEnd("`r", "`n")

if ($runE0) {
    $reportLines += ""
    $reportLines += "===== Gate2 E0 summary.csv ====="
    $reportLines += (Get-Content -Raw $sumE0Gate2).TrimEnd("`r", "`n")
    $reportLines += ""
    $reportLines += "===== Gate3 E0 summary.csv ====="
    $reportLines += (Get-Content -Raw $sumE0Gate3).TrimEnd("`r", "`n")
}
if ($runE1 -and $focusId -ne "E1") {
    $reportLines += ""
    $reportLines += "===== Gate2 E1 summary.csv ====="
    $reportLines += (Get-Content -Raw $sumE1Gate2).TrimEnd("`r", "`n")
    $reportLines += ""
    $reportLines += "===== Gate3 E1 summary.csv ====="
    $reportLines += (Get-Content -Raw $sumE1Gate3).TrimEnd("`r", "`n")
}
if ($runE2 -and $focusId -ne "E2") {
    $reportLines += ""
    $reportLines += "===== Gate2 E2 summary.csv ====="
    $reportLines += (Get-Content -Raw $sumE2Gate2).TrimEnd("`r", "`n")
    $reportLines += ""
    $reportLines += "===== Gate3 E2 summary.csv ====="
    $reportLines += (Get-Content -Raw $sumE2Gate3).TrimEnd("`r", "`n")
}

$reportText = ($reportLines -join "`n") + "`n"
Write-Utf8NoBom -PathValue $reportPath -Content $reportText

Write-Host "Report written: $reportPath"
Write-Host "Balanced realdata run completed."

