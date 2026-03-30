param(
    [string]$RepoRoot = ".",
    [string]$ReleasePrepDir = "docs/release/gate12a_first_replication_checkpoint",
    [string]$OutputRoot = "dist/zenodo",
    [string]$BundleName = "gate12a_first_replication_checkpoint_bundle"
)

$repoRootPath = (Resolve-Path -Path $RepoRoot).Path
$releasePrepPath = Join-Path $repoRootPath $ReleasePrepDir
$outputRootPath = Join-Path $repoRootPath $OutputRoot
$bundleRootPath = Join-Path $outputRootPath $BundleName
$zipPath = Join-Path $outputRootPath ($BundleName + ".zip")
$fileListPath = Join-Path $releasePrepPath "BUNDLE_FILE_LIST.txt"

if (-not (Test-Path $releasePrepPath)) {
    throw "Release prep directory not found: $releasePrepPath"
}

if (-not (Test-Path $fileListPath)) {
    throw "Bundle file list not found: $fileListPath"
}

if (Test-Path $bundleRootPath) {
    Remove-Item -Recurse -Force $bundleRootPath
}

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

New-Item -ItemType Directory -Force -Path $bundleRootPath | Out-Null

$payloadFiles = Get-Content -Path $fileListPath | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

foreach ($relativePath in $payloadFiles) {
    $sourcePath = Join-Path $repoRootPath $relativePath
    if (-not (Test-Path $sourcePath)) {
        throw "Bundle source file missing: $relativePath"
    }

    $destinationPath = Join-Path $bundleRootPath $relativePath
    $destinationDir = Split-Path -Parent $destinationPath
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    Copy-Item -Path $sourcePath -Destination $destinationPath -Force
}

$rootDocs = @(
    "CHECKPOINT_README.md",
    "BUNDLE_FILE_LIST.txt",
    "ZENODO_METADATA_DRAFT.md",
    "MANUAL_UPLOAD_RUNBOOK.md",
    "make_sha256sums.ps1"
)

foreach ($name in $rootDocs) {
    $sourcePath = Join-Path $releasePrepPath $name
    if (-not (Test-Path $sourcePath)) {
        throw "Release prep file missing: $sourcePath"
    }
    Copy-Item -Path $sourcePath -Destination (Join-Path $bundleRootPath $name) -Force
}

$hashRows = @()
Get-ChildItem -Path $bundleRootPath -Recurse -File | Sort-Object FullName | ForEach-Object {
    $relativePath = $_.FullName.Substring($bundleRootPath.Length + 1).Replace("\", "/")
    if ($relativePath -eq "SHA256SUMS.txt") {
        return
    }
    $hash = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $hashRows += "$hash  $relativePath"
}

Set-Content -Path (Join-Path $bundleRootPath "SHA256SUMS.txt") -Value $hashRows -Encoding ascii

New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
Compress-Archive -Path (Join-Path $bundleRootPath "*") -DestinationPath $zipPath -Force

Write-Output "Bundle directory: $bundleRootPath"
Write-Output "Bundle zip: $zipPath"
