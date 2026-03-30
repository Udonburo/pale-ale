param(
    [string]$BundleRoot = ".",
    [string]$FileList = "BUNDLE_FILE_LIST.txt",
    [string]$Output = "SHA256SUMS.txt"
)

$bundleRootPath = Resolve-Path -Path $BundleRoot
$fileListPath = Join-Path $bundleRootPath $FileList
$outputPath = Join-Path $bundleRootPath $Output

if (-not (Test-Path $fileListPath)) {
    throw "File list not found: $fileListPath"
}

$rows = @()

Get-Content -Path $fileListPath | ForEach-Object {
    $relativePath = $_.Trim()
    if ($relativePath -eq "") {
        return
    }

    $fullPath = Join-Path $bundleRootPath $relativePath
    if (-not (Test-Path $fullPath)) {
        throw "Bundle file missing: $relativePath"
    }

    $hash = (Get-FileHash -Path $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $rows += "$hash  $relativePath"
}

Set-Content -Path $outputPath -Value $rows -Encoding ascii
Write-Output "Wrote $outputPath"
