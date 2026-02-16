[CmdletBinding()]
param(
    [string]$PythonPath = "python",
    [string]$OutputDir = "dist",
    [string]$Features = "python-inspect",
    [switch]$Offline,
    [switch]$NoRelease,
    [switch]$SkipInstall,
    [string]$InstallPythonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Run-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$Label
    )

    Write-Host "[build-wheel] $Label" -ForegroundColor Cyan
    Write-Host "  $Executable $($Args -join ' ')" -ForegroundColor DarkGray
    & $Executable @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cargoToml = Resolve-ExistingPath -PathValue (Join-Path $repoRoot "Cargo.toml") -Label "Cargo.toml"

if (-not (Get-Command $PythonPath -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $PythonPath"
}

$absOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $repoRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $absOutputDir | Out-Null

$gitSha = "unknown"
try {
    $resolved = (& git -C $repoRoot rev-parse --short HEAD 2>$null).Trim()
    if ($resolved) {
        $gitSha = $resolved
    }
}
catch {
}

$buildStartUtc = (Get-Date).ToUniversalTime()
$buildArgs = @(
    "-m", "maturin", "build",
    "-m", $cargoToml,
    "-o", $absOutputDir,
    "-F", $Features
)
if (-not $NoRelease) {
    $buildArgs += "--release"
}
if ($Offline) {
    $buildArgs += "--offline"
}
Run-Step -Executable $PythonPath -Args $buildArgs -Label "Build wheel via maturin"

$candidates = Get-ChildItem -Path $absOutputDir -Filter "*.whl" -File |
    Where-Object { $_.LastWriteTimeUtc -ge $buildStartUtc.AddSeconds(-5) } |
    Sort-Object LastWriteTimeUtc -Descending
if (-not $candidates) {
    $candidates = Get-ChildItem -Path $absOutputDir -Filter "*.whl" -File |
        Sort-Object LastWriteTimeUtc -Descending
}
$wheel = $candidates | Select-Object -First 1
if (-not $wheel) {
    throw "No wheel found in output directory: $absOutputDir"
}

$wheelHash = (Get-FileHash -Algorithm SHA256 -Path $wheel.FullName).Hash.ToLowerInvariant()

if (-not $SkipInstall) {
    $targetPython = if ($InstallPythonPath) { $InstallPythonPath } else { $PythonPath }
    if (-not (Get-Command $targetPython -ErrorAction SilentlyContinue)) {
        throw "Install target python not found: $targetPython"
    }

    Run-Step -Executable $targetPython -Args @("-m", "pip", "uninstall", "-y", "pale-ale-core", "pale-ale") -Label "Uninstall old wheel distributions"
    Run-Step -Executable $targetPython -Args @("-m", "pip", "install", "--force-reinstall", "--no-deps", $wheel.FullName) -Label "Install wheel"

    $verifyScript = "import pale_ale_core; symbols=sorted([n for n in dir(pale_ale_core) if n.startswith('spin3_')]); required=['spin3_distance','spin3_struct','spin3_components']; missing=[n for n in required if n not in symbols]; print('core_path=' + pale_ale_core.__file__); print('symbols=' + ','.join(symbols)); assert not missing, 'missing symbols: ' + ','.join(missing)"
    Run-Step -Executable $targetPython -Args @("-c", $verifyScript) -Label "Verify import and symbols"
}

$provenancePath = Join-Path $absOutputDir "wheel_provenance.json"
$provenance = [ordered]@{
    built_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    core_git_sha = $gitSha
    wheel_path = $wheel.FullName
    wheel_sha256 = $wheelHash
    wheel_mtime_utc = $wheel.LastWriteTimeUtc.ToString("yyyy-MM-ddTHH:mm:ssZ")
    features = $Features
    release = (-not $NoRelease)
    offline = [bool]$Offline
}
$provenance | ConvertTo-Json -Depth 3 | Set-Content -Path $provenancePath -Encoding UTF8

Write-Host ""
Write-Host "[build-wheel] Done" -ForegroundColor Green
Write-Host "  core_git_sha=$gitSha"
Write-Host "  wheel_sha256=$wheelHash"
Write-Host "  provenance=$provenancePath"
