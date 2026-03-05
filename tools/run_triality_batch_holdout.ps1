Param(
    [int]$StartIndex = 1,
    [int]$EndIndex = 50,
    [int]$Seed = 7,
    [int]$PermR = 2000,
    [double]$MinCoverage = 0.30,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

if ($StartIndex -lt 0) {
    throw "-StartIndex must be >= 0"
}
if ($EndIndex -lt $StartIndex) {
    throw "-EndIndex must be >= -StartIndex"
}
if ($PermR -lt 1) {
    throw "-PermR must be >= 1"
}
if ($MinCoverage -lt 0.0 -or $MinCoverage -gt 1.0) {
    throw "-MinCoverage must be in [0,1]"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

$dateStamp = Get-Date -Format "yyyy-MM-dd"
$runsRoot = Join-Path $repoRoot "runs\triality_batch_holdout"
$inputsRoot = Join-Path $runsRoot "inputs"
$attestBatchRoot = Join-Path $repoRoot "attestations\triality\batch"
$attestBatchRootRel = "attestations/triality/batch"
$resultsPath = Join-Path $runsRoot "results.jsonl"
$sample0Report = Join-Path $repoRoot ("attestations\triality\{0}_eval_hf0_teacher_forcing_primaryE.txt" -f $dateStamp)

New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $inputsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $attestBatchRoot | Out-Null

function Write-Utf8NoBom {
    param(
        [string]$PathValue,
        [string]$Content
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($PathValue, $Content, $utf8NoBom)
}

function Append-Jsonl {
    param(
        [string]$PathValue,
        [hashtable]$Payload
    )
    $json = ($Payload | ConvertTo-Json -Compress)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($PathValue, $json + "`n", $utf8NoBom)
}

function Parse-ReportValue {
    param(
        [string]$ReportText,
        [string]$Pattern
    )
    $m = [System.Text.RegularExpressions.Regex]::Match(
        $ReportText,
        $Pattern,
        [System.Text.RegularExpressions.RegexOptions]::Multiline
    )
    if ($m.Success) {
        return $m.Groups[1].Value.Trim()
    }
    return $null
}

function Parse-ReportNumber {
    param([string]$RawValue)
    if ([string]::IsNullOrWhiteSpace($RawValue)) {
        return $null
    }
    if ($RawValue -eq "NA") {
        return $null
    }
    $out = 0.0
    if ([double]::TryParse($RawValue, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$out)) {
        return $out
    }
    return $null
}

Write-Host "[0/4] Preparing deterministic HF input files for holdout indices..."
@'
from datasets import load_dataset
from pathlib import Path
import json
import sys

repo_id = "llm-semantic-router/halueval-spans"
split = "train"
revision = "main"
start = int(sys.argv[1])
end = int(sys.argv[2])
inputs_root = Path(sys.argv[3])
inputs_root.mkdir(parents=True, exist_ok=True)

ds = load_dataset(repo_id, split=split, revision=revision)
if end >= len(ds):
    raise RuntimeError(f"requested end index {end} exceeds split length {len(ds)}")

for idx in range(start, end + 1):
    row = ds[idx]
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    raw = {
        "repo_id": repo_id,
        "split": split,
        "revision": revision,
        "index": idx,
        "raw": row,
    }
    (inputs_root / f"hf{idx}_prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    (inputs_root / f"hf{idx}_answer.txt").write_text(answer, encoding="utf-8", newline="\n")
    (inputs_root / f"hf{idx}_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

print(f"Prepared indices {start}..{end} under {inputs_root.as_posix()}")
'@ | python - $StartIndex $EndIndex $inputsRoot

if (-not $SkipSmoke) {
    Write-Host "[1/4] Smoke test on existing sample0 with primary-score E..."
    $sample0Ndjson = Join-Path $repoRoot "runs\triality_real\hf0_triplets.ndjson"
    $sample0Labels = Join-Path $repoRoot "runs\triality_real\hf0_labels.jsonl"
    $sample0LabelsMeta = Join-Path $repoRoot "runs\triality_real\hf0_labels_meta.json"
    if ((-not (Test-Path $sample0Ndjson)) -or (-not (Test-Path $sample0Labels)) -or (-not (Test-Path $sample0LabelsMeta))) {
        throw "Sample0 files missing. Expected: runs/triality_real/hf0_triplets.ndjson + labels + labels_meta"
    }
    python tools/eval_triality_token.py `
      --ndjson $sample0Ndjson `
      --labels-jsonl $sample0Labels `
      --labels-meta-json $sample0LabelsMeta `
      --min-label-coverage $MinCoverage `
      --perm-R $PermR `
      --seed $Seed `
      --primary-score E `
      --out $sample0Report

    $sample0Text = Get-Content -Raw $sample0Report
    $sample0PEmpRaw = Parse-ReportValue -ReportText $sample0Text -Pattern "(?m)^\s*perm_p_empirical=([^\r\n]+)$"
    $sample0PEmp = Parse-ReportNumber -RawValue $sample0PEmpRaw
    Write-Host ("Sample0 primary(E) p_emp={0}" -f $sample0PEmp)
}
else {
    Write-Host "[1/4] Smoke test skipped by -SkipSmoke."
}

Write-Host "[2/4] Running holdout loop..."
if (Test-Path $resultsPath) {
    Remove-Item -Force $resultsPath
}
Write-Utf8NoBom -PathValue $resultsPath -Content ""

for ($idx = $StartIndex; $idx -le $EndIndex; $idx++) {
    $idxDir = Join-Path $runsRoot ("hf{0}" -f $idx)
    New-Item -ItemType Directory -Force -Path $idxDir | Out-Null

    $promptPath = Join-Path $inputsRoot ("hf{0}_prompt.txt" -f $idx)
    $answerPath = Join-Path $inputsRoot ("hf{0}_answer.txt" -f $idx)
    $tripletsPath = Join-Path $idxDir "triplets.ndjson"
    $metaPath = Join-Path $idxDir "meta.json"
    $labelsPath = Join-Path $idxDir "labels.jsonl"
    $labelsMetaPath = Join-Path $idxDir "labels_meta.json"
    $evalReport = Join-Path $attestBatchRootRel ("{0}_eval_hf{1}_primaryE.txt" -f $dateStamp, $idx)

    Write-Host ("[idx={0}] extract -> label -> eval" -f $idx)
    try {
        python tools/extract_triality_triplets.py `
          --prompt-file $promptPath `
          --target-answer-file $answerPath `
          --deterministic `
          --seed $Seed `
          --out $tripletsPath

        if (-not (Test-Path $metaPath)) {
            throw "missing extractor meta.json for idx=$idx"
        }
        $meta = Get-Content -Raw $metaPath | ConvertFrom-Json
        $ratio = [double]$meta.exact_token_match_ratio
        if ($ratio -lt 0.98) {
            Append-Jsonl -PathValue $resultsPath -Payload @{
                idx = $idx
                status = "skip_token_match"
                exact_token_match_ratio = $ratio
                AUPRC_E = $null
                AUPRC_best_baseline = $null
                delta = $null
                p_emp = $null
            }
            Write-Host ("  skip: exact_token_match_ratio={0}" -f $ratio)
            continue
        }

        python tools/labels_from_halueval_spans.py `
          --sample-id $idx `
          --triplets-ndjson $tripletsPath `
          --answer-file $answerPath `
          --out $labelsPath `
          --no-improve-low-coverage

        if (-not (Test-Path $labelsMetaPath)) {
            throw "missing labels meta for idx=$idx"
        }
        $labelsMeta = Get-Content -Raw $labelsMetaPath | ConvertFrom-Json
        $coverage = [double]$labelsMeta.final_alignment_coverage_ratio
        if ($coverage -lt $MinCoverage) {
            Append-Jsonl -PathValue $resultsPath -Payload @{
                idx = $idx
                status = "skip_coverage"
                exact_token_match_ratio = $ratio
                coverage = $coverage
                AUPRC_E = $null
                AUPRC_best_baseline = $null
                delta = $null
                p_emp = $null
            }
            Write-Host ("  skip: coverage={0}" -f $coverage)
            continue
        }

        python tools/eval_triality_token.py `
          --ndjson $tripletsPath `
          --labels-jsonl $labelsPath `
          --labels-meta-json $labelsMetaPath `
          --min-label-coverage $MinCoverage `
          --perm-R $PermR `
          --seed $Seed `
          --primary-score E `
          --out $evalReport

        $reportText = Get-Content -Raw $evalReport
        $auprcERaw = Parse-ReportValue -ReportText $reportText -Pattern "(?m)^E:V_Sminus_Vnext,[^,\r\n]*,[^,\r\n]*,[^,\r\n]*,[^,\r\n]*,([^,\r\n]*),[^,\r\n]*$"
        $bestRaw = Parse-ReportValue -ReportText $reportText -Pattern "(?m)^best_baseline_auprc=([^\r\n]+)$"
        $deltaRaw = Parse-ReportValue -ReportText $reportText -Pattern "(?m)^delta_auprc_primary_vs_best_baseline=([^\r\n]+)$"
        $pEmpRaw = Parse-ReportValue -ReportText $reportText -Pattern "(?m)^\s*perm_p_empirical=([^\r\n]+)$"
        $verdictRaw = Parse-ReportValue -ReportText $reportText -Pattern "(?m)^verdict=([^\r\n]+)$"

        $parsedAUPRCE = (Parse-ReportNumber -RawValue $auprcERaw)
        $parsedBest = (Parse-ReportNumber -RawValue $bestRaw)
        $parsedDelta = (Parse-ReportNumber -RawValue $deltaRaw)
        $parsedPEmp = (Parse-ReportNumber -RawValue $pEmpRaw)

        Append-Jsonl -PathValue $resultsPath -Payload @{
            idx = $idx
            status = "ok"
            exact_token_match_ratio = $ratio
            coverage = $coverage
            AUPRC_E = $parsedAUPRCE
            AUPRC_best_baseline = $parsedBest
            delta = $parsedDelta
            p_emp = $parsedPEmp
            verdict = $verdictRaw
            report = $evalReport
        }
    }
    catch {
        $msg = $_.Exception.Message
        Append-Jsonl -PathValue $resultsPath -Payload @{
            idx = $idx
            status = "error"
            error = $msg
            AUPRC_E = $null
            AUPRC_best_baseline = $null
            delta = $null
            p_emp = $null
        }
        Write-Host ("  error: {0}" -f $msg)
        continue
    }
}

Write-Host "[3/4] Exporting commit-ready holdout summary under attestations/triality..."
@'
import json
import statistics
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
date_stamp = sys.argv[3]

rows = [json.loads(x) for x in results_path.read_text(encoding="utf-8").splitlines() if x.strip()]
out_dir.mkdir(parents=True, exist_ok=True)
out_jsonl = out_dir / f"{date_stamp}_holdout_primaryE_results.jsonl"
out_txt = out_dir / f"{date_stamp}_holdout_primaryE_summary.txt"

out_jsonl.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    encoding="utf-8",
    newline="\n",
)

status = {}
for r in rows:
    key = str(r.get("status", "unknown"))
    status[key] = status.get(key, 0) + 1

ok = [r for r in rows if r.get("status") == "ok"]
skip_cov = [r for r in rows if r.get("status") == "skip_coverage"]
skip_tm = [r for r in rows if r.get("status") == "skip_token_match"]
err = [r for r in rows if r.get("status") == "error"]

def vals(key):
    out = []
    for r in ok:
        v = r.get(key)
        if v is not None:
            out.append(float(v))
    return out

auprc_e = vals("AUPRC_E")
base = vals("AUPRC_best_baseline")
delta = vals("delta")
p_emp = vals("p_emp")

lines = []
lines.append(f"date={date_stamp}")
lines.append("experiment=triality_holdout_primaryE")
lines.append(f"source_results_jsonl={results_path.as_posix()}")
lines.append(f"copied_results_jsonl={out_jsonl.as_posix()}")
lines.append("holdout_indices=1..50")
lines.append("primary_score=E:V_Sminus_Vnext")
lines.append("")
lines.append("counts:")
lines.append(f"  total_rows={len(rows)}")
for k in sorted(status.keys()):
    lines.append(f"  status_{k}={status[k]}")
lines.append(f"  ok_rows={len(ok)}")
lines.append(f"  skip_coverage_rows={len(skip_cov)}")
lines.append(f"  skip_token_match_rows={len(skip_tm)}")
lines.append(f"  error_rows={len(err)}")
if auprc_e:
    lines.append("")
    lines.append("ok_row_stats:")
    lines.append(f"  auprc_e_mean={statistics.mean(auprc_e):.17e}")
    lines.append(f"  auprc_e_median={statistics.median(auprc_e):.17e}")
    lines.append(f"  baseline_auprc_mean={statistics.mean(base):.17e}")
    lines.append(f"  delta_mean={statistics.mean(delta):.17e}")
    lines.append(f"  delta_median={statistics.median(delta):.17e}")
    lines.append(f"  p_emp_min={min(p_emp):.17e}")
    lines.append(f"  p_emp_median={statistics.median(p_emp):.17e}")
    lines.append(f"  count_delta_ge_0p02={sum(1 for x in delta if x >= 0.02)}")
    lines.append(f"  count_p_emp_le_0p05={sum(1 for x in p_emp if x <= 0.05)}")

if skip_cov:
    lines.append("")
    lines.append("skip_coverage_indices=" + ",".join(str(x["idx"]) for x in sorted(skip_cov, key=lambda r: r["idx"])))

if err:
    lines.append("")
    lines.append("errors:")
    for r in sorted(err, key=lambda x: x.get("idx", -1)):
        lines.append(f"  idx={r.get('idx')} error={r.get('error')}")

out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(out_jsonl.as_posix())
print(out_txt.as_posix())
'@ | python - $resultsPath (Join-Path $repoRoot "attestations/triality") $dateStamp

Write-Host "[4/4] Done."
Write-Host ("results_jsonl={0}" -f $resultsPath)
Write-Host ("batch_attestations={0}" -f $attestBatchRoot)
