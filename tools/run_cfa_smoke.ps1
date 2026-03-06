Param(
    [string]$CfaJsonl = "data/cfa/cfa_v1.jsonl",
    [int]$SampleId = 1,
    [int]$Seed = 7,
    [string]$PrimaryScore = "F",
    [int]$PermR = 500,
    [double]$MinCoverage = 0.30,
    [string]$ModelId = "Qwen/Qwen2.5-1.5B",
    [string]$TokenizerModel = "Qwen/Qwen2.5-1.5B"
)

$ErrorActionPreference = "Stop"

if ($PrimaryScore -notin @("A", "B", "C", "D", "E", "F")) {
    throw "-PrimaryScore must be one of A,B,C,D,E,F"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

function Invoke-OrThrow {
    param([string]$Description)
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

$dateStamp = Get-Date -Format "yyyy-MM-dd"
$outDir = Join-Path $repoRoot "runs\cfa_smoke"
$attestDir = Join-Path $repoRoot "attestations\triality"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $attestDir | Out-Null

if (-not (Test-Path $CfaJsonl)) {
    Write-Host "CFA dataset not found. Generating default dataset..."
    python tools/generate_cfa_dataset.py --out $CfaJsonl --meta-out "data/cfa/cfa_v1_meta.json" --n-worlds 120 --seed $Seed
    Invoke-OrThrow -Description "generate_cfa_dataset.py"
}

$promptPath = Join-Path $outDir ("sample{0}_prompt.txt" -f $SampleId)
$answerPath = Join-Path $outDir ("sample{0}_answer.txt" -f $SampleId)
$tripletsPath = Join-Path $outDir ("sample{0}_triplets.ndjson" -f $SampleId)
$labelsPath = Join-Path $outDir ("sample{0}_labels.jsonl" -f $SampleId)
$labelsMeta = Join-Path $outDir ("sample{0}_labels_meta.json" -f $SampleId)
$reportPath = Join-Path $attestDir ("{0}_cfa_sample{1}_primary{2}.txt" -f $dateStamp, $SampleId, $PrimaryScore)

@"
import json
import sys
from pathlib import Path

cfa = Path(sys.argv[1])
sample_id = int(sys.argv[2])
prompt_path = Path(sys.argv[3])
answer_path = Path(sys.argv[4])

row = None
with cfa.open('r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if int(obj.get('sample_id', -1)) == sample_id:
            row = obj
            break

if row is None:
    raise SystemExit(f'sample_id={sample_id} not found in {cfa}')

prompt = str(row.get('prompt', ''))
answer = str(row.get('answer', ''))
if not prompt or not answer:
    raise SystemExit('sample has empty prompt/answer')

prompt_path.write_text(prompt, encoding='utf-8', newline='\n')
answer_path.write_text(answer, encoding='utf-8', newline='\n')
print('variant=' + str(row.get('variant')))
print('world_type=' + str(row.get('world_type')))
print('has_defect=' + str(row.get('has_defect')))
"@ | python - $CfaJsonl $SampleId $promptPath $answerPath
Invoke-OrThrow -Description "prepare sample prompt/answer"

python tools/extract_triality_triplets.py `
  --prompt-file $promptPath `
  --target-answer-file $answerPath `
  --deterministic `
  --seed $Seed `
  --model-id $ModelId `
  --out $tripletsPath
Invoke-OrThrow -Description "extract_triality_triplets.py"

python tools/labels_from_cfa_spans.py `
  --cfa-jsonl $CfaJsonl `
  --sample-id $SampleId `
  --triplets-ndjson $tripletsPath `
  --tokenizer-model $TokenizerModel `
  --out $labelsPath `
  --min-coverage $MinCoverage `
  --fail-below-coverage
Invoke-OrThrow -Description "labels_from_cfa_spans.py"

python tools/eval_triality_token.py `
  --ndjson $tripletsPath `
  --labels-jsonl $labelsPath `
  --labels-meta-json $labelsMeta `
  --min-label-coverage $MinCoverage `
  --perm-R $PermR `
  --seed $Seed `
  --primary-score $PrimaryScore `
  --out $reportPath
Invoke-OrThrow -Description "eval_triality_token.py"

Write-Host "CFA smoke run completed."
Write-Host "triplets=$tripletsPath"
Write-Host "labels=$labelsPath"
Write-Host "report=$reportPath"
