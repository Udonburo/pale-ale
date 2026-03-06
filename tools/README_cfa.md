# CFA Triality Workflow

Constraint Frustration Arena (CFA) builds closed-world pairs where text is locally fluent but globally inconsistent.

## 1) Generate CFA dataset

```powershell
python tools/generate_cfa_dataset.py --out data/cfa/cfa_v1.jsonl --meta-out data/cfa/cfa_v1_meta.json --n-worlds 200 --seed 7
```

Each row includes:
- `sample_id`
- `world_type` (`genealogy`, `temporal`, `reachability`)
- `variant` (`consistent`, `frustrated`)
- `prompt`
- `answer`
- `defect_spans` (character-level ground truth for frustrated rows)

## 2) Teacher-forcing extraction (existing tool)

```powershell
python tools/extract_triality_triplets.py --prompt-file runs/cfa/prompt.txt --target-answer-file runs/cfa/answer.txt --deterministic --seed 7 --out runs/cfa/triplets.ndjson
```

## 3) Create token/step labels from CFA spans

```powershell
python tools/labels_from_cfa_spans.py --cfa-jsonl data/cfa/cfa_v1.jsonl --sample-id 1 --triplets-ndjson runs/cfa/triplets.ndjson --out runs/cfa/labels_step.jsonl
```

## 4) Evaluate

```powershell
python tools/eval_triality_token.py --ndjson runs/cfa/triplets.ndjson --labels-jsonl runs/cfa/labels_step.jsonl --labels-meta-json runs/cfa/labels_step_meta.json --primary-score F --perm-R 2000 --seed 7 --min-label-coverage 0.30
```

## 5) One-command smoke

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_cfa_smoke.ps1 -CfaJsonl data/cfa/cfa_v1.jsonl -SampleId 1 -PrimaryScore F -PermR 500 -Seed 7
```

## 6) Preregistered batch verification (Primary=E)

This is the verification-phase run:
- Primary score is fixed to `E`
- `seed=7`
- `perm-R=2000`
- Skip rules:
  - `exact_token_match_ratio < 0.98`
  - `coverage < 0.30`
- To satisfy the prereg gate (`>=15` valid per class), generate at least `n-worlds=15`
  (recommended `n-worlds=100`).

```powershell
python tools/generate_cfa_dataset.py --out data/cfa/cfa_v1.jsonl --meta-out data/cfa/cfa_v1_meta.json --n-worlds 100 --seed 7
python tools/run_cfa_batch_primaryE.py --cfa-jsonl data/cfa/cfa_v1.jsonl --device auto --model-id Qwen/Qwen2.5-1.5B
```

Outputs:
- sample results: `runs/cfa_batch_primaryE/results.jsonl`
- per-sample eval reports: `attestations/triality/cfa_batch/`
- dataset report: `attestations/triality/YYYY-MM-DD_cfa_batch_primaryE_report.txt`

If you need to rerun only aggregation:

```powershell
python tools/aggregate_cfa_batch.py --results-jsonl runs/cfa_batch_primaryE/results.jsonl --cfa-jsonl data/cfa/cfa_v1.jsonl
```

## 7) Case Study Visualization (sample pair)

Default interesting pair:
- frustrated: `sample_id=127`
- consistent contrast: auto-read from `contrast_sample_id` (expected `126`)

```powershell
python tools/plot_cfa_case_pair.py --cfa-jsonl data/cfa/cfa_v1.jsonl --sample-id 127 --device auto --model-id Qwen/Qwen2.5-1.5B --perm-R 0 --seed 7
```

Outputs:
- `runs/cfa_case_study/sample127_vs_126/token_table_127.csv`
- `runs/cfa_case_study/sample127_vs_126/token_table_126.csv`
- `runs/cfa_case_study/sample127_vs_126/pair_overlay_127_vs_126.csv`
- `runs/cfa_case_study/sample127_vs_126/plot_case_127.png`
- `runs/cfa_case_study/sample127_vs_126/plot_pair_compare_127_126.png`
- `runs/cfa_case_study/sample127_vs_126/case_meta_127_126.json`
- `attestations/triality/case_study/case_summary_127_126.md`

Representative set (Top/Median/Bottom frustrated deltas, 15 cases total):

```powershell
python tools/run_cfa_case_representative.py --results-jsonl runs/cfa_batch_primaryE/results.jsonl --cfa-jsonl data/cfa/cfa_v1.jsonl --batch-report attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt --group-size 5 --seed 7 --model-id Qwen/Qwen2.5-1.5B --device auto --topk 128 --perm-R 0 --min-coverage 0.30
```

Local metric-only check for one token table:

```powershell
python tools/eval_local_span.py --token-table-csv runs/cfa_case_study/sample127_vs_126/token_table_127.csv --topk 10 --percentile 0.90
```

Note:
- If HuggingFace access is blocked, extraction can fail at model load. Use a locally cached model id/path.
