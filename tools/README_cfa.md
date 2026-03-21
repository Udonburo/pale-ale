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

## 8) Gate4 parity / smoke validation

This validates the Rust Gate4 sink against Python-computed expectations on a small CFA subset.
- generates fresh teacher-forced triplets + labels
- packs `Gate4RunInputV1`
- runs `pale-ale gate4 run` twice
- checks sample/token parity and byte-identical rerun artifacts

```powershell
python tools/run_gate4_validation_smoke.py --cfa-jsonl data/cfa/cfa_v1.jsonl --model-id Qwen/Qwen2.5-1.5B --device auto
```

Outputs:
- input JSON: `runs/gate4_validation_smoke/gate4_input.json`
- Gate4 artifacts: `runs/gate4_validation_smoke/gate4_out_a/` and `gate4_out_b/`
- parity attestation: `attestations/triality/gate4_validation/YYYY-MM-DD_gate4_parity_smoke.txt`

Core CI does **not** run this workflow. It depends on local/generated sample directories and model access.

Representative-set smoke (reuses existing `runs/cfa_batch_primaryE/samples/`):

```powershell
python tools/run_gate4_representative_smoke.py
```

Outputs:
- input JSON: `runs/gate4_representative_smoke/gate4_input.json`
- Gate4 artifacts: `runs/gate4_representative_smoke/gate4_out_a/` and `gate4_out_b/`
- parity attestation: `attestations/triality/gate4_validation/YYYY-MM-DD_gate4_representative_smoke.txt`

This is also local/heavy validation, not core CI.

## 8a) Gate4 fixture parity (core CI)

Core CI uses a committed, model-free fixture instead of the local smoke paths above.

```powershell
python tools/run_gate4_fixture_parity.py
```

Inputs:
- `fixtures/gate4/core/gate4_input.json`
- `fixtures/gate4/core/cfa_subset.txt`
- `specs/internal/SPEC.internal.draft.md`

What it checks:
- `pale-ale gate4 run` parity against Python expectations
- canonical identity hashing via `gate4 hash-identity`
- deterministic rerun byte equality for:
  - `manifest.json`
  - `gate4_token_features.csv`
  - `gate4_sample_summary.csv`
  - `gate4_run_summary.csv`

## 9) Gate4 batch ingestion (official packer)

Use `build_gate4_input.py` when you already have extracted sample directories and want a deterministic `Gate4RunInputV1` without running smoke validation.

Explicit sample ids:

```powershell
python tools/build_gate4_input.py --samples-root runs/cfa_batch_primaryE/samples --sample-ids 0 1 2 3 --out runs/gate4_batch_ingestion/gate4_input.json --selection-manifest-out runs/gate4_batch_ingestion/batch_selection_manifest.json
```

Directory walk with filters:

```powershell
python tools/build_gate4_input.py --samples-root runs/cfa_batch_primaryE/samples --all-samples --variant frustrated --limit 25 --out runs/gate4_batch_ingestion/gate4_input.json --selection-manifest-out runs/gate4_batch_ingestion/batch_selection_manifest.json
```

Deterministic selection rules:
- sample directories are discovered as `sample_<id>`
- discovered ids are sorted ascending
- `variant` filter is applied after discovery
- `offset` / `limit` are applied after filtering
- output `sample_ids` are unique and sorted ascending

Artifacts:
- `gate4_input.json`
- optional `batch_selection_manifest.json`

## 10) Gate4 batch ingestion (one-shot run)

Use `run_gate4_batch_ingestion.py` to:
- select sample dirs
- build `gate4_input.json`
- verify that `samples_root` prompt/answer + metadata still match the claimed raw `--cfa-jsonl`
- compute canonical `dataset_hash_blake3`
- compute canonical `spec_hash_*_blake3`
- run `pale-ale gate4 run`

```powershell
python tools/run_gate4_batch_ingestion.py --samples-root runs/cfa_batch_primaryE/samples --all-samples --limit 4 --out-dir runs/gate4_batch_ingestion_smoke --dataset-revision-id cfa_v1_batch_smoke_v1
```

Outputs:
- `runs/gate4_batch_ingestion_smoke/gate4_input.json`
- `runs/gate4_batch_ingestion_smoke/batch_selection_manifest.json`
- `runs/gate4_batch_ingestion_smoke/batch_run_manifest.json`
- `runs/gate4_batch_ingestion_smoke/gate4_out/manifest.json`
- `runs/gate4_batch_ingestion_smoke/gate4_out/gate4_token_features.csv`
- `runs/gate4_batch_ingestion_smoke/gate4_out/gate4_sample_summary.csv`
- `runs/gate4_batch_ingestion_smoke/gate4_out/gate4_run_summary.csv`

## 11) Gate4 artifact sufficiency (artifact-only reanalysis)

Use `check_gate4_negative_stability.py` to recompute the existing negative-stability diagnostic
from Gate4 artifacts only.

Inputs:
- `manifest.json`
- `gate4_token_features.csv`
- `gate4_sample_summary.csv`
- `gate4_run_summary.csv`

Example:

```powershell
python tools/check_gate4_negative_stability.py --gate4-out-dir runs/gate4_artifact_sufficiency/gate4_out --out attestations/triality/gate4_validation/2026-03-09_gate4_negative_stability_from_artifacts.txt
```

One-shot full batch:

```powershell
python tools/run_gate4_artifact_sufficiency.py --samples-root runs/cfa_batch_primaryE/samples --cfa-jsonl data/cfa/cfa_v1.jsonl --out-dir runs/gate4_artifact_sufficiency --dataset-revision-id cfa_v1_full200_gate4_v1
```

Outputs:
- `runs/gate4_artifact_sufficiency/gate4_input.json`
- `runs/gate4_artifact_sufficiency/batch_selection_manifest.json`
- `runs/gate4_artifact_sufficiency/batch_run_manifest.json`
- `runs/gate4_artifact_sufficiency/gate4_out/manifest.json`
- `runs/gate4_artifact_sufficiency/gate4_out/gate4_token_features.csv`
- `runs/gate4_artifact_sufficiency/gate4_out/gate4_sample_summary.csv`
- `runs/gate4_artifact_sufficiency/gate4_out/gate4_run_summary.csv`
- `attestations/triality/gate4_validation/YYYY-MM-DD_gate4_negative_stability_from_artifacts.txt`

This is local/heavy validation, not core CI.
