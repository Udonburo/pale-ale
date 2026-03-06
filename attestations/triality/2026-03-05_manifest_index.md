# Triality Commit Manifest (2026-03-05)

This file fixes the exact scope, evidence, and verification status of the current CFA / triality work before the next design step.

## Scope

Base commit before this manifest set:
- `git rev-parse --short HEAD` -> `aaeecf5`

Worktree snapshot at manifest time:
- `git status --short` entries after public cleanup: `19`
- `.gitignore` diff: targeted ignore added for `attestations/triality/cfa_batch/`
- `runs/` outputs remain untracked by design

Commit scope in this batch:
- CFA dataset generator and span-label tooling
- teacher-forcing triality extraction updates
- preregistered CFA batch runner + aggregator
- case-study plotting + representative-set runner
- Gate4 feature contract draft
- selected commit-ready attestations copied under `attestations/triality/`

Local-only verbose outputs not intended for commit:
- `attestations/triality/cfa_batch/` (`200` per-sample CFA eval reports)

## Commands Executed

Dataset + prereg batch:

```powershell
python tools/generate_cfa_dataset.py --out data/cfa/cfa_v1.jsonl --meta-out data/cfa/cfa_v1_meta.json --n-worlds 100 --seed 7
python tools/run_cfa_batch_primaryE.py --cfa-jsonl data/cfa/cfa_v1.jsonl --device auto --model-id Qwen/Qwen2.5-1.5B
python tools/aggregate_cfa_batch.py --results-jsonl runs/cfa_batch_primaryE/results.jsonl --cfa-jsonl data/cfa/cfa_v1.jsonl
```

Single pair case study:

```powershell
python tools/plot_cfa_case_pair.py --cfa-jsonl data/cfa/cfa_v1.jsonl --sample-id 127 --device auto --model-id Qwen/Qwen2.5-1.5B --perm-R 0 --seed 7
```

Representative set:

```powershell
python tools/run_cfa_case_representative.py --results-jsonl runs/cfa_batch_primaryE/results.jsonl --cfa-jsonl data/cfa/cfa_v1.jsonl --batch-report attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt --group-size 5 --seed 7 --model-id Qwen/Qwen2.5-1.5B --device auto --topk 128 --perm-R 0 --min-coverage 0.30
```

Verification:

```powershell
python -m py_compile tools/generate_cfa_dataset.py tools/extract_triality_triplets.py tools/labels_from_cfa_spans.py tools/eval_triality_token.py tools/run_cfa_batch_primaryE.py tools/aggregate_cfa_batch.py tools/plot_cfa_case_pair.py tools/eval_local_span.py tools/run_cfa_case_representative.py
python tools/eval_local_span.py --token-table-csv runs/cfa_case_study/sample127_vs_126/token_table_127.csv --topk 10 --percentile 0.90
```

## Artifact Index

Primary dataset / aggregate evidence:
- `data/cfa/cfa_v1.jsonl`
- `data/cfa/cfa_v1_meta.json`
- `attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt`
- `attestations/triality/case_study/index.md`
- `attestations/triality/case_study/representative_set_summary.md`
- `docs/gate4_feature_contract_draft.md`

Attestation counts:
- `attestations/triality/case_study/` top-level files: `17`

## SHA256

Core data / evidence:
- `data/cfa/cfa_v1.jsonl` -> `76658a4ee2230460ae7525ffc5c488fe600b1b9ae51df3cc77d1b868e874e7d2`
- `data/cfa/cfa_v1_meta.json` -> `11db5557242ac4a110b77ca1511dc3417799942e2beca4b114ab5d2735599461`
- `runs/cfa_batch_primaryE/results.jsonl` -> `76c444794c55379a8475dd97f1d6e6784939c646ea05cf79b79878d3fb6697ab`
- `attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt` -> `59c8d217f41686983fc19199dd0baa415f1adde18f72880400a5fda2ce8ef6b3`
- `attestations/triality/case_study/index.md` -> `a5b54bbf701c864e9809a8b211e62a9524a17ac6f684733fdda888d6f4918bb9`
- `attestations/triality/case_study/representative_set_summary.md` -> `79e69ef8a4297da66daf1aef1f5dcaa79238cf3686efc1b080712c81f2051d50`
- `docs/gate4_feature_contract_draft.md` -> `6eb3ef9691935ee586071a0d187619702299c2407fff106020f4ba98da1b37b6`

Scripts / docs:
- `tools/generate_cfa_dataset.py` -> `fb1c9d2c9512276f194b2c0d74d29c2ca984f2a80cc9a6e8c13b9e3867ad5d65`
- `tools/extract_triality_triplets.py` -> `d30c2415762aa89fb000e45dc62d5a9824500676dbd5ae94568524c38e8d0035`
- `tools/labels_from_cfa_spans.py` -> `e9d451669334dab1dfaeebdda48d378941af50bd1c73ef5625b9f3f8c66557eb`
- `tools/eval_triality_token.py` -> `efc648f44a5c5f04fcf68e95f85864f5b2bbb8436e0d1ad3e4b71790f6607cf3`
- `tools/run_cfa_batch_primaryE.py` -> `fcfba00378f9320c2f3cb33ca1908ce1464387b0ea257ff0daa15367d14b5617`
- `tools/aggregate_cfa_batch.py` -> `f59c09fe7f03490d9e2915f5a5d86b573dddb9a4292cb167923db5d69bd3e280`
- `tools/plot_cfa_case_pair.py` -> `df236deea84c493851c088b77092c8b33643e3369315657789afb5b80e91e5a6`
- `tools/eval_local_span.py` -> `2a29a8914ca1163b95ff9669df70dd1b4d9f1ee0375a0b0101741ec8a9ee2b29`
- `tools/run_cfa_case_representative.py` -> `3d776dbe013b26266adb5370124c7b1b0029c51cfd5731a573668bae8f1f3404`
- `tools/run_cfa_smoke.ps1` -> `bcc53484a75aac7ec5cd32dc76ead92e15fd876d8d979fdfaea2ff4fd273cd21`
- `tools/README_cfa.md` -> `adcb3a616ed405fce3aef1e5635ec3b54da673e74fcd19894086e688f3b5efb9`
- `tools/README_realdata_ab.md` -> `208783230087c470de33af06a2e5731a4dba990d723eac050f3d0b90bedbc443`
- `docs/cfa_prereg_template.md` -> `a496b5d412994f61f5fc355b6746079aa96a081120becdef2308d8dfd3a3f8a0`
- `docs/cfa_case_study_template.md` -> `0506fc0b2f9bed7dcfd9457036cb3175dd9ab264f81ebf057a765c790de24264`

## Validation Summary

Static validation:
- `py_compile` passed for all new/updated Python tools in the CFA / triality path

Runtime validation:
- CFA prereg batch completed end-to-end with `200/200` sample reports under `attestations/triality/cfa_batch/`
- Aggregate report verdict: `GO`
- Representative case-study batch completed for `15` frustrated samples (`top=5`, `median=5`, `bottom=5`)
- Local metric check on `sample127_vs_126`:
  - `Hit@10 = 6`
  - `First-hit distance = 0`
  - `First-hit after defect distance = 0`

## Commit Notes

Expected tracked scope in this commit:
- `.gitignore`
- `tools/`
- `docs/`
- `attestations/triality/`

Expected untracked / ignored runtime outputs not part of commit:
- `runs/`
- any HF model cache
- `attestations/triality/cfa_batch/`

Recommended commit themes:
- CFA dataset + batch verification
- case-study tooling + representative-set local metrics
- Gate4 feature contract draft
