# Real-data A/B + Triality (halueval-spans)

Generates Gate2RunInputV1 from `data/realdata/halueval_spans_train.jsonl`, then runs Gate2/Gate3.

Methods:
- E0: `chunk_sequential_48x8_v1` (`mean_then_chunk`, fixed 48 steps)
- E1: `gaussian_proj_d8_seed7_v1` (per-unit projection)
- E2: `e8_softsnap_chunk48_k3_beta12_v1` (fixed 48 steps)
- Unitization: `sentence_split_v1`, `sentence_split_v2_min4`

Run balanced E1:
```powershell
powershell -ExecutionPolicy Bypass -File tools/run_realdata_ab.ps1 -N0 100 -N1 100 -UnitizationId sentence_split_v2_min4 -E1Only
```

Run balanced E2:
```powershell
powershell -ExecutionPolicy Bypass -File tools/run_realdata_ab.ps1 -N0 100 -N1 100 -UnitizationId sentence_split_v2_min4 -E2Only
```

Fetch/refresh HF source JSONL:
```powershell
python tools/fetch_hf_halueval_spans.py --out data/realdata/halueval_spans_train.jsonl --n 1000
```

Triality extraction + labeling + eval:
```powershell
python tools/extract_triality_triplets.py --prompt "Hello" --deterministic
python tools/labels_from_halueval_spans.py --sample-id 0 --triplets-ndjson runs/triality_smoke/triplets_out.ndjson --out runs/triality_smoke/labels_step.jsonl
python tools/eval_triality_token.py --ndjson runs/triality_smoke/triplets_out.ndjson --labels-jsonl runs/triality_smoke/labels_step.jsonl --labels-meta-json runs/triality_smoke/labels_step_meta.json --min-label-coverage 0.30 --perm-R 2000 --seed 7
```

Holdout batch (primary endpoint E):
```powershell
powershell -ExecutionPolicy Bypass -File tools/run_triality_batch_holdout.ps1 -StartIndex 1 -EndIndex 50 -PermR 2000 -Seed 7
```

Key attestations:
- `attestations/realdata_ab/YYYY-MM-DD_*`
- `attestations/triality/YYYY-MM-DD_eval_hf0_teacher_forcing_primaryE.txt`
- `attestations/triality/YYYY-MM-DD_holdout_primaryE_summary.txt`
