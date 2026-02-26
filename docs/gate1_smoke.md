# Gate1 Smoke Run Guide

This page fixes two reproducible smoke fixtures for Gate1 closeout under `SPEC.phase4.md` `v4.0.0-ssot.9`.

## Fixtures

- Tiny E2E fixture: `docs/examples/gate1_smoke_tiny.json`
  - Purpose: validate end-to-end execution path quickly.
  - Includes: antipodal, collinear, and missing label.
  - Note: tiny `n` may produce `run_valid=false` because sanity/collapse logic is still active by design.
  - Expected in current implementation: `run_invalid_reason=dominant_link_collapse`.
- Sanity-16 fixture: `docs/examples/gate1_smoke_sanity16.json`
  - Purpose: exercise K=16 sanity slice behavior in a realistic shape.
  - Uses 16 samples so sanity sampling runs at full `K`.
  - Expected in current implementation: `run_valid=true`.

## Command

Use one of the fixtures with `pale-ale gate1 run`:

```bash
pale-ale gate1 run \
  --input docs/examples/gate1_smoke_sanity16.json \
  --out ./tmp/gate1-smoke-out \
  --evaluation-mode supervised_v1 \
  --dataset-revision-id "smoke_sanity16_v1" \
  --dataset-hash-blake3 "dataset_hash_placeholder" \
  --spec-hash-raw-blake3 "spec_hash_raw_placeholder" \
  --spec-hash-blake3 "spec_hash_lf_placeholder" \
  --unitization-id "sentence_split_v1" \
  --rotor-encoder-id "encoder_revision_placeholder" \
  --rotor-encoder-preproc-id "preproc_id_placeholder" \
  --vec8-postproc-id "vec8_postproc_id_placeholder"
```

## Expected artifacts

`--out` directory must contain all four files:

- `manifest.json`
- `summary.csv`
- `link_topk.csv`
- `link_sanity.md`

## What to check

- All four artifacts are present.
- `manifest.json` passes the built-in validator (orchestrator already fails hard if invalid).
- `summary.csv` has exactly one data row.
- `link_topk.csv` rows are sorted by `(sample_id, ans_unit_id, rank, doc_unit_id)` ascending.
- Re-running with the same input and identity should produce deterministic bytes.

## Minimal deterministic check

```bash
diff -u ./tmp/runA/manifest.json ./tmp/runB/manifest.json
diff -u ./tmp/runA/summary.csv ./tmp/runB/summary.csv
diff -u ./tmp/runA/link_topk.csv ./tmp/runB/link_topk.csv
diff -u ./tmp/runA/link_sanity.md ./tmp/runB/link_sanity.md
```
