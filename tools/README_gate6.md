# Gate6-A Workflow

Gate6-A builds a native local observation object from raw hidden-space triplets and then reuses the existing Gate5 triad-loop runner through a transient adapter.

Canonical artifacts:

- `manifest.json`
- `step_index.jsonl`
- `native_object_arrays.npz`
- `compatibility_input.json`
- `checksums.json`
- `gate5_boundary_input_provenance.json` in downstream Gate5 reruns

Important:

- `compatibility_input.json` is the canonical compatibility lane
- it uses `compat_vectors.{V_local8,Splus_local8,Sminus_local8}`
- it intentionally does not reuse legacy `V_8d` / `Splus_8d` / `Sminus_8d` names
- `tools/run_gate5_spike.py` can consume this file directly and will materialize a temporary Gate4-shaped adapter only for the Gate5 CLI call
- `manifest.json` includes `rank_local_counts` plus a scorecard-compatible `boundary_outcome_counts` summary
- downstream Gate5 reruns persist canonical provenance in both `manifest.json` and `gate5_boundary_input_provenance.json`

Gate6-B smoke:

- `tools/run_gate6_native_object_consumer.py` reads only `manifest.json`, `step_index.jsonl`, and `native_object_arrays.npz`
- it does not consume `compatibility_input.json`
- it builds an object-native edge-plane holonomy consumer directly from `coords_local` and `gram_raw`

## 1) Build Gate6-A artifacts on CFA native-raw samples

```powershell
python tools/build_gate6_native_local_span.py --samples-root runs/cfa_batch_primaryE_native_raw/samples --all-samples --out-dir runs/gate6_cfa_full
```

## 2) Build Gate6-A artifacts on Seam native-raw samples

```powershell
python tools/build_gate6_native_local_span.py --samples-root runs/gate5_seam_64e2e_native_raw/samples --all-samples --out-dir runs/gate6_seam_full
```

## 3) Run the fixed Gate5 triad loop on Gate6-A compatibility input

For CFA:

```powershell
python tools/run_gate5_spike.py --input runs/gate6_cfa_full/compatibility_input.json --out-dir runs/gate5_cfa_gate6a_v0 --run-id gate5_cfa_gate6a_v0 --dataset-revision-id cfa_gate6a_v0 --cfa-jsonl data/cfa/cfa_v1.jsonl
```

For Seam:

```powershell
python tools/run_gate5_spike.py --input runs/gate6_seam_full/compatibility_input.json --out-dir runs/gate5_seam_gate6a_v0 --run-id gate5_seam_gate6a_v0 --dataset-revision-id seam_gate6a_v0 --seam-jsonl runs/gate5_seam_64e2e/seam_v0.jsonl --evaluation-mode-id unsupervised_v1
```

## 4) Build a standing scorecard against the FWHT baseline

For CFA:

```powershell
python tools/build_gate5_boundary_scorecard.py --surface cfa --out runs/gate6_cfa_boundary_standing.md --csv-out runs/gate6_cfa_boundary_standing.csv --run "label=fwht_baseline;gate5_out=runs/gate5_cfa_spike;input=runs/gate5_cfa_ingestion/gate4_input.json" --run "label=gate6a_v0;gate5_out=runs/gate5_cfa_gate6a_v0;input=runs/gate6_cfa_full/compatibility_input.json;boundary_manifest=runs/gate6_cfa_full/manifest.json"
```

For Seam:

```powershell
python tools/build_gate5_boundary_scorecard.py --surface seam --out runs/gate6_seam_boundary_standing.md --csv-out runs/gate6_seam_boundary_standing.csv --run "label=fwht_baseline;gate5_out=runs/gate5_seam_64e2e/gate5_out;input=runs/gate5_seam_64e2e/gate4_prep/gate4_input.json" --run "label=gate6a_v0;gate5_out=runs/gate5_seam_gate6a_v0;input=runs/gate6_seam_full/compatibility_input.json;boundary_manifest=runs/gate6_seam_full/manifest.json"
```

## 5) Minimal helper regression

```powershell
python -m py_compile tools/build_gate6_native_local_span.py tools/test_build_gate6_native_local_span.py tools/run_gate5_spike.py
python tools/test_build_gate6_native_local_span.py
```

These tests cover:

- sign tie-break
- rank drop
- reconstruction
- rerun determinism
- legacy-name non-reuse in the canonical compatibility artifact

## 6) Run the object-native Gate6-B consumer smoke

For CFA smoke:

```powershell
python tools/run_gate6_native_object_consumer.py --gate6-dir runs/gate6_cfa_smoke --out-dir runs/gate6b_cfa_smoke --run-id gate6b_cfa_smoke
```

For Seam smoke:

```powershell
python tools/run_gate6_native_object_consumer.py --gate6-dir runs/gate6_seam_smoke --out-dir runs/gate6b_seam_smoke --run-id gate6b_seam_smoke
```

For full matched Gate6 native objects:

```powershell
python tools/run_gate6_native_object_consumer.py --gate6-dir runs/gate6_cfa_full --out-dir runs/gate6b_cfa_full --run-id gate6b_cfa_full
python tools/run_gate6_native_object_consumer.py --gate6-dir runs/gate6_seam_full --out-dir runs/gate6b_seam_full --run-id gate6b_seam_full
```

The primary metric is `edge_plane_loop_projective_chordal_v1`.
It is built from the holonomy of the three edge-plane normals:

- `span(V, Splus)`
- `span(Splus, Sminus)`
- `span(Sminus, V)`

## 7) Evaluate object-native Seam pairs directly

This evaluator does not use `compat_local8`.
It reads the Gate6-B token telemetry and the existing Seam pair definition, then emits
Gate5-style quietness headlines for:

- `edge_plane_loop_projective_chordal_v1`
- `score_F_gram_loop_v1`

Run:

```powershell
python tools/evaluate_gate6_native_object_seam_pairs.py --token-csv runs/gate6b_seam_full/gate6b_token_telemetry.csv --seam-jsonl runs/gate5_seam_64e2e/seam_v0.jsonl --out-dir runs/gate6b_seam_pairs_full --run-id gate6b_seam_pairs_full
```

Outputs:

- `manifest.json`
- `gate6b_seam_pair_summary.csv`
- `gate6b_seam_family_summary.csv`
- `gate6b_seam_report.md`
- `checksums.json`

Headline vocabulary is aligned with Gate5:

- `mean_delta_max_*`
- `mean_delta_p90_*`
- `mean_iqr_normalized_delta_max_*`
- `mean_top10_inflation_*_vs_clean_p90`

Rows with `loop_outcome != none` are deterministically skipped from pair statistics.
