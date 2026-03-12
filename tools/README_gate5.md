# Gate5 Workflow

Gate5 consumes the existing `Gate4RunInputV1` boundary from `gate4_input.json`.

It does not change Gate4. It reads the same proxy observables and emits transport-loop telemetry.

## 1) Prepare or reuse `gate4_input.json`

For CFA, reuse the existing Gate4 batch ingestion flow:

```powershell
python tools/run_gate4_batch_ingestion.py --samples-root runs/cfa_batch_primaryE/samples --all-samples --cfa-jsonl data/cfa/cfa_v1.jsonl --out-dir runs/gate5_cfa_ingestion --dataset-revision-id cfa_v1_gate5_v0
```

This writes:

- `runs/gate5_cfa_ingestion/gate4_input.json`

## 2) Run Gate5 spike on CFA

```powershell
python tools/run_gate5_spike.py --input runs/gate5_cfa_ingestion/gate4_input.json --out-dir runs/gate5_cfa_spike --run-id gate5_cfa_spike --dataset-revision-id cfa_v1_gate5_v0 --cfa-jsonl data/cfa/cfa_v1.jsonl
```

Outputs:

- `runs/gate5_cfa_spike/manifest.json`
- `runs/gate5_cfa_spike/gate5_token_telemetry.csv`
- `runs/gate5_cfa_spike/gate5_sample_summary.csv`
- `runs/gate5_cfa_spike/gate5_attestation_report.txt`
- `runs/gate5_cfa_spike/gate5_aggregate_report.md`

## 3) Generate Seam Challenge Set v0

```powershell
python tools/generate_seam_challenge.py --out data/seam/seam_v0.jsonl --meta-out data/seam/seam_v0_meta.json --n-pairs 64 --seed 7
```

The generator emits paired rows:

- `clean_consistent`
- `seam_perturbed_consistent`

Each row carries:

- `pair_id`
- `perturbation_family`
- `perturbation_spans`

## 4) Run Gate5 spike on Seam

Gate5 still consumes `gate4_input.json`, so the Seam side must first be extracted and packed through the same teacher-forcing path as CFA.

Use the dedicated Seam builder:

```powershell
python tools/build_seam_gate4_input.py --seam-jsonl data/seam/seam_v0.jsonl --out-dir runs/seam_gate5 --device auto
```

This writes:

- `runs/seam_gate5/gate4_input.json`
- `runs/seam_gate5/samples/sample_*/`
- `runs/seam_gate5/seam_selection_manifest.json`
- `runs/seam_gate5/seam_build_manifest.json`

Notes:

- By default the builder enforces the paired quietness contract and fails if a subset drops either the clean or perturbed side of a `pair_id`.
- Use `--allow-incomplete-pairs` only for debugging or partial extraction, not for the canonical Seam quietness run.

Once a Seam `gate4_input.json` exists:

```powershell
python tools/run_gate5_spike.py --input runs/seam_gate5/gate4_input.json --out-dir runs/gate5_seam_spike --run-id gate5_seam_spike --dataset-revision-id seam_v0_gate5 --seam-jsonl data/seam/seam_v0.jsonl --evaluation-mode-id unsupervised_v1
```

Notes:

- `aggregate_gate5_spike.py` uses the Seam JSONL as a sidecar for pair linkage and perturbation spans.
- Seam aggregation now also emits structured sidecars next to the markdown report:
  - `*_seam_pair_summary.csv`
  - `*_seam_family_summary.csv`
- If no canonical dataset hash path is available, the attestation report will note a local fallback identity source instead of claiming full attestation.

## 5) Re-aggregate from existing Gate5 artifacts

```powershell
python tools/aggregate_gate5_spike.py --gate5-out-dir runs/gate5_cfa_spike --out runs/gate5_cfa_spike/gate5_aggregate_report.md --cfa-jsonl data/cfa/cfa_v1.jsonl
```

or

```powershell
python tools/aggregate_gate5_spike.py --gate5-out-dir runs/gate5_seam_spike --out runs/gate5_seam_spike/gate5_aggregate_report.md --surface seam --seam-jsonl data/seam/seam_v0.jsonl
```

## 6) Boundary Liberation Smoke

For post-Gate5 boundary comparison work, first preserve raw native triplets during extraction:

```powershell
python tools/build_seam_gate4_input.py --seam-jsonl data/seam/seam_v0.jsonl --out-dir runs/seam_gate5_native_source --device auto --emit-native-raw
```

Then build a native-local-span candidate on the same samples:

```powershell
python tools/build_native_local_span_gate4_input.py --samples-root runs/seam_gate5_native_source/samples --all-samples --out-dir runs/seam_gate5_native --coordinate-rule anchored_projection_v0
```

or the centered affine diagnostic variant:

```powershell
python tools/build_native_local_span_gate4_input.py --samples-root runs/seam_gate5_native_source/samples --all-samples --out-dir runs/seam_gate5_centered_affine --coordinate-rule centered_affine_local_span_v1
```

or the first rank-3 origin-span candidate:

```powershell
python tools/build_native_local_span_gate4_input.py --samples-root runs/seam_gate5_native_source/samples --all-samples --out-dir runs/seam_gate5_origin_span --coordinate-rule origin_span_projection_v2
```

These builders emit:

- `gate4_input.json`
- `native_local_span_boundary_steps.ndjson`
- `native_local_span_build_manifest.json`

Notes:

- `anchored_projection_v0` and `centered_affine_local_span_v1` are useful dead-zone diagnostics, but both can collapse a triplet into an affine rank-2 plane.
- `origin_span_projection_v2` is the first candidate that preserves the raw normalized triplet span and can materialize rank-3 frames when the source vectors support it.

## 7) Dead-Zone Diagnosis

To compare the inherited 8D baseline against a richer boundary candidate under the same Gate5 comparator:

```powershell
python tools/diagnose_boundary_dead_zone.py --baseline-input runs/seam_gate5_native_source/gate4_input.json --candidate-input runs/seam_gate5_native/gate4_input.json --candidate-boundary-steps runs/seam_gate5_native/native_local_span_boundary_steps.ndjson --out-dir runs/seam_gate5_dead_zone_diag
```

This emits:

- `baseline_gate5_diagnostic.csv`
- `candidate_gate5_diagnostic.csv`
- `matched_boundary_diagnostics.csv`
- `baseline_transport_sample_summary.csv`
- `candidate_transport_sample_summary.csv`
- `boundary_dead_zone_report.md`

Use this before broad CFA/Seam reruns to identify whether a candidate boundary is already flat at:

- emitted boundary coordinates
- comparator input normalization
- edge / loop transport residual stages

## 8) Gate5 Failure-Mode Autopsy

After a full CFA Gate5 run, inspect where rotor loses globally but still wins locally:

```powershell
python tools/analyze_gate5_autopsy.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_autopsy --sample-ids 137 147 149 11 167
```

This emits:

- `gate5_autopsy_selected_cases.csv`
- `gate5_autopsy_selected_top_tokens.csv`
- `gate5_autopsy_frustrated_world_summary.csv`
- `gate5_autopsy_report.md`

Use this when the next question is not "can Gate5 run?" but:

- where rotor wins despite losing globally
- whether rotor has an early-hit / before-span bias
- which world types carry the strongest rotor-vs-F split

## 9) Span-Dilation Sensitivity

To test whether rotor is reacting to a near-defect neighborhood rather than missing the defect entirely:

```powershell
python tools/analyze_gate5_span_dilation.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_span_dilation --sample-ids 137 147 149 11 167 --k-values 0 1 2 3
```

This emits:

- `gate5_span_dilation_global_summary.csv`
- `gate5_span_dilation_world_summary.csv`
- `gate5_span_dilation_selected_cases.csv`
- `gate5_span_dilation_report.md`

Interpretation:

- span dilation is a calibration / signal-shape diagnostic only and does not by itself justify comparator promotion
- if rotor improves sharply for small `k`, the problem is likely early-hit / label-alignment mismatch
- if rotor stays weak as `k` grows, the problem is more likely in the residual itself
- if only some `world_type` groups improve, the remaining issue is more benchmark- or motif-specific than global

## 10) Genealogy-Only Autopsy

To isolate the persistent `world_type=genealogy` failure mode on the fixed FWHT baseline:

```powershell
python tools/analyze_gate5_autopsy.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_genealogy_autopsy --world-type genealogy --output-prefix gate5_genealogy_autopsy
```

This emits:

- `gate5_genealogy_autopsy_selected_cases.csv`
- `gate5_genealogy_autopsy_top_tokens.csv`
- `gate5_genealogy_autopsy_world_summary.csv`
- `gate5_genealogy_autopsy_report.md`

Use this when the next question is:

- whether genealogy is mainly a before-span failure
- whether rotor overshoots after the defect span
- whether within-span mass is too weak even when token scores are live

## 11) Field-Side Diagnostics

To inspect mass placement and decay shape on the existing FWHT baseline, without rerunning extraction:

```powershell
python tools/analyze_gate5_field_diagnostics.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_field
```

This emits:

- `gate5_field_diagnostics_sample_summary.csv`
- `gate5_field_diagnostics_world_summary.csv`
- `gate5_field_diagnostics_selected_cases.csv`
- `gate5_field_diagnostics_report.md`

Use this when the next question is:

- whether rotor is mainly early / prefix-heavy
- whether useful signal concentrates near defect start
- whether genealogy remains a mass-placement failure even after dilation diagnostics
- whether a fixed set of case-study sample ids shows the same field-shape pattern as the full frustrated population
