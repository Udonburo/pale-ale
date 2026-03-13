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
## 12) Aggregation Candidate Comparison

To compare simple field readers on the fixed FWHT baseline and fixed rotor comparator:

```powershell
python tools/analyze_gate5_aggregation_candidates.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_aggregation_candidates --k-values 0 3 --sample-ids 137 147 149 11 167
```

This emits:

- `gate5_aggregation_candidates_global_summary.csv`
- `gate5_aggregation_candidates_world_summary.csv`
- `gate5_aggregation_candidates_genealogy_summary.csv`
- `gate5_aggregation_candidates_selected_cases.csv`
- `gate5_aggregation_candidates_report.md`
- `gate5_aggregation_candidates_decision.md`

The comparison stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- `k=0,3` only

Candidates currently compared:

- `raw_token_rotor`
- `inside_to_after_ratio`
- `first_after_defect_score_ranknorm`
- `prefix_penalized_inside_mass_w1_ranknorm`
- `prefix_penalized_inside_mass_w3_ranknorm`

Scale handling:

- `inside_to_after_ratio` is compared on raw scores because it is already scale-free
- scalar / mass candidates use sample-wise rank-normalized token scores before aggregation so `score_F_loop` and `rotor_loop_chordal_v1` are not compared on incompatible raw magnitudes

Use this when the next question is:

- whether a better reader of the current rotor field exists without changing the boundary
- whether reachability / temporal improvements survive field-side aggregation
- whether genealogy remains the limiting failure mode

## 13) Aggregation Reader Failure-Mode Analysis

To isolate why `first_after_defect_score_ranknorm` is strong at `k=3` but weak at `k=0`:

```powershell
python tools/analyze_gate5_aggregation_failure_mode.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_aggregation_failure_mode --sample-ids 137 147 149 11 167 --k-values 0 3
```

This emits:

- `gate5_aggregation_failure_mode_sample_summary.csv`
- `gate5_aggregation_failure_mode_world_summary.csv`
- `gate5_aggregation_failure_mode_selected_cases.csv`
- `gate5_aggregation_failure_mode_report.md`
- `gate5_aggregation_failure_mode_decision.md`

The analysis stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- `first_after_defect_score_ranknorm` only
- `k=0,3` only

Use this when the next question is:

- whether the strong `k=3` reader is a true early-signal / label-mismatch effect
- whether the `k=3` gain is too dependent on rank normalization
- whether genealogy is the main blocker even after reader refinement

## 14) Genealogy Reader Refinement

To test genealogy-focused reader refinements on the fixed FWHT baseline and fixed rotor comparator:

```powershell
python tools/analyze_gate5_genealogy_reader_refinement.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_genealogy_reader_refinement --sample-ids 137 147 149 11 167 --k-values 0 3
```

This emits:

- `gate5_genealogy_reader_refinement_sample_summary.csv`
- `gate5_genealogy_reader_refinement_world_summary.csv`
- `gate5_genealogy_reader_refinement_selected_cases.csv`
- `gate5_genealogy_reader_refinement_report.md`
- `gate5_genealogy_reader_refinement_decision.md`

The comparison stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- `k=0,3` only

Candidates currently compared:

- `first_after_defect_score_ranknorm`
- `post_start_mass_w1_ranknorm`
- `post_start_mean_w3_ranknorm`
- `before_penalized_first_after_w1_ranknorm`

Use this when the next question is:

- whether genealogy can be improved without giving back reachability / temporal gains
- whether a short post-start window is better than a single first-after read
- whether a before-band penalty helps genealogy without collapsing selected win cases

## 15) Reader Failure Autopsy

To diagnose why `post_start_mass_w1_ranknorm` remains weak at `k=0` even when it helps at `k=3`:

```powershell
python tools/analyze_gate5_reader_failure_autopsy.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_reader_failure_autopsy --sample-ids 137 147 149 11 167 --k-values 0 3
```

This emits:

- `gate5_reader_failure_autopsy_sample_summary.csv`
- `gate5_reader_failure_autopsy_world_summary.csv`
- `gate5_reader_failure_autopsy_selected_wins.csv`
- `gate5_reader_failure_autopsy_genealogy_cases.csv`
- `gate5_reader_failure_autopsy_temporal_cases.csv`
- `gate5_reader_failure_autopsy_report.md`
- `gate5_reader_failure_autopsy_decision.md`

The analysis stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- `k=0,3` only
- the post-start reader family only

Candidates compared:

- `first_after_defect_score_ranknorm`
- `post_start_mass_w1_ranknorm`
- `post_start_mass_w1_prefix_penalized_ranknorm`
- `post_start_mean_w2_ranknorm`

Use this when the next question is:

- why `post_start_mass_w1_ranknorm` still fails on `genealogy k=0`
- whether `temporal k=0` is a real side effect or just a reporting artifact
- whether the post-start family is still refinable, needs a genealogy-specific branch, or should be stopped

## 16) Genealogy-Specific Reader Lab

To test narrow genealogy-specific readers on the fixed FWHT baseline and fixed rotor comparator:

```powershell
python tools/analyze_gate5_genealogy_specific_reader.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_genealogy_specific_reader --sample-ids 137 147 149 11 167 --k-values 0 3
```

This emits:

- `gate5_genealogy_specific_reader_sample_summary.csv`
- `gate5_genealogy_specific_reader_world_summary.csv`
- `gate5_genealogy_specific_reader_selected_cases.csv`
- `gate5_genealogy_specific_reader_report.md`
- `gate5_genealogy_specific_reader_decision.md`

The analysis stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- `k=0,3` only
- genealogy-specific readers only

Candidates compared:

- `first_after_defect_score_ranknorm`
- `genealogy_post_start_mass_w1_ranknorm`
- `genealogy_post_start_mean_w2_ranknorm`
- `genealogy_before_penalized_first_after_w1_ranknorm`

Guardrails:

- worst genealogy failures are the baseline reader `genealogy k=0` rows with the lowest `delta_rotor_vs_F`, top 5
- temporal `k=0` must not be worse than the current post-start reader by more than `0.02`
- zero genealogy frustrated rows or unresolved requested sample ids fail fast

Use this when the next question is:

- whether a genealogy-only reader line is justified
- whether genealogy can be improved at `k=0` without paying back temporal `k=0`
- whether global selected rotor-win cases remain explainable under a genealogy-specific reader

## 17) Genealogy Label Geometry Lab

To compare diagnostic label geometries on the fixed FWHT baseline and fixed rotor comparator:

```powershell
python tools/analyze_gate5_genealogy_label_geometry.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_genealogy_label_geometry --sample-ids 137 147 149 11 167
```

This emits:

- `gate5_genealogy_label_geometry_sample_summary.csv`
- `gate5_genealogy_label_geometry_global_summary.csv`
- `gate5_genealogy_label_geometry_world_summary.csv`
- `gate5_genealogy_label_geometry_selected_cases.csv`
- `gate5_genealogy_label_geometry_report.md`
- `gate5_genealogy_label_geometry_decision.md`

The analysis stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- canonical CFA labels remain unchanged
- diagnostic label geometries only

Geometries compared:

- `inside_span`
- `onset_only`
- `start_neighborhood_w1`
- `start_neighborhood_w3`
- `prefix_only_w1`
- `prefix_only_w3`
- `symmetric_dilation_k1`
- `symmetric_dilation_k3`

Guardrails:

- zero frustrated rows fails fast
- zero frustrated `genealogy` rows fails fast
- unresolved requested sample ids fail fast
- selected cases include both existing global rotor wins and baseline genealogy worst failures

Use this when the next question is:

- whether genealogy's persistent failure is better explained by supervision geometry mismatch
- whether prefix/onset/start-neighborhood geometries help genealogy more than temporal/reachability
- whether the genealogy problem should move to benchmark/label-geometry work rather than reader work

## 18) Genealogy Supervision Policy Summary

To build a fixed policy summary from the existing genealogy label-geometry diagnostic:

```powershell
python tools/build_genealogy_supervision_policy_summary.py --label-geometry-out-dir runs/gate5_cfa_spike_genealogy_label_geometry --out-dir runs/gate5_cfa_spike_genealogy_supervision_policy
```

This emits:

- `genealogy_supervision_policy_summary.csv`
- `genealogy_supervision_policy_report.md`
- `genealogy_supervision_policy_decision.md`

This step does not:

- change canonical CFA labels
- change the Gate5 boundary
- change the Gate5 comparator

It exists to formalize the handling rule:

- canonical genealogy evaluation remains `inside_span`
- diagnostic geometry remains supplementary only
- canonical and diagnostic views must not be merged into one aggregate

Use this when the next question is:

- how genealogy should be reported going forward
- which geometry is canonical vs diagnostic
- how to keep benchmark policy separate from model-side iteration

## 19) Genealogy Residual Autopsy

To test whether genealogy's persistent failure is mostly benchmark-geometry mismatch or whether residual-side weakness remains:

```powershell
python tools/analyze_gate5_genealogy_residual_autopsy.py --gate5-out-dir runs/gate5_cfa_spike --out-dir runs/gate5_cfa_spike_genealogy_residual_autopsy
```

This emits:

- `genealogy_residual_shape_summary.csv`
- `genealogy_residual_selected_failures.csv`
- `genealogy_residual_autopsy_decision.md`

The analysis stays fixed on:

- FWHT baseline only
- `rotor_loop_chordal_v1` only
- canonical `inside_span`
- diagnostic `prefix_only_w3`

It does not:

- add a new boundary
- add a new reader
- add a new residual
- rewrite canonical labels

Use this when the next question is:

- whether genealogy is mostly explained by supervision geometry mismatch
- whether a residual-side weakness remains even after the diagnostic prefix view
- which genealogy failures still remain negative under `prefix_only_w3`
