# Gate5 Minimal Scope SSOT

Status: Draft
Date: 2026-03-10

This document freezes the minimal Gate5 scope against the current repository state. It is intentionally narrow.

Reference decision record: `gate5_decision_record.md`

## 1. Purpose

Gate5 is an experimental transport-based loop residual scorer.

Gate5 is not:

- a proof layer
- a verdict layer
- a fusion layer
- a rewrite of Gate4
- a universal hallucination detector

Gate5 exists to test one bounded hypothesis only:

> Transport-based loop residuals on shared projected proxy observables may separate seam-like local variation from contradiction-like defect better than current distance-based score families under controlled settings.

Prior for Gate5 v0:

- any effect, if present, is expected to be modest
- no uplift over current distance-based families should be assumed
- null results and tradeoffs are first-class outcomes
- this scope should stay minimal precisely because the expected effect may be small

## 2. Implementation-Binding Facts

These items are confirmed from the current repository and must be treated as binding facts for Gate5 planning.

### 2.1 Input Boundary

- The canonical input boundary already exists as `Gate4RunInputV1` in `crates/diagnose/src/gate4.rs`.
- The on-disk form already exists as `gate4_input.json`, produced by `tools/build_gate4_input.py` and referenced in `tools/README_cfa.md`.
- Each `Gate4TokenStepInputV1` already carries the required 8D proxy observables:
  - `V_8d`
  - `Splus_8d`
  - `Sminus_8d`
- Gate4 validates these fields as finite 8-vectors before computing its own outputs.
- Gate4 also already carries token labels and the baseline fields `baseline_logprob` and `baseline_entropy`.

### 2.2 Existing Gate4 Alignment Facts

- Existing `score_F_loop` is token-step aligned and computed per token step from `V_8d`, `Splus_8d`, and `Sminus_8d`.
- Existing `score_E_v_sminus_vnext` is transition-aligned and only defined for `step -> step+1`.
- Gate4's existing transition missing enum is only `none|final_step_no_successor`.
- Gate5 must not rewrite these Gate4 definitions or schemas.

### 2.3 Existing Rotor / Algebra Facts

- `crates/rotor/src/lib.rs` already provides `simple_rotor29_doc_to_ans` for 8D-to-rotor edge construction.
- The current edge-construction constants in that path are:
  - `tau_wedge = 1e-6`
  - `tau_antipodal_dot = 1.0 - 1e-6`
- The current branch order is fixed in code as antipodal -> collinear -> normal.
- Same-direction collinear inputs already materialize as identity-like rotors and are marked internally by `is_collinear = true`.
- Antipodal inputs already branch to `RotorStep::AntipodalAngleOnly`; they do not currently materialize a reusable rotor.
- `crates/rotor/src/even128.rs` already provides the reusable algebra path for Gate5 composition:
  - `embed_simple29_to_even128`
  - `left_fold_mul_time_reversed_normalize_once`
  - `Even128::identity`
  - `inner`
  - `scalar_part`
  - `normalize`
- Current composition order is fixed by code and tests: `left_fold_mul_time_reversed_normalize_once(&[R1, R2, R3]) = normalize(R3 * R2 * R1)`.

## 3. Provisional Engineering Choices Frozen For Gate5 v0

These are not claims about existing Gate5 code. They are the minimal v0 engineering decisions that Gate5 implementation should follow.

### 3.1 Gate5 Identity

- Gate5 is telemetry-only.
- Gate5 emits no threshold, verdict, or learned fusion output.
- Gate5 evaluates a loop residual family on top of existing Gate4 proxy observables.

### 3.2 Canonical Comparison Path

- Gate5 v0 shall consume `Gate4RunInputV1` directly.
- Gate5 v0 shall not use `gate4_token_features.csv` as its primary computational input.
- Gate5 v0 shall not fall back to raw triplets as the primary input boundary.

### 3.3 Frozen Edge-Construction Constants

Gate5 v0 shall call the Rust rotor path with explicit constants:

- `tau_wedge = 1e-6`
- `tau_antipodal_dot = 1.0 - 1e-6`

Policy:

- Gate5 v0 shall not rely on `RotorConfig::default()` implicitly.
- The explicit constants above are chosen because they match the current repository behavior exactly.
- If these constants ever need to change, Gate5 should treat that as a versioned behavior change, not a silent default drift.

### 3.4 Algebra Path

Gate5 v0 shall use the existing Rust rotor/algebra implementation as follows:

1. Construct edge rotors from 8D proxy observables using `simple_rotor29_doc_to_ans`.
2. Define the token-step loop at step `t` as:
   - `R1 = rotor(V_t -> Splus_t)`
   - `R2 = rotor(Splus_t -> Sminus_t)`
   - `R3 = rotor(Sminus_t -> V_t)`
3. Embed each materialized edge rotor into `Even128` with `embed_simple29_to_even128`.
4. Compose with `left_fold_mul_time_reversed_normalize_once([R1, R2, R3])`, which fixes the loop product as `normalize(R3 * R2 * R1)`.
5. Compare against identity in the same normalized `Even128` space.

Gate5 v0 shall use sign-invariant residuals only.

Implementation architecture rule:

- Gate5 v0 SHALL reuse the existing Rust rotor/algebra path via a minimal Rust helper or diagnose-side compute-only entrypoint.
- Pure Python rotor/algebra reimplementation is not allowed in v0.

### 3.5 Exact Residual Formulas

For every token step with three materialized edges:

- `R_loop = left_fold_mul_time_reversed_normalize_once([R1, R2, R3])`
- `I = Even128::identity()`
- `a = min(1.0, abs(inner(R_loop, I)))`

Primary metric:

- `rotor_loop_chordal_v1 = sqrt(max(0.0, 2.0 * (1.0 - a)))`

Auxiliary metric:

- `rotor_loop_nonscalar_norm_v1 = sqrt(sum_{k=1..127} R_loop.coeffs[k]^2)`

Conditional telemetry-only metric:

- `rotor_loop_identity_gap_v1 = 1.0 - a`

Retention rule:

- `rotor_loop_identity_gap_v1` is not retained as an independent primary metric in v0.
- `rotor_loop_identity_gap_v1` SHALL NOT be emitted as a headline metric in v0.
- In the normalized `Even128` path, it is monotone-equivalent to `rotor_loop_chordal_v1`.
- `rotor_loop_nonscalar_norm_v1` is telemetry-only in v0.
- `rotor_loop_nonscalar_norm_v1` SHALL NOT be used as a headline metric unless it is shown to carry non-redundant ranking information under the implemented path.
- No replacement high-grade metric is frozen in v0. A high-grade residual family is a post-v0 candidate only if it is implemented and shown to add non-redundant information.

### 3.6 Closed Outcome Enums

Gate5 v0 shall use a closed `edge_outcome` enum:

- `materialized`
- `collinear_identity`
- `antipodal_angle_only`
- `vec8_nonfinite_component`
- `vec8_zero_or_nonfinite_norm`
- `rotor_nonfinite_theta`
- `rotor_renorm_failure`

Gate5 v0 shall use a closed `loop_outcome` enum:

- `none`
- `partial_loop_missing`
- `invalid_loop_product`

Policy:

- Same-direction collinear identity-like edges are valid and shall not be marked missing.
- If any of the three required edges is unusable, the loop residual is missing and `loop_outcome=partial_loop_missing`.
- Gate5 v0 shall not fabricate angle-only loop fallbacks.
- Algebra-space failures after edge materialization map to `invalid_loop_product`.

### 3.7 Alignment Policy For Existing Comparisons

- Direct comparison against existing `F` is allowed because both are token-step aligned.
- Existing `E` may be reported only in an explicitly separate transition-aligned section.
- Gate5 shall not fabricate a token-aligned proxy for `E`.

### 3.8 Evaluation Scope

Gate5 v0 evaluation scope is limited to:

- CFA
- Seam Challenge Set v0

### 3.9 Seam Challenge Set v0

The repo does not currently provide a seam generator, so Gate5 v0 defines a new minimal deterministic challenge generator.

Allowed perturbation families:

- punctuation injection
- casing perturbation
- spacing perturbation
- harmless fragmentation triggers

Required classes:

- `clean_consistent`
- `seam_perturbed_consistent`

Required recording:

- deterministic seed
- perturbation family
- perturbation spans
- JSONL output plus companion metadata JSON

Required pair structure:

- the generator SHALL emit stable paired examples
- each pair SHALL have exactly one `clean_consistent` row and one `seam_perturbed_consistent` row
- each pair SHALL carry a stable `pair_id`
- the perturbed row SHALL carry `source_sample_id` pointing to its clean source sample
- paired evaluation SHALL operate only on complete clean/perturbed pairs

Required seam-side evaluation contract:

- Seam is evaluated as a paired quietness problem, not as contradiction-positive detection
- required paired quietness outputs SHALL include at least:
  - paired delta of sample `max`
  - paired delta of sample `p90`
  - top-k spike inflation relative to the clean pair mate
  - at least one robust scale-normalized quietness summary using `MAD` or `IQR`
- `Hit@10`, `first-hit distance`, and token-level AUPRC are not primary Seam-side decision metrics unless an explicit Seam-local labeling scheme is separately introduced

Explicit exclusion for v0:

- no synonym substitution

### 3.10 Output Contract

Gate5 v0 SHALL define deterministic token-level, sample-level, and run-level artifacts before implementation proceeds.

Artifact set:

- `gate5_token_telemetry.csv`
- `gate5_sample_summary.csv`
- `manifest.json`
- attestation report text artifact
- aggregate report text or markdown artifact

Encoding and ordering:

- UTF-8 with LF line endings
- float format id: `sci_17e_v1`
- missing score sentinel id: `empty_string_v1`
- token telemetry rows sorted by `(sample_id ASC, step ASC)`
- sample summary rows sorted by `(sample_id ASC)`
- manifest contains exactly one JSON object

Closed enum string values:

- `edge_outcome`: `materialized|collinear_identity|antipodal_angle_only|vec8_nonfinite_component|vec8_zero_or_nonfinite_norm|rotor_nonfinite_theta|rotor_renorm_failure`
- `loop_outcome`: `none|partial_loop_missing|invalid_loop_product`
- `transition_missing_reason`: `none|final_step_no_successor`

Required columns for `gate5_token_telemetry.csv` in order:

- `run_id`
- `sample_id`
- `variant`
- `world_type`
- `step`
- `absolute_pos`
- `token_id`
- `token_text`
- `answer_char_start`
- `answer_char_end`
- `label_token`
- `label_transition`
- `defect_span_id`
- `label_coverage_ratio`
- `exact_token_match_ratio`
- `transition_missing_reason`
- `edge_outcome_r1_v_to_splus`
- `edge_outcome_r2_splus_to_sminus`
- `edge_outcome_r3_sminus_to_v`
- `loop_outcome`
- `score_A_logprob`
- `score_B_entropy`
- `score_E_v_sminus_vnext`
- `score_F_loop`
- `rotor_loop_chordal_v1`
- `rotor_loop_nonscalar_norm_v1`

Token telemetry missing-value policy:

- `score_E_v_sminus_vnext` serializes as empty string when `transition_missing_reason=final_step_no_successor`
- `rotor_loop_chordal_v1` serializes as empty string when `loop_outcome!=none`
- `rotor_loop_nonscalar_norm_v1` serializes as empty string when `loop_outcome!=none`
- `rotor_loop_identity_gap_v1` is not part of the required token telemetry schema in v0

Required columns for `gate5_sample_summary.csv` in order:

- `run_id`
- `sample_id`
- `variant`
- `world_type`
- `n_token_steps`
- `n_transition_steps`
- `n_loop_steps_valid`
- `n_loop_steps_missing`
- `positive_token_count`
- `positive_transition_count`
- `label_coverage_ratio`
- `exact_token_match_ratio`
- `triplets_sha256`
- `labels_sha256`
- `auprc_A`
- `auprc_B`
- `auprc_E`
- `auprc_F`
- `auprc_rotor_loop_chordal_v1`
- `best_token_baseline_name`
- `delta_auprc_rotor_loop_chordal_v1_vs_F`
- `hit_at_10_F`
- `hit_at_10_rotor_loop_chordal_v1`

Sample summary missing-value policy:

- `gate5_sample_summary.csv` uses one schema for both CFA and Seam Challenge runs; it does not fork by benchmark
- the universal summary fields are:
  - `run_id`
  - `sample_id`
  - `variant`
  - `world_type`
  - `n_token_steps`
  - `n_transition_steps`
  - `n_loop_steps_valid`
  - `n_loop_steps_missing`
  - `positive_token_count`
  - `positive_transition_count`
  - `label_coverage_ratio`
  - `exact_token_match_ratio`
  - `triplets_sha256`
  - `labels_sha256`
- the label-dependent comparison fields are:
  - `auprc_A`
  - `auprc_B`
  - `auprc_E`
  - `auprc_F`
  - `auprc_rotor_loop_chordal_v1`
  - `best_token_baseline_name`
  - `delta_auprc_rotor_loop_chordal_v1_vs_F`
  - `hit_at_10_F`
  - `hit_at_10_rotor_loop_chordal_v1`
- for any run or sample with no contradiction-positive labels defined for that evaluation surface, all label-dependent comparison fields serialize as empty string
- Seam Challenge v0 rows therefore remain present in `gate5_sample_summary.csv`, with universal fields populated and label-dependent comparison fields serialized as empty string

Required keys for `manifest.json`:

- `spec_version`
- `method_id`
- `dataset_revision_id`
- `dataset_hash_blake3`
- `spec_hash_raw_blake3`
- `spec_hash_blake3`
- `code_git_commit`
- `build_target_triple`
- `rustc_version`
- `evaluation_mode_id`
- `run_id`
- `n_samples_total`
- `n_token_rows_total`
- `n_transition_rows_total`
- `n_loop_rows_valid`
- `n_loop_rows_missing`
- `model_id`
- `model_revision`
- `seed`
- `perm_r`
- `primary_score`
- `proj_id`
- `splus_def_id`
- `sminus_def_id`
- `token_telemetry_schema_id`
- `sample_summary_schema_id`
- `float_format_id`
- `transition_label_mode_id`
- `edge_outcome_enum_id`
- `loop_outcome_enum_id`
- `score_missing_sentinel_id`
- `input_json_sha256`
- `token_telemetry_sha256`
- `sample_summary_sha256`

## 4. Research Interpretation Boundary

Gate5 outputs may support only the following claim forms:

- transport-based residuals may be more seam-stable under controlled settings
- transport-based residuals did not improve seam stability
- transport-based residuals trade seam quietness against defect localization

Interpretation guard:

- "no clear gain" is an acceptable primary outcome
- small apparent gains should be treated skeptically unless they survive seam-side quietness checks
- Gate5 should not be expanded in scope just to force a larger effect

Gate5 outputs may not support the following claims:

- proof of KAGAMI
- world-first theorem
- proof of nonassociativity
- universal hallucination detector
- solved hallucination detection

## 5. Scope Guard

The following constraints are frozen for this Gate5 scope:

- Do not modify Gate4 logic.
- Do not rewrite Gate4 score families.
- Do not change Gate4 CSV schema in this task.
- Do not add thresholding or verdict output to Gate5.
- Do not add learned fusion to Gate5.
- Do not add benchmark-specific hacks inside Gate5.
- Prefer codepath consistency over convenience.

