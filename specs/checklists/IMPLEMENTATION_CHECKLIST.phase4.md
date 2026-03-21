# IMPLEMENTATION_CHECKLIST.phase4.md - Phase 4 Gate 1 Implementation Plan (v4.0.0-ssot.9)

**Spec (SSOT):** `SPEC.phase4.md` - `v4.0.0-ssot.9` (FROZEN)  
**Primary Deliverable:** Gate 1 `rotor_diagnostics_v1`  
**Non-Negotiables:** determinism, auditability, no silent failure, strict gate separation

This checklist is implementation-constraining. Any behavior change requires a spec bump and new ids.

---

## 0. Preconditions (Do once)

### 0.1 SSOT freeze and spec hash
- [ ] Emit `spec_version = v4.0.0-ssot.9`.
- [ ] Emit dual spec hashes:
  - [ ] `spec_hash_raw_blake3` with `spec_hash_raw_input_id = spec_text_raw_utf8_v1`
  - [ ] `spec_hash_blake3` with `spec_hash_input_id = spec_text_utf8_lf_v1`
- [ ] Hash input contracts:
  - [ ] raw (`spec_text_raw_utf8_v1`): hash file bytes verbatim — no line-ending conversion, no whitespace trimming, no BOM stripping
  - [ ] lf (`spec_text_utf8_lf_v1`): convert CRLF→LF only, then hash — no other transformations
  - [ ] do not trim trailing whitespace in either path
- [ ] Add CI check to recompute hash and fail on mismatch.

### 0.2 Fixed K=16 sanity sampling
- [ ] Emit `link_sanity_rng_id`.
- [ ] Emit `link_sanity_seed`.
- [ ] Emit `link_sanity_sampling_id`.
- [ ] Fix deterministic sampler in code.
- [ ] Emit `link_sanity_selected_sample_ids` in `manifest.json`.
- [ ] Write selected K sample ids in `link_sanity.md` header.
- [ ] Add snapshot test: for fixed `n=32` synthetic indices, selected `K=16` ids must exactly match expected output.

### 0.3 Evaluation mode and run scaffold
- [ ] Emit `evaluation_mode_id = supervised_v1|unsupervised_v1`.
- [ ] Supervised mode must apply label-missing exclusions.
- [ ] Unsupervised mode must skip AUROC and keep diagnostics.

### 0.4 Dataset/code/build identity
- [ ] Emit `dataset_revision_id`.
- [ ] Emit `dataset_hash_blake3`.
- [ ] Emit `code_git_commit`.
- [ ] Emit `build_target_triple`.
- [ ] Emit `rustc_version`.

**DoD (Section 0):** run artifacts are self-verifiable (`spec_version` + dual spec hash + dataset identity + mode + deterministic sanity slice).

---

## 1. PR1 - Core Math + Vec8 Contract + Golden Vectors

Scope: pure math and input contract, unit tests only.

### 1.1 Vec8 acceptance contract (SPEC §4)
- [ ] For every input vec8:
  - [ ] abort on NaN/Inf -> `excluded_reason=non_finite_vec8`
  - [ ] abort on zero/non-finite norm -> `excluded_reason=zero_or_nonfinite_norm`
  - [ ] normalize otherwise
- [ ] Record `normalized_count`, `max_norm_err`.
- [ ] Implement `vec8_total = doc_unit_count + ans_unit_count` exactly.
- [ ] Implement `normalized_rate = normalized_count / vec8_total`.
- [ ] Implement unit count contract:
  - [ ] `doc_unit_count = len(doc_vec8_sample)`
  - [ ] `ans_unit_count = len(ans_vec8_sample)`

### 1.2 `SimpleRotor29` construction and degeneracy
Implement ids:
- [ ] `rotor_construction_id = simple_rotor29_uv_v1`
- [ ] `theta_source_id = theta_uv_atan2_v1`
- [ ] `bivector_basis_id = lex_i_lt_j_v1`
- [ ] `antipodal_policy_id = antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)`

Checklist:
- [ ] Fixed bivector ordering and wedge coefficients.
- [ ] Compute `dot`, `wedge_norm`, `theta_uv`.
- [ ] Enforce branch order and comparison operators from SSOT.
- [ ] Enforce antipodal angle-only materialization ban.
- [ ] In normal branch only: compute `s`, `sin_half`, `b`, `r_pre`.
- [ ] Enforce trig bans (`acos` forbidden, no `sin/cos` half-angle path).
- [ ] Enforce rotor mapping direction: `u=doc_vec8`, `v=ans_vec8` (doc→ans, SPEC §8.3).
- [ ] Optional: log `max_theta_source_gap` between `theta_uv` and `theta_rotor` (SPEC §3.1.2, non-mandatory audit diagnostic).

### 1.3 Rotor renorm and distance
- [ ] Renormalize materialized rotor vectors and record renorm diagnostics.
- [ ] `excluded_reason=rotor_renorm_failure` on renorm failure.
- [ ] Implement `proj_chordal_v1` exactly (`inner`, `abs`, clamp, non-negative `d2`, `sqrt`).

### 1.4 Golden vectors (Appendix D)
- [ ] D1-D4 tests implemented with numeric tolerances.
- [ ] D3 verifies no rotor materialization in antipodal angle-only.

### 1.5 PR1 tests
- [ ] Vec8 acceptance unit tests:
  - [ ] non-finite
  - [ ] zero norm
  - [ ] already unit
  - [ ] non-unit normalization accounting
- [ ] Core math tests:
  - [ ] branch coverage for antipodal/collinear/normal
  - [ ] distance NaN guard

**DoD (PR1):** math core and vec8 contract are deterministic, drift-proof, and fully unit-tested.

---

## 2. PR2 - Linking Canonicalization + Top-1 Strict + Link Sanity

Scope: `links_topk` shaping and deterministic top-1 behavior.

### 2.1 Canonicalization (SPEC §6.6)
Implement id:
- [ ] `links_topk_canonicalization_id = link_rank_doc_canon_v1`

Checklist per `(sample_id, ans_unit_id)`:
- [ ] drop out-of-range `ans_unit_id` -> `count_invalid_ans_unit_id`
- [ ] drop out-of-range `doc_unit_id` -> `count_invalid_doc_unit_id`
- [ ] drop invalid rank (`0` or `>k`) -> `count_invalid_rank_links`
- [ ] sort by `(rank asc, doc_unit_id asc)`
- [ ] dedup `(ans_unit_id, doc_unit_id)` keep smallest rank -> `count_link_dedup`
- [ ] compute `available_links` after canonicalization
- [ ] if multiple `rank==1`: top-1 picks canonical first (`doc_unit_id` min)
- [ ] emit optional `count_multi_rank1_links`

### 2.2 Top-1 strictness (SPEC §6.5/§7.1)
Implement id:
- [ ] `top1_policy_id = strict_rank1_or_missing_v1`

Checklist:
- [ ] define `steps_total = #answer_units_in_sample` (independent of link availability)
- [ ] if `steps_total == 0` -> `excluded_reason=no_answer_units`
- [ ] top-1 uses only canonicalized `rank==1`
- [ ] if `available_links==0` -> `missing_link_step=true`, increment `count_missing_link_steps`
- [ ] if `available_links>0` and no `rank==1` -> `missing_top1_step=true`, increment `count_missing_top1_steps`
- [ ] never substitute rank>1 in top-1 track
- [ ] compute `missing_link_step_rate`, `missing_top1_step_rate`
- [ ] fixed threshold: `max_missing_link_step_rate = 0.20`
- [ ] if `missing_link_step_rate > max_missing_link_step_rate` -> rotor-vector metrics are `metric_missing` with `metric_missing_reason=too_many_missing_link_steps` (track-scoped)

### 2.3 Link sanity + automated collapse gates (SPEC §6.4)
Implement id:
- [ ] `link_sanity_id = sanity16_single_judgment_v1`

Checklist:
- [ ] representative unit = minimum `ans_unit_id` with canonicalized top-1
- [ ] else `NO_LINK` and count as unrelated
- [ ] enforce FAIL rule: `unrelated > 6` -> `run_invalid_reason=link_sanity_fail`
- [ ] compute entropy collapse on same K=16 slice:
  - [ ] `H_norm > 0.95` -> `run_invalid_reason=random_like_link_collapse`
  - [ ] `max_share > 0.50` -> `run_invalid_reason=dominant_link_collapse`

### 2.4 PR2 tests
- [ ] malformed links fixtures (invalid ids/ranks/duplicates/unsorted)
- [ ] top-1 missing vs missing-top1 separation
- [ ] multi-rank1 deterministic tie-break
- [ ] sanity fail and collapse-gate trigger fixtures

**DoD (PR2):** link processing is deterministic, top-1 is strict, sanity logic is reproducible and gateable.

---

## 3. PR3 - Aggregation + Metrics + Missing/Exclusion Semantics

Scope: Track A/B computations, metric-missing flow, track-scoped invalidation.

### 3.1 Top-K aggregation algorithms (SPEC §7)
- [ ] Top-1 track fully strict.
- [ ] Trimmed-Best exact algorithm:
  - [ ] `k_eff = min(k, available_links)`
  - [ ] `m = ceil(k_eff * p)`
  - [ ] select by `(rank asc, doc_unit_id asc)`
  - [ ] mean rotor vectors
  - [ ] renormalize
  - [ ] on failure -> `excluded_reason=trimmed_best_zero_or_nonfinite_norm` (trimmed track only)
- [ ] Emit Trimmed-Best stability diagnostics:
  - [ ] `trimmed_rbar_norm_pre_p50`
  - [ ] `trimmed_rbar_norm_pre_p10`
  - [ ] `trimmed_rbar_norm_pre_p01`
  - [ ] `trimmed_failure_rate`
- [ ] Emit `trimmed_best_id = trimmed_best_v1(k=8,p=0.5,key=rank,tie=doc_unit_id)`

### 3.2 M1-M5 and compressed sequence rules (SPEC §8)
- [ ] Implement M1-M5 for both tracks where applicable.
- [ ] M1 missing reason order:
  - [ ] `missing_top1_link`
  - [ ] `missing_links_for_theta`
  - [ ] `missing_theta`
- [ ] M2/M3/M4 use compressed ordered sequences exactly.
- [ ] M4 uses `eps_dist` rule and dual outputs.
- [ ] Emit required rates including `rate_missing_top1_steps`.
- [ ] Emit `degenerate_path_rate` per track (numerator: `degenerate_path==true` count; denominator: samples where `wandering_ratio` is not `metric_missing` for that track).
- [ ] Emit optional `trimmed_degenerate_path_rate` for Trimmed-Best track.

### 3.3 Metric missing vs exclusion split (SPEC §9/§10)
- [ ] Keep `excluded_reason` and `metric_missing_reason` separate.
- [ ] Enforce closed enums for both.
- [ ] enforce fail-fast per sample:
  - [ ] if `count_antipodal_drop / steps_total > max_antipodal_drop_rate` -> `ABORT SAMPLE`
  - [ ] emit `excluded_reason=excess_antipodal_drop_rate`
  - [ ] exclude sample from all supervised metrics and unsupervised aggregates; keep diagnostics counters
- [ ] Enforce track-scoped invalidation:
  - [ ] trimmed failures do not invalidate top-1 unless run-level invalid reason is triggered.

### 3.4 PR3 tests
- [ ] metric-missing reason precedence tests
- [ ] track-scope tests (trimmed failure, top-1 survives)
- [ ] missing-links and missing-top1 propagation tests
- [ ] closed-enum validation tests

**DoD (PR3):** metrics and failure semantics are correct, deterministic, and audit-friendly per track.

---

## 4. PR4 - Run Gating + Monitoring + AUROC + Confounds + Audit

Scope: run-level gates and reporting completeness.

### 4.1 Run invalidation order and supervised accounting (SPEC §9)
- [ ] Enforce gate order:
  1. link sanity invalidation
  2. quantile/collapse gates (only if still valid)
  3. exclusion ceiling
- [ ] Implement supervised accounting:
  - [ ] `n_supervised_eligible`
  - [ ] `n_supervised_excluded_primary`
  - [ ] `n_supervised_used_primary`
  - [ ] `primary_exclusion_rate`
  - [ ] `label_missing_rate`
- [ ] `no_supervised_eligible_samples` handling.

### 4.2 Quantiles and collapse gates (SPEC §11)
- [ ] Quantile algorithm:
  - [ ] `total_cmp` sort
  - [ ] nearest-rank index contract: `idx = ceil(p * n) - 1`, clamped to `[0, n-1]`
  - [ ] fixed p set: `{0.01, 0.50, 0.90, 0.99}`
- [ ] Top-1-only quantile gating and naming scope.
- [ ] fixed Top-1 quantile population:
  - [ ] include steps where canonicalized `rank==1` exists and normalized finite `u,v` exist
  - [ ] exclude `missing_top1_step` and `missing_link_step`
  - [ ] include collinear and antipodal-angle-only steps
- [ ] `empty_quantile_population_primary` handling.
- [ ] `quantiles_missing_ok` and `quantile_reference_only` behavior.
- [ ] If run already invalid from link sanity, emit `quantile_reference_only=true` and skip quantile-based invalidation.
- [ ] Collapse gate checks:
  - [ ] `rate_collinear > 0.80`
  - [ ] `rate_antipodal_drop > 0.20`
  - [ ] `wedge_norm_p99 < tau_wedge`
- [ ] Antipodal warning channel:
  - [ ] warning population fixed to samples with `steps_total > 0` and not aborted
  - [ ] per-sample value: `rate_antipodal_angle_only = count_antipodal_angle_only / steps_total`
  - [ ] p50/p90 over per-sample rates
  - [ ] compute `share_samples_antipodal_angle_only_gt_0_50`
  - [ ] if `share_samples_antipodal_angle_only_gt_0_50 > 0.20` then set warning
  - [ ] `run_warning=antipodal_angle_only_high`
- [ ] Representation robustness diagnostics:
  - [ ] `vec8_eff_dim_pr` (participation ratio)
  - [ ] Trimmed-Best robustness diagnostics from PR3 are propagated to summary/manifest

### 4.3 AUROC contract (SPEC §10)
Implement ids:
- [ ] `rank_method_id = average_rank_total_cmp_v1`
- [ ] `auc_algorithm_id = mann_whitney_rank_sum_v1`

Checklist:
- [ ] rank direction fixed (rank 1 = smallest score)
- [ ] tie handling average rank
- [ ] emit `auc_ties = average_rank` in `manifest.json`
- [ ] ensure `auc_ties` is consistent with `rank_method_id = average_rank_total_cmp_v1`
- [ ] Mann-Whitney formula
- [ ] single-class undefined handling
- [ ] dual AUC for wandering ratio
- [ ] Optional non-gating confidence outputs:
  - [ ] `auc_primary_bootstrap_ci95_low/high`
  - [ ] `auc_primary_permutation_p`
  - [ ] `stats_id = bootstrap_ci_v1+permutation_test_v1|none`

### 4.4 Confound quarantine (SPEC §12)
- [ ] Spearman with required tie/missing contract.
- [ ] Length tertile AUROC stratification.
- [ ] Exclusion rate by length tertile.
- [ ] Emit `length_confound_warning` non-gating flag (e.g. `abs(rho_len_max_theta) > 0.70`).

### 4.5 Determinism and audit outputs (SPEC §13)
- [ ] Enforce sample-level parallelism only.
- [ ] Enforce fixed-order reductions and `total_cmp` where required.
- [ ] Enforce `determinism_scope` value.
- [ ] Enforce float serialization contract (`sci_17e_v1`).
- [ ] Enforce artifact ordering contract:
  - [ ] `link_topk.csv` row order fixed `(sample_id, ans_unit_id, rank, doc_unit_id)` ascending
  - [ ] `summary.csv` column order fixed via schema
  - [ ] `manifest.json` key order deterministic
- [ ] Emit run identity/provenance:
  - [ ] `spec_hash_raw_blake3`, `spec_hash_raw_input_id`
  - [ ] `spec_hash_blake3`, `spec_hash_input_id`
  - [ ] `dataset_revision_id`, `dataset_hash_blake3`
  - [ ] `code_git_commit`, `build_target_triple`, `rustc_version`

### 4.6 Manifest required-field completeness (SPEC §13.5)
- [ ] Validate fixed schema ids: `summary_schema_id = summary_csv_schema_v1`, `link_topk_schema_id = link_topk_csv_schema_v1`.
- [ ] Add a machine check that validates presence of all required fields from SSOT §13.5.
- [ ] Add machine check for required non-§13.5 manifest fields from other sections (e.g., `auc_ties = average_rank` from SPEC §10.2).
- [ ] Validator must also assert required flags/accounting beyond §13.5:
  - [ ] `quantiles_missing_ok`, `quantile_reference_only`
  - [ ] warning fields when applicable (`run_warning`, `length_confound_warning`)
  - [ ] supervised accounting fields (`n_supervised_eligible`, `n_supervised_used_primary`, `n_supervised_excluded_primary`, `primary_exclusion_rate`, `label_missing_rate`)
- [ ] Recommended implementation: hard-coded required list id `manifest_required_fields_v4_0_0_ssot_9` + test.
- [ ] Validate all required `*_id` values are emitted and non-empty.
- [ ] Validate threshold/epsilon fields and run accounting fields are emitted.
- [ ] Validate identity/provenance fields and sanity sampling metadata fields are emitted.

### 4.7 PR4 tests
- [ ] quantile fixture tests (including empty population)
- [ ] collapse-gate trigger tests
- [ ] AUROC hand-calculated tie fixtures
- [ ] confound output schema tests
- [ ] manifest-field completeness test
- [ ] deterministic ordering tests for `link_topk.csv` and `summary.csv`
- [ ] optional stats-output tests (bootstrap/permutation path)

**DoD (PR4):** run gating, AUROC, monitoring, confounds, and audit outputs fully match SSOT.

---

## 5. Artifact Contract and Smoke Acceptance

### 5.1 Required artifacts
- [x] `manifest.json`
- [x] `summary.csv`
- [x] `link_topk.csv`
- [x] `link_sanity.md`

### 5.2 Required content checks
- [x] manifest contains all required ids/thresholds/reason fields/accounting fields/identity fields
- [x] manifest contains `link_sanity_selected_sample_ids`
- [ ] manifest validator checks `auc_ties = average_rank`
- [x] manifest validator checks `quantiles_missing_ok` and `quantile_reference_only`
- [ ] summary includes all required metric outputs, exclusions, metric-missing counts, diagnostics
- [x] `link_topk.csv` row order and `summary.csv` column order are schema-compliant (`link_topk_csv_schema_v1`, `summary_csv_schema_v1`)
- [x] no NaN/Inf in machine artifacts

### 5.3 End-to-end smoke run
- [x] run a tiny fixture dataset (8-32 samples)
- [x] verify deterministic outputs across repeated runs
- [ ] verify forced invalid/warning fixtures produce expected reasons

**Exit criteria:** `rotor_diagnostics_v1` runs end-to-end with deterministic, audit-complete artifacts.

### 5.4 Closeout notes (post PR4c)
- [x] Added smoke fixtures:
  - [x] `docs/examples/gate1_smoke_tiny.json`
  - [x] `docs/examples/gate1_smoke_sanity16.json`
- [x] Added smoke guide: `docs/gate1_smoke.md`
- [ ] Follow-up issue candidate: add/validate `auc_ties = average_rank` field in `manifest.json` (SSOT §10.2 / checklist §4.3, §4.6, §5.2).

---

## 6. Coverage Matrix (SSOT -> Checklist)

- [ ] §3 Data model -> PR1
- [ ] §4 Vec8 contract -> PR1
- [ ] §5 Distance -> PR1
- [ ] §6 Linking + sanity -> PR2
- [ ] §7 Top-K aggregation -> PR3
- [ ] §8 Metrics -> PR3
- [ ] §9 Exclusions/run validity -> PR3/PR4
- [ ] §10 AUROC -> PR4
- [ ] §11 Monitoring/collapse gates -> PR4
- [ ] §12 Confound quarantine -> PR4
- [ ] §13 Determinism/audit/manifest -> PR4
- [ ] Appendix D golden vectors -> PR1

---

## 7. Change Control

- [ ] Any behavior change requires spec bump and new ids.
- [ ] No silent behavior drift.
- [ ] Every PR adds tests that fail under drift.

---
