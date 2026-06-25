# Gate12C-1 First Empirical Execution Plan

Status: pre-execution empirical plan
Role: frozen plan for the first real Gate12C-1 equal-rank alpha execution over the canonical twelve Gate12A source runs, not an execution memo, not an analysis implementation PR, not a Gate12B overlay, and not a physical nonassociativity claim
Date: 2026-06-25

This plan proceeds from:

- `231_GATE12C_ASSOCIATOR_FEASIBILITY_AND_EQUAL_RANK_ALPHA_CONTRACT.md`
- `232_GATE12C_FIRST_REAL_ARTIFACT_FEASIBILITY_CENSUS_PLAN.md`
- `233_GATE12C_FIRST_REAL_ARTIFACT_FEASIBILITY_CENSUS_MEMO.md`
- `234_GATE12C_EQUAL_RANK_ALPHA_IMPLEMENTATION_CONTRACT.md`
- `tools/run_gate12c_compressed_overlap_associator.py`
- `docs/gate12c_compressed_overlap_associator_runbook.md`

## 0. Execution Boundary

This memo is a docs-only execution plan.

It does not:

- run Gate12C-1
- inspect real associator values
- add analysis code
- add a summary tool
- consume Gate12B overlays
- change Gate12A or Gate12B semantics
- define a physical nonassociativity claim
- tune thresholds after seeing data

The first real execution is not authorized until the separate synthetic-only summary-tool PR in Section 15 is merged.

## 1. Canonical Twelve-Case Grid

The execution grid is exactly the canonical Gate12A surface frozen by `232` and verified by `233`:

```text
4 dense-transformer model lines x 3 rendering families = 12 Gate12A runs
```

No case may be substituted. Missing or malformed source artifacts stop the affected execution path as a contract failure; they must not be replaced by exploratory runs.

| Case | Model | Family | Source Gate12A directory | Source run id | Preflight eligible cycles |
| --- | --- | --- | --- | --- | ---: |
| case_01 | `qwen_qwen2_5_0_5b` | `transcript_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k` | 320 |
| case_02 | `qwen_qwen2_5_0_5b` | `briefing_200r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k` | 500 |
| case_03 | `qwen_qwen2_5_0_5b` | `archive_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k` | 320 |
| case_04 | `qwen_qwen2_5_3b_instruct` | `transcript_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k` | 320 |
| case_05 | `qwen_qwen2_5_3b_instruct` | `briefing_200r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k` | 500 |
| case_06 | `qwen_qwen2_5_3b_instruct` | `archive_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k` | 320 |
| case_07 | `meta_llama_llama_3_2_3b_instruct` | `transcript_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k` | 320 |
| case_08 | `meta_llama_llama_3_2_3b_instruct` | `briefing_200r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k` | 500 |
| case_09 | `meta_llama_llama_3_2_3b_instruct` | `archive_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k` | 320 |
| case_10 | `qwen_qwen3_4b` | `transcript_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k` | 320 |
| case_11 | `qwen_qwen3_4b` | `briefing_200r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k` | 500 |
| case_12 | `qwen_qwen3_4b` | `archive_128r` | `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k` | 320 |

The preflight memo records all twelve as contract pass and `empirical_surface_status = pass_declared_minimum`. The Gate12C-0 census found 4,560 eligible rank-3 cycles and 27,360 stable nontrivial root/q cuts. That census did not compute Gate12C-1 associator values.

## 2. Runner Provenance

The first empirical execution must use the runner as merged to `main` after PR `#116`.

```text
post_merge_main_commit = 8d5613bffe5b6c91d0956c812404072eb76e98c6
runner_script = tools/run_gate12c_compressed_overlap_associator.py
runner_script_sha256 = b363fd874a0538dc548853e97e8ec17c0eb84be5658f6e2f01f60d2a12789c3e
schema_version = gate12c_compressed_overlap_associator_v1
method_id = gate12c_compressed_overlap_associator_v1
run_mode = gate12a_residual_bearing_explicit_triangle_equal_rank_alpha_v1
```

If any of these provenance values differ, do not execute under this plan. Amend the plan first.

## 3. Frozen Runner Settings

The frozen orientation-null settings are:

```text
orientation_null_seed = "gate12c1_first_empirical_orientation_null_v1"
orientation_null_requested_draw_count = 255
orientation_null_max_attempt_count = 1024
```

All numerical tolerances remain the runner defaults:

| Setting | Value |
| --- | ---: |
| `tau_overlap_sv_min` | `1e-8` |
| `tau_overlap_singular_value_abs_error` | `1e-8` |
| `tau_transport_reconstruction_fro` | `1e-8` |
| `tau_ordinary_associator_fro` | `1e-10` |
| `tau_no_compression_associator_fro` | `1e-10` |
| `tau_split_rel` | `1e-3` |
| `tau_gauge_operator_covariance_fro` | `1e-8` |
| `tau_gauge_scalar_delta_abs` | `1e-10` |
| `epsilon` | `1e-12` |

No setting may be changed after the first real Gate12C-1 run starts.

## 4. Primary Row Effect

The primary row-level effect size is:

```text
log_null_ratio =
  log((compressed_overlap_associator_fro + epsilon) /
      (orientation_null_median + epsilon))
```

The `epsilon` in this formula is the frozen runner default `1e-12`.

This row effect is an input to the predeclared hierarchy only. It is not a row-level discovery claim.

## 5. Primary Eligibility

A row is eligible for primary aggregation only when all of the following hold:

```text
aggregation_eligible == true
orientation_null_status == complete
orientation_null_scale_degenerate == false
orientation_null_median > epsilon
```

Additionally, all required primary values must be present and finite:

```text
compressed_overlap_associator_fro
orientation_null_median
log_null_ratio
gate12a_holonomy_residual_fro
edge_compatibility_gap_max
```

Rows with JSON `null`, NaN, infinity, incomplete nulls, scale-degenerate nulls, or nonpositive null median under the `epsilon` guard are excluded from primary aggregation. There is no imputation.

## 6. Hierarchy

The primary aggregation hierarchy is:

1. `cycle/q`: median `log_null_ratio` across all three valid roots.
2. `source-block/q`: median across valid cycles in the source block.
3. `run/q`: median across source blocks in the run.

All three roots are required for a `cycle/q` aggregate. A cycle/q with fewer than three valid roots is excluded from primary aggregation and counted in coverage diagnostics.

The two compression ranks are separate co-primary endpoints:

```text
q = 1
q = 2
```

They must not be pooled into one primary statistic.

## 7. Coverage Requirements

For each run and each q:

- at least 90% of preflight-eligible cycles must have all three valid roots
- at least 90% of source blocks must be represented
- no imputation is allowed

Coverage failures do not become positive or negative evidence. They must be reported as coverage failures in the result memo.

## 8. Per-Run Directional Test

For each run and q, apply a one-sided exact sign test to source-block scores:

```text
H1: source-block score > 0
```

Zero ties are excluded before the sign test. If no nonzero block scores remain, the run/q test is non-informative and cannot support directional excess.

There are 24 predeclared run-by-q tests:

```text
12 runs x 2 q values = 24 tests
```

Apply Holm correction across only these 24 tests.

A run supports directional excess only when:

- the `q = 1` run median is positive
- the `q = 2` run median is positive
- the Holm-adjusted p-value for `q = 1` is below `0.05`
- the Holm-adjusted p-value for `q = 2` is below `0.05`

Single-q support is not sufficient for a run-level directional-excess call.

## 9. Outcome Categories

The first empirical result memo must classify the run grid into exactly one of:

- `strong_broad`: 12/12 runs support directional excess
- `broad_replicated`: at least 10/12 runs support directional excess, every family has at least 3/4 supporting models, and every model has at least 2/3 supporting families
- `partial_or_structured`: 1-9 runs support directional excess, or support is localized by model/family pattern
- `no_directional_support`: 0/12 runs support directional excess
- `mixed_q`: systematic disagreement between `q = 1` and `q = 2`

If `mixed_q` and another category both appear plausible, `mixed_q` takes precedence until the q disagreement is described.

## 10. Secondary Telemetry

The following are descriptive telemetry only:

- hierarchical median robust z
- empirical p-value distributions
- compressed associator relative magnitude
- root spread
- `q = 1` versus `q = 2` difference
- scale-degenerate null rates
- incomplete-null rates

These checks must not be converted into new discovery thresholds inside the result memo.

## 11. Existing-Defect Separation

The matched orientation null preserves each edge singular spectrum. Therefore, the primary null comparison is not a test against arbitrary edge-strength destruction.

Secondary separation checks must include:

1. A low-holonomy secondary surface:
   - sort cycles within each run by `(gate12a_holonomy_residual_fro, cycle_id)`
   - select the first `floor(N/4)` cycles per run
   - retain q-specific aggregation
2. q-specific Spearman correlations with:
   - `gate12a_holonomy_residual_fro`
   - `edge_compatibility_gap_max`

These separation checks are descriptive. They do not replace the primary hierarchy or create row-level claims.

## 12. Multiple-Testing Boundary

No row-level discovery claims are authorized.

Only the 24 predeclared run/q sign tests receive Holm correction. Secondary telemetry, low-holonomy checks, and correlation checks remain descriptive.

Do not add post-hoc families of tests after inspecting real outputs.

## 13. Blind Execution Discipline

The execution operator must not inspect per-run associator outputs before all 12 runs complete.

The following are frozen once the first real run starts:

- orientation-null seed
- orientation-null draw count
- orientation-null max attempts
- numerical tolerances
- aggregation hierarchy
- coverage thresholds
- directional test
- outcome categories

Contract failures stop execution. Examples include missing artifacts, reconstruction mismatch, ordinary null failure, no-compression null failure, gauge covariance failure, source mutation, or invalid deterministic artifact output.

Valid null-like data outcomes do not stop execution. Examples include no directional excess, zero aggregation-eligible rows, insufficient valid null draws recorded correctly, or scale-degenerate null summaries.

## 14. Gate12B Overlay Boundary

Gate12B overlay remains forbidden until the Gate12C-1 empirical result memo is committed.

No Gate12B candidate label, flat/high band, observer-relative band, source-facing annotation, or Gate12B-derived subset may influence:

- execution selection
- row eligibility
- q selection
- null construction
- primary aggregation
- outcome category

## 15. Required Summary-Tool PR Before Execution

Before real execution, create and merge a separate synthetic-only summary-tool PR:

```text
tools/summarize_gate12c1_first_empirical_grid.py
tools/test_summarize_gate12c1_first_empirical_grid.py
```

That PR must use synthetic fixtures only. It must not run real Gate12C-1 outputs or inspect real associator values.

The summary tool must implement this plan's primary eligibility, hierarchy, coverage checks, sign tests, Holm correction, outcome categories, secondary telemetry, and deterministic result-table emission.

## 16. Planned Result Memo

The first empirical execution result must be recorded in:

```text
workstream/236_GATE12C1_FIRST_EMPIRICAL_RESULT_MEMO.md
```

The result memo must report:

- exact runner provenance
- exact source directories and source manifests
- source immutability checks
- command settings
- completion status for all 12 runs
- coverage by run/q
- 24 run/q sign tests before and after Holm correction
- run-level directional-excess support calls
- grid outcome category
- secondary telemetry
- existing-defect separation checks
- explicit non-claims

## 17. Short Sentence

The first Gate12C-1 empirical execution is frozen as a blind, twelve-run, equal-rank-only test of whether compressed-overlap associator magnitudes exceed their matched spectrum-preserving orientation null under a block-aware hierarchy, with q=1 and q=2 kept separate and co-primary.
