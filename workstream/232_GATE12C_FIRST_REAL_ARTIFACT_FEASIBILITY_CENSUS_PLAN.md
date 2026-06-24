# Gate12C First Real-Artifact Feasibility Census Plan

Status: empirical execution plan
Role: bounded plan for the first real Gate12C-0 census over the canonical Gate12A upstream surface used by the Gate12B paper line, not a Gate12C-1 implementation contract, not a Gate12B overlay, not a new model run, and not a physical nonassociativity claim
Date: 2026-06-24

This memo proceeds from:

- `231_GATE12C_ASSOCIATOR_FEASIBILITY_AND_EQUAL_RANK_ALPHA_CONTRACT.md`
- `tools/inspect_gate12c_associator_feasibility.py`
- `229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md`
- `docs/gate12b_observer_relative_coarse_grained_closure_runbook.md`

## 0. Executive Decision

The next Gate12C step is not Gate12C-1 implementation.

The next step is a real-artifact Gate12C-0 census over the canonical twelve Gate12A discrete-connection inputs already used as the upstream surface for the bounded Gate12B paper line.

The exact grid is:

```text
4 dense-transformer model lines x 3 rendering families = 12 Gate12A runs
```

The first empirical question is only:

> Do the existing Gate12A-defined residual-bearing explicit triangles contain any equal-rank, common-rank `r >= 2` cycles with stable nontrivial SVD cuts under the frozen Gate12C-0 tolerances?

This census must be completed before authorizing Gate12C-1.

## 1. Merge and Implementation Boundary

The Gate12C-0 implementation is merged on `main` through PR `#112`.

Canonical implementation files:

```text
tools/inspect_gate12c_associator_feasibility.py
tools/test_inspect_gate12c_associator_feasibility.py
```

The census must use the merged `main` implementation.

It must not:

- modify Gate12A or Gate12B semantics
- implement compressed associator values
- implement Gate12C-1
- add rectangular rank-mismatch support
- consume Gate12B candidate rows
- run model inference
- regenerate Gate8 or Gate12A artifacts
- tune thresholds after looking at results

## 2. Canonical Real-Artifact Grid

The census surface is the upstream Gate12A discrete-connection grid recorded by the Gate12B evidence manifest.

### 2.1 Models

```text
qwen_qwen2_5_0_5b
qwen_qwen2_5_3b_instruct
meta_llama_llama_3_2_3b_instruct
qwen_qwen3_4b
```

### 2.2 Families

```text
transcript_128r
briefing_200r
archive_128r
```

### 2.3 Source directory template

```text
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_{model}_{family}_gate9k/
```

The intended twelve source directories are:

```text
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k/

runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k/

runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k/

runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k/
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k/
```

These directories are local generated artifacts and are not expected to be tracked by git.

## 3. Inventory Before Execution

Before running the preflight, create an explicit inventory table with one row per intended source directory.

Required columns:

```text
case_id
model
family
source_gate12a_dir
source_dir_exists
required_files_complete
source_run_id
source_schema_version
source_code_git_commit
source_checksums_present
source_checksums_status
```

Missing directories must be recorded as missing.

Do not regenerate a missing Gate12A source and do not substitute an exploratory run merely to complete the grid.

If a canonical source directory name differs only because a manifest records an equivalent established path, record the exact discovered path and the evidence connecting it to the canonical case. Do not silently guess.

## 4. Frozen Execution Settings

Use the merged Gate12C-0 defaults without data-dependent tuning:

```text
tau_overlap_sv_min = 1e-8
tau_overlap_singular_value_abs_error = 1e-8
tau_transport_reconstruction_fro = 1e-8
tau_ordinary_associator_fro = 1e-10
tau_split_rel = 1e-3
epsilon = 1e-12
```

Use:

```text
--min-eligible-cycles 1
```

The value `1` is an existence-only threshold.

It means only:

> at least one equal-rank `r >= 2` cycle has at least one root/q configuration with both required SVD cuts stable

It is not:

- a publication threshold
- a statistical sufficiency threshold
- a theory-selection threshold
- a Gate12C-1 success criterion

The actual counts remain the primary census output.

## 5. Report-Wording Hardening Before the Real Run

The current generated read text contains a generic sentence stating that empirical feasibility remains unknown until the preflight is run on real artifacts.

Before materializing real-run reports, replace that sentence with wording that remains correct for synthetic and real inputs:

> Gate12C-1 is not implemented. This preflight reports only artifact-surface eligibility under the caller-declared minimum and does not measure compressed associator behavior.

Add or update a focused regression assertion for the new wording.

This is report-boundary hardening only. It must not alter any calculation, status, tolerance, eligibility rule, or artifact schema.

## 6. Execution Order

### 6.1 Representative smoke

First run the existing Qwen2.5-0.5B transcript source used by the Gate12B runbook:

```text
runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k/
```

Suggested output:

```text
runs/gate12c_associator_feasibility_qwen2_5_0_5b_transcript_min1/
```

If the smoke produces a contract failure, stop the grid run and diagnose the implementation/artifact boundary before proceeding.

An empirical-surface failure is not a reason to stop the rest of the grid. It is a valid census outcome.

### 6.2 Full canonical grid

After the smoke passes the contract boundary, run every available canonical source directory in the twelve-case grid.

Suggested output template:

```text
runs/gate12c_associator_feasibility_{model_short}_{family_short}_min1/
```

Use a separate output directory per source case.

Do not overwrite any prior Gate12C output directory. If an output already exists, verify it by manifest, builder hash, settings, and checksums or choose a clearly versioned new directory.

## 7. Source Immutability

Before and after each run, verify the source Gate12A artifact directory remains unchanged.

At minimum, hash the seven required Gate12C input files:

```text
manifest.json
node_local_object_registry.jsonl
node_local_object_arrays.npz
transport_relation_registry.jsonl
transport_operator_arrays.npz
explicit_triangle_cycle_registry.jsonl
triangle_holonomy_registry.jsonl
```

Record pre/post equality.

Do not write temporary files under the source directory.

## 8. Per-Run Required Readout

For each executed case, record:

```text
case_id
model
family
source_gate12a_run_id
source_gate12a_dir
gate12c_output_dir
contract_feasibility_status
empirical_surface_status
min_eligible_cycles
reconstructed_edge_count
failed_edge_reconstruction_count
overlap_singular_value_max_abs_error
transport_reconstruction_max_fro_error
ordinary_associator_max_fro
ordinary_associator_failed_count
total_gate12a_residual_bearing_explicit_triangle_count
defined_equal_rank_triangle_count
common_rank_1_triangle_count
common_rank_2_triangle_count
common_rank_3_triangle_count
common_rank_ge_4_triangle_count
eligible_equal_rank_common_rank_ge_2_cycle_count
probe_configuration_count
stable_both_inner_cut_count
near_degenerate_left_cut_count
near_degenerate_right_cut_count
near_degenerate_both_cut_count
eligible_cycle_count_with_at_least_one_stable_q
source_immutability_status
output_checksums_status
```

No Gate12B candidate labels belong in this table.

## 9. Cross-Run Summary

Create a deterministic local summary artifact under:

```text
runs/gate12c_associator_feasibility_canonical_grid_min1/
```

Suggested files:

```text
manifest.json
gate12c_canonical_grid_inventory.csv
gate12c_canonical_grid_summary.csv
gate12c_canonical_grid_summary.json
gate12c_canonical_grid_read.md
checksums.json
```

This summary may be built with a one-off bounded script in the execution branch if needed, but no generic summarizer should be promoted without a separate contract and tests.

The summary must preserve one row per intended case, including missing or incomplete cases.

Generated `runs/` outputs remain untracked unless an intentional evidence-package decision is made later.

## 10. Interpretation Boundary

The first real census may establish only one of the following bounded outcomes:

### Outcome A: eligible surface exists broadly

Multiple model/family runs pass the contract boundary and expose stable nontrivial cuts.

This authorizes drafting Gate12C-1 implementation scope. It does not establish a nonzero or meaningful associator.

### Outcome B: eligible surface exists narrowly

Only some models or families expose stable nontrivial cuts.

This becomes an empirical boundary result. Gate12C-1, if authorized, must remain restricted to that predeclared eligible surface.

### Outcome C: no usable equal-rank surface

The canonical grid has no stable nontrivial equal-rank cuts under the frozen settings.

This kills Gate12C-1 equal-rank alpha under the current local-object construction. It does not authorize threshold tuning or immediate rectangular generalization.

### Outcome D: contract reconstruction fails

Stored Gate12A artifacts do not reconstruct under the merged Gate12C-0 checks.

This is an artifact-contract or implementation issue and must be resolved before any associator work.

## 11. Prohibited Post-Hoc Moves

After seeing the first census, do not:

- lower `tau_split_rel` to create eligibility
- choose only archive cases because Gate12B archive results were stronger
- drop model/family failures from the summary
- substitute Gate12B high/flat candidates for the full eligible surface
- interpret rank abundance as associator evidence
- implement rectangular support solely because equal-rank eligibility is weak
- call stable-cut availability a Type-III defect

## 12. Tracked Result Memo

After execution, add a new tracked memo:

```text
workstream/233_GATE12C_FIRST_REAL_ARTIFACT_FEASIBILITY_CENSUS_MEMO.md
```

The memo must include:

- exact local source paths used
- exact merged Gate12C builder commit and hash
- exact tolerances and declared minimum
- intended-case inventory including missing cases
- per-case census table
- contract failures, if any
- empirical surface outcomes
- source immutability result
- explicit non-claims
- next-step decision

Do not commit generated `runs/` directories with the memo.

## 13. Authorization Boundary for Gate12C-1

Gate12C-1 implementation is not automatically authorized by one passing case.

After the twelve-case census, review:

```text
model breadth
family breadth
rank distribution
stable-cut abundance
near-degeneracy rate
contract reconstruction stability
```

Only then decide whether to open a Gate12C-1 implementation contract or record a kill/boundary result.

## 14. Short Sentence

The next step is not to calculate an associator. It is to determine, without threshold tuning or Gate12B selection, whether the canonical real Gate12A surface contains a stable equal-rank domain on which an associator probe could be defined at all.
