# Gate12B First Observer-Relative Coarse-Grained Closure Smoke Memo

Status: first runner smoke memo draft
Role: first local Gate12B read-only secondary audit smoke over one existing Gate12A transcript surface, not a family comparison, not a checkpoint revision, not a release claim, and not a Gate12A schema change
Date: 2026-05-05

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`

## 0. Scope

This memo records one first Gate12B runner smoke.

It does:

- run the Gate12B secondary audit over one existing Gate12A transcript surface
- verify that the runner reads existing Gate12A artifacts without mutating them
- verify that observer x scale matrix output is produced
- verify that the bounded array-level admissible reparameterization check runs
  when transport arrays are present
- record the local smoke counts

It does not:

- claim family-wide Gate12B behavior
- compare transcript, briefing, and archive families
- revise the Gate12A memo line
- promote invariant candidates into a tracked law
- turn residual bands into correctness labels

## 1. Evidence Base

Source Gate12A run:

- `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k/`

Gate12B output run:

- `runs/gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_smoke/`

Command:

```powershell
python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k `
  --out-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_smoke `
  --top-k 3
```

This is a CPU-only secondary audit.
It does not run model inference.

## 2. Smoke Result

The emitted manifest records:

- `run_id = gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_smoke`
- `schema_version = gate12b_observer_relative_coarse_grained_closure_v1`
- `secondary_audit_mode = read_only_gate12a_artifacts_v1`
- `primitive_mode = observer_x_scale_x_admissible_gauge_transform_v1`
- `gauge_language_boundary = basis_preserving_local_reparameterization_v1`
- `source_gate12a_run_id = gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k`

The current local counts are:

- `input_triangle_count = 320`
- `defined_triangle_count = 320`
- `flat_count = 80`
- `tense_count = 160`
- `high_tension_count = 80`
- `observer_scale_matrix_row_count = 982`
- `invariant_signature_candidate_count = 108`
- `gauge_total_check_count = 3840`
- `gauge_stable_check_count = 3840`
- `gauge_unstable_check_count = 0`
- `gauge_variant_signature_candidate_count = 108`
- `gauge_arrays_available = true`
- `gauge_transform = basis_coordinate_reversal_v1`

The gauge summary records:

- `nontrivial_transform_evaluated = true`
- `max_residual_delta_abs = 4.440892098500626e-16`
- `tau_gauge_residual_delta = 1e-08`

## 3. Candidate Reading

The `108` invariant candidates are candidate rows only.
They are not a law and not correctness labels.

The runner now requires candidate support across:

- independent observer scopes
- multiple scale modes
- at least one non-triangle coarse scale

Observer views with identical cycle membership are grouped and counted as one
observer scope for candidate support.

The `108` gauge-stable candidate rows are also conditional on:

- transport arrays being present
- `basis_coordinate_reversal_v1` being evaluated
- all corresponding candidate gauge-stability checks remaining stable

So this smoke verifies that the first candidate extraction and bounded
gauge-stability path execute on one representative transcript surface.
It does not establish portability beyond that surface.

## 4. Verification

Focused tests were run:

```text
python -m unittest tools.test_run_gate12a_discrete_connection_audit tools.test_run_gate12a_triangle_phenotype_tag_prep tools.test_run_gate12b_observer_relative_coarse_grained_closure
```

Result:

- `Ran 12 tests`
- `OK`

## 5. Next Move

The next honest move is a three-family Gate12B comparison over the same
`Qwen/Qwen2.5-0.5B` family set:

- `transcript_v1`
- `briefing_v1`
- `archive_v1`

That comparison should remain separate from this first smoke and should preserve
the same read-only secondary audit boundary.

## 6. Short Sentence

The first Gate12B local smoke ran over one existing Qwen2.5-0.5B transcript
Gate12A surface, emitted observer x scale and gauge-stability artifacts, and
produced candidate rows under the tightened observer-scope and multi-scale
support rules without changing the source Gate12A artifacts.
