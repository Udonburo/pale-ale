# Gate12B Observer-Relative Coarse-Grained Closure Runbook

This runbook describes the first Gate12B secondary audit runner.
It is a local read-only audit over an existing Gate12A discrete-connection run.

## Scope

The Gate12B runner reads existing Gate12A artifacts and emits a new secondary
audit surface.

It does not:

- modify Gate12A source artifacts
- change Gate12A residual, transport, or holonomy definitions
- collapse the result into one scalar score
- treat high residual as automatic failure
- add hidden graph relations
- use external physics terminology as code semantics

## Inputs

Required input files under `--gate12a-dir`:

- `manifest.json`
- `explicit_triangle_cycle_registry.jsonl`
- `triangle_holonomy_registry.jsonl`
- `transport_relation_registry.jsonl`

Optional input:

- `transport_operator_arrays.npz`

When the optional transport arrays are present, the runner evaluates
`basis_coordinate_reversal_v1` as a bounded admissible local reparameterization
check.

When the optional transport arrays are absent, the runner records that no
nontrivial array-level reparameterization was evaluated and does not emit
gauge-stable candidates.

## Representative Smoke

The first representative local smoke uses the existing Qwen2.5-0.5B
`transcript_v1` Gate12A run:

```powershell
python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k `
  --out-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_smoke `
  --top-k 3
```

This command is CPU-only and does not run model inference.

## Outputs

The output directory contains:

- `manifest.json`
- `observer_scale_closure_matrix.csv`
- `observer_scale_closure_matrix.json`
- `invariant_signature_candidates.jsonl`
- `gauge_stability_matrix.csv`
- `gauge_stability_summary.json`
- `gauge_variant_signature_candidates.jsonl`
- `gate12b_observer_relative_coarse_grained_closure.md`
- `checksums.json`

Generated outputs should remain under `runs/` unless they are deliberately
promoted through a later packaging decision.

Never set `--out-dir` to the same directory as `--gate12a-dir`, or to any
child directory inside `--gate12a-dir`.
The runner rejects those aliases because Gate12A source artifact directories
are read-only inputs and must not receive Gate12B outputs.

## Three-Family Comparison

The first bounded family comparison keeps the model fixed to
`Qwen/Qwen2.5-0.5B` and runs the same Gate12B secondary audit over the current
Gate12A `transcript_v1 / briefing_v1 / archive_v1` family set:

```powershell
python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k `
  --out-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_family_compare `
  --top-k 3

python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k `
  --out-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_briefing_family_compare `
  --top-k 3

python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k `
  --out-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_family_compare `
  --top-k 3
```

This comparison is still a read-only secondary audit. It compares candidate
surfaces; it does not promote invariant candidates into a law.

## Archive Strict-Support Sensitivity

To test whether an archive-family candidate signal survives stricter support,
run the archive surfaces with explicit support settings. For example:

```powershell
python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir <gate12a-archive-dir> `
  --out-dir <gate12b-archive-strict-out-dir> `
  --top-k 3 `
  --min-observer-support 2 `
  --min-scale-support 3
```

Use separate output directories for each setting. Do not overwrite the baseline
family-compare run when performing sensitivity checks.

## Observer Mode Sets

The default observer set is:

- `core_v1`

For bounded sensitivity checks, the runner also supports:

- `cycle_motif_expansion_v1`

`cycle_motif_expansion_v1` keeps the core observers and adds ordered
cycle-motif observers derived from `ordered_relation_kind_path`:

- `residual_first_leg`
- `residual_second_leg`
- `residual_third_leg`

This mode set is useful when testing whether a candidate signal survives three
independent observer scopes without changing Gate12A source artifacts.

Example:

```powershell
python tools\run_gate12b_observer_relative_coarse_grained_closure.py `
  --gate12a-dir <gate12a-archive-dir> `
  --out-dir <gate12b-archive-motif-strict-out-dir> `
  --top-k 3 `
  --min-observer-support 3 `
  --min-scale-support 3 `
  --observer-mode-set cycle_motif_expansion_v1
```

Treat this as an explicit sensitivity setting. It does not replace the default
`core_v1` observer set.

## Candidate Reading

`invariant_signature_candidates.jsonl` is not a triangle-only top-k list.

Candidate rows require:

- support across independent observer scopes
- support across multiple scale modes
- at least one non-triangle coarse scale support

Observer views with identical cycle membership are reported together but counted
as one observer scope.

`gauge_variant_signature_candidates.jsonl` is stricter.
It is emitted only when a nontrivial array-level admissible reparameterization
was evaluated and all corresponding gauge-stability checks for the candidate
remain stable.

## Run Summary

Use the run summarizer when comparing multiple Gate12B output directories:

```powershell
python tools\summarize_gate12b_runs.py `
  --out-dir runs\gate12b_summary_example `
  --run-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_motif_obs3_scale3_topk1 `
  --run-dir runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_motif_obs3_scale3_topk3
```

It reads existing Gate12B runs and writes:

- `gate12b_run_summary.csv`
- `gate12b_run_summary.json`
- `manifest.json`

The summary includes candidate counts, dominant relation signatures by band,
observer/scale support distributions, gauge unstable counts, builder hash
status, and output checksum status.

## Source Inspection Queue

Use the source inspection queue builder when returning Gate12B candidates to
Gate12A text-surface rows:

```powershell
python tools\build_gate12b_source_inspection_queue.py `
  --out-dir runs\gate12b_archive_candidate_source_inspection_queue_motif_topk3 `
  --per-band-limit 2 `
  --case qwen2_5_0_5b_archive `
    runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_archive_motif_obs3_scale3_topk3 `
    runs\gate12a_triangle_text_surface_audit_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k
```

Repeat `--case` for each model/family surface.

The queue writes:

- `gate12b_source_inspection_queue.csv`
- `gate12b_source_inspection_queue.jsonl`
- `gate12b_source_inspection_queue.md`
- `gate12b_source_inspection_queue_status.json`
- `manifest.json`
- `checksums.json`

The exact support/conflict anchor text flags are inspection helpers only.
They are not semantic labels.

## Source-Facing Annotation

Use the source-facing annotation helper after building an inspection queue:

```powershell
python tools\annotate_gate12b_source_inspection_queue.py `
  --queue-jsonl runs\gate12b_archive_candidate_source_inspection_queue_motif_topk3\gate12b_source_inspection_queue.jsonl `
  --out-dir runs\gate12b_archive_source_facing_annotation_motif_topk3 `
  --derive-from-queue `
  --annotator codex_source_surface_v1
```

The helper writes:

- `gate12b_source_annotations.jsonl`
- `gate12b_source_annotations.csv`
- `gate12b_source_annotation_summary.json`
- `gate12b_source_annotation_summary.csv`
- `gate12b_source_annotation.md`
- `manifest.json`
- `checksums.json`

Allowed tags:

- `support-following`
- `conflict-following`
- `non-gluing`
- `ambiguous`

These tags describe the queued row's source-facing direction.
They are not answer-quality labels.

## Minimal Verification

Run the focused Python tests:

```powershell
python -m unittest tools.test_run_gate12b_observer_relative_coarse_grained_closure
python -m unittest tools.test_summarize_gate12b_runs
python -m unittest tools.test_build_gate12b_source_inspection_queue
python -m unittest tools.test_annotate_gate12b_source_inspection_queue
```

For the current Gate12B line plus its immediate Gate12A dependencies:

```powershell
python -m unittest `
  tools.test_run_gate12a_discrete_connection_audit `
  tools.test_run_gate12a_triangle_phenotype_tag_prep `
  tools.test_run_gate12b_observer_relative_coarse_grained_closure `
  tools.test_summarize_gate12b_runs `
  tools.test_build_gate12b_source_inspection_queue `
  tools.test_annotate_gate12b_source_inspection_queue
```

Inspect the smoke summary:

```powershell
Get-Content runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_smoke\manifest.json
Get-Content runs\gate12b_observer_relative_coarse_grained_closure_qwen2_5_0_5b_transcript_smoke\gauge_stability_summary.json
```

## Reading Boundary

Read Gate12B conservatively:

- observer-relative movement is not itself failure
- coarse-grained stability is a candidate signal, not a law
- gauge-stable candidates depend on the bounded transform actually evaluated
- no family-wide or model-wide claim follows from one representative smoke
- Gate12A remains the source artifact family, and Gate12B is a secondary audit
