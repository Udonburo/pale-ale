# Gate8 Label And Provenance Rules

Status: Draft
Role: Tracked RFC / implementation-binding contract
Date: 2026-03-18

## 0. Purpose

Gate8 must not repeat the old label-geometry mismatch problem.

This document fixes how labels and provenance are represented so the benchmark remains geometry-aligned and reproducible.

## 1. Four-Layer Separation

Every Gate8 sample must preserve distinct IDs for:

- `world_id`
- `rendering_id`
- `target_id`
- `sample_id`

Meaning:

- `world_id` identifies the closed underlying truth state
- `rendering_id` identifies retrieval realization and perturbation pattern
- `target_id` identifies the answer behavior being evaluated
- `sample_id` identifies the concrete benchmark row

Target variation should not force a new world by default.

In conflict cells, multiple `target_id` variants should usually share:

- a stable `world_id`
- and, unless the benchmark is explicitly testing rendering variation, the same `rendering_id`

For quietness controls, `clean_support` and `surface_noisy_clean` should usually share:

- the same `world_id`
- different `rendering_id` values

That is the minimum same-world negative control needed to measure surface-noise quietness without world-difficulty confound.

No layer may silently overwrite another.

## 2. Required Sample-Level Fields

Every sample must carry at least:

- `sample_id`
- `cell_id`
- `world_id`
- `world_ordinal`
- `world_type`
- `rendering_id`
- `target_id`
- `answer_target_type`
- `is_conflict_intended`
- `is_surface_noise_only`
- `retrieval_chunk_ids`
- `retrieval_conflict_chunk_ids`
- `retrieval_support_chunk_ids`

`world_ordinal` and `world_type` are the minimum stable construction fields when later generation depends on synthetic world realization.

## 3. Required Token Or Span Labels

Every answer-bearing sample must support:

- `label_token`
- `label_span_conflict`
- `label_span_support`
- `label_span_defect`

Minimum rules:

- `label_token` remains the token-level detection label
- `label_span_conflict` marks the answer span that follows or reflects conflict
- `label_span_support` marks the answer span supported by retrieval and world truth
- `label_span_defect` marks the answer span where gluing failure is intended or observed

These labels may coincide, but they must not be assumed identical.

## 4. Provenance Binding

Every benchmark run must bind:

- generator script path
- generator script sha256
- code git commit
- taxonomy schema version
- label contract version
- input world specification sha256
- retrieval rendering plan sha256
- answer target plan sha256

For constitution-only scaffold runs, the last three bindings must still exist.

They are satisfied by explicit staged placeholder artifacts:

- `world_plan.json`
- `rendering_plan.json`
- `target_plan.json`

These placeholders are not omissions.

They are implementation-binding declarations that:

- the four-layer contract has been frozen
- the concrete world / rendering / target artifacts are not yet materialized
- later generation stages must replace placeholders with realized artifacts under the same layer split

## 5. Required Artifacts

At minimum, a Gate8 generation run should emit:

- `manifest.json`
- `conflict_plan.json`
- `label_contract.json`
- `world_plan.json`
- `rendering_plan.json`
- `target_plan.json`
- `sample_index.jsonl`
- `checksums.json`

Later generation stages may add retrieval and answer artifacts, but these are the minimum skeleton outputs.

## 6. Determinism

Gate8 generation must be deterministic under fixed:

- world truth source
- taxonomy cell definition
- rendering plan
- target selection rule
- seed
- code revision

If a later natural-language rendering model is involved, that generator layer must still be reproducibly pinned and attested.

## 7. Semi-Closed Meaning

Semi-closed means:

- the benchmark author knows the world truth exactly
- retrieval realizations are rendered from that truth or from controlled contradiction plans
- answer targets are fixed by contract
- labels are not inferred from candidate outputs

Semi-closed does not mean:

- trivial closed-world symbolic toy
- open-world factual uncertainty

## 8. Anti-Mismatch Rule

A sample is invalid if:

- the intended conflict type cannot be reconstructed from provenance
- span labels cannot be explained from world truth plus retrieval rendering
- answer target type and defect span label disagree without explicit justification

This invalidity belongs to the benchmark, not to the candidate model.

## 9. Candidate Freeze Reminder

Gate8 provenance must record that the benchmark was designed for the fixed comparison set:

- `score_F_gram_loop_v1`
- `sigma_gap_tailkeep_weighted_gram_loop_v2`
- `sigma_sqrtgap_tailkeep_object_v2`
- `progression_anisotropic_closure_v3`

This is part of the experiment definition.
