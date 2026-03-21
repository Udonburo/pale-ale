# Gate9A Execution Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9A first-pass execution read, not a standing verdict
Date: 2026-03-21

This first tracked Gate9A smoke read executes the graph-gauge consumer defined in:

- `26_GATE9_GRAPH_GAUGE_CONSTITUTION.md`

## 0. Scope

This file records the first committed-code smoke execution of Gate9A.

It is not:

- a Gate8 standing re-trial
- a candidate-promotion file
- a field-aggregation file
- a claim that Gate9 already explains the geometry of conflict

It is:

- a tracked handoff for the first Gate9A consumer
- a code-bound smoke read of the object-level failure surface
- the current scientific judgment on what this smoke run does and does not earn

The tracked evidence package is:

- `runs/gate8c_smoke_transcript_candidate_execution/manifest.json`
- `runs/gate9a_smoke_from_gate8c/manifest.json`
- `runs/gate9a_smoke_from_gate8c/gate9a_failure_surface.md`
- `runs/gate9a_smoke_from_gate8c/edge_transport_by_type.csv`
- `runs/gate9a_smoke_from_gate8c/cycle_summary_by_cell.csv`
- `runs/gate9a_smoke_from_gate8c/anchor_conditioned_closure_by_cell.csv`

## 1. Source And Bind

This smoke run consumes the Gate8C smoke execution bundle:

- `source_gate8_run_id = gate8c_smoke_transcript_candidate_execution`
- `source_rendering_family_id = transcript_v1`
- `source_gate8_code_git_commit = 0050ebc8df66e5ceabe1441d6215d26ac40be1aa`

The Gate9A consumer bind is:

- `method_id = gate9a_graph_gauge_consumer_v1`
- `code_git_commit = d68dd28`
- `projector_public_kind = factorized_projector_public_v1`
- `cycle_token_selector = terminal_token_state_v1`
- `aggregation_ban_inherited = true`
- `quietness_pairing_rule = shared_world_id_v1`

The declared public object remains:

- projector / rank / metadata

The declared primary observables remain:

- `edge_transport_defect`
- `small_cycle_holonomy`
- `anchor_conditioned_closure`

## 2. What Landed

Gate9A now emits a deterministic graph-gauge failure surface over:

- `token_state`
- `support_chunk`
- `conflict_chunk`
- `answer_state`

with the declared edge types:

- `temporal_transition`
- `support_anchor`
- `conflict_anchor`
- `answer_projection`
- `quietness_pair`

The main implementation discipline is preserved in this smoke run:

- explicit cycle closure only; no implicit return legs
- projector public, basis auxiliary
- failure enums before aggregate interpretation
- inherited Gate8 court discipline remains frozen downstream

## 3. Smoke Read

### 3.1 Edge Transport Defect

The edge failure surface is live and non-degenerate.

The main smoke read is:

- `temporal_transition mean_edge_transport_defect = 0.563989`
- `answer_projection mean_edge_transport_defect = 0.484533`
- `support_anchor mean_edge_transport_defect = 0.448075`
- `conflict_anchor mean_edge_transport_defect = 0.334080`
- `quietness_pair mean_edge_transport_defect = 0.078603`

This earns a narrow but real statement:

- the object-level graph-gauge failure surface is not collapsing into noise
- `quietness_pair` remains materially quieter than the heavier local transport edges

### 3.2 Small-Cycle Holonomy

Small-cycle holonomy is emitted on explicit closed cycles, but the smoke read is not yet aligned with the hoped explanatory line.

For `support_answer_terminal_token_cycle`:

- `clean_support mean_holonomy_defect = 0.818136`
- `surface_noisy_clean mean_holonomy_defect = 0.828794`
- `direct_contradiction mean_holonomy_defect = 0.064387`
- `distributed_incompatibility mean_holonomy_defect = 0.110330`

For `conflict_answer_terminal_token_cycle`:

- `direct_contradiction mean_holonomy_defect = 0.070547`
- all other cells are `missing_conflict_anchor` in this smoke bundle

So the current smoke result is not:

- contradiction-side or distributed-incompatibility-side holonomy cleanly rising above cleaner cells

It is only:

- explicit cycle machinery is working
- the present smoke bundle does not yet produce an explanatory holonomy read

### 3.3 Anchor-Conditioned Closure

Anchor-conditioned closure is implemented, but this smoke bundle does not make it discriminative.

Where the anchor exists:

- all emitted `anchor_conditioned_closure_defect` values are `0.000000`

Where the anchor does not exist:

- the output is `missing_conflict_anchor`

So the current earned sentence is:

- the observable is alive as infrastructure
- it is not yet scientifically useful on this smoke bundle

## 4. Current Scientific Judgment

The correct first-pass judgment is:

- Gate9A implementation succeeded as object-level infrastructure
- the present smoke read remains infrastructure-heavy rather than scientifically mature
- Gate9A has not yet earned the right to say that it explains Gate8 standing more naturally
- Gate9A has not yet earned the right to say that distributed incompatibility is now legible

The strongest honest sentence is:

- `Gate9A first pass succeeded as deterministic failure-surface infrastructure, but its scientific read remains immature.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the Gate9 constitution is now executable
- the graph-gauge consumer can emit deterministic node/edge/cycle/closure artifacts
- explicit cycles and failure enums survive contact with real Gate8C artifacts
- the public primitive can remain projector-first in execution, not just in prose

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- Gate9 already explains Gate8 standing
- distributed incompatibility is now cleanly separated
- holonomy is already the decisive readout
- anchor-conditioned closure is already informative
- the current smoke numbers should be treated as repo-level judgment

## 7. Next Honest Move

The next honest move is not:

- score shaping
- explanatory overclaim
- field aggregation
- spectral branding

The next honest move is:

- preserve this smoke run as the first tracked Gate9A handoff
- keep the scientific judgment cold
- move next only through a narrow Gate9B holonomy spec, now tracked in `28_GATE9B_SMALL_CYCLE_HOLONOMY_STUDY.md`
