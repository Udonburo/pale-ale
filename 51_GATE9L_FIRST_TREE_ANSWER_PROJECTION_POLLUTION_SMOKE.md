# Gate9L First-Tree Answer-Projection Pollution Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9L first-tree residual-pollution read, not decomposition verdict or operator opening
Date: 2026-03-21

This first tracked Gate9L smoke read executes the first-tree audit defined in:

- `50_GATE9L_FIRST_TREE_ANSWER_PROJECTION_POLLUTION.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate9L first-tree answer-projection pollution audit.

It is not:

- a trusted-tree bypass success
- tree-choice sensitivity execution
- operator opening
- a new metric settlement

It is:

- a tracked handoff for the first deterministic tree or forest build under the Gate9K policy
- a code-bound read on whether cleaner-side residual pollution is concretely `answer_projection`
- the current scientific judgment on what blocks a first honest bypass read

The tracked evidence package is:

- `runs/gate9l_first_tree_answer_projection_pollution_smoke_from_gate9k/manifest.json`
- `runs/gate9l_first_tree_answer_projection_pollution_smoke_from_gate9k/first_tree_edge_registry.jsonl`
- `runs/gate9l_first_tree_answer_projection_pollution_smoke_from_gate9k/first_tree_residual_pollution_registry.jsonl`
- `runs/gate9l_first_tree_answer_projection_pollution_smoke_from_gate9k/first_tree_residual_pollution_by_cell.csv`
- `runs/gate9l_first_tree_answer_projection_pollution_smoke_from_gate9k/first_tree_residual_pollution_status.json`
- `runs/gate9l_first_tree_answer_projection_pollution_smoke_from_gate9k/gate9l_first_tree_answer_projection_pollution_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate9K logging bundle:

- `source_gate9k_run_id = gate9k_trusted_tree_residual_chord_logging_smoke_from_gate9j`
- `source_gate9k_code_git_commit = 4746ca692c2782bd67dc6eaa6f5d838009be33d8`

It builds the first forest over the recovered Gate9A graph:

- `source_gate9a_dir = runs/gate9f_conflict_anchor_recovery_smoke_from_gate9e/gate9a_recovered_from_gate9f`

The Gate9L audit bind is:

- `method_id = gate9l_first_tree_answer_projection_pollution_audit_v1`
- `code_git_commit = e71bfcc0071f0d506186714e9a69132062032c36`

## 2. What Landed

Gate9L builds one deterministic first forest under the declared trusted-edge policy.

It does not yet score a bypass verdict.

It only asks:

- after that first forest is built, what remains in the residual set on cleaner and conflict cells

The target blocker is narrow:

- `cleaner_answer_projection_residual_pollution`

## 3. Smoke Read

### 3.1 The First Forest Builds Cleanly

The status payload is:

- `trusted_forest_build_status = built`
- `trusted_forest_cycle_free_status = clear`
- `trusted_tree_selected_edge_count = 220`
- `trusted_tree_skipped_edge_count = 0`

So the declared trusted-edge policy does produce a clean first forest on this recovered graph.

This first pass does not yet touch tree-choice sensitivity.

It only establishes that a deterministic first tree or forest can be built without doctrine drift.

### 3.2 Cleaner Residual Pollution Is Purely Answer-Projection

The decisive cleaner-side read is:

- `cleaner_residual_answer_projection_edge_count = 8`
- `cleaner_residual_conflict_anchor_edge_count = 0`
- `residual_cleaner_pollution_source_status = answer_projection_only`
- `cleaner_answer_projection_residual_pollution_status = triggered`

The by-cell summary is even sharper:

- `clean_support / answer_projection mean_edge_transport_defect = 0.588568`
- `surface_noisy_clean / answer_projection mean_edge_transport_defect = 0.583639`

So the first forest does not remove cleaner-side residual burden.

It isolates that burden cleanly enough to name it.

### 3.3 Conflict Residual Bridges Still Exist

Conflict cells retain anomaly-side residual bridges:

- `conflict_residual_answer_projection_edge_count = 8`
- `conflict_residual_conflict_anchor_edge_count = 16`
- `conflict_residual_chord_bridge_status = clear`

The conflict-side residual set is therefore not empty or collapsed.

The issue is not lack of anomaly-side residual chords.

The issue is that cleaner-side answer-projection residue still survives alongside them.

### 3.4 Bypass Readiness Remains Denied

The final Gate9L status is:

- `bypass_readiness_status = denied`
- `next_named_blocker = cleaner_answer_projection_residual_pollution`
- `tree_choice_dependence_status = not_yet_executed`

This is the correct landing point.

Gate9L does not say the trusted-tree hypothesis failed.

It says the first honest bypass read is blocked before any sensitivity study, because the cleaner-side residual set is still polluted by `answer_projection`.

## 4. Current Scientific Judgment

The correct Gate9L smoke judgment is:

- Gate9L succeeded as a first-tree residual-pollution audit
- the declared trusted-edge policy already builds a clean deterministic forest
- cleaner-side residual pollution is now explicitly named as `answer_projection_only`
- conflict residual bridges remain available
- the next honest blocker is therefore `cleaner_answer_projection_residual_pollution`

The strongest honest sentence is:

- `Gate9L shows that a first trusted forest can be built cleanly, but the residual set remains polluted on cleaner cells by answer-projection chords, so bypass readiness stays denied before any tree-choice study begins.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the first-tree construction problem is no longer abstract
- cleaner-side residual pollution is explicitly named
- future work can attack `answer_projection` pollution directly without pretending the blocker is still vague

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- trusted-tree decomposition already reduces cleaner-cell dominance
- the residual side is now anomaly-clean
- tree-choice stability is acceptable
- operator admission should open

## 7. Next Honest Move

The next honest move is not:

- a bypass victory claim
- operator opening
- tree-choice sensitivity inflation

The next honest move is:

- attack `cleaner_answer_projection_residual_pollution` directly as the next named blocker before any broader trusted-tree verdict is attempted
