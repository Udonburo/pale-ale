# Gate9B Execution Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9B first-pass execution read, not explanatory settlement
Date: 2026-03-21

This first tracked Gate9B smoke read executes the narrow holonomy study defined in:

- `28_GATE9B_SMALL_CYCLE_HOLONOMY_STUDY.md`

## 0. Scope

This file records the first committed-code smoke execution of Gate9B.

It is not:

- a Gate8 standing re-trial
- a Gate9 success verdict
- a field or spectral opening
- permission to add a third cycle motif without spec revision

It is:

- a tracked handoff for the first Gate9B comparison consumer
- a code-bound smoke read of the narrow two-cycle holonomy study
- the current scientific judgment on what Gate9B now sharpens and what it still does not earn

The tracked evidence package is:

- `runs/gate9a_smoke_from_gate8c/manifest.json`
- `runs/gate9b_smoke_from_gate9a/manifest.json`
- `runs/gate9b_smoke_from_gate9a/cycle_focus_registry.jsonl`
- `runs/gate9b_smoke_from_gate9a/quietness_cycle_pair_registry.jsonl`
- `runs/gate9b_smoke_from_gate9a/cycle_motif_by_cell.csv`
- `runs/gate9b_smoke_from_gate9a/falsifier_status.json`
- `runs/gate9b_smoke_from_gate9a/gate9b_holonomy_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate9A smoke bundle:

- `source_gate9a_run_id = gate9a_smoke_from_gate8c`
- `source_gate9a_code_git_commit = d68dd28f22e747fed63f46a1fbf897dddff1aaa3`

That Gate9A smoke bundle itself remains bound to:

- `source_gate8_run_id = gate8c_smoke_transcript_candidate_execution`
- `source_gate8_code_git_commit = 0050ebc8df66e5ceabe1441d6215d26ac40be1aa`
- `source_rendering_family_id = transcript_v1`

The Gate9B consumer bind is:

- `method_id = gate9b_small_cycle_holonomy_study_v1`
- `code_git_commit = 649e4eb`

The narrow study freeze remains:

- allowed cycle types = `support_answer_terminal_token_cycle`, `conflict_answer_terminal_token_cycle`
- cycle registry primary = true
- per-cell aggregate supplemental = true

## 2. What Landed

Gate9B now adds a narrow comparison layer over Gate9A cycle output.

The public artifacts are:

- per-cycle focus registry
- quietness-pair cycle registry
- per-cell motif summary as support telemetry
- explicit falsifier status

The main implementation discipline is preserved in this smoke run:

- existing two cycle motifs only
- per-cycle registry remains primary
- no new public primitive was introduced
- no connection Laplacian or field layer was opened

## 3. Smoke Read

### 3.1 Quietness Pair Comparison

The quietness-pair layer is live.

For `support_answer_terminal_token_cycle`:

- `mean_abs_quietness_delta = 0.010658`
- `mean_surface_noisy_minus_clean_defect = 0.010658`

For `conflict_answer_terminal_token_cycle`:

- both quietness pairs resolve to `paired_cycle_failure:missing_conflict_anchor`

So the narrow earned sentence is:

- quietness-pair comparison is now executable
- but one of the two licensed motifs already collapses there because the anchor is absent

### 3.2 Support-Cycle Read

For `support_answer_terminal_token_cycle`, the smoke read is:

- `clean_support mean_holonomy_defect = 0.818136`
- `surface_noisy_clean mean_holonomy_defect = 0.828794`
- `direct_contradiction mean_holonomy_defect = 0.064387`
- `distributed_incompatibility mean_holonomy_defect = 0.110330`

This triggers:

- `cleaner_cell_dominance`
- `distributed_incompatibility_failure`

It does not trigger:

- `direct_contradiction_escape`

So the support-cycle read is now sharper than Gate9A was, but still not good news.

It says:

- cleaner cells still dominate the main support-cycle holonomy read
- distributed incompatibility is still not the proving ground yet

### 3.3 Conflict-Cycle Read

For `conflict_answer_terminal_token_cycle`, the smoke read is:

- `direct_contradiction mean_holonomy_defect = 0.070547`
- `clean_support = missing_conflict_anchor`
- `distributed_incompatibility = missing_conflict_anchor`
- `surface_noisy_clean = missing_conflict_anchor`

This triggers:

- `missing_anchor_collapse`

Other falsifiers remain:

- `insufficient_data`

So the conflict-cycle read does not yet fail by scalar noise.

It fails more simply:

- the licensed motif is not sufficiently instantiated on this smoke bundle to carry the study

## 4. Current Scientific Judgment

The correct first-pass Gate9B judgment is:

- Gate9B implementation succeeded as a narrow comparison layer
- Gate9B sharpened the falsifier surface rather than clearing it
- Gate9B has not yet earned explanatory holonomy status
- Gate9B still does not make distributed incompatibility legible in the desired way

The strongest honest sentence is:

- `Gate9B first pass succeeded as narrow holonomy-comparison infrastructure, but the current smoke read still triggers the main falsifiers.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the Gate9B narrow study is now executable
- the two licensed motifs can be compared without reopening the graph-gauge law
- quietness-pair cycle comparison can be emitted directly rather than inferred indirectly
- falsifiers can be evaluated explicitly rather than narratively

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- holonomy has now become an explanatory read
- distributed incompatibility is now cleanly separated
- the conflict-cycle motif is ready for mainline use
- Gate9 may now open connection Laplacian
- a third motif should be added ad hoc during implementation

## 7. Next Honest Move

The next honest move is not:

- adding a third motif without revising the spec
- softening the falsifiers
- reopening field or spectral language

The next honest move is:

- preserve this smoke run as the first tracked Gate9B handoff
- keep the two-motif study frozen
- only then decide whether a fuller tracked Gate9B execution is worth the cost under the same falsifiers
