# Gate9G Anchor-Conditioned Triviality Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9G anchor-conditioned blocker read, not anchor redesign or operator opening
Date: 2026-03-21

This first tracked Gate9G smoke read executes the narrow blocker audit defined in:

- `40_GATE9G_ANCHOR_CONDITIONED_TRIVIALITY.md`

The next redesign-candidate spec is now recorded in:

- `42_GATE9H_ANCHOR_COVERAGE_GAP_REDESIGN.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate9G anchor-conditioned triviality audit.

It is not:

- anchor redesign
- closure redesign
- a graph-wide operator opening
- a cleaner-cell-dominance study

It is:

- a tracked handoff for naming the anchor-conditioned blocker after Gate9F recovery
- a code-bound read on whether surviving anchor-conditioned rows are non-trivial or collapsed
- the current scientific judgment on whether Gate9C criterion `4.2 Non-Trivial Anchor-Conditioned Read` is still denied

The tracked evidence package is:

- `runs/gate9g_anchor_conditioned_triviality_smoke_from_gate9f/manifest.json`
- `runs/gate9g_anchor_conditioned_triviality_smoke_from_gate9f/anchor_conditioned_triviality_registry.jsonl`
- `runs/gate9g_anchor_conditioned_triviality_smoke_from_gate9f/anchor_conditioned_triviality_by_cell_anchor.csv`
- `runs/gate9g_anchor_conditioned_triviality_smoke_from_gate9f/anchor_conditioned_triviality_status.json`
- `runs/gate9g_anchor_conditioned_triviality_smoke_from_gate9f/gate9g_anchor_conditioned_triviality_read.md`

## 1. Source And Bind

This smoke run consumes the recovered Gate9A bundle from Gate9F:

- `source_gate9a_run_id = gate9a_recovered_from_gate9f`
- `source_gate9a_code_git_commit = efd9be68fcec991edff57547e7e02a8af1ed4050`

The Gate9G audit bind is:

- `method_id = gate9g_anchor_conditioned_triviality_audit_v1`
- `code_git_commit = 6586002abc9dbb773f66015e7a15a88702e3ed69`

The source Gate8 execution remains:

- `recovered_gate8_execution`

## 2. What Landed

Gate9G now emits:

- row-level anchor-conditioned triviality registry
- per-cell and per-anchor summary for triviality status
- deterministic blocker status payload

This is the first point where the repo stops saying only that anchor-conditioned read is weak and instead asks whether it is actually trivial on the rows where it survives.

## 3. Smoke Read

### 3.1 Non-Missing Rows

Gate9G sees:

- `n_non_missing_rows = 12`
- `n_full_anchor_span_collapse_rows = 12`
- `n_nontrivial_signal_candidate_rows = 0`

So every surviving anchor-conditioned row on the recovered bundle collapses the same way.

### 3.2 Collapse Signature

On all non-missing rows:

- `anchor_rank = 3`
- `answer_conditioned_rank = 3`
- `token_conditioned_rank = 3`
- `anchor_conditioned_closure_defect` stays at numerical zero

The emitted triviality label is:

- `full_anchor_span_collapse`

This matters because the rows are not empty.

Coverage remains non-trivial.

For example, `anchor_answer_coverage` and `anchor_token_coverage` differ materially across rows.

But the current closure observable still collapses once both conditioned ranks saturate the anchor span.

### 3.3 Missing Rows

The only non-surviving rows are the structurally missing cleaner-side conflict rows:

- `clean_support / conflict`
- `surface_noisy_clean / conflict`

Those remain:

- `missing_or_insufficient`

They are not the main Gate9G blocker.

The main blocker is that the surviving rows are trivial, not absent.

## 4. Current Scientific Judgment

The correct Gate9G smoke judgment is:

- Gate9G succeeded as a blocker audit
- the current anchor-conditioned read is not merely weak on the recovered bundle
- it is trivial under the current construction
- the denial of `non-trivial anchor-conditioned read` therefore remains earned

The strongest honest sentence is:

- `After Gate9F recovery, the surviving anchor-conditioned rows still collapse to a full-anchor-span triviality signature, so Gate9C criterion 4.2 remains denied for an object-level reason rather than a missingness reason.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the next admission blocker is now named cleanly
- the blocker is `full_anchor_span_collapse` in the current anchor-conditioned observable
- future work should treat this as a blocker of read design, not as a missingness problem

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- anchor redesign should now be improvised
- operator admission should open anyway
- cleaner-cell dominance is resolved

## 7. Next Honest Move

The next honest move is not:

- operator opening
- field language
- anchor rescue by vague narrative

The next honest move is:

- keep this blocker named as `full_anchor_span_collapse`
- decide whether the next narrow line should attack anchor-conditioned redesign or cleaner-cell dominance first
