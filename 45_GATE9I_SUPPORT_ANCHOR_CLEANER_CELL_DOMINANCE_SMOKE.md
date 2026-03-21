# Gate9I Support-Anchor Cleaner-Cell Dominance Smoke

Status: tracked smoke execution snapshot
Role: tracked Gate9I cleaner-cell dominance read, not blocker resolution or operator opening
Date: 2026-03-21

This first tracked Gate9I smoke read executes the narrow dominance audit defined in:

- `44_GATE9I_SUPPORT_ANCHOR_CLEANER_CELL_DOMINANCE.md`

## 0. Scope

This file records the first committed-code smoke execution of the Gate9I support-anchor cleaner-cell dominance audit.

It is not:

- blocker resolution
- final redesign settlement
- operator opening
- a new anchor or cycle family

It is:

- a tracked handoff for the next named blocker after Gate9H
- a code-bound read on whether cleaner-cell dominance is quietness noise or a real support-side pattern
- the current scientific judgment on what subblocker sits inside the dominance line

The tracked evidence package is:

- `runs/gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h/manifest.json`
- `runs/gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h/support_anchor_cleaner_dominance_registry.jsonl`
- `runs/gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h/support_anchor_quietness_pairs.csv`
- `runs/gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h/support_anchor_cleaner_dominance_by_cell.csv`
- `runs/gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h/support_anchor_cleaner_dominance_status.json`
- `runs/gate9i_support_anchor_cleaner_cell_dominance_smoke_from_gate9h/gate9i_support_anchor_cleaner_dominance_read.md`

## 1. Source And Bind

This smoke run consumes the first Gate9H redesign bundle:

- `source_gate9h_run_id = gate9h_anchor_coverage_gap_redesign_smoke_from_gate9g`
- `source_gate9h_code_git_commit = 80f0a3c4916982fd5b130a96670c656068401157`

The Gate9I audit bind is:

- `method_id = gate9i_support_anchor_cleaner_cell_dominance_audit_v1`
- `code_git_commit = 29ab3cdcfb4f0e182f85515f0360755b636ad6a8`

The audited redesign candidate remains:

- `anchor_coverage_gap_abs_v1`

## 2. What Landed

Gate9I does not redesign the object again.

It holds the Gate9H candidate fixed and asks a narrower question:

- why does cleaner-cell dominance still trigger on the support-anchor family

The audit decomposes that blocker through:

- support-side per-cell means
- quietness-paired cleaner rows
- the support-side split between `direct_contradiction` and `distributed_incompatibility`

## 3. Smoke Read

### 3.1 Cleaner-Cell Dominance Still Holds

The support means are:

- `support_clean_mean_gap = 0.148147`
- `support_surface_noisy_mean_gap = 0.150462`
- `support_direct_mean_gap = 0.136027`
- `support_distributed_mean_gap = 0.089808`

The status payload therefore reads:

- `support_anchor_cleaner_dominance_status = triggered`

So Gate9H's blocker is not gone.

The support-anchor redesign line still peaks on cleaner-side rows.

### 3.2 Quietness Noise Does Not Explain It

The paired cleaner deltas are small:

- `mean_abs_quietness_pair_gap_delta = 0.010537`

And the noisy cleaner rows do not fall below the conflict-side support maximum:

- `surface_noisy_corroboration_status = corroborated`
- `support_surface_noisy_mean_gap = 0.150462`
- `support_conflict_max_mean_gap = 0.136027`

So the current dominance line is not honestly readable as a quietness-only artifact.

The strongest narrow sentence is:

- `dominance_explained_as_quietness_noise_status = denied`

### 3.3 Conflict-Side Split Is Real

The support-side conflict rows are not flat.

They split:

- `direct_contradiction = 0.136027`
- `distributed_incompatibility = 0.089808`

That yields:

- `distributed_underactivation_status = triggered`

So the blocker decomposes into:

- a cleaner-side plateau that survives surface noise
- a conflict-side drop concentrated on `distributed_incompatibility`

### 3.4 Current Subblocker

The status payload closes as:

- `next_named_subblocker = distributed_underactivation`

This is the right Gate9I landing point.

Gate9I does not settle the dominance blocker.

It names the inner support-side split that now explains where to cut next.

## 4. Current Scientific Judgment

The correct Gate9I smoke judgment is:

- Gate9I succeeded as a blocker-decomposition audit
- support-anchor cleaner-cell dominance is still real on the Gate9H redesign line
- that dominance is not explained by quietness noise
- the next named subblocker is `distributed_underactivation`

The strongest honest sentence is:

- `Gate9I shows that support-anchor cleaner-cell dominance is not a quietness artifact; it is corroborated on surface-noisy cleaner rows and sharpened by distributed-underactivation on the conflict side.`

## 5. What This Smoke Run Earns

This smoke run earns the right to say:

- the cleaner-cell dominance blocker is now internally decomposed
- quietness pairing does not rescue the support-anchor dominance line
- future work can attack `distributed_underactivation` directly as the next named subblocker

## 6. What This Smoke Run Does Not Earn

This smoke run does not earn the right to say:

- the redesign line is now conflict-led
- cleaner-cell dominance is resolved
- operator admission should open
- the support-anchor redesign candidate is now final

## 7. Next Honest Move

The next honest move is not:

- another metric redesign
- operator opening
- a return to abstract admission debate

The next honest move is:

- attack `distributed_underactivation` as the next named subblocker inside support-anchor cleaner-cell dominance
