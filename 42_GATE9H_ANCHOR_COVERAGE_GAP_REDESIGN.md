# Gate9H Anchor-Coverage-Gap Redesign

Status: narrow redesign spec, first implementation planned
Role: Gate9H redesign-candidate spec, not operator opening or final anchor redesign settlement
Date: 2026-03-21

Gate9H proceeds from:

- `40_GATE9G_ANCHOR_CONDITIONED_TRIVIALITY.md`
- `41_GATE9G_ANCHOR_CONDITIONED_TRIVIALITY_SMOKE.md`

## 0. Why This Exists

Gate9G named the blocker cleanly.

What is now known is:

- surviving anchor-conditioned rows are not merely weak
- they collapse under `full_anchor_span_collapse`
- the current closure-defect observable is trivial on the recovered bundle

So the next honest move is not:

- vague anchor rescue
- operator opening anyway
- a jump back to cleaner-cell stories without replacing the trivial object

It is:

- testing the smallest redesign candidate that stays on the same anchor-conditioned bundle

## 1. Scope

Gate9H studies only:

- a redesign candidate built from existing `anchor_answer_coverage` and `anchor_token_coverage`
- the same recovered Gate9A bundle already audited by Gate9G
- whether a non-trivial anchor-conditioned read can be recovered without changing the law

It does not:

- introduce a new anchor lane
- introduce a new closure convention
- reopen operator admission
- settle the final redesign

## 2. Public Question

The Gate9H question is:

- can `coverage_gap_abs := |anchor_answer_coverage - anchor_token_coverage|` serve as a minimal non-trivial redesign candidate where closure defect collapses

More concretely:

- does the candidate escape zero-level collapse
- does it stay available on the rows recovered by Gate9F
- does it immediately run into cleaner-cell dominance instead

## 3. Public Object

Gate9H must emit a deterministic row registry that includes at least:

- `cell_id`
- `anchor_kind`
- legacy triviality status
- `anchor_answer_coverage`
- `anchor_token_coverage`
- `coverage_gap_abs`
- redesign candidate status

The public object is not a final metric.

It is:

- a redesign-candidate audit under the frozen law

## 4. Gate9H Falsifiers

Gate9H must keep these falsifiers explicit:

- the candidate still collapses to numerical zero
- the candidate exists only by reintroducing missingness ambiguity
- the candidate becomes immediately cleaner-cell dominated
- the candidate would require new anchor semantics after all

## 5. What This Audit Can Earn

At most, Gate9H can earn the right to say:

- the triviality blocker can be escaped by a minimal redesign candidate

Or:

- even the minimal redesign candidate fails and anchor-conditioned redesign remains blocked

It still does not earn:

- final redesign settlement
- operator opening
- field or spectral language

## 6. Current Memory Hook

The shortest acceptable sentence is:

- replace the collapsed object with the smallest candidate before escalating the story
