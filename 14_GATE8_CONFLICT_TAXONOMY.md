# Gate8 Conflict Taxonomy

Status: Draft
Role: Tracked RFC / benchmark taxonomy
Date: 2026-03-18

## 0. Purpose

Gate8 should be designed as a collision taxonomy, not as a loose pile of retrieval examples.

The benchmark surface is about epistemic collision geometry:

- support
- contradiction
- distributed incompatibility
- surface noise without semantic break

## 1. The Four Cells

### A. `clean_support`

Definition:

- retrieval is consistent with world truth
- retrieval chunks mutually support one another
- no intended contradiction exists

Purpose:

- negative control for quietness

What should happen:

- all viable candidates stay relatively quiet
- no candidate should need conflict to produce useful signal here

### B. `direct_contradiction`

Definition:

- at least one retrieval chunk explicitly contradicts world truth or the dominant support set

Purpose:

- first-order conflict detection

What should happen:

- static candidates should rise near the conflict span
- dynamic candidate may or may not improve, but should at least remain interpretable

### C. `distributed_incompatibility`

Definition:

- no single retrieval chunk is a decisive contradiction alone
- incompatibility emerges only when multiple chunks are glued together

Purpose:

- more natural global non-integrability

What should happen:

- this is the most likely cell where dynamic regime dependence can appear
- `gate7c` is specifically being watched here

### D. `surface_noisy_clean`

Definition:

- retrieval surface is noisy
- semantics remain compatible with world truth
- noise may include paraphrase, order wobble, omission, stylistic mismatch, or distractor clutter

Purpose:

- retrieval-space extension of Seam

What should happen:

- quietness must remain materially intact
- if candidates spike here, Gate8 is not teaching the right lesson

## 2. Generation Layers

Each cell is generated through four explicit layers:

1. `world truth`
2. `retrieval rendering`
3. `answer target`
4. `span labels`

The taxonomy belongs primarily to layers 1 and 2.

## 3. Allowed Answer Target Types

Minimum answer-target regimes:

- `consistent_answer`
- `conflict_following_wrong_answer`
- `unsupported_bridge_answer`

Not every cell requires every answer type.

Recommended default pairing:

- `clean_support` -> `consistent_answer`
- `direct_contradiction` -> `consistent_answer`, `conflict_following_wrong_answer`
- `distributed_incompatibility` -> `consistent_answer`, `unsupported_bridge_answer`
- `surface_noisy_clean` -> `consistent_answer`

## 4. Geometry Read By Cell

### `clean_support`

- tests quietness preservation

### `direct_contradiction`

- tests local defect visibility

### `distributed_incompatibility`

- tests global glue failure without trivial local contradiction

### `surface_noisy_clean`

- tests whether retrieval rendering noise is mistaken for conflict topology

## 5. What This Taxonomy Avoids

This taxonomy is intentionally trying to avoid:

- benchmark zoo drift
- retrieval QA score-chasing
- arbitrary example curation without provenance
- post hoc labeling after candidate outputs are seen

## 6. Standing Questions

For every cell, the benchmark should support these questions:

- does `gate6f` keep operational standing?
- does `gate6h` remain a meaningful pure-object candidate?
- does `gate7c` gain relative ground in conflict-heavy cells?
- does `F` regain or lose relative authority?

## 7. Exit Condition

If these four cells cannot be kept distinct in labels and provenance, the benchmark is not ready.

That is a benchmark-contract failure, not a model result.
