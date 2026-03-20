# Gate8B Standing Generalization

Status: spec fixed, first rendering-family implementation landed
Role: post-Gate8 standing portability spec, not bridge spec
Date: 2026-03-20

## 0. Why This Exists

Gate8 has already taught the main thing it was able to teach on the current family.

That thing is not:

- a successful explanatory bridge

That thing is:

- `gate7c revival` persisted under court repair, scale-up, and repeated bridge failure

So the next question should no longer be:

- can one more bridge rescue the story

The next question should be:

- does the fixed-court `gate7c revival` survive a controlled regime shift

This document therefore opens a new workstream:

- standing generalization under a frozen court

## 1. Primary Question

The only primary question of Gate8B is:

- `gate7c revival survives regime shift?`

This is a portability question, not an explanation question.

It does not ask:

- why `gate7c` wins
- which bridge explains the win
- whether field language can now be promoted

## 2. Court Freeze

The standing court remains fully frozen.

The following are fixed:

- candidates: `F`, `gate6f`, `gate6h`, `gate7c`
- standing evaluator
- label provenance rules
- mixed-granularity caveat
- same-world quietness discipline

The following remain forbidden:

- adding a new ranking candidate
- changing evaluator logic
- changing aggregation rules
- mixing bridge diagnostics back into standing
- introducing a classifier, fusion layer, or rescue scorer

## 3. Aggregation Ban

The aggregation ban remains active.

This workstream is not allowed to answer portability by:

- combining bridge outputs with standing metrics
- adding neighboring summaries until one story survives
- replacing fixed-court tables with a new aggregate verdict

The fixed standing table remains the court.

## 4. One New Thing Only

Gate8B is allowed to add exactly one new regime axis.

For the first pass, that axis should be:

- one new template/rendering family

Not this:

- new conflict family plus new rendering family at the same time
- new truth construction
- new ontology of contradiction

Reason:

- template/rendering shift preserves taxonomy and truth while still testing whether the standing result is tied to one rhetorical packaging of the same benchmark logic

## 5. Why Template/Rendering First

Template/rendering family is the cleanest first shift because it can hold fixed:

- world truth
- cell taxonomy
- answer target types
- conflict intent

while changing:

- prompt surface form
- retrieval chunk phrasing
- memo ordering
- rhetorical packaging of support and contradiction

So the main disturbance is regime, not benchmark identity.

That makes portability easier to read than a simultaneous taxonomy change.

## 6. Required Invariants

The first generalization family must preserve:

- the four Gate8 cells as the court-facing taxonomy
- `consistent_answer`, `conflict_following_wrong_answer`, and other existing answer-target semantics where applicable
- truth-side construction for the same world families
- quietness pairability for clean/noisy rows

It may vary:

- rendering order
- retrieval voice
- support/conflict memo presentation
- surface template shape

It may not vary:

- the fixed candidate set
- evaluator semantics
- bridge status
- court-facing aggregation rules

## 7. Expected Reads

### 7.1 If `gate7c` Survives

If `gate7c` remains ahead under the new family, the earned statement is small:

- the dynamic revival is less likely to be benchmark-local to the original Gate8 rendering family

This still does not earn:

- bridge success
- field aggregation
- universal superiority

### 7.2 If `gate7c` Collapses

If `gate7c` falls back under the new family, the earned statement is:

- the current Gate8 revival was regime-specific

That is still useful.

It means the current standing result should be read as local to one family, not as portable court behavior.

### 7.3 If the Read Is Mixed

If conflict standing partly survives but the quietness side degrades, or vice versa, the read remains unresolved.

That does not license rescue operations.

It only licenses one further tightly-scoped generalization decision later.

## 8. Falsifiers

This workstream weakens or fails under any of the following:

- `gate7c` falls back to `F` on the new family
- the ranking becomes unstable enough under family shift that the current revival no longer looks like a court-level fact
- `gate7c` keeps conflict-side wins only by materially breaking the quietness side
- quietness remains acceptable but the conflict-side standing gain disappears

More sharply:

- if the family shift makes `gate7c` lose the very revival Gate8 was supposed to preserve, the portability claim fails
- if the family shift preserves only one court while collapsing the other, the result is not yet a clean generalization

## 9. What Stays Out

Gate8B v0 is not allowed to quietly become:

- bridge v4
- a new benchmark constitution rewrite
- a multi-family expansion wave
- field aggregation
- a transport-motif comparison

Those are different workstreams.

## 10. Non-Promotion Rule

Even a successful Gate8B portability read does not immediately earn:

- candidate promotion beyond the current fixed court
- replacement of `F` as a permanent guardrail
- a claim that `gate7c` is regime-invariant in general

At most, it earns:

- the right to say that the current `gate7c revival` survives one controlled rendering-family shift under a frozen court

## 11. Implementation Gate

No code should be written for Gate8B until this spec is accepted.

Implementation is allowed only if:

- the candidate set is explicitly frozen
- the evaluator freeze is explicit
- the aggregation ban is explicit
- the new addition is one template/rendering family only
- the falsifiers are written first

Until then, this remains spec-only.

## 12. Implementation Note

The first implementation pass has now been limited to exactly one new rendering family:

- `briefing_v1`

This implementation keeps fixed:

- candidates: `F`, `gate6f`, `gate6h`, `gate7c`
- standing evaluator semantics
- aggregation ban
- diagnostic-only bridge status

The implementation scope is intentionally narrow:

- add one rendering-family switch at generation/materialization time
- carry `rendering_family_id` through benchmark and candidate-batch provenance
- do not change the court-facing standing logic

Commit-bound smoke validation was run at:

- `runs/gate8b_smoke_briefing_commitbind_constitution`
- `runs/gate8b_smoke_briefing_commitbind_benchmark`
- `runs/gate8b_smoke_briefing_commitbind_candidate_execution`

with code commit:

- `4ac4b97`

This smoke run is only an implementation bind check.

It does not settle the Gate8B scientific read.

The real question remains:

- does `gate7c revival` survive a proper regime-shift evaluation under the new family
