# Gate7 Motif Bake-Off And Outcome

Status: Draft
Role: Tracked standing snapshot / implementation-facing closeout
Date: 2026-03-18

## 0. Scope

Gate7 was intentionally constrained to dynamic motif comparison.

It did not include:

- field aggregation
- burst aggregation
- persistence topology
- benchmark broadening

This was deliberate.
The point of Gate7 was to decide whether any object-native dynamic law earns the right to unlock a later field stage.

## 1. Bake-Off Rules

The Gate7 bake-off used the following discipline:

- same Gate6 native object source
- same CFA aggregate surface
- same Seam pair evaluator
- one dynamic law at a time
- no aggregation rescue

Practical unlock emphasis:

- Seam tail mattered more than raw CFA excitement
- especially `mean_delta_p90` and `mean_top10_inflation`

Interpretation rule:

- a dynamic law may be interesting without earning field aggregation
- field aggregation remains locked unless the dynamic law clears the tail discipline

## 2. Candidate Lines

### 2.1 `gate7a` progression leakage

Method:

- `progression_leak_v1 = 1 - ||P_t v_{t+1}||^2 / ||v_{t+1}||^2`

Primary artifacts:

- [`runs/gate7a_cfa_full/gate7a_aggregate_summary.md`](runs/gate7a_cfa_full/gate7a_aggregate_summary.md)
- [`runs/gate7a_seam_pairs_full/gate7a_seam_report.md`](runs/gate7a_seam_pairs_full/gate7a_seam_report.md)

Read:

- dynamic signal clearly alive
- strong quietness on `mean_delta_max`
- strong quietness on robust-normalized max
- weaker `p90`
- slightly worse `top10_inflation`

Decision:

- `mixed keep`

### 2.2 `gate7b` projector closure

Method:

- `progression_closure_v2 = 1 - ||P_{t+1} P_t v_{t+1}||^2 / ||v_{t+1}||^2`

Primary artifacts:

- [`runs/gate7b_cfa_full/gate7b_aggregate_summary.md`](runs/gate7b_cfa_full/gate7b_aggregate_summary.md)
- [`runs/gate7b_seam_pairs_full/gate7b_seam_report.md`](runs/gate7b_seam_pairs_full/gate7b_seam_report.md)

Read:

- did not improve the Gate7a ambiguity
- CFA degraded relative to both `F` and `gate7a`
- Seam `p90` and `top10_inflation` remained bad

Decision:

- `clean negative`

### 2.3 `gate7c` anisotropic projector closure

Method:

- `A_t = B_t diag(sigma_i / sigma_1) B_t^T`
- `progression_anisotropic_closure_v3 = 1 - ||A_{t+1} A_t v_{t+1}||^2 / ||v_{t+1}||^2`

Primary artifacts:

- [`runs/gate7c_cfa_full/gate7c_aggregate_summary.md`](runs/gate7c_cfa_full/gate7c_aggregate_summary.md)
- [`runs/gate7c_seam_pairs_full/gate7c_seam_report.md`](runs/gate7c_seam_pairs_full/gate7c_seam_report.md)

Read:

- strongest CFA result in the Gate7 family so far
- better `p90` than `gate7a` and `gate7b`
- still does not clear the tail discipline because `top10_inflation` remains worse than the `F` guardrail
- `mean_delta_max` is good, but not enough to unlock a field phase by itself

Decision:

- `mixed keep`

## 3. Current Standing

| line | CFA read | Seam read | decision |
|---|---|---|---|
| `gate7a` progression leak | mixed | mixed with real quietness signal | keep |
| `gate7b` projector closure | degraded | insufficient tail behavior | kill |
| `gate7c` anisotropic closure | best Gate7 CFA | still mixed on seam tail | keep but not promoted |

## 4. Outcome

Gate7 did produce real dynamic signal.

What it did not produce is a dynamic law strong enough to unlock field aggregation.

Repo-level decision:

- Gate7 dynamic motif comparison is now sufficiently explored for this phase
- field aggregation remains locked
- no Gate7 candidate currently displaces the static operational winner from Gate6

The best current dynamic research candidate is:

- `gate7c`

But that is not the same thing as a field unlock.

## 5. What This Means

The important negative result is:

- dynamic signal exists
- but dynamic signal does not yet have a clean independent win condition over seam tail

That means the repo should not reintroduce the old failure mode of amplifying weak mixed signal through aggregation.

This is exactly why field aggregation stays off.

## 6. Handoff

After this bake-off, the correct next move is not field construction.

The correct next move is benchmark and object expansion under fixed winners.

That means:

- operational static mainline remains Gate6 `gate6f`
- research north star remains Gate6 `gate6h`
- dynamic research candidate remains Gate7 `gate7c`
- next expansion should be a benchmark-broadening workstream, not aggregation rescue

That benchmark constitution workstream begins in:

- [`13_GATE8_BENCHMARK_CONSTITUTION.md`](13_GATE8_BENCHMARK_CONSTITUTION.md)
- [`14_GATE8_CONFLICT_TAXONOMY.md`](14_GATE8_CONFLICT_TAXONOMY.md)
- [`15_GATE8_LABEL_AND_PROVENANCE_RULES.md`](15_GATE8_LABEL_AND_PROVENANCE_RULES.md)
