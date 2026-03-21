# Gate6 Local Design Memo

Status: Draft
Role: Tracked RFC / not frozen
Date: 2026-03-17

Naming note:
This RFC family uses `Gate6-A` as a working label for the current observation-redesign workstream.
It does not, by itself, resume repo-wide numbering for every post-Gate5 workstream.
Until the broader architecture is updated explicitly, downstream units remain workstreams rather than pre-frozen new gates.

## 0. Why Gate6 Exists

Gate5 closed the boundary-side scalar-shaping line.

What Gate5 established:

- transport-style loop residuals are computable
- they are not identical to distance-sum scores
- FWHT-8D plus boundary-side scalar shaping did not achieve simultaneous:
  - CFA-side useful localization
  - Seam-side quietness

Therefore Gate6 is not a score redesign.

Gate6 is an observation redesign.

Its task is to replace the current compressed proxy boundary with a native local observation object that preserves the geometry actually present at each token step.

## 1. Central Reframe

Old question:

- Which scalar score best detects defect?

New question:

- What is the smallest deterministic local object that preserves token-step geometry well enough for transport experiments to be meaningful?

This means:

- Gate4 fixed a canonical observation sink at the proxy level.
- Gate5 showed that transport residuals are feasible on that sink.
- Gate6 now asks whether the sink itself is too lossy.

Gate6 should therefore be understood as boundary liberation / native observation design, not as:

- a better score
- a better cap
- a better boundary scalar
- a proof layer

## 2. What Gate6 Must And Must Not Change

### Gate6 must change

- the observation boundary
- the representation of local token-step geometry
- the artifact family used for downstream transport experiments

### Gate6 must not change

- the philosophical role of Gate4 as the first observation SSOT
- the interpretation of Gate5 as a feasibility study
- the discipline of CFA vs Seam side-by-side evaluation
- the visibility of canonical `F` as a guardrail readout during boundary judging
- the requirement for deterministic, attested outputs

### Gate6 does not yet claim

- KAGAMI proof
- nonassociative validation
- universal hallucination detection
- superiority in open-world settings

## 3. The New Primitive

The primary Gate6 output is not a scalar.

The primary Gate6 output is a token-step local object:

`O_t^(6) = (r_t, B_t, P_t, C_t, G_t, Sigma_t, flags_t)`

Where:

- `r_t` = effective local rank in `1..3`
- `B_t` = deterministic local orthonormal frame as an auxiliary representation
- `P_t` = projector onto the local span as the primary invariant representation
- `C_t` = coordinates of `V_t`, `Splus_t`, `Sminus_t` in the local frame
- `G_t` = `3 x 3` Gram matrix of the normalized raw observables
- `Sigma_t` = singular values of the local construction matrix
- `flags_t` = degeneracy, gauge, and rank-drop metadata

The key design principle is:

Gate6 emits objects first. Scalar scores are downstream consumers.

## 4. Why This Is The Right Level

Gate5 showed that scalar shaping at the boundary is exhausted.

That implies the remaining leverage is upstream:

- what vectors enter the transport computation
- what local structure is preserved
- what gauge freedoms are controlled
- what invariants are available before scalarization

Therefore the next mainline is:

1. build a local observation object
2. expose both invariant and transport-ready views of that object
3. compare downstream transport experiments without changing the object

This is the point where pale-ale stops being score hunting and becomes a deterministic local-geometry lab.

## 5. Two-Lane Output Philosophy

Gate6 should emit two compatible lanes from the same local object.

### Lane A: Native object lane

For research and future downstream motif and field work:

- full local frame, projector, coordinates, and flags
- minimal loss of local geometry
- suitable for future motif families and field aggregation

### Lane B: Compatibility lane

For immediate benchmarking against Gate5:

- local coordinates padded into a canonical 8D compatibility view
- lets us run the same transport motif as Gate5 while changing only the boundary
- keeps the comparison honest

This is critical.

Gate6-A should not change the transport law and the observation boundary at the same time in a way that prevents attribution.

The first Gate6 experiment is:

same motif, new boundary

not:

- new motif plus new boundary
- new benchmark plus new boundary
- new theory plus new boundary

## 6. Native Construction Principle

At each token step `t`, Gate6 begins from full hidden-space observables:

- `V_t`
- `Splus_t`
- `Sminus_t`

These live in the native hidden dimension `d_model`.

Gate6 constructs a local object from these vectors directly.

The point is not to preserve all of `R^d_model`.
The point is to preserve the geometry actually spanned at the current token step.

Since only three vectors are involved, the relevant local span has rank at most 3.

That means Gate6 does not need a global 8D compression at all.

It needs:

- a deterministic local span
- a deterministic gauge policy
- a transport-ready local coordinate chart

## 7. Why Projector Beats Basis

`B_t` is necessary but not primary.

A basis is fragile:

- sign flips
- column swaps
- near-collinear instability
- orientation jitter

The projector `P_t = B_t @ B_t.T` is the invariant object.

Therefore:

- `P_t` is the canonical mathematical object
- `B_t` is an auxiliary gauge-fixed realization
- downstream transport should depend on invariant structure as much as possible

This is the main guard against beautiful but non-reproducible local geometry.

## 8. First Gate6 Goal

Gate6-A has one job only:

Replace FWHT-8D with a native local span object, then rerun the existing triad-loop experiment without changing the loop motif.

If that wins, the boundary mattered.

If that does not win, the problem is upstream of scalar shaping but not solved by native local span alone.

Either result is valuable.

## 9. What Counts As Progress

Gate6 is a success if at least one of the following happens:

- seam false spikes decrease relative to Gate5 FWHT-8D
- defect localization does not materially collapse
- rank and degeneracy behavior becomes more interpretable
- metric redundancy decreases because the object is richer

Gate6 is not a success merely because:

- the math sounds deeper
- the object is more elegant
- the notation looks more geometric

## 10. Current Mainline Placement

The cleanest current reading is:

- Gate1-3 = exploratory probes
- Gate4 = canonical observation sink
- Gate5 = transport feasibility study on compressed proxies

The current post-Gate5 mainline workstreams are:

- observation redesign
- transport motif comparison
- field aggregation
- benchmark expansion

This RFC family binds the first of those workstreams.
Its current working label is `Gate6-A`.

This document does not pre-freeze later names such as `Gate7` or `Gate8`.
If the observation-redesign workstream hardens into a new canonical stage, that broader naming update must be made explicitly across the architecture docs.

## 11. Current Standing Snapshot

The empirical standing of the Gate6 workstream is now fixed separately in:

- [`10_GATE6_STANDING_AND_OUTCOME.md`](10_GATE6_STANDING_AND_OUTCOME.md)

The architecture-level consequence is:

- the repo now carries an `operational candidate`
- and a separate `research north star`

These do not need to be the same consumer.

At the current snapshot:

- operational candidate = `gate6f`
- research north star = `gate6h`

This is the intended reading until a later workstream either unifies them or explicitly replaces one of them.
