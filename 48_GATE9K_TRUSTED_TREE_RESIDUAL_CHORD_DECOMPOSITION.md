# Gate9K Trusted-Tree / Residual-Chord Decomposition

Status: decomposition hypothesis spec, first implementation landed and first logging smoke execution recorded
Role: Gate9K trusted-tree / residual-chord hypothesis spec, not operator opening or metric settlement
Date: 2026-03-21

Gate9K proceeds from:

- `46_GATE9J_DISTRIBUTED_UNDERACTIVATION_AUDIT.md`
- `47_GATE9J_DISTRIBUTED_UNDERACTIVATION_SMOKE.md`

The first Gate9K logging consumer now exists in:

- `tools/run_gate9k_trusted_tree_residual_chord_logging.py`

The first tracked Gate9K logging smoke read is now recorded in:

- `49_GATE9K_TRUSTED_TREE_RESIDUAL_CHORD_LOGGING_SMOKE.md`

## 0. Context And Blocker

Gate9I and Gate9J have now named the active support-anchor blockers more sharply.

What is currently known is:

- `support_anchor_cleaner_cell_dominance` still blocks redesign admission
- `distributed_underactivation` is real on the support-anchor conflict side
- that underactivation is no longer treated as family-wide
- the active narrow branch is now `distributed_consistent_answer_compression`

This means the next honest move does not need:

- a new metric family
- graph-wide operator opening
- a return to scalar rescue

It may instead test a stricter decomposition hypothesis:

- whether a trusted-tree / residual-chord view can suppress cleaner-cell dominance structurally enough to make the remaining anomaly-side branch more legible under the frozen law

## 1. Hypothesis

The Gate9K hypothesis is:

- extract a trusted tree or trusted forest from edges already treated as operationally safe
- treat transport along that trusted backbone as a gauge-fixed identity baseline
- read only the residual anomaly-side chords as the active non-closure burden

In that view:

- trusted structure is pushed into baseline transport
- anomaly-side deviation is pushed into residual chords
- cleaner-cell dominance may weaken if the clean support backbone stops acting as the main cost-bearing path

This is not a scalar masking proposal.

It is a decomposition hypothesis about where transport burden should live.

## 2. Scope

Gate9K studies only:

- decomposition of the existing graph-gauge law into trusted-tree transport plus residual chords
- whether that decomposition reduces cleaner-cell dominance on the support-anchor line
- whether distributed underactivation becomes more legible on the residual side

It does not:

- open any graph-wide operator
- settle a new metric family
- reopen global redesign
- change the frozen node/edge law

## 3. Trusted Edges

The initial trusted-edge policy is intentionally narrow.

Trusted edges are:

- `temporal_transition`
- `support_anchor`

These edges define the first candidate trusted tree or trusted forest.

The point is not to declare them metaphysically pure.

The point is:

- they are the current best candidate backbone for safe transport baselining

## 4. Residual Chord Set

Residual chords are the edges not absorbed into the trusted tree or forest.

The initial anomaly-side candidate set is:

- `conflict_anchor`
- optionally `answer_projection` when needed to close the declared comparison path

Gate9K therefore asks whether anomaly-side burden becomes more concentrated when those chords are read against a trusted backbone rather than against the full support-dominated graph.

## 5. Doctrine Continuity

Gate9K must remain explicit on one doctrinal point:

- trusted-tree suppression is not allowed to degenerate into scalar masking

In practice that means:

- trusted transport must be represented as gauge-fixed identity along declared tree paths
- projector / transport structure must remain intact
- the implementation may not simply zero out edge weights or post-mask scalar outputs

If the decomposition becomes scalar suppression, Gate9K fails.

## 6. Falsifiers

Gate9K must keep the following falsifiers explicit.

### 6.1 Tree Choice Dependence

If small perturbations of tree selection under the same trusted-edge policy produce unstable residual verdicts, the hypothesis fails.

Perfect invariance is not required on first pass.

Practical stability is required.

### 6.2 Distributed Underactivation Not Improved

If the trusted-tree baseline does not make the distributed anomaly-side branch any more legible, the hypothesis fails.

The point is not merely to re-express the same underactivation in new language.

### 6.3 Cleaner-Cell Dominance Remains

If cleaner-side structure still dominates the residual readout after trusted-tree decomposition, the hypothesis fails.

The decomposition must not leave the same blocker untouched while renaming it.

### 6.4 Scalar-Masking Degeneration

If the decomposition is implemented as scalar edge masking rather than gauge-fixed identity transport on trusted paths, the hypothesis fails.

## 7. Verdict Granularity

Gate9K must not collapse into one headline scalar.

Its verdict must remain granular at least along these axes:

- trusted-edge policy used
- residual chord set used
- cell-level effect on cleaner-cell dominance
- cell-level effect on distributed underactivation
- tree-choice stability
- scalar-masking degeneration check

The verdict must therefore remain a structured audit, not a one-number bypass score.

## 8. Operator Admission Non-Promotion

Gate9K does not promote operator admission.

Even a positive Gate9K result would earn only:

- permission to treat trusted-tree / residual-chord decomposition as a serious local hypothesis

It would not earn:

- graph-wide operator opening
- spectral branding
- field language
- admission reversal by narrative shortcut

So Gate9K begins and ends under:

- `operator admission still denied`

## 9. What This Spec Can Earn

At most, Gate9K can earn the right to say:

- a trusted-tree / residual-chord decomposition is worth auditing on the frozen law as a local bypass hypothesis

It still does not earn:

- implementation success
- blocker resolution
- operator opening

## 10. Next Steps

The first implementation obligations, if Gate9K is executed, are:

1. declare and log the trusted-edge policy explicitly
2. declare and log the residual chord set explicitly
3. emit verdict slices with tree-choice sensitivity visible
4. refuse any implementation that achieves suppression by scalar masking

## 11. Current Memory Hook

The shortest acceptable sentence is:

- test trusted-tree decomposition as hypothesis, not as rescue metric
