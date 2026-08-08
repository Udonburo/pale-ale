# Gate12C-2 Development Checkpoint

Status date: 8 August 2026

Code checkpoint: [`42b1f3b`](https://github.com/Udonburo/pale-ale/commit/42b1f3bdbdac4138a7eb8d2b30c8bff25436d302)

Epistemic status: development infrastructure and payload identity verified;
scientific calibration not yet decided

## Short version

Gate12C-2 is building a measurement instrument for distinguishing a structural
process signal from artifacts introduced by the comparison procedure itself.
The current checkpoint contains the synthetic controls, graph-constrained null
machinery, decision hierarchy, deterministic execution infrastructure, and a
verified read boundary for an existing development payload.

The existing payload contains nine configurations, 768 outer experiments, 768
shards, and nine indices. A production extractor and a separately implemented
verifier agreed on all four frozen commitment families for every
configuration: 36 of 36 comparisons matched, with no protected-surface
mutation.

That is an execution and evidence-lineage result. It is **not** a scientific
calibration result and does not establish that the proposed observable is
valid, useful, or ready for held-out use.

## Why Gate12C-2 exists

Gate12C-1 tested whether compressed-overlap parenthesization sensitivity on a
frozen LLM replay-artifact surface exceeded a spectrum-preserving orientation
comparison. It returned no directional support under its predeclared
hierarchy. More importantly for the next stage, that comparison could not by
itself separate a process-originating defect from inflation caused by breaking
shared-node graph coupling in the null construction.

Gate12C-2 therefore treats measurement identification as a prerequisite. It
asks whether a graph-constrained comparison procedure can:

1. control false promotion under a realizable true-null regime;
2. detect a known-direction effect generated upstream of the test statistic;
3. identify inflation caused specifically by a graph-unconstrained comparison;
4. remain stable under nested accepted-valid inner-draw prefixes; and
5. carry the complete 24-endpoint decision hierarchy used by the intended
   claim.

## Frozen development controls

The development laboratory separates three roles:

- **S0 — true null:** graph-realizable exchangeable inputs used to measure the
  false-promotion behaviour of the complete pipeline.
- **S1 — known-direction control:** a graph-realizable end-to-end mechanism
  that introduces a graded effect upstream of the statistic.
- **S2 — null-inflation stressor:** an intentionally graph-unconstrained
  orientation comparison used to test whether the instrument attributes a
  direction to null inflation rather than to the observed process.

S2 is a diagnostic stressor, not a valid null candidate. The only initially
active null candidate is **N1**, a role-constrained frame reassignment that is
checked for joint graph realizability. N2 and N3 remain closed; they are not
parallel candidates from which the most favourable result may be selected.

## What is implemented

- FP64 object-reference and batched numerical paths;
- independent direct and block-Gram realizability checks;
- S0, graded S1, and paired S2 development generators;
- deterministic seed namespaces and accepted-valid draw ordering;
- nested 255 / 511 / 1023 accepted-valid draw prefixes;
- the full 12-case by 2-rank endpoint hierarchy;
- endpoint, family-wise, run, and claim-level decision plumbing;
- sharded, resumable, merge-order-invariant execution infrastructure;
- strict missing, degenerate, nonfinite, and coverage accounting;
- a separate Process Triage development line with outcome firewalls and
  grouped evaluation infrastructure.

Useful implementation entry points include:

- [`tools/gate12c2_synthetic_lab.py`](../../tools/gate12c2_synthetic_lab.py)
- [`tools/gate12c2_draw_stability.py`](../../tools/gate12c2_draw_stability.py)
- [`tools/gate12c2_n1_fidelity.py`](../../tools/gate12c2_n1_fidelity.py)
- [`tools/gate12c2_development_shards.py`](../../tools/gate12c2_development_shards.py)
- [`tools/gate12c2_original_baseline_commitments.py`](../../tools/gate12c2_original_baseline_commitments.py)
- [`tools/verify_gate12c2_original_baseline_commitments.py`](../../tools/verify_gate12c2_original_baseline_commitments.py)
- [`tools/process_triage_evaluator.py`](../../tools/process_triage_evaluator.py)

The corresponding `tools/test_gate12c2_*` and `tools/test_process_triage_*`
files contain unit, property-style, lineage, and adversarial regression tests.

## Verified Task 2 boundary

The current machine-verification chain reports:

| Check | Verified status |
| --- | ---: |
| Configurations | 9 / 9 |
| Outer experiments | 768 / 768 |
| Shards | 768 / 768 |
| Indices | 9 / 9 |
| Four commitment families across configurations | 36 / 36 matched |
| Commitment mismatches | 0 |
| Partial or temporary payload artifacts | 0 |
| Unexpected payload artifacts | 0 |
| Protected-surface mutations | 0 |
| Scientific values emitted during verification | false |
| Stability analysis authorized | false |

The four independently rederived commitment families are:

- outer-experiment ID surface;
- result-commitment surface;
- scientific-projection surface; and
- timing-excluded semantic-index commitment.

These commitments establish that the extractor and independent verifier read
the same complete computational surface. They do not evaluate the scientific
direction of that surface.

The raw 768-outer payload and internal execution receipts are not part of this
public repository. The public checkpoint therefore exposes the implementation
and claim boundary, not a standalone reproduction package for that payload.

## What remains open

The checkpoint does **not** claim any of the following:

- a selected production inner-draw count;
- a passed S0 false-promotion gate;
- a passed S1 power gate;
- a passed S2 inflation-identification gate;
- a valid or promoted N1 scientific result;
- a replacement resource-qualification result;
- a real held-out or confirmatory result;
- operational value on an external agent-process benchmark;
- an empirical result about a multi-agent population; or
- a safety, deployment, model-quality, correctness, deception, or causal
  guarantee.

The original aggregate resource evidence is permanently indeterminate because
the original long run completed its payload but failed during closeout. The
payload was subsequently sealed without mutation and its baseline commitments
were independently verified. A replacement resource qualification is a
separate operational question; it is not silently treated as completed here.

## Planned scientific sequence

The next scientific sequence is intentionally short:

1. read the verified baseline commitments through a bounded analyzer;
2. select the smallest inner-draw prefix satisfying the frozen stability
   criteria, without using favourable FPR or power as the selection rule;
3. run and judge development S0, S1, and S2 calibration;
4. retain N1 only if it passes its frozen realizability, calibration, and
   nuisance-fidelity gates; and
5. consider locked or external evaluation only after those decisions are
   closed.

## Relevance to multi-agent safety

Failures in agent networks can propagate through interaction, delegation, and
shared state. A graph-aware detector can appear to identify such propagation
while actually responding to an invalid comparison that destroys the same
coupling structure it is meant to measure. Gate12C-2 addresses that measurement
problem: it tests whether direction can be attributed to the observed process
rather than to the instrument's null construction.

This is potentially useful for system-level monitoring, cascade attribution,
and review triage in multi-agent deployments. The present checkpoint is still
an LLM-artifact development testbed, however. Extending it to realistic,
multi-principal agent populations is proposed future empirical work, not an
already earned result.

## Related public surfaces

- [Gate12C-1 negative-control technical note](../../papers/gate12c-negative-control/README.md)
- [Trace Triage live demo](https://pale-ale-trace-triage.vercel.app/)
- [Trace Triage source and claim boundaries](../../apps/trace-triage-demo/README.md)
- [Gate12B public-facing summary](gate12b-public-facing-summary.md)
- [Repository orientation](../../ABOUT/README.md)
