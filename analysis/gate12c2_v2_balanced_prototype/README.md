# Gate12C-2 v2 balanced N1 prototype

This directory contains one bounded, post-locked synthetic development study.
It asks whether exact donor balancing reduces finite-draw instability enough to
justify writing a v2 qualification specification, while checking that the
changed assignment design does not worsen measured N1 geometry.

It does **not** revise the frozen `RETIRE_OR_DEMOTE` result, authorize a fresh
locked suite, or open a real held-out surface.

## Fixed comparison

- schedules: retained iid derangements versus randomized balanced
  1-factorization cycles;
- fresh synthetic configurations: `S0`, `S1_PRIMARY`, and `S2`;
- surface: 12 independent datasets, six representative cases, `q=1,2`;
- draws: 44 per schedule, organized as four complete 11-draw cycles;
- primary precision evidence: disagreement between the independent first and
  second 22-draw halves;
- secondary precision evidence: complete-cycle bootstrap MCSE at 22 and 44
  draws;
- fidelity evidence: edge, product-spectrum, block-Gram, cross-feature
  correlation, donor-collision, realizability, and S2 spectrum controls;
- hard caps: 216 shards, 900 cumulative seconds, and 120 MB.

The only terminal decisions are `ADVANCE_TO_V2_SPECIFICATION` and
`STOP_N1_OR_REDESIGN_WITHOUT_NEW_LOCKED_SUITE`. Neither decision authorizes
locked or real-data execution.

## Completed result

The bounded run completed on 2026-08-13 and independently validated as
`STOP_N1_OR_REDESIGN_WITHOUT_NEW_LOCKED_SUITE`. Exact donor balancing preserved
the declared fidelity controls but improved independent 22-draw-half
reproducibility in only one of three configurations. The scientific result and
claim boundary are in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Execution

Run from the repository root after committing the implementation:

```powershell
python -m analysis.gate12c2_v2_balanced_prototype.prototype `
  --spec analysis/gate12c2_v2_balanced_prototype/prototype_spec.json `
  --locked-study "$env:GATE12C2_LOCKED_RESULT_ROOT\study.json" `
  --output $env:GATE12C2_PROTOTYPE_OUTPUT
```

Validate the completed output independently:

```powershell
python -m analysis.gate12c2_v2_balanced_prototype.prototype `
  --spec analysis/gate12c2_v2_balanced_prototype/prototype_spec.json `
  --output $env:GATE12C2_PROTOTYPE_OUTPUT `
  --validate-only
```
