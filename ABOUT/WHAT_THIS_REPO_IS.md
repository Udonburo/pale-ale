# What this repository studies

`pale-ale` is Aoi Kawasaki's research repository for exact stochastic simulation
and structural measurement of learned systems. Its central interest is how a
representation affects what a computation preserves, reveals, and costs.

The published work follows two lines.

## Exact representations for stochastic computation

The Array-RQMC studies examine which parts of randomized inputs and ranked
populations a simulator needs to represent explicitly.

- [Exact Binary Projections for Array-RQMC](../publications/binary-array-rqmc/README.md)
  characterizes a consumed joint binary law and maintains state order while
  preserving finite-seed paths.
- [Exact State-Ranked Array-RQMC](../publications/exact-state-ranked-array-rqmc/README.md)
  propagates a realized population through primal or dual counts, recovers its
  next rank intervals, analyzes shared transformation demand, and compares six
  compiled executors. It has been submitted to MCMA; the public version remains
  an author preprint.

The second paper connects an exactness guarantee to an implementation decision:
what can be shared, what still needs work, and which executor is faster in the
reported conditions. Its companion retains the measured source, original
observations, corrected wrappers, and reproduction commands.

## Structural observation and measurement of learned systems

The earlier studies define observation surfaces on language-model replay
artifacts, formulate transport and closure consistency, and test the boundaries
of the resulting measurements.

The [FP32 replay study](../publications/structural-replay-fp32/README.md)
provides evidence under a fixed precision and execution regime. The
[transport formulation](../publications/transport-first-defect-telemetry/README.md),
[observer-relative audit](../publications/observer-relative-closure-signatures/README.md),
and [parenthesization-defect null test](../publications/compression-interleaved-parenthesization-defects/README.md)
address distinct mathematical and empirical questions.

Later work tests [iterative capability](../publications/local-mapping-without-iterative-closure/README.md)
and [measurement reproducibility](../publications/sensitivity-without-reproducibility/README.md).
The distinction between detecting a controlled change and reproducing a
measurement under fresh estimation is part of the scientific result.

These are claims about specified models, inputs, and observation regimes.
The paper and its evidence determine the scope of each conclusion.

## How to use the repository

| Layer | What it provides |
| --- | --- |
| [Publications](../publications/README.md) | Citable records, archived packages, DOIs, and distribution links. |
| [Papers](../papers/) | Manuscript sources and computational companions. |
| [Analysis](../analysis/) and [docs](../docs/) | Technical reports, checks, and reproduction guides. |
| [Tools](../tools/), [src](../src/), and [crates](../crates/) | Research implementations and utilities. |
| [Apps](../apps/) | Interactive prototypes, including Amber. |
| [Workstream](../workstream/README.md) | The numbered history of earlier investigations. |

Use the publication-specific entry point to read or reproduce a result.
The [Workstream and Gate guide](WORKSTREAM_AND_GATES.md) explains historical
names when they appear in code or reports. Published releases preserve their
original identities; new working changes do not revise archived observations.
