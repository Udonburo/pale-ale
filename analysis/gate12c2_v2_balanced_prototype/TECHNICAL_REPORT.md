# Balanced donor exposure does not rescue Gate12C-2 N1 stability

Date: 2026-08-13

Status: completed synthetic development result

Decision: `STOP_N1_OR_REDESIGN_WITHOUT_NEW_LOCKED_SUITE`

## Answer first

Exact donor balancing is not a sufficient repair for N1. A randomized
1-factorization schedule eliminated donor-exposure imbalance and preserved the
measured edge, product-spectrum, block-Gram, collision, realizability, and S2
spectrum controls. It did not produce a consistent finite-draw precision gain.

Across independent first and second 22-draw halves, balanced sampling reduced
the median dataset-level disagreement only for S2. It slightly missed the
development target for S1_PRIMARY and worsened S0. At 44 draws, the balanced
schedule also failed the predeclared decision-precision requirement because no
S0 dataset had an uncertainty interval contained inside the development
equivalence margin.

This result rules out uneven iid donor exposure as the main explanation for the
locked instability. It does not prove that the exact residual decomposition is
wrong, nor does it prove a mathematical no-go for every graph-realizable null.
It leaves the adequacy of N1 geometry and of the current estimand unresolved.

## Fixed development surface

- fresh synthetic data only: S0, S1_PRIMARY, and S2;
- 12 independent datasets per configuration;
- six representative cases and q=1,2;
- iid and balanced schedules on the same observed cohorts;
- 44 draws per schedule, split into four independently randomized 11-draw
  cycles;
- 216 physical shards;
- primary precision comparison: independent first versus second 22-draw
  halves;
- secondary precision comparison: complete-cycle bootstrap MCSE;
- execution caps: 900 cumulative seconds, 120 MB, and 216 shards.

The surface and thresholds were fixed before shard generation. The run finished
in 403.929 seconds and occupies 25,118,457 bytes across 235 files, including
state and manifest.

## Results

### Primary reproducibility result

The criterion required a balanced-to-iid median disjoint-half error ratio at or
below 0.80 in at least two of the three configurations.

| Configuration | iid median error | Balanced median error | Ratio | Bootstrap 95% interval | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| S0 | 0.046961 | 0.066268 | 1.411 | 0.484–3.909 | worse |
| S1_PRIMARY | 0.124902 | 0.100869 | 0.808 | 0.324–1.731 | target missed |
| S2 | 0.037234 | 0.027163 | 0.730 | 0.282–1.723 | target met |

Only one configuration met the target. Every interval includes 1, so the study
does not establish a configuration-independent variance reduction. Endpoint
results were heterogeneous: balancing helped some case/q combinations and
harmed others, with no uniform q or ambient-dimension pattern.

### Dataset-level decision precision

At 44 draws, the criterion required at least 75% clear decisions in every
configuration.

| Configuration | iid clear fraction | Balanced clear fraction | Balanced median effect | Balanced median cycle MCSE |
| --- | ---: | ---: | ---: | ---: |
| S0 | 0.000 | 0.000 | 0.034323 | 0.044146 |
| S1_PRIMARY | 1.000 | 1.000 | -2.835532 | 0.063387 |
| S2 | 1.000 | 0.917 | 0.131415 | 0.024279 |

The failure is concentrated in the no-effect safety configuration. The
balanced S0 center uses most of the ±0.05 development equivalence budget before
uncertainty is added, and its cycle MCSE is not lower than iid. Simply adding a
balanced schedule therefore does not yield a qualified null-centered
instrument.

### Fidelity and controls

All non-precision gates passed.

| Gate | Result | Evidence |
| --- | --- | --- |
| Exact donor exposure | pass | balanced maximum exposure CV and range were both 0 |
| Donor-collision fidelity | pass | pair-rate differences ≤0.00229; triple-rate differences ≤0.00050 |
| S0 measured geometry non-worsening | pass | median quantile deltas ≤0.00169; median correlation deltas ≤0.01295 |
| Joint realizability | pass | maximum error 0 |
| S2 spectrum preservation | pass | maximum error 1.22e-15 |

These controls matter because they show that the negative precision result is
not explained by a broken implementation or a gross change in the measured
null geometry. They do not establish equivalence for every unmeasured aspect of
shared-node geometry.

## Scientific interpretation

The retained audit identified finite-draw precision as the dominant diagnosed
mechanism in the consumed locked result while declining to exonerate N1
geometry. This prototype narrows that statement:

> Finite-draw instability is real, but it is not primarily repaired by equalizing
> donor exposure through balanced derangement cycles.

The synthetic signal configurations remain directionally separable, yet the
null-centered S0 regime remains too variable at the independent-dataset grain.
Good power and injected-inflation identification therefore still do not make
N1 a stable scientific instrument.

The historical Gate12C-1 empirical pattern remains exactly where it was:
observed under the old graph-unconstrained null, but not identified. This
prototype used fresh synthetic data and did not open a new real held-out
surface.

## Consequence

Do not spend another attempt on more iid draws, another balanced schedule, a
retuned 0.30 threshold, or a replacement locked suite. Any scientifically
distinct continuation must change the inferential object, not just the draw
allocator. At minimum it would need:

1. one cohort-level statistic rather than treating graph blocks as independent;
2. an explicit conditional target for product and shared-node geometry, with a
   demonstrated equivalence margin;
3. an error budget derived from the downstream decision rather than inherited
   from the failed 9-versus-15 comparison; and
4. development evidence that the no-effect regime is stable before any fresh
   held-out observation is opened.

The exact tail–propagation–cancellation decomposition remains usable. N1 as
currently formulated does not advance.

## Reproducibility

The durable result is retained outside the repository at:

```text
<retained-results-root>/gate12c2-v2-balanced-prototype-v0.1
```

Key hashes:

```text
spec:     5d30c4b2090a6271e0d77a9d2e990c026a1e8bd540a483a0d1b5ba91bc7d995c
analysis: 443c4febb48f2bc3c73828fe988a6c1b08e6a401e1800ba38f99c56c11f7f475
manifest: 36e9a956a482a882e8a4aca6f6e7e86eae4dc74cf19697adbd82902d96d9aaf6
```

The independent validator rechecked the locked source hash, implementation
hashes, complete file surface, all manifest hashes, all 216 shard identities
and assignment reconstructions, geometry summaries, controls, and table row
counts. It returned the same terminal decision.
