# Gate13 Track C — Review 2 candidate protocol

## 1. Status and scope

This is one complete **estimand-design candidate** for human review. Its
terminal state is `REVIEW2_READY_FOR_HUMAN_AUTHORIZATION`; Track C itself,
A3, and formal Gate13 remain closed. Nothing here is an execution
authorization.

No Track C scientific execution was performed in Review 2. There was no Modal
call, model download/load, GPU allocation, model forward, activation
collection, or inspection of a Track C coupling outcome. The only empirical
calculation used existing Qwen3.6 fresh-B activations for the prespecified
8/12/16/24 downsampling grid. The sensitivity calculation was model-free.

The protocol resolves the previous analysis-unit mismatch as follows:

> One independent fresh naturality-square block is one row of the analysis.
> Its cloud-level operator packets are measurements used to construct one
> representation scalar, and its behavior episodes are measurements used to
> construct one functional outcome. Halves, layers, nodes, activation samples,
> paths, and episodes are never treated as independent analysis units.

## 2. Frozen target

The sole target is `Qwen/Qwen3.6-27B` at the exact frozen revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. A model substitution, revision
change, layer change, or tokenizer/template change invalidates provenance and
must fail closed.

The representation layers are exactly `[21, 43, 62]`, with equal weight. They
were not selected in Review 2.

## 3. Experimental unit and block construction

The experimental unit is one independent fresh `natural_rule_v1` naturality
square. Each planned block has:

- one fresh opaque codebook;
- fresh demonstration identities;
- fresh map and behavior episode seeds;
- one prospectively assigned rollout depth;
- two disjoint map-estimation halves;
- a behavior-evaluation set fully disjoint from both map halves; and
- the same block-level codebook and demonstration semantics across all three
  sets.

No codebook, demonstration identity, or episode seed may be shared across
blocks. No new template variant is allowed. The codebook, demonstrations,
seeds, block identifier, and depth must all be written into a hash-bound future
block manifest before the first authorized Track C call. Review 2 does not
create that manifest or authorize those calls.

Difficulty is rollout depth only. The candidate has 20 planned blocks, five at
each depth `2`, `4`, `6`, and `8`. Depth is not tuned to obtain any target
failure rate. The planned depth counts and all block identities are immutable
after any outcome is generated. Failed blocks are not replaced, and no blocks
may be added after outcome observation.

Each map half contains 24 paired samples at each of the five frozen square
nodes. Thus a full map block has `2 × 5 × 24 = 240` activation forwards. Map
samples are paired across nodes within a half but remain subsamples of their
single parent block.

Each behavior block has 24 paired episodes. Within an episode, both externally
equivalent paths use the same episode semantics and terminate in the same
forced-choice endpoint probe. Map identities and behavior identities are fully
disjoint.

## 4. Primary representation estimand

Let `b` index blocks, `l ∈ {21,43,62}` index frozen layers, and
`h ∈ {1,2}` index the map-estimation half. All frames and fitted maps below are
native to half `h`.

For each square node, estimate the frozen rank-four centered frame from half
`h`. Fit the frozen ridge maps on paired coordinates, compose the two maps
along path `P`, and separately compose the two maps along path `Q`. Both paths
have the same source and target. Define the half-native return action

```text
Delta_blh = P_blh - Q_blh.
```

Do not compare or multiply `Delta` matrices from different halves. Instead,
cross-fit the weighting distribution into the gauge in which `Delta_blh` was
estimated:

1. take the source-node activations from the opposite half `-h`;
2. project them into half `h`'s source frame;
3. center those projected coordinates and form their held-out covariance
   `Sigma_blh`; and
4. compute

```text
R_blh = tr(Delta_blh Sigma_blh Delta_blh^T) / tr(Sigma_blh).
```

The primary block scalar is the symmetric, equal-layer aggregate

```text
R_b = (1/3) sum_l [(1/2) sum_h R_blh].
```

This construction is gauge invariant: under a source rotation `G` and target
rotation `H`, `Delta` transforms as `H^T Delta G` and the covariance in that
same source gauge transforms as `G^T Sigma G`; cyclic trace invariance removes
both rotations. This argument applies within one half-native coordinate
system. It does not license a product of matrices expressed in unrelated
half-specific gauges.

`R_b` is a source-distribution-weighted representation-geometric discrepancy.
It is not itself a causal transition or evidence for a hidden causal register.

## 5. Primary behavioral estimand

For behavior episode `e` in block `b`, let `m_beP` and `m_beQ` be the
correct-versus-other logit margins at the common endpoint probe after the two
externally equivalent paths:

```text
m_bep = logit(correct)_bep - logit(other)_bep.
```

The primary behavioral target is the block-level RMS path discrepancy

```text
Y_b = sqrt((1/E_b) sum_e (m_beP - m_beQ)^2),   E_b = 24.
```

The endpoint scoring rule, correct/other labels, and tokenization checks must
be fixed in the future block manifest before calls. Missing episodes are not
imputed, retried after outcome inspection, or replaced. A block without its
complete prespecified behavior packet fails integrity and cannot enter the
primary analysis.

## 6. Frozen nuisance model

Define the block-level mean path-averaged margin

```text
M_b = (1/(2E_b)) sum_e (m_beP + m_beQ).
```

The nuisance-only design is exactly:

```text
intercept + rollout depth + M_b.
```

There are no other covariates, interactions, nonlinear depth terms, model
selection steps, or outcome-dependent transformations. On the final qualified
cohort, depth, `M_b`, and `R_b` are each centered and divided by their sample
standard deviation. Zero variance, deficient rank, or a condition number above
`1e6` fails closed. These invertible column transformations do not change the
OLS predictions or the primary statistic.

## 7. Primary prediction statistic

For every qualified block, fit both models on all other qualified blocks and
predict the held-out block:

```text
Nuisance: Y_b ~ 1 + depth_b + M_b
Full:     Y_b ~ 1 + depth_b + M_b + R_b
```

Pool the squared held-out residuals only after every block has been held out
once. Freeze

```text
T = 1 - SSE_full_LOBO / SSE_nuisance_LOBO.
```

`T` may be negative and is never truncated. Larger values indicate greater
held-out incremental predictive value. The implementation uses the exact OLS
PRESS identity, which is algebraically identical to refitting each
leave-one-block-out fold. There is no episode-level, half-level, or layer-level
replication in this test.

## 8. One null test

The only primary null procedure is a one-sided Freedman–Lane residual
permutation that preserves the nuisance structure:

1. fit the frozen nuisance model to all qualified blocks;
2. retain its fitted values and residuals;
3. permute residuals only within exact rollout-depth strata;
4. add permuted residuals to the nuisance fitted values; and
5. rerun the complete nuisance and full LOBO prediction pipeline, including
   both held-out SSEs and their relative reduction.

Use exactly 99,999 permutations with seed `13602027`. The p-value is

```text
p = (1 + count(T_perm >= T_observed)) / 100000.
```

The confirmatory threshold is one-sided `alpha = 0.05`. There is one primary
representation feature, one primary outcome, and one primary test. Spectra,
`S`, `H_path`, `H_edge`, binary accuracy, and broken-square outputs are
secondary diagnostics or validity controls and receive no confirmatory
p-values.

## 9. Fresh-distribution validity gates

All representation gates are evaluated without revealing `Y_b`. The qualified
block set is finalized before the primary behavior outcome is unsealed.

| Gate | Frozen requirement |
|---|---|
| Split-half qualification | For the two layer-averaged half features, `abs(log(max(R_b1,1e-12)/max(R_b2,1e-12))) <= log(4)` |
| Rank and conditioning | Every half/layer frame, edge, and path has rank 4; every edge/path condition number is at most `1e6` |
| Exact-square packet reproducibility | At every layer, the maximum P/Q singular-spectrum RMSE between halves is at most `0.20` |
| Broken-square sensitivity | At every layer, `min(broken delta over halves) > max(2F, F+0.05, max(exact delta over halves)+0.05)`, where `F` is the split-half spectral floor |
| Minimum qualified blocks | At least 16 overall and at least 4 at each rollout depth |
| Analysis design | Nuisance and full designs have full rank, nonzero scaled columns, and condition number at most `1e6` |

A block must pass every per-block representation gate at all three frozen
layers. The broken square remains a validity control and never becomes the
primary feature or outcome. If the minimum-qualified gate fails, the campaign
stops without adding blocks, changing layers, altering thresholds, tuning
difficulty, or changing the estimand.

## 10. Outcome-blind sample-size evidence

The fixed-grid downsampling used only the existing Qwen3.6 fresh-B activation
NPZ files. For sizes below full support it used 24 deterministic replicates;
the full-support point has one available realization. Seed `13602026` gives
schedule hash
`78569ffac2a3cc7826482cf3e3f51e77a4a71413ac34baab843fadfd17ff44f3`.

| Samples/node/half | Complete pass | Median absolute log error to n=24 | Median block half log-ratio | Median spectral floor |
|---:|---:|---:|---:|---:|
| 8  | 2/24 (0.083) | 1.630 | 1.329 | 0.162 |
| 12 | 4/24 (0.167) | 1.467 | 0.946 | 0.139 |
| 16 | 3/24 (0.125) | 1.196 | 1.596 | 0.115 |
| 24 | 1/1 (1.000)  | 0.000 | 0.204 | 0.071 |

The full-support existing-B scalar was `0.00018199685461385876`. Because n=24
has only one historical block, its 1/1 result is not an estimate of the future
block qualification probability. Likewise, the lower-grid replicate rates are
diagnostics of subsampling stability within one block, not independent-block
pass rates. The data support n=24 as the only eligible cloud size; they do not
justify a smaller cloud or a claim of future gate reliability.

## 11. Model-free sensitivity and selection

The design simulation used no model output. It crossed block counts
`16/20/24/28/32`, cloud sizes `8/12/16/24`, and behavior episodes
`8/16/24/32`, with 1,000 simulations per cell and effect-grid point. The seed
was `13602028`. The effect grid was latent nuisance-adjusted correlation
`0.2` through `0.9` in steps of `0.1`.

Declared planning noise was standardized representation noise SD `0.35` at
n=24, scaling as `sqrt(24/n)`, and standardized behavior noise SD `0.50` at
E=24, scaling as `sqrt(24/E)`. These are sensitivity assumptions, not estimates
from a Track C outcome and not predictions of the real effect distribution.

Under the n=24 empirical rule, reference RMS relative SE at most 0.15, and the
USD 65 planning ceiling, three cells were feasible:

| Blocks | Cloud | Episodes | Grid MDE at 80% | Planning upper USD |
|---:|---:|---:|---:|---:|
| 16 | 24 | 24 | 0.8 | 45.254 |
| 16 | 24 | 32 | 0.8 | 53.453 |
| 20 | 24 | 24 | 0.8 | 56.528 |

The frozen selection rule first minimizes the grid MDE, then maximizes planned
block count, then minimizes cost. It therefore selects **20 blocks, n=24, and
E=24**.

Sensitivity remains limited to large effects:

| Latent partial correlation | Power, 20 blocks | Power, minimum 16 blocks |
|---:|---:|---:|
| 0.2 | 0.077 | 0.075 |
| 0.3 | 0.139 | 0.128 |
| 0.4 | 0.259 | 0.186 |
| 0.5 | 0.398 | 0.309 |
| 0.6 | 0.560 | 0.431 |
| 0.7 | 0.778 | 0.598 |
| 0.8 | 0.925 | 0.814 |
| 0.9 | 0.991 | 0.962 |

The 95th-percentile model-free null thresholds for `T` were 0.1295 at 20
blocks and 0.1764 at 16 blocks. Simulation Monte Carlo error, assumed-noise
misspecification, and an unknown true effect make these numbers planning
sensitivities only. A null or imprecise future result cannot be repaired by
adding blocks or changing the estimand.

## 12. Forward and cost forecast

The map forecast is `20 × 2 halves × 5 nodes × 24 = 4,800` forwards. For a
block of depth `d`, 24 behavior episodes require
`2 paths × 24 × (d + 1)` forwards: `d` self-fed transition calls plus one
common endpoint probe per path. With five blocks at each depth:

| Component | Forecast forwards |
|---|---:|
| Map activations | 4,800 |
| Behavior paths and probes | 5,760 |
| Total scientific forwards | 10,560 |

The historical fresh-B provider cost was USD 1.02497632 for 240 forwards, or
USD 0.004270734667 per forward. Adding the historical acquisition/preflight
reference of USD 0.12368778 yields a linear forecast of USD 45.22264586. The
frozen 25% contingency produces a planning upper of USD 56.528307325, below a
declared future hard ceiling of USD 65.00.

This forecast does not allocate an A100, reserve budget, call Modal, or grant
execution authority. A future human authorization would have to bind the exact
budget and stop behavior. Runtime/provider drift beyond the contingency is a
stop condition, not permission to reduce samples or blocks after outcomes.

## 13. Conditional future order of operations

If—and only if—a separate human authorization is later issued, the order is:

1. verify every provenance hash and the exact frozen model revision;
2. instantiate and hash-bind all 20 blocks, both disjoint map halves, disjoint
   behavior episodes, and the prospective depth assignment before calls;
3. collect the map packets for all planned blocks;
4. run the frozen representation gates while behavior outcomes remain absent or
   sealed;
5. stop if fewer than 16 blocks or fewer than four blocks at any depth qualify;
6. evaluate the frozen behavior packets only for the finalized qualified set;
7. verify complete behavior packets and analysis-design gates; and
8. run the single frozen LOBO statistic and its 99,999-permutation null test.

No outcome may trigger a new block, replacement, threshold relaxation, new
template, layer change, alternate primary feature, alternate outcome, or
additional primary test.

## 14. Claim boundary

- Raw overlap and `R_b` remain representation geometry, not a causal
  transition.
- Visible-state use does not establish a hidden causal register.
- Qualification is not uniqueness.
- The protocol does not claim first use of naturality, holonomy, or
  intermediate-state edits.
- A positive primary result would establish only prospective block-level
  incremental prediction for this exact frozen target and distribution. It
  would not by itself identify a unique mechanism or a hidden causal register.

Prior-art collisions and the resulting wording constraints are recorded in
`track_c_prior_art_collision_matrix.md`.
