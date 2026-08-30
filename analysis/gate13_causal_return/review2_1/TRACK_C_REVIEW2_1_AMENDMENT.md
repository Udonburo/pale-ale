# Track C Review 2.1 prospective estimand amendment

Date: 2026-08-26
Scope: model-free estimand, null-calibration, rendering, and staged-execution design only

## 1. Status and historical boundary

This is a prospective amendment to the Review 2 candidate at commit
`4818d0cd313f09e9ae6304bae7c647424dcfcb9f`. It does not rewrite Review 2.
The Review 2 directory and the closed panel artifacts remain byte-for-byte
unchanged and are bound by the hashes in
`track_c_estimand_lock_candidate_v2.json`.

The immutable state is:

```text
PANEL_A2_AND_B_PASS = IMMUTABLE TERMINAL PANEL STATE
Track C             = CLOSED_PENDING_FRESH_PROTOCOL
A3                  = CLOSED
Formal Gate13       = CLOSED
```

Review 2.1 is not an execution authorization. Its scientific implementation and
final audit used only local Python/NumPy and made no Modal call, Qwen3.6 load,
GPU allocation, Track C forward, activation collection, hidden intervention,
Track C outcome generation/inspection, A3 work, prompt rescue, layer/model
change, or closed-artifact edit. During repository validation, however, an
over-broad Rust workspace test reached a cached, unrelated embedding-model
load/forward before it was interrupted. This violated the broader task
prohibition and makes the overall Review 2.1 package blocked even though the
scientific null audit passed. The exact incident is recorded in
`REVIEW2_1_REPORT.md` and the v2 lock.

## 2. Independent block design

One independent fresh `natural_rule_v1` naturality-square block is one and only
one analysis row. Episodes are repeated measurements used to construct a block
quantity; they are never independent statistical units.

The prospective design remains 20 blocks, with rollout depths `2, 4, 6, 8`
and five blocks assigned to each level. Eligibility requires at least 16 blocks
overall and at least four at every depth. Those counts are only an eligibility
floor: rank, conditioning, leverage, representation variance, nuisance SSE,
permutation support, and every scientific gate remain mandatory.

Every block has a fresh opaque codebook, fresh demonstration identities, fresh
map and behavior seeds, one prospectively assigned depth, two disjoint map
halves, and a behavior ledger disjoint from both map halves. The single
block-level codebook and demonstration semantics bind both stages. Blocks do
not share identities or seeds. No fresh template variant, outcome-adaptive
block addition, failed-block replacement, target-failure-rate tuning, or prompt
rescue is permitted.

## 3. Primary representation amplitude

For block `b`, frozen layer `l` in `[21, 43, 62]`, and training half `h`, fit
the rank-four return-action operator `Delta_blh` only in half-`h` native source
and target frames. Project source activations from the opposite half into the
half-`h` source frame, center them there, and form the held-out covariance
`Sigma_blh` in that same coordinate system. Define the descriptive energy

```text
E_blh = tr(Delta_blh Sigma_blh Delta_blh^T) / tr(Sigma_blh).
```

The one primary representation feature is the amplitude

```text
R_b = sqrt((1/3) sum_l ((1/2) sum_h E_blh)).
```

All six unsquared `E_blh` values and their mean remain descriptive packet
fields. They are not additional tests. Opposite-half source activations are
mandatory. `Delta_blh` and `Sigma_blh` must be expressed in the same
training-half source gauge; matrices from unrelated half-specific gauges may
never be multiplied. Orthogonal source/target gauge changes leave each energy
and `R_b` invariant. Raw overlap remains representation geometry, not a causal
transition.

## 4. Primary functional outcome

Before any future forward, freeze all 24 behavior episode IDs per block, both
path IDs, every transition/probe case ID, correct and other answer tokens, and
the common score slot. For episode `e` and path `p`,

```text
m_bep = logit(correct)_bep - logit(other)_bep
```

at the frozen common endpoint probe. The sole outcome is

```text
Y_b = sqrt(mean_e[(m_beP - m_beQ)^2]).
```

The exact episode ledger is all-or-nothing. No episode-level inference,
replacement, or post-outcome selection is allowed.

## 5. Leakage-free nuisance

The existing map packet schema records `target_state` and the two
`candidate_logits` for every required map forward. It therefore supports a
semantically direct competence score without a new forward or a separate
competence ledger. For the four exact-square nodes `N`, define

```text
C_b^M = (1/2) sum_h [ (1/(4*24))
        sum_{node in N} sum_{s=1}^{24}
        (logit(target)-logit(other))_{b,h,node,s} ].
```

This uses exactly 192 already-required exact-square map rows per block, weights
the two halves equally, and uses zero broken-square or behavior rows. Broken
square remains a validity control. The behavior ledger `E_b` cannot construct
any nuisance regressor. No disjoint future `K_b` ledger is required, so no
additional competence cost is introduced.

The nuisance columns are exactly:

```text
intercept
depth 4 indicator
depth 6 indicator
depth 8 indicator
map-derived competence C_b^M
```

Depth 2 is the reference. Numeric depth trends and behavior-derived mean
margins are forbidden.

## 6. Exact OLS and LOBO estimand

For every held-out block, center and divide `C_b^M` and `R_b` by their sample
standard deviations fitted only on that fold's training blocks. Fit the frozen
nuisance OLS and nuisance-plus-`R_b` OLS on those training blocks and predict
the held-out block. Aggregate squared errors once across blocks:

```text
T_LOBO = 1 - SSE_full^LOBO / SSE_nuisance^LOBO.
```

The directional coefficient `beta_R` is the raw-amplitude `R_b` coefficient
from the frozen full-cohort OLS
`Y ~ 1 + categorical_depth + C_b^M + R_b`. Its sign is unchanged by positive
centering/scaling. For every permuted outcome this full OLS coefficient is
refitted as well.

The result is positive only if all qualification and numerical gates pass and
all three conditions hold:

```text
T_LOBO > 0
beta_R > 0
one-sided permutation p <= 0.05
```

The fail-closed decision order is qualification/predictor gates, outcome
variance, nuisance SSE, positive incremental value, coefficient direction,
then significance. Frozen terminals are:

```text
INSUFFICIENT_QUALIFIED_BLOCKS
INSUFFICIENT_DEPTH_STRATUM
RANK_DEFICIENT_DESIGN
EXCESSIVE_LEVERAGE
INVALID_PERMUTATION_SUPPORT
NO_REPRESENTATION_FEATURE_VARIANCE
NO_OUTCOME_VARIANCE
DEGENERATE_NUISANCE_SSE
NO_POSITIVE_INCREMENT
WRONG_DIRECTION
NOT_SIGNIFICANT
```

None may be reframed as positive evidence.

## 7. Directional Freedman–Lane null

Fit the nuisance model on all qualified blocks, retain its fitted values, and
permute its residuals only within the four frozen depth strata. Add a permuted
residual vector to the nuisance fit to form each synthetic outcome. On every
synthetic outcome, refit both nuisance and full models throughout the complete
LOBO pipeline, with fold-local scaling. The implementation precomputes
design-only linear prediction operators from explicit fold fits; applying an
operator to a new synthetic outcome is algebraically the exact refit, not reuse
of observed predictions. Tests compare it against explicit OLS refits.

To implement the one-sided direction without a second test, define

```text
T_perm^+ = T_perm  if refitted beta_R_perm > 0
           -infinity otherwise.
```

Then, with `B=99,999`,

```text
p = (1 + #{T_perm^+ >= T_obs}) / (1 + B).
```

The root seed is `13602027`. Before Stage M, the finite schedule family is
fixed by the named SHA-256 derivation over this root seed and every possible
ordered outcome-blind qualified block-ID set. Map qualification selects exactly
one family member without using behavior. Minimum support is
`(4!)^4 - 1 = 331,775`, above 99,999.

There is one feature, one outcome, and one primary test. Spectra, `S`,
`H_path`, `H_edge`, binary accuracy, broken-square outputs, and the six energy
components remain secondary or validity-only.

## 8. Numerical, rank, and leverage gates

Every full-cohort design and every LOBO training design must have full column
rank, positive residual degrees of freedom, and 2-norm condition number at most
`1e6`. `R_b` and `Y_b` variance and nuisance LOBO SSE must exceed the frozen
relative floor `1e-12 * max(1, relevant squared scale)`.

For a design with `p` columns and `n` rows, every diagonal hat value must obey

```text
h_max <= min(0.80, 3p/n).
```

Three times the average leverage is a design-only dominance screen. The
absolute cap keeps `1/(1-h)` at or below five. It applies separately to the
nuisance and full designs in the full cohort and every LOBO training fold. It
was frozen before the final audit and stress-calibrated without Track C data.

## 9. Final null-calibration firewall

Development used seed `2026082601`; schedule development uses seed
`2026082602`. After code, generators, thresholds, terminal definitions, and
the audit seed were fixed, the final audit uses seed `2026082603` exactly once:

```text
2,000 independently seeded null datasets per eligible scenario
999 unique within-depth permutations per dataset
nominal alpha = 0.05
required empirical FPR interval = [0.035, 0.065]
```

The required scenarios are balanced 20, balanced 16, independent
outcome-blind eligible qualification masks, depth-stratified
heteroskedasticity, and below-threshold leverage stress. Every replicate applies
the complete joint positive rule. A separate 2,000-mask audit proves
ineligible masks terminate without outcome generation or testing. Two
minimal-eligible cases enumerate all 331,775 nonidentity permutations. Fixed
terminal cases cover rank deficiency, zero predictor/outcome variance,
degenerate nuisance SSE, excessive leverage, invalid support, and insufficient
counts. Fixed-seed reproducibility is required.

The committed calibration records the validator source SHA and threshold-lock
SHA. After the final audit is read, its algorithm, generator, seed, thresholds,
terminal definitions, and failed result cannot be changed or replaced. Any
required failure makes Review 2.1 terminally blocked.

## 10. Path surface and deterministic serialization

The intended P/Q difference is operation order only. Before the first future
forward, the frozen renderer must materialize all 480 P/Q behavior episode
pairs with the exact tokenizer and compare:

```text
canonical natural_rule_v1 template
message count
operation count
operation-token and operation-segment multisets
codebook tokens
special-token count
answer prefix
score slot
total rendered input-token count
non-operation token sequence
```

The ledger retains exact UTF-8 rendered bytes as hexadecimal, exact token-ID
arrays, their counts and hashes, and every mismatch field. The full ledger is
hashed. Broken square is a semantic positive control and is explicitly not a
serialization control.

This candidate selects exact P/Q matching, not a matched no-op fallback. If
any pair cannot meet the contract, the state is `REVIEW2_1_BLOCKED` before any
model forward; no surface repair, alternate control, or scientific execution
may begin.

## 11. All-at-once freeze and Stage M seal

Before Stage M, one validated campaign manifest must freeze the exact model and
tokenizer revision, runtime image/dependencies, chat-template bytes/hash,
tokenizer hash, score position, single-token checks, all 20 block IDs, map and
behavior case IDs, templates, codebooks, demonstration identities, map halves,
all seeds, depth assignment, randomized block-interleaved execution orders,
map qualification rules, exact-resume rules, analysis seed, and schedule-family
algorithm. Map and behavior IDs are disjoint. All 4,800 Stage M and 5,760
possible Stage E forward IDs exist before the first forward. Accepted IDs can
never be duplicated or replaced.

Stage M evaluates every predictor-side gate before Stage E can become eligible:
minimum overall/per-depth counts, full and foldwise analysis rank, leverage,
permutation support, nondegenerate `R_b`, split-half validity, frame rank,
conditioning, exact-square reproducibility, broken-square sensitivity,
path-surface validity, and artifact completeness. Any failure is
`MAP_COMPLETE_NOT_QUALIFIED` and terminates without block replacement or Stage
E.

During `SEALED_WAIT_FOR_BILLING_CYCLE`, the human-visible report is restricted
to qualification state, qualified counts by depth, artifact hashes, forward
accounting, and billing/resume state. Sealed map artifacts may be hash-verified
and backed up. Until Stage E completes or the campaign terminates permanently,
do not expose block-level `R_b`, layer return patterns, depth summaries,
spectra, packet/operator summaries, or exploratory predictor plots.

## 12. Credit staging and last-amendment boundary

The operational state machine and cost accounting are frozen in
`track_c_credit_staged_execution_plan.md`. A billing wait is operational only;
it cannot change IDs, samples, blocks, the estimand, or any scientific rule.

Review 2.1 is the last design amendment before an execution decision. A ready
state means inspect once, SHA-lock, and decide whether to issue a separate
human authorization. A blocked state means stop without weakening or redesigning
the primary analysis; there is no Review 2.2 rescue lane. The speculative
graded-cobordism lane is outside this amendment and is not implemented.

Visible-state use does not establish a hidden causal register. Qualification
is not uniqueness. This work makes no first-use claim for naturality, holonomy,
or intermediate-state edits, and does not claim a causal transition from raw
representation overlap.
