# Track C Review 2.1 report

## A. Repository and authority verification

- Worktree: clean repository worktree (local absolute path omitted)
- Branch: `codex/gate13-track-c-review2-1`
- Historical Review 2 HEAD: `4818d0cd313f09e9ae6304bae7c647424dcfcb9f`
- Pre-edit status: clean
- Read-only planning-note SHA-256: `4677d79cedc9671219c7c8fd693d08188d771d43f1c70b2b3885fd66eec2997a` (matched)
- Historical Review 2 and panel closeout hashes: all matched the v2 lock
- Original `pale-ale` worktree: not modified
- Modal, GPU, Qwen3.6, activation collection, Track C outcome, A3, and Formal Gate13: untouched

A repository-validation incident prevents authorization readiness. An
over-broad `cargo test --workspace` reached the existing-cache branch of
`crates/cli/tests/cli.rs`; `json_embed_vector_invariants` completed successfully,
which entails an unrelated cached embedding-model load/forward. The suite was
interrupted when three model-capable eval tests exceeded 60 seconds. The test
code uses `--offline`, no download occurred, no matching process remained, and
this was not Qwen3.6 or Track C. It nevertheless violates the absolute general
model-load/forward prohibition, so this package fails closed.

## B. Review 2.1 changes

- Preserved Review 2 as a historical candidate; added a separate `review2_1` directory.
- Replaced the energy mean with the single square-root amplitude `R_b` while retaining six descriptive energy components.
- Removed behavior-derived nuisance leakage and bound `C_b^M` to required map-stage logits.
- Froze categorical-depth OLS, fold-local scaling, directional `beta_R`, LOBO SSE reduction, and direction-aware Freedman–Lane testing.
- Added exact surface/serialization, full campaign freeze, Stage M sealing, credit-stage, exact-resume, and fail-closed contracts.
- Added deterministic validation, complete null calibration, and targeted tests.

## C. Final estimand and nuisance definition

```text
R_b = sqrt((1/3) sum_{l in {21,43,62}} (1/2) sum_{h in {1,2}}
      tr(Delta_blh Sigma_blh Delta_blh^T) / tr(Sigma_blh))

Y_b = sqrt(mean_e[(m_beP - m_beQ)^2])

C_b^M = 0.5 * (mean exact-node map margin in half 1
             + mean exact-node map margin in half 2)
```

`C_b^M` uses 192 required exact-square map rows, zero behavior rows, zero
broken-square rows, and zero additional forwards. The nuisance design is
intercept plus three categorical-depth indicators plus `C_b^M`.

## D. Directional decision rule

```text
T_LOBO = 1 - SSE_full_LOBO / SSE_nuisance_LOBO
positive only if all gates pass, T_LOBO > 0, beta_R > 0, and p <= 0.05
p = (1 + #{T_perm >= T_obs and beta_R_perm > 0}) / (1 + B)
B = 99,999
```

`beta_R` is the raw-amplitude coefficient in the full-cohort frozen full OLS.
Every permuted coefficient and every LOBO fit is refitted. Negative direction,
negative predictive increment, or any failed gate is not positive evidence.

## E. Null-calibration design and observed result

The final audit ran once with seed `2026082603`, 2,000 datasets per eligible
scenario and 999 permutations per dataset. The frozen acceptable interval was
`[0.035, 0.065]`.

| Scenario | Positives / 2,000 | Empirical FPR | Result |
|---|---:|---:|---|
| Balanced 20, nuisance correlated | 115 | 0.0575 | PASS |
| Balanced 16, nuisance correlated | 100 | 0.0500 | PASS |
| Independent eligible qualification masks | 89 | 0.0445 | PASS |
| Heteroskedastic within depth strata | 92 | 0.0460 | PASS |
| Leverage stress below gate | 114 | 0.0570 | PASS |

The 2,000 ineligible masks generated no outcomes and ran no primary tests.
Both 331,775-permutation exact enumerations passed, as did deterministic
reproducibility and every frozen terminal case. Scientific calibration substate:
`REVIEW2_1_READY_FOR_HUMAN_AUTHORIZATION`. Overall state remains blocked by the
validation incident in section A; the audit is not rerun or replaced.

## F. Leverage and degeneracy gates

- Full rank and positive residual degrees of freedom in the full cohort and every LOBO training fold.
- 2-norm condition number at most `1e6`.
- `h_max <= min(0.80, 3p/n)` for nuisance/full, full-cohort/fold designs.
- Relative variance and nuisance-SSE floor `1e-12 * max(1, relevant squared scale)`.
- At least 16 blocks, at least four per depth, and at least 99,999 unique nonidentity within-depth permutations.
- Explicit terminals for zero outcome/representation variance, rank, SSE, leverage, support, count, direction, increment, and significance failures.

## G. Credit-staged execution contract

The entire 20-block campaign, all 10,560 possible forward IDs, both execution
orders, all 480 exact P/Q render pairs, and the analysis schedule family must be
frozen before Stage M. Stage E can become eligible only after every map-side
gate passes. Billing waits expose counts/hashes/accounting only. Accepted IDs
cannot be duplicated or replaced; resume runs only missing frozen IDs.

The operational plan remains hypothetical. Because this package is blocked,
neither Stage M nor any later state may be entered.

## H. Updated forward and cost forecast

| Component | Forwards | Historical-linear planning forecast |
|---|---:|---:|
| Stage M | 4,800 | USD 20.62321418 including fixed allowance |
| Stage E, 16 blocks (4/depth) | 4,608 | USD 19.67954534 |
| Stage E, 20 blocks (5/depth) | 5,760 | USD 24.59943168 |
| Full 20-block campaign | 10,560 | USD 45.22264586 |
| Full campaign, 25% contingency | — | USD 56.528307325 |
| Hard planning ceiling | — | USD 65.00 |

These are forecasts, not charges, credits, allocations, or authorization.

## I. Files changed

Only `analysis/gate13_causal_return/review2_1/` was added:

```text
TRACK_C_REVIEW2_1_AMENDMENT.md
track_c_estimand_lock_candidate_v2.json
track_c_review2_1_validator.py
track_c_null_calibration.json
track_c_credit_staged_execution_plan.md
REVIEW2_1_REPORT.md
tests/test_track_c_review2_1_validator.py
```

## J. Tests and validators run

- Review 2.1 targeted scientific tests excluding the blocked package inventory test: 12/12 PASS.
- Historical Review 2 targeted tests: 8/8 PASS.
- Prior Gate12/Gate13-adjacent targeted Python tests: 63/63 PASS.
- Final null audit: PASS on every required scientific condition.
- Rust unit tests: 15/15 PASS.
- Rust integration suite: ABORTED and not rerun after the prohibited cached-model branch was identified.
- Package validator: FAIL closed with `estimand lock contains execution authority or activity`, because the v2 authority ledger truthfully records the cached-model validation incident.

No existing test was weakened, deleted, or changed.

## K. Commit(s)

One local blocked-candidate commit is to be recorded on
`codex/gate13-track-c-review2-1`. It is not pushed.

## L. Git status

The intended post-commit state is clean. Exact commit and status are reported
at handoff.

## M. Unresolved blocker

The estimand, calibration, leverage design, surface contract, and budget are
scientifically feasible under the frozen audit. The unresolved blocker is the
task-contract/provenance breach in section A. Per the last-amendment rule, the
primary analysis is not weakened or redesigned, no Review 2.2 is opened, and
Track C remains closed.

`REVIEW2_1_BLOCKED`
