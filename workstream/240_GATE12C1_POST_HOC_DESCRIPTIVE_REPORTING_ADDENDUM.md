# Gate12C-1 Post-Hoc Descriptive Reporting Addendum

Date: 2026-07-12

Status: frozen post-hoc descriptive reporting boundary

## 1. Purpose

This addendum fixes the descriptive quantities that may be added to the
Gate12C-1 companion paper after inspection of the completed result. It is not
a prospective analysis plan, does not alter the frozen confirmatory test, and
does not convert the observed reverse direction into confirmatory evidence.

The sole input is the complete family of 24 predeclared endpoint
`run_q_median` values already recorded in:

- `workstream/236_GATE12C1_FIRST_EMPIRICAL_RESULT_MEMO.md`

No generated root-, cycle-, or block-level artifact is reopened for this
descriptive addition.

## 2. Authorized Descriptive Reporting

The paper may report only the following additions:

1. the full sign pattern across all 24 predeclared endpoint summaries;
2. the minimum, maximum, and median of the 24 endpoint-level hierarchical
   log-ratio summaries;
3. the values obtained by exponentiating each of those 24 summaries, followed
   by the minimum, maximum, and median on that transformed endpoint-summary
   scale; and
4. the complete 24-value table and a zero-referenced figure containing all
   endpoints without selection.

For the even endpoint count, the median is the arithmetic mean of the two
central ordered values, matching Python `statistics.median`. Log-scale values
may be displayed to six decimal places. Exponentiated values may be displayed
to three decimal places. These are descriptive effect summaries, not raw-row
ratios, confidence intervals, or test statistics.

## 3. Frozen Non-Exploration Boundary

This addendum does not authorize:

- a reverse-direction significance test;
- selection or exclusion of endpoints after inspection;
- new subgroup summaries by model, rendering family, compression rank, or
  any combination of those fields;
- new q-difference, model-difference, or family-difference hypotheses;
- new correlation, regression, clustering, or threshold analyses;
- aggregate positive/negative block totals across the endpoint family;
- re-analysis of root-, cycle-, block-, or null-draw telemetry; or
- causal language about suppression, coherence, or graph consistency.

Any such analysis belongs to a separately frozen prospective Gate12C-2 plan
and held-out data.

## 4. Interpretation Boundary

The authorized summaries describe the direction and scale of the 24 already
defined endpoint summaries. They do not establish a reverse-direction effect,
identify a mechanism, or distinguish a genuinely smaller observed defect from
inflation of the graph-unconstrained null caused by destroying shared-node
coupling.
