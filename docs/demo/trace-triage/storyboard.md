# pale-ale Trace Triage Storyboard

First-contact demo concept for long LLM and agent evaluation trace review prioritization.

Core message:

> Scalar-only evaluation says pass. pale-ale shows the 3 trace rows a human should inspect first.

pale-ale does not decide correctness. It tells a reviewer where to look first.

## 1. Hero / before-after

Purpose: show the failure mode and the triage output in the first screen, before any detailed trace view.

### Left: Scalar-only evaluation

- Final answer: Acceptable
- LLM-as-judge: Pass
- Similarity check: Pass
- Hidden: where the source constraint changed inside the trace

### Right: pale-ale Trace Triage

- Review recommended
- 3 rows shortlisted
- Output: review targets, not a model score
- Inspect first:
  - Row 03 - retrieved context omitted exclusion
  - Row 04 - requirement broadened
  - Row 06 - final answer conflicts with source

Screen copy:

> Scalar-only evaluation says pass. pale-ale shows the 3 trace rows a human should inspect first.

## 2. Trace timeline

| Row | Stage | Trace content | Review signal |
| --- | --- | --- | --- |
| 01 | Prompt | Contractor asks whether they can expense a monitor. | None shown. |
| 02 | Source policy | Company-issued equipment only; manager pre-approval required; personal devices excluded. | Source constraints captured. |
| 03 | Retrieved context | Preserves company-issued equipment and manager pre-approval, but omits personal-device exclusion. | Source coverage weakened. |
| 04 | Intermediate judgment | Treats receipt as sufficient. | Requirement broadened. |
| 05 | Draft answer | Allows personal monitor after purchase. | Source-linked conflict. |
| 06 | Final answer | Says receipt is enough. | Source-linked conflict. |
| 07 | Evaluator note | Scalar-only checks pass, but source constraints were not preserved. | Review context. |

Timeline behavior:

- Keep all 7 rows visible.
- Mark rows 03, 04, and 06 as the first review targets.
- Let the reviewer open row 05 from the row 06 card as supporting context, without making it part of the first 3-row shortlist.

## 3. Reviewer shortlist cards

### Row 03 - Source coverage weakened

Why shortlisted:
Retrieved context omitted a constraint that mattered.

Reviewer question:
Did retrieval drop the personal-device exclusion?

### Row 04 - Requirement broadened

Why shortlisted:
The trace changed "manager pre-approval required" into "receipt may be enough."

Reviewer question:
Where did the policy condition change?

### Row 06 - Source-linked conflict

Why shortlisted:
The final answer permits a personal monitor while the source excludes personally selected devices.

Reviewer question:
Should the answer be revised or require a policy citation?

## 4. Evidence comparison

Use three compact cards so the reviewer can compare the source, retrieved context, and final answer without reading the whole trace first.

### Source policy

Company-issued equipment only; manager pre-approval required; personal devices excluded.

Highlighted spans:

- `manager pre-approval required`
- `company-issued equipment only`
- `personal devices excluded`

### Retrieved context

Company-issued equipment may be expensed with manager pre-approval.

Highlighted spans:

- `manager pre-approval`
- `company-issued equipment`
- `personal-device exclusion missing`

### Final answer

You can expense a personal monitor after purchase if you keep the receipt; the receipt is enough.

Highlighted spans:

- `personal monitor`
- `after purchase`
- `receipt is enough`

## 5. Same-review-budget comparison

If a reviewer can inspect only 3 rows:

For this synthetic comparison only, "review-relevant" means one of the first-inspect target rows named in this storyboard: rows 03, 04, and 06. Row 05 remains supporting context for row 06, but it is not counted as a first-inspect target here.

Fixed illustrative selections:

| Review method | Fixed rows inspected | Review-relevant rows found |
| --- | --- | ---: |
| Random baseline | 01, 03, 07 | 1 relevant row |
| Final-output heuristic | 05, 06, 07 | 1 relevant row |
| pale-ale shortlist | 03, 04, 06 | 3 review-relevant rows |

Label:

> Synthetic illustrative example, not a benchmark result.

## 6. Technical appendix

Demo input:

- A synthetic policy/RAG/evaluation trace with 7 ordered rows.
- Source constraints attached to the policy row.
- Row-level review signals for source coverage, requirement broadening, and source-linked conflict.
- A scalar-only evaluator note that still records pass-like outcomes.

Demo output:

- A review recommendation.
- Three shortlisted trace rows.
- Reviewer questions tied to those rows.
- Evidence spans linked to source, retrieved context, and final answer.

Output boundary:

- The output is a set of review targets, not a model score.
- The example supports a first-contact explanation of trace triage.
- The demo does not ask the viewer to trust a hidden classifier; it asks the viewer to inspect the linked rows.

## 7. Claim boundaries

This demo shows review triage, not model scoring.

- Not a benchmark
- Not a correctness classifier
- Not a model-quality score
- Not a detector of deception
- Not a claim about model internals
- Synthetic illustrative example; bounded evidence linked separately
