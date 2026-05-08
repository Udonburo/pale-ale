# Gate12B Manuscript Source Plan

Status: manuscript-source plan draft
Role: bounded handoff from Gate12B workstream evidence to expanded manuscript draft, not a checkpoint revision, not a public evidence package claim, not a new experiment, and not a Gate12A/Gate12B schema change
Date: 2026-05-06

This memo proceeds from:

- `228_GATE12B_PAPER_OUTLINE_AND_CLAIM_BOUNDARY.md`
- `workstream/229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md`

## 0. Scope

This memo records the shift from workstream memo voice to manuscript source
voice.

The paper draft should not be another long workstream note. The manuscript
source should follow the conservative technical-report structure used by the
existing Zenodo paper-source line.

## 1. Current Status

The manuscript source has moved from an initial source outline into an expanded bounded
technical-report draft. The current paper surface is now the primary Gate12B
manuscript text, while this workstream note remains only a compact handoff.

The Gate12B manuscript draft is:

```text
papers/gate12b-observer-relative-closure/main.tex
```

Supporting orientation:

```text
papers/gate12b-observer-relative-closure/README.md
```

The current draft includes:

- external-reader definitions
- compact positioning for readers outside the workstream
- controlled artifact regime
- a lightweight read-only pipeline figure
- observer / scale / bounded artifact-level reparameterization framing
- residual-band construction
- archive relation-signature result
- sensitivity and boundary results
- source-facing annotation summaries
- abbreviated actual source-facing example rows
- non-archive sensitivity
- threats to validity
- explicit small-N and residual effect-size boundaries
- manifest-level reproducibility boundary
- compact evidence package appendix

The manuscript keeps the original boundary from `228` and `229`.

## 2. Boundary

This pass does not:

- add model inference
- add observer modes
- change Gate12A or Gate12B artifacts
- package generated `runs/` artifacts
- create a public evidence bundle
- turn source-facing tags into answer-quality labels
- claim a universal interpretability law
- claim model quality
- claim answer correctness
- claim a physical invariant
- claim a weight-level causal mechanism

## 3. Next Step

The next step is not more evidence collection. It is manuscript editing:

```text
230 expanded manuscript draft -> 231 manuscript polish / PDF build verification
```

The paper should keep the claim from `228` as its ceiling and the evidence
package from `229` as its artifact boundary.

Final production formatting may be generated through Prism or another LaTeX
authoring/export workflow. The checked-in `main.tex` is the current canonical
paper-voice content source, not a claim that the final public PDF must be
produced directly from this file without a final formatting and build
verification step.

## 4. Short Sentence

Gate12B now has an expanded manuscript draft in the existing
Zenodo-report style: the workstream remains the evidence memory, while
`main.tex` becomes the manuscript surface.
