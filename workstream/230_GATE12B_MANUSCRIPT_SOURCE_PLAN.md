# Gate12B Manuscript Source Plan

Status: manuscript-source plan draft
Role: bounded handoff from Gate12B workstream evidence to paper-source skeleton, not a full paper draft, not a checkpoint revision, not a release claim, not a new experiment, and not a Gate12A/Gate12B schema change
Date: 2026-05-06

This memo proceeds from:

- `228_GATE12B_PAPER_OUTLINE_AND_CLAIM_BOUNDARY.md`
- `229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md`

## 0. Scope

This memo records the shift from workstream memo voice to manuscript source
voice.

The paper draft should not be another long workstream note. The manuscript
source should follow the conservative technical-report structure used by the
existing Zenodo paper-source line.

## 1. Added Paper Source Surface

The Gate12B manuscript skeleton is:

```text
papers/gate12b-observer-relative-closure/main.tex
```

Supporting orientation:

```text
papers/gate12b-observer-relative-closure/README.md
```

The source starts from paper voice:

- title
- abstract
- introduction
- what we claim
- what we do not claim
- controlled artifact regime
- Gate12B operational framework
- main archive-family evidence
- sensitivity and boundary results
- source-facing evidence
- non-archive sensitivity
- limitations
- artifact manifest and reproducibility
- conclusion
- appendix handoff to the evidence package manifest

The current `main.tex` has been expanded from skeleton into a v0.2 manuscript
draft. The expansion adds external-reader definitions, method detail, result
subsections, source-facing annotation context, threats to validity, and
claim-to-artifact reproducibility mapping while preserving the bounded claim
from `228` and the evidence-package boundary from `229`.

## 2. Boundary

This pass does not:

- add model inference
- add observer modes
- change Gate12A or Gate12B artifacts
- package generated `runs/` artifacts
- create a release bundle
- turn source-facing tags into answer-quality labels
- claim a universal interpretability law

## 3. Next Step

The next step is not more evidence collection. It is manuscript editing:

```text
230 paper-source skeleton -> 231 manuscript draft pass / table tightening
```

The paper should keep the claim from `228` as its ceiling and the evidence
package from `229` as its artifact boundary.

Final production formatting may be generated through Prism or another LaTeX
authoring/export workflow. The checked-in `main.tex` is the current canonical
paper-voice skeleton, not a claim that the final release PDF must be produced
directly from this file without a production pass.

## 4. Short Sentence

Gate12B now has a paper-source skeleton in the existing Zenodo-report style:
the workstream remains the evidence memory, while `main.tex` becomes the
paper-facing surface.
