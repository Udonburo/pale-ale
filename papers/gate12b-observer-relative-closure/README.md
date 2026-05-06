# Gate12B Observer-Relative Closure Manuscript Source

Status: manuscript-source skeleton
Scope: paper-facing LaTeX source for the bounded Gate12B archive-family closure-signature result

This directory holds the manuscript source surface for the Gate12B paper line.
It is intentionally separate from `workstream/` memos: the memos establish the
evidence chain and claim boundary, while `main.tex` uses paper voice.

Current source:

- `main.tex`

Build target:

```powershell
pdflatex main.tex
```

Production note:

This LaTeX source is a manuscript skeleton and canonical paper-voice content
source. Final production formatting may be generated through Prism or another
LaTeX authoring/export workflow.

Boundary:

- no generated `runs/` artifacts are stored here
- no new experiment is introduced by this skeleton
- the claim is bounded to the current Gate12B artifact study
- source-facing tags remain source-facing annotations, not answer-quality labels
- the evidence package is specified by
  `workstream/229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md`

Before release, pair the final manuscript with:

- a paper PDF
- checksum material
- release-side provenance notes
- an intentionally packaged evidence bundle or stable evidence record
