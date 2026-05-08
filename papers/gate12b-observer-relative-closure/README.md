# Observer-Relative Closure Signatures on Replay Artifact Graphs: A Bounded Source-Facing Audit of Existing LLM Artifacts

Status: published bounded technical report source
Scope: LaTeX manuscript source for the bounded Gate12B archive-family closure-signature result
Zenodo DOI: https://doi.org/10.5281/zenodo.20080003

This directory holds the manuscript source surface for the Gate12B paper line.
It is intentionally separate from `workstream/` memos: the memos establish the
evidence chain and claim boundary, while `main.tex` uses paper voice.

Current source:

- `main.tex`

Build target:

```powershell
pdflatex main.tex
```

PDF build requires a local LaTeX installation. This repository does not commit
generated `.aux`, `.log`, `.out`, `.toc`, or PDF build products for this draft.

Production note:

This LaTeX source is the current canonical paper-voice content source. Final
production formatting may be generated through Prism or another LaTeX
authoring/export workflow.

Boundary:

- no generated `runs/` artifacts are stored here
- no new experiment is introduced by this manuscript draft
- the claim is bounded to the current Gate12B artifact study
- source-facing tags remain source-facing annotations, not answer-quality labels
- the evidence package is specified by
  `workstream/229_GATE12B_PAPER_EVIDENCE_PACKAGE_MANIFEST.md`
- generated `runs/` artifacts remain local evidence and are not committed here
- the manuscript records a manifest-level evidence map; a public evidence
  package can be assembled separately from recorded artifact directories,
  manifests, checksums, and queue outputs

Public deposit:

- DOI: https://doi.org/10.5281/zenodo.20080003
- release package:
  `zenodo-release-gate12b-observer-relative-closure-signatures/`
- generated `runs/` artifacts are not included in the public deposit
