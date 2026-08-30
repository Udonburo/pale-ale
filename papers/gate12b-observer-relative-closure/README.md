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

Release-candidate note:

After Gate12C-1 negative-control integration, this source is ready for
release-candidate review when the manuscript static checks pass. The
release-candidate boundary preserves the current abstract and main Gate12B
claim, treats Gate12C-1 only as a negative-control boundary, and does not add
generated `runs/` artifacts.

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
  `publications/observer-relative-closure-signatures/zenodo/`
- generated `runs/` artifacts are not included in the public deposit

The publication target directory is a frozen historical release snapshot with
its own checksum set. Editing this manuscript source directory does not
refresh the Zenodo package, update packaged files, or update
`CHECKSUMS-SHA256.txt`.
