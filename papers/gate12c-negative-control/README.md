# Gate12C-1 Negative-Control Technical Note

Status: first technical-note draft
Scope: conservative paper source for the Gate12C-1 negative-control result

This directory contains a concise draft technical-note source for the Gate12C-1
negative-control result. It is not a Zenodo package, not a refreshed release
folder, and not a checksum set.

The draft is based on tracked repository workstream records:

- `workstream/236_GATE12C1_FIRST_EMPIRICAL_RESULT_MEMO.md`
- `workstream/237_GATE12C1_RESULT_CLOSEOUT_AND_NEXT_BOUNDARY.md`
- `workstream/238_GATE12C1_NEGATIVE_CONTROL_AND_PUBLIC_POSITIONING_PLAN.md`

It does not include generated `runs/` artifacts. It does not run a new
experiment, inspect generated runs, inspect individual cycles, run a Gate12B
overlay, modify Gate12A/B/C runners, create a Zenodo package, or update
`CHECKSUMS-SHA256.txt`.

Gate12C-1 is treated here only as a predeclared higher-order negative control.
The draft does not claim a physical-structure discovery, safety or deployment
guarantee, model-quality ranking, correctness classification, weight-level
causal mechanism, or Gate12C-1 as positive evidence for Gate12B.

Build target, once a local LaTeX installation is available:

```powershell
pdflatex main.tex
```

Build products such as `.aux`, `.log`, `.out`, `.toc`, and generated PDFs are
not committed by this draft.
