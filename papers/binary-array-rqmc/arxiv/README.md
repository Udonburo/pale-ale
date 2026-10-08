# Array-RQMC: retained September 2026 manuscript

**Exact Binary Projections for Array-RQMC: Joint Laws and Pathwise-Preserving Execution**  
Aoi Kawasaki - Preprint, 25 September 2026

This historical revision remains at its original source path. The current
paper is
[Exact State-Ranked Array-RQMC: Representations and Shared Work](../../exact-state-ranked-array-rqmc/README.md),
with its public paper DOI and fixed reproduction companion. The earlier
[Zenodo v1.0.0](https://doi.org/10.5281/zenodo.22728405) remains a separate,
published edition with its own sources and archived files.

## Read and reproduce

[`main.md`](main.md) is this edition's canonical manuscript. The local
[`repro/`](repro/README.md), figures, PDF builder, and layout belong to the same
edition. They do not import source files from the parent Zenodo tree.

The revision adds introductory context and binary-field notation, connects the
joint-law result to an exact stopped-chain counterexample, and separates the
main execution-cost results from cross-method cost accounting in Appendix C.
The measured kernels and original benchmark observations are unchanged.

The companion has 22 tests. It includes all 720 original timing rows from the
order-maintenance study and a paired-round bootstrap reanalysis of Table 3.
The stopped-chain example is an added mathematical calculation, not a new
performance experiment. The full original calibration/validation archive and
baseline orchestration are not included.

The manuscript cites this publicly retrievable
[companion source commit](https://github.com/Udonburo/pale-ale/blob/2d7e0c8e961d45e108ae2792d1bb3316800875d5/papers/binary-array-rqmc/repro/README.md).
Its location at an earlier Git revision remains valid. This directory retains
its own copy of those reproduction sources for independent builds.

## Build this historical PDF

Use Python 3.11 or later in a separate environment. From this directory:

```text
python -m pip install -r pdf-requirements.txt
python build_pdf.py
```

The result is `output/pdf/binary-array-rqmc.pdf`. Pandoc and Typst are supplied
by the pinned dependencies; no TeX installation is required. The builder
preserves the manuscript's mathematical nodes and pseudocode. For a separate
PDF tool installation, pass `--tools-dir PATH`.

The 14-page PDF contains the complete manuscript and both figures. The title
page's date identifies this manuscript revision.
The build follows Markdown -> Pandoc -> Typst -> PDF; the reproduction ZIP
is a separate companion.

## Check this edition

```text
python -m pip install -r repro/requirements.txt
python -m unittest discover -s repro -v
python repro/render_results.py
```

Tests check finite laws, complete paths, ordering, stopped-chain distributions,
and the timing reanalysis. The renderer regenerates saved-input tables and
figures; it does not run a new performance experiment. See the companion README
for the distinction between saved-result reproduction and new smoke timings.

## Recover a historical source snapshot

This source tree and its earlier revisions remain available in Git under their
original paths. Generated PDFs stay under ignored `output/`. The earlier Zenodo
deposit and its builders are independent. Use the current state-ranked paper's
fixed release for the current reproduction package.

The parent `CITATION.cff` describes Zenodo v1.0.0. It is intentionally not copied
into this edition as though that DOI identified the revised manuscript.

## Licenses

The manuscript, figures, documentation, and saved data are CC BY 4.0; code and
layout are MPL-2.0. See [`LICENSES.txt`](LICENSES.txt) and [`repro/LICENSE`](repro/LICENSE).
AI assistance is disclosed after the Conclusion in the manuscript.
