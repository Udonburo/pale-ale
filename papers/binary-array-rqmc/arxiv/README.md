# Array-RQMC: arXiv submission manuscript

**Exact Binary Projections for Array-RQMC: Joint Laws and Pathwise-Preserving Execution**  
Aoi Kawasaki - Preprint, 25 September 2026

This is the independent manuscript prepared for the initial arXiv submission.
No arXiv identifier or peer-review outcome is asserted. The earlier
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

## Build the submission PDF

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
page's date is a manuscript date, not a Zenodo version or an arXiv version label.
The build follows Markdown -> Pandoc -> Typst -> PDF. See the
[arXiv PDF submission guidance](https://info.arxiv.org/help/submit_pdf.html)
for the upload format; the reproduction ZIP is a separate companion.

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

## GitHub distribution

Keep this source tree in Git and use ordinary commits or release tags for its
history. After committing the desired source revision, a source-only companion
can be generated from the repository root with:

```text
git archive --format=zip --prefix=binary-array-rqmc-arxiv/ --output=papers/binary-array-rqmc/arxiv/output/arxiv-source.zip HEAD:papers/binary-array-rqmc/arxiv
```

The generated ZIP and submission PDF stay under ignored `output/`. They can be
attached to the GitHub Release for the arXiv edition. The earlier Zenodo deposit
and its builders are independent. No new Zenodo record or version is required
by this source layout.

The parent `CITATION.cff` describes Zenodo v1.0.0. It is intentionally not copied
into this edition as though that DOI identified the revised manuscript.
Record the actual arXiv identifier in the publication index after submission.

## Licenses

The manuscript, figures, documentation, and saved data are CC BY 4.0; code and
layout are MPL-2.0. See [`LICENSES.txt`](LICENSES.txt) and [`repro/LICENSE`](repro/LICENSE).
AI assistance is disclosed after the Conclusion in the manuscript.
