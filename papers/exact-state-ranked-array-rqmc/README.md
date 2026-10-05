# Exact State-Ranked Array-RQMC: Representations and Shared Work

Aoi Kawasaki. Author preprint, 6 October 2026; not peer reviewed.

[Paper and supplement](https://doi.org/10.5281/zenodo.23166655) ·
[Fixed public companion v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/exact-state-ranked-array-rqmc-v1.0.0) ·
[License scope](LICENSES.txt)

This is the public reproduction companion for manuscript version
`20261005-estimator`: the 20-page main paper and 6-page reader supplement.
It contains the exact measured source, all 4,080 timing observations, canonical
trajectories, matrices and direction columns, work counters, diagnostic inputs,
current corrected executors, and the sources that regenerate the tables,
figures and PDFs. The paper develops primal and dual exact population execution,
and analyzes the dual counter's shared transformation demand. Direct basis is
the practical large-population choice in the reported comparisons.

## Reproduce

Extract the release ZIP into an empty directory, or use this directory from
the tagged repository checkout. Use Python 3.11 and run from this directory:

```text
python -m pip install -r requirements.txt -r pdf-requirements.txt
python verify_public.py
python reproduce_review.py --quick --require-identical
```

The twelve-stage command checks scalar finite-field enumeration, manuscript
Algorithm 3, corrected model domains, the independent point-sorting bridge,
the worked example, saved-source trajectories, exact conditional workloads,
numerical aggregation, mechanism diagrams and the supplement's error example,
then builds both PDFs. It never repeats performance timing experiments.
`--quick` checks 12 measured-source method trajectories (1,800 steps); the
separate current-wrapper regression still checks all 17 cells and six methods
(102 trajectories, 13,800 steps). Omit `--quick` for all 816 measured-source
trajectories (110,400 steps). This is distinct from collecting new timings.

The recorded byte-identical PDF build uses Windows, Python 3.11, Pandoc 3.9,
Typst 0.15.0 and the pinned scientific packages. `pypandoc_binary` supplies
Pandoc and `typst` supplies the compiler and its embedded fonts. No compiler,
font bundle or scientific runtime is redistributed. On another environment,
omit `--require-identical` to inspect a nonidentical PDF build; the exact
scientific checks remain in force. Use `--skip-pdf` for numerical checks alone.
Reproduction writes generated check reports and figures into the extracted
copy. Run `verify_public.py` before reproduction to check the distributed bytes.

## Which source produced the observations?

The [study guide](review_checks/primal_dual/README.md) describes the methods,
conditions, seeds, input map and output contract. In particular:

| Files | Role |
| --- | --- |
| `review_checks/primal_dual/confirmation/source/` | Exact code that produced the reported 4,080 timings. Its hashes are checked against the original metadata. |
| `review_checks/primal_dual/confirmation/data/` | All original timings, fixed matrices and 136 saved population realizations. Six methods per realization give 816 replay trajectories. |
| `review_checks/primal_dual/executors.py` | Current wrapper, correcting even-capacity repair readouts and rejecting tandem word widths below four. |
| `review_checks/primal_dual/model-wrapper.diff.txt` | Exact measured/current wrapper difference; the published 17 model tables and checked outputs agree. |
| `review_checks/primal_dual/workloads/` | Original fixed batches, shifts and staged timing observations; exact expectations are stored as rational numbers. |
| `archive/retained-dense-inputs.zip` | Unedited source and saved observations needed for the supplement's earlier dense-study error example. |

Table 2 ratios are formed within each seed, after taking the technical-repeat
median, and then summarized across eight seeds. The five technical repeats
are not counted as independent seeds. Saved-source replay verifies deterministic
histograms and readouts; it does not promise the same elapsed time on another host.

## Relationship to the review companion

The published PDFs mention `array-rqmc-review-20261005-estimator.zip` and say
that a public deposit identifier had not yet been assigned. Those sentences
describe the pre-publication review package. This release supplies its public
counterpart and the paper DOI above. The PDFs and their Markdown sources are
unchanged from that reviewed version; the publication record supplies the
current distribution link without revising the scientific text.

`PUBLIC_RELEASE.json` identifies both source archives by SHA-256 and lists
every selected file preserved byte-for-byte. All six-method study inputs,
measured source and observations are included. The dispatcher selects the
public dense-input archive instead of extracting the full private archive.
Private editorial correspondence, submission notes, earlier manuscript PDFs
and unrelated historical packages are omitted. The original distributed review
archives retain their names and bytes; this public ZIP has a distinct identity.

The historical dense analyzer retains its input records and can regenerate
the supplement's error table and CDF. It also checks earlier timing and sampled
RSS observations, which are not pooled with the main paper's six-method study.
No new timing observation has been substituted for an original measurement.

## Contents and citation

- [Main source](main.md) and [unchanged PDF](output/pdf/implicit-array-rqmc.pdf).
- [Supplement source](retained_studies.md) and [unchanged PDF](output/pdf/retained-studies.pdf).
- [Public release provenance](PUBLIC_RELEASE.json) and [citation metadata](CITATION.cff).
- [Third-party attribution](THIRD_PARTY.md) and [per-file licensing](LICENSES.txt).

For the paper, cite DOI **10.5281/zenodo.23166655**. For the implementation,
also identify the public companion tag **exact-state-ranked-array-rqmc-v1.0.0**
and the commit recorded on its GitHub Release.
