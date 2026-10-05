# Exact Binary Projections for Array-RQMC

**Joint Laws and Pathwise-Preserving Execution** - Aoi Kawasaki

This directory keeps two independently buildable historical editions.
The current paper and companion are
[Exact State-Ranked Array-RQMC: Representations and Shared Work](../exact-state-ranked-array-rqmc/README.md).

| Edition | Manuscript and code | Public identity |
| --- | --- | --- |
| Zenodo v1.0.0, 13 September 2026 | [`main.md`](main.md), [`repro/`](repro/README.md), and the build scripts in this directory | [DOI 10.5281/zenodo.22728405](https://doi.org/10.5281/zenodo.22728405); [original GitHub tag](https://github.com/Udonburo/pale-ale/tree/binary-array-rqmc-v1.0.0/papers/binary-array-rqmc) |
| Retained revision, 25 September 2026 | [`arxiv/`](arxiv/README.md), with its own manuscript, figures, build, and reproduction sources | Historical preparation directory; use the current state-ranked paper for journal preparation |

The Zenodo edition is the earlier public record. The September revision incorporates
subsequent exposition, evidence, and presentation revisions. They do not share
an editable manuscript, figure directory, or runtime source directory. Editing
or building one does not update the other. This navigation page is the only
changed file in the original source layout; the other original files match the
v1.0.0 tag.

The [published v1.0.0 deposit](../../publications/binary-array-rqmc/README.md)
retains its original files and paths. The fixed companion commit
[`2d7e0c8`](https://github.com/Udonburo/pale-ale/blob/2d7e0c8e961d45e108ae2792d1bb3316800875d5/papers/binary-array-rqmc/repro/README.md)
remains the public source identity cited by the September revision. Its separate
directory retains the original `arxiv/` path for reproducibility.

## Build and reproduce the Zenodo edition

Use Python 3.11 or later. From this directory:

```text
python -m pip install -r pdf-requirements.txt -r repro/requirements.txt
python build_pdf.py
python -m unittest discover -s repro -v
```

The original companion has 19 tests. For the September revision and its 22-test
companion, use the commands in [`arxiv/README.md`](arxiv/README.md).

For an exact historical deposit rebuild, use `build_publication.py` from the
extracted original capsule. Its embedded metadata and sources bind v1.0.0.
The legacy publication and review builders remain here for that edition;
they rebuild that historical edition.

## Historical source layout

Ordinary Git history records the September revision under `arxiv/`. Each tree's
`output/` and `tmp/` are ignored generated files. The publication index identifies
the published release separately from these historical manuscript sources.

The existing Zenodo v1.0.0 deposit is retained in Git for compatibility. This is
not a template for adding another full deposit directory on every revision.
The parent `CITATION.cff` describes the published Zenodo edition.

Code is MPL-2.0; manuscript, figures, documentation, and data are CC BY 4.0.
See the license scope supplied with the edition being used.
