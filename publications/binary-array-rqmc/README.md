# Exact Binary Projections for Array-RQMC

**Joint Laws and Pathwise-Preserving Execution**

Author: Aoi Kawasaki. Preprint; not peer reviewed.

Published on Zenodo on **13 September 2026**, version **1.0.0**:
[10.5281/zenodo.22728405](https://doi.org/10.5281/zenodo.22728405).
The [GitHub Release](https://github.com/Udonburo/pale-ale/releases/tag/binary-array-rqmc-v1.0.0)
mirrors the same nine deposit files and identifies the repository snapshot.

- [Read the published PDF](zenodo/binary-array-rqmc.pdf)
- [Download the reproducibility capsule](zenodo/reproducibility-capsule.zip)
- [Deposit files and scope](zenodo/README.txt)
- [Working manuscript and build instructions](../../papers/binary-array-rqmc/README.md)
- [Code, 19 tests, and saved-data regeneration](../../papers/binary-array-rqmc/repro/README.md)
- [Citation metadata](../../papers/binary-array-rqmc/CITATION.cff)

## Contribution and scope

The paper characterizes the complete rank-ordered binary innovation law for a
specified LMS-scrambled Sobol construction. An equal-covariance nonlinear
counterexample explains why preserving covariance alone is insufficient.
Direct generation preserves the ideal joint law; state-order maintenance
preserves finite-seed paths and outputs for the stated stopped scalar chains,
including absorption and floating-point ties.

The measured warm-time reductions are 37.2% at 512 paths and 67.5% at 4,096
paths relative to direct generation with full state sorting. The eight model
laws and their two input encodings are not sixteen independent laws. Total
cost comparisons include startup and calibration and do not establish a
universal advantage over CRN or antithetic sampling.

## Use the code

Use Python 3.11 in a separate environment. From the repository root:

```sh
python -m pip install -r papers/binary-array-rqmc/repro/requirements.txt
python -m unittest discover -s papers/binary-array-rqmc/repro -v
```

The companion includes exact-law and pathwise checks, the three array kernels,
and saved inputs for the manuscript's tables and figures. It is not a complete
archive of the original calibration, bootstrap, or timing observations.
An optional smoke driver produces new local engineering timings, not new
manuscript results. See the companion README before interpreting them.

## Published bytes and development

`zenodo/` is the exact nine-file published deposit. Keep these files unchanged;
the SHA-256 inventory and ZIP identify the archived version. The manuscript
sources and computational companion are also browsable under `papers/`.
Later development does not replace the DOI-bound sources inside the capsule.

Tag `binary-array-rqmc-v1.0.0` identifies the initial repository release.
Use the attached `reproducibility-capsule.zip` for the self-contained paper
companion. GitHub's automatically generated source archives contain the whole
repository and are not the capsule.

The deposit was assembled before its source was committed to GitHub. Statements
inside it that no package-containing Git commit is asserted describe that
archival boundary; they are retained, not retroactively rewritten. Its reserved-
DOI wording likewise describes preparation. The DOI above is now public.

For development builds, select a separate destination rather than overwrite
the published package:

```sh
python papers/binary-array-rqmc/build_publication.py --output-dir output/binary-array-rqmc-build
```

Install the PDF dependencies from `papers/binary-array-rqmc/pdf-requirements.txt`
first. This build command does not run new scientific experiments or publish.
The separate review ZIP is historical review material, not the published capsule.

Publication materials and saved data are **CC BY 4.0**; code is **MPL-2.0**.
These are per-file scopes, not alternative licenses for every file. See
[LICENSES.txt](zenodo/LICENSES.txt).

No arXiv submission or peer-review outcome is asserted.
