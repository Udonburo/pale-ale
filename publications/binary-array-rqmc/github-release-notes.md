# Exact Binary Projections for Array-RQMC

**Joint Laws and Pathwise-Preserving Execution**

Aoi Kawasaki · Version 1.0.0 · Preprint, not peer reviewed.

**Archival DOI:** [10.5281/zenodo.22728405](https://doi.org/10.5281/zenodo.22728405)

This release pairs the published manuscript and reproducibility capsule with
their first tagged repository snapshot. All nine attached files are identical
to the Zenodo deposit; no scientific results or archived files are revised.

## What the paper establishes

- The complete rank-ordered binary innovation law for a specified
  LMS-scrambled Sobol construction, with direct generation of that ideal law.
- An equal-covariance counterexample explaining why the joint law matters for
  nonlinear outputs.
- State-order maintenance preserving finite-seed paths and outputs for the
  stated stopped scalar chains, including absorption and floating-point ties.

Across eight fixed benchmark cases, order maintenance reduced warm execution
time by **37.2% at 512 paths** and **67.5% at 4,096 paths**, relative to direct
generation with full state sorting. Startup, calibration, workload, and reuse
affect total cost; this is not a universal advantage over CRN or antithetic
sampling, or a new estimator with lower variance.

## Downloads and reproduction

- **binary-array-rqmc.pdf** — the 13-page preprint.
- **reproducibility-capsule.zip** — self-contained manuscript/build sources,
  three array kernels, 19 tests, saved table/figure inputs, and checksums.
- **README.txt, REPOSITORY.md, LICENSES.txt, release_manifest.json,
  CHECKSUMS-SHA256.txt, zenodo-description.md, zenodo-metadata.json** — the
  remaining original deposit files.

The [tagged computational companion](https://github.com/Udonburo/pale-ale/tree/binary-array-rqmc-v1.0.0/papers/binary-array-rqmc/repro)
is also browsable. With Python 3.11, extract the capsule and run from its
`binary-array-rqmc/` directory:

```sh
python -m pip install -r repro/requirements.txt
python -m unittest discover -s repro -v
```

The companion supports exact-law/ordering checks and saved-data regeneration.
It does **not** contain the complete original raw calibration, bootstrap, or
timing history. GitHub's automatically generated whole-repository source ZIP
and tarball are separate from the attached reproducibility capsule.

The deposit predates this GitHub release. Its original preparation-time wording
about a reserved DOI and the absence of a package-containing Git commit is
preserved; the DOI is now public and this tag supplies the repository binding.

## Licenses and file identity

Publication materials and saved data: **CC BY 4.0**. Code: **MPL-2.0**.
These are per-file scopes, not alternative licenses for every file; see
`LICENSES.txt`.

SHA-256:

```text
6616bef66bf1ad481c44d0bfeabd5797916eaa56b4a129a6338d2cc791734dd2  binary-array-rqmc.pdf
c74077b4114c0fccb0ea426676677191b13374fc899d95aa098c1e9d6e758ec8  reproducibility-capsule.zip
```

No arXiv submission or peer-review outcome is asserted.
