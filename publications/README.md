# Publications

Papers, technical notes, and archived research outputs from `pale-ale`.
Each publication page links its manuscript, code, license, and reproduction
scope. A DOI identifies an archived version; it is not a peer-review claim.

## Papers and notes

| Date | Publication | Archive | GitHub Release |
| --- | --- | --- | --- |
| 2026-09-13 | [Exact Binary Projections for Array-RQMC: Joint Laws and Pathwise-Preserving Execution](binary-array-rqmc/README.md) | [DOI](https://doi.org/10.5281/zenodo.22728405) | [v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/binary-array-rqmc-v1.0.0) |
| 2026-08-31 | [Sensitivity Without Reproducibility](sensitivity-without-reproducibility/README.md) | [DOI](https://doi.org/10.5281/zenodo.22180751) | [v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/sensitivity-without-reproducibility-v1.0.0) |
| 2026-08-18 | [Local Mapping Without Iterative Closure](local-mapping-without-iterative-closure/README.md) | [DOI](https://doi.org/10.5281/zenodo.21992852) | [v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/local-mapping-without-iterative-closure-v1.0.0) |
| 2026-07-14 | [Compression-Interleaved Parenthesization Defects in LLM Replay Artifact Graphs](compression-interleaved-parenthesization-defects/README.md) | [DOI](https://doi.org/10.5281/zenodo.21355572) | [v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/gate12c1-parenthesization-defects-v1.0.0) |
| 2026-05-08 | [Observer-Relative Closure Signatures on Replay Artifact Graphs](observer-relative-closure-signatures/README.md) | [DOI](https://doi.org/10.5281/zenodo.20080003) | [v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/gate12b-observer-relative-closure-signatures-v1.0.0) |
| 2026-04-14 | [Transport-First Defect Telemetry on Replay Artifact Graphs](transport-first-defect-telemetry/README.md) | [DOI](https://doi.org/10.5281/zenodo.19569052) | [v0.1.0](https://github.com/Udonburo/pale-ale/releases/tag/transport-first-defect-telemetry-v0.1.0) |
| 2026-04-09 | [Structural Replay in Dense Transformers Under a Frozen FP32 Regime](structural-replay-fp32/README.md) | [DOI](https://doi.org/10.5281/zenodo.19483162) | [v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/paper-v1.0.0) |

## Other public records

These are dataset or software releases, not papers. Repository-only technical
reports are listed [separately](../README.md#technical-reports).

| Record | Type | Downloads |
| --- | --- | --- |
| [Gate12A First Replication Checkpoint](gate12a-first-replication-checkpoint/README.md) | Dataset / empirical checkpoint | [Zenodo](https://doi.org/10.5281/zenodo.19340221) · [GitHub](https://github.com/Udonburo/pale-ale/releases/tag/gate12a-first-replication-checkpoint-2026-03-31) |
| [Gate2 Telemetry E2E v1.0.1](gate2-telemetry-v1-0-1/README.md) | Software | [GitHub](https://github.com/Udonburo/pale-ale/releases/tag/v1.0.1) |

## Verify an archive

Machine-readable identifiers, repository paths, and distribution targets are
in [catalog.json](catalog.json). To check the catalog and tracked package
checksums, run from the repository root:

```sh
python publications/validate_catalog.py
```

This checks local files; it does not rerun the studies or verify remote
downloads. Each publication page describes the available reproduction work.

## Repository layout

```text
publications/
  catalog.json
  <publication-slug>/
    README.md     publication page and distribution links
    zenodo/       exact tracked Zenodo deposit, when available
papers/
  <paper-slug>/   working manuscript and computational companion
```

Already published deposits are preserved byte-for-byte. Development may
continue under `papers/`; it does not replace a DOI-bound capsule. Future
platform-specific packages belong under the same publication directory, and
only actual distribution targets are added to the catalog.

For historical deposits containing untracked run artifacts, the publication
page links to the archive rather than presenting an incomplete local copy.
The [2026 layout migration](MIGRATION.md) records earlier path changes;
historical tags retain their original layouts and published assets.
