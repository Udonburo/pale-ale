# Publications

This directory is the platform-neutral home for public research outputs from
`pale-ale`. It gives every publication one stable repository location even
when the same work is distributed through Zenodo, a GitHub Release, arXiv,
OSF, or another archive.

## Directory contract

```text
publications/
  catalog.json
  <publication-slug>/
    README.md
    zenodo/        exact tracked Zenodo upload package, when available
    arxiv/         optional platform-specific package
    osf/           optional platform-specific package
```

The publication directory is the identity layer. Platform-specific upload
bundles live beneath it and may differ when a platform imposes different file
or metadata requirements. Do not create another top-level release directory
for a new platform.

The files inside an already published target bundle remain frozen. A later
version receives a new target package or an explicitly versioned update; it is
not silently rewritten. Manuscript development may continue under `papers/`,
while the corresponding publication landing page records the exact source and
release bindings.

## Published works

| Publication | Date | DOI | GitHub Release | Repository landing |
| --- | --- | --- | --- | --- |
| *Structural Replay in Dense Transformers Under a Frozen FP32 Regime* | 2026-04-09 | [10.5281/zenodo.19483162](https://doi.org/10.5281/zenodo.19483162) | [`paper-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/paper-v1.0.0) | [`structural-replay-fp32/`](structural-replay-fp32/README.md) |
| *Transport-First Defect Telemetry on Replay Artifact Graphs* | 2026-04-14 | [10.5281/zenodo.19569052](https://doi.org/10.5281/zenodo.19569052) | [`transport-first-defect-telemetry-v0.1.0`](https://github.com/Udonburo/pale-ale/releases/tag/transport-first-defect-telemetry-v0.1.0) | [`transport-first-defect-telemetry/`](transport-first-defect-telemetry/README.md) |
| *Observer-Relative Closure Signatures on Replay Artifact Graphs* | 2026-05-08 | [10.5281/zenodo.20080003](https://doi.org/10.5281/zenodo.20080003) | [`gate12b-observer-relative-closure-signatures-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/gate12b-observer-relative-closure-signatures-v1.0.0) | [`observer-relative-closure-signatures/`](observer-relative-closure-signatures/README.md) |
| *Compression-Interleaved Parenthesization Defects in LLM Replay Artifact Graphs* | 2026-07-14 | [10.5281/zenodo.21355572](https://doi.org/10.5281/zenodo.21355572) | [`gate12c1-parenthesization-defects-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/gate12c1-parenthesization-defects-v1.0.0) | [`compression-interleaved-parenthesization-defects/`](compression-interleaved-parenthesization-defects/README.md) |
| *Local Mapping Without Iterative Closure* | 2026-08-18 | [10.5281/zenodo.21992852](https://doi.org/10.5281/zenodo.21992852) | [`local-mapping-without-iterative-closure-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/local-mapping-without-iterative-closure-v1.0.0) | [`local-mapping-without-iterative-closure/`](local-mapping-without-iterative-closure/README.md) |
| *Sensitivity Without Reproducibility* | 2026-08-31 | [10.5281/zenodo.22180751](https://doi.org/10.5281/zenodo.22180751) | [`sensitivity-without-reproducibility-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/sensitivity-without-reproducibility-v1.0.0) | [`sensitivity-without-reproducibility/`](sensitivity-without-reproducibility/README.md) |

Machine-readable metadata and historical path mappings are recorded in
[`catalog.json`](catalog.json). Run `python publications/validate_catalog.py`
after adding or relocating a publication target.

## 2026 layout migration

The six publication packages originally accumulated under two conventions:
root-level `zenodo-release*` directories and later `zenodo/<slug>/`
directories. They were moved without content changes into the uniform layout
above. Published Zenodo records and GitHub Release assets were not changed.
Historical commits and release tags continue to preserve the paths that were
current when each release was created.

The file counts, byte counts, and content-tree identities before and after the
move are recorded in [`MIGRATION.md`](MIGRATION.md).
