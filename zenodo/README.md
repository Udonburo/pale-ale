# Archival Release Capsules

This directory is the indexed home for new DOI-bound release capsules tracked
on the repository's default branch. Each capsule is self-contained and keeps
its own manifest, checksums, licensing, and provenance records.

## Releases

| Release | DOI | GitHub release | Capsule |
| --- | --- | --- | --- |
| Gate12C-1: Compression-Interleaved Parenthesization Defects | [10.5281/zenodo.21355572](https://doi.org/10.5281/zenodo.21355572) | [`gate12c1-parenthesization-defects-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/gate12c1-parenthesization-defects-v1.0.0) | [`gate12c1-parenthesization-defects/`](gate12c1-parenthesization-defects/README.txt) |
| Local Mapping Without Iterative Closure | [10.5281/zenodo.21992852](https://doi.org/10.5281/zenodo.21992852) | [`local-mapping-without-iterative-closure-v1.0.0`](https://github.com/Udonburo/pale-ale/releases/tag/local-mapping-without-iterative-closure-v1.0.0) | [`local-mapping-without-iterative-closure/`](local-mapping-without-iterative-closure/README.txt) |

New capsules should use `zenodo/<release-slug>/` rather than adding another
top-level `zenodo-release-*` directory.

Capsule contents are tracked byte-for-byte from their published release tags.
Internal status fields may record the pre-publication freeze stage; the DOI and
release index are authoritative for current publication status.

## Historical Compatibility Paths

The following top-level directories are frozen historical release snapshots:

- [`../zenodo-release/`](../zenodo-release/README.txt)
- [`../zenodo-release-transport-first-defect-telemetry/`](../zenodo-release-transport-first-defect-telemetry/README.txt)
- [`../zenodo-release-gate12b-observer-relative-closure-signatures/`](../zenodo-release-gate12b-observer-relative-closure-signatures/README.txt)

They remain in place because repository checks, tracked documentation, and
external links already refer to those paths. In particular,
`tools/run_eval_checks.py` requires
`zenodo-release/CHECKSUMS-SHA256.txt`. Do not move, rename, or replace these
directories with symlinks as part of routine release organization.

The immutable Gate12C-1 release tag preserves its original capsule path,
`zenodo-release-gate12c1-parenthesization-defects/`. The byte-identical copy on
`main` lives here under the current directory policy; this organizational
change does not rewrite the published tag or the Zenodo record.
