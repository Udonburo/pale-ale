# Releases And Artifacts

This repo separates the paper-facing release surface from the longer moving research memory.

## What Should Be Frozen For A Public Release

For a DOI-bound report line, the frozen public package should include:

- a rendered manuscript PDF
- the matching manuscript source, bibliography, and figure sources
- manifest and checksum material
- a frozen implementation snapshot
- artifact bundles, or stable links to separately archived bundles

That is the release surface a reader should be able to cite, download, and inspect without needing to reconstruct the line from the entire repo history.

## What Lives Where In The Repo

Today, the ingredients already present in the repo are mainly:

- [`../publications/README.md`](../publications/README.md): the platform-neutral publication catalog and directory policy
- [`../publications/structural-replay-fp32/`](../publications/structural-replay-fp32/README.md): the April 2026 Gate12A frozen technical report
- [`../publications/transport-first-defect-telemetry/`](../publications/transport-first-defect-telemetry/README.md): the separate mathematical telemetry note
- [`../publications/observer-relative-closure-signatures/`](../publications/observer-relative-closure-signatures/README.md): the Gate12B bounded technical report
- [`../publications/compression-interleaved-parenthesization-defects/`](../publications/compression-interleaved-parenthesization-defects/README.md): the Gate12C-1 paper and reproducibility capsule
- [`../publications/local-mapping-without-iterative-closure/`](../publications/local-mapping-without-iterative-closure/README.md): the graph-XOR capability-boundary report
- [`../publications/sensitivity-without-reproducibility/`](../publications/sensitivity-without-reproducibility/README.md): the operator-instrument measurement-boundary report
- [`../docs/reproduce_gate12a.md`](../docs/reproduce_gate12a.md): the short release/reproducibility guide
- [`../workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md`](../workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md)
- [`../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md`](../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md)
- [`../tools/`](../tools/): current narrow runner and audit surface

Local `runs/` directories are generated working outputs and are not the tracked
public evidence surface. Public release navigation should point to the DOI-bound
release bundles, selected manifests, checksums, and commit bindings first.

## Publication Directory Policy

Every public work has one landing directory under
`publications/<publication-slug>/`. Platform-specific upload bundles live one
level below it, for example `zenodo/`, `arxiv/`, or `osf/`. Adding a platform
extends the existing publication directory instead of creating a new
top-level naming convention.

The six packages published before this policy were consolidated into that
layout without changing their contents. Published Git tags and Zenodo records
remain immutable and continue to preserve the files and historical paths from
their release commits. Historical authority files may therefore mention an
older repository path; do not rewrite those frozen records merely to follow a
later layout migration.

## How GitHub And Zenodo Should Line Up

The clean versioning pattern is:

1. freeze the artifact-backed repo state with a tagged GitHub release
2. let Zenodo archive that release and mint the DOI-backed record
3. add DOI-aware paper/source updates if needed
4. tag the paper-facing source state separately if it is materially different from the artifact-freeze point

This keeps four things from getting blurred together:

- the repo state that generated the artifacts
- the frozen release state sent to Zenodo
- the paper source state
- the concept DOI versus any one version-specific DOI

## What Readers Should Be Able To Find Quickly

A public reader landing on the repo should be able to find, in a few clicks:

- what the project is
- what the current report claims
- where the tracked memory lives
- where the artifacts live
- which implementation snapshot matches the release
- which paper source matches the release

That is why the root [`../README.md`](../README.md) is now treated as a landing page and this `ABOUT/` folder exists at top level.
