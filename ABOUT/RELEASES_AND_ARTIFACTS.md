# Releases And Artifacts

This repo separates the paper-facing release surface from the longer moving research memory.

## What Should Be Frozen For A Public Release

For the Gate12A report line, the frozen public package should include:

- `paper.pdf`
- full paper source (`main.tex`, bibliography, figure source)
- manifest and checksum material
- a frozen implementation snapshot
- artifact bundles, or stable links to separately archived bundles

That is the release surface a reader should be able to cite, download, and inspect without needing to reconstruct the line from the entire repo history.

## What Lives Where In The Repo

Today, the ingredients already present in the repo are mainly:

- [`../zenodo-release/README.txt`](../zenodo-release/README.txt): the April 2026 Gate12A frozen technical report release bundle
- [`../zenodo-release-transport-first-defect-telemetry/README.txt`](../zenodo-release-transport-first-defect-telemetry/README.txt): the separate mathematical telemetry-note release bundle
- [`../docs/reproduce_gate12a.md`](../docs/reproduce_gate12a.md): the short release/reproducibility guide
- [`../workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md`](../workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md)
- [`../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md`](../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md)
- [`../tools/`](../tools/): current narrow runner and audit surface

Local `runs/` directories are generated working outputs and are not the tracked
public evidence surface. Public release navigation should point to the DOI-bound
release bundles, selected manifests, checksums, and commit bindings first.

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
