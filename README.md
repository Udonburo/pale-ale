# pale-ale

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19340221.svg)](https://doi.org/10.5281/zenodo.19340221)

**pale-ale is a public-facing research sandbox for structural approaches to local observation, consistency, and learned systems.**

Its current public checkpoints are LLM-centered. The main report surface is the Gate12A line: a frozen FP32 dense-transformer replay regime used to test one narrow part of a broader structural research program through structural replay evidence and boundary cases.

## About

This repo is organized around a staged research line.

- **Gate** means a named checkpoint in the line: what is established, what is open, and what is still denied
- **Workstream** means the numbered tracked memory that records how the line moved from one checkpoint to the next
- the current paper-facing surface is Gate12A, not the entire historical repo at once

This repo is a public-facing experimental surface for a broader structural research program. Its public scope is intentionally narrower than that broader canvas: the checkpoints tracked here are currently framed around LLMs and related learned-system artifacts unless they are explicitly widened. Public claims are intentionally checkpointed narrowly.

If you want the plain-language orientation first, go to [`ABOUT/`](ABOUT/README.md).

## If You Are Here For...

### The current report and artifacts

- [`workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md`](workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md)
- [`workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md`](workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md)
- [`runs/`](runs/)
- [`tools/`](tools/)

### A human-readable explanation of the repo

- [`ABOUT/README.md`](ABOUT/README.md)
- [`ABOUT/WHAT_THIS_REPO_IS.md`](ABOUT/WHAT_THIS_REPO_IS.md)
- [`ABOUT/WORKSTREAM_AND_GATES.md`](ABOUT/WORKSTREAM_AND_GATES.md)
- [`ABOUT/RELEASES_AND_ARTIFACTS.md`](ABOUT/RELEASES_AND_ARTIFACTS.md)
- [`ABOUT/FORWARD_DIRECTIONS.md`](ABOUT/FORWARD_DIRECTIONS.md)

### The full tracked research memory

- [`workstream/README.md`](workstream/README.md)

## Current Release Shape

The intended frozen release surface for the Gate12A report is:

- `paper.pdf`
- paper source (`main.tex`, bibliography, figure source)
- manifest and checksum material
- a frozen implementation snapshot
- artifact bundles or stable links to those bundles

The repo is being shaped so that this release surface is easy to find and does not require reconstructing the entire workstream history first.

## Repo Map

- [`ABOUT/`](ABOUT/README.md): human-facing explanation and release guidance
- [`workstream/`](workstream/README.md): numbered tracked research memory
- [`runs/`](runs/): artifacts, manifests, checksums, statuses, and replay outputs
- [`tools/`](tools/): current narrow Python-side runner and audit surface
- [`src/`](src/) and [`crates/`](crates/): retained implementation and infrastructure
- [`docs/`](docs/) and [`specs/`](specs/): longer-form design and specification material

## Scope

The current public claim surface is intentionally narrow relative to the broader research canvas:

- study local observation, replay, transport, closure, and boundary behavior under declared surfaces
- use LLMs as the current public sandbox rather than as the final scientific boundary
- keep current empirical claims narrower than the surrounding mathematical ambition
- use current checkpoints as testbeds rather than as a final ontology
- separate frozen report surfaces from broader exploratory directions
- avoid letting future extensions silently rewrite current evidence

Future mathematical and empirical directions are visible in [`ABOUT/FORWARD_DIRECTIONS.md`](ABOUT/FORWARD_DIRECTIONS.md), but they are not treated as already-earned results.
