Title: Compression-Interleaved Parenthesization Defects in LLM Replay
Artifact Graphs: A Predeclared Null Test

Author: Aoi Kawasaki
Package date: 2026-07-14
Package status: archival release capsule
DOI: https://doi.org/10.5281/zenodo.21355572

This capsule contains the manuscript, the frozen protocol and descriptive
reporting boundary, exact code snapshots, and the generated records needed to
audit the 24 endpoint results from the Gate12C-1 predeclared null test. No new
scientific experiment was run to prepare this package.

The package is associated with the reserved Zenodo DOI above and the matching
immutable Git release tag:

  gate12c1-parenthesization-defects-v1.0.0

Contents
--------

- gate12c1-parenthesization-defects.pdf
  Final rendered manuscript. The build log is free of overfull boxes,
  underfull boxes, and unresolved references; all eleven pages were rendered
  to PNG and visually reviewed.

- gate12c1-parenthesization-defects.tex
  TeX source used to build the manuscript.

- protocol/
  The frozen execution plan, first empirical result memo, and the post-hoc
  descriptive reporting addendum.

- code/
  Exact recorded byte snapshots of the measurement runner and grid summarizer,
  plus their shared feasibility helper. The two recorded scripts came from
  different Windows checkout contexts and intentionally have different line
  endings; code/README.txt records the hashes and explains the provenance.

- LICENSES.txt
  File-level mapping between CC BY 4.0 for non-code publication materials and
  MPL 2.0 for the bundled source code. The full MPL 2.0 text is in LICENSE.

- results/case_manifest.json
  Canonical twelve-case manifest used for execution.

- results/runner-manifests/
  Per-case runner manifests and checksums for all twelve completed cases.

- results/summary/gate12c1_run_q_tests.csv
  Complete endpoint-level table for the 24 predeclared endpoints.

- results/summary/gate12c1_block_q_scores.jsonl
  Block-level score export underlying the endpoint aggregation.

- results/summary/
  The complete generated summary surface: cycle- and block-level scores,
  endpoint tests, secondary telemetry, case inventory, grid summary, grid
  manifest and checksums, and the tracked summary readout.

- environment_receipt.json
  A transparent post-execution reconstruction of the relevant software and
  operating-system versions. No contemporaneous environment lock was found;
  the receipt says so explicitly and must not be read as exact execution-time
  proof.

- release_manifest.json
  Provenance, expected hashes, package scope, and release-state record.

- pdf_build_receipt.json
  Build-engine provenance, PDF digest, page count, and visual-QA record.

- CHECKSUMS-SHA256.txt
  Package-wide SHA-256 checksums. Generated only after all files are final.

Licensing
---------

Unless a file states otherwise, every capsule file except the contents of
code/ and the MPL license text in LICENSE is distributed under Creative
Commons Attribution 4.0 International (CC BY 4.0). Source code in code/ is
distributed under the Mozilla Public License 2.0 (MPL 2.0). See LICENSES.txt
for the file-level mapping and license URLs, and LICENSE for the full MPL 2.0
text.

Reuse should cite https://doi.org/10.5281/zenodo.21355572. This citation
request does not replace or add to the license terms.

Repository: https://github.com/Udonburo/pale-ale
