# Compression-Interleaved Parenthesization Defects in LLM Replay Artifact Graphs: A Predeclared Null Test

Status: DOI-bearing release-ready companion research paper source
Protocol name: Gate12C-1
Zenodo DOI: https://doi.org/10.5281/zenodo.21355572

This directory contains the paper source for a predeclared test of whether a
compression-interleaved parenthesization defect is larger in LLM replay
artifact cycles than under an edge-spectrum-matched, graph-unconstrained
orientation null.

The paper reports two distinct levels of result:

- confirmatory: all 24 endpoints were informative and coverage-complete, and
  none met the frozen support rule for positive directional excess;
- descriptive: all 24 endpoint median log-ratios were negative, but the
  reverse-direction hypothesis was not predeclared and is not treated as
  confirmatory evidence. On the post-hoc endpoint-summary scale fixed before
  aggregation, exponentiated values range from 0.028 to 0.236 with median
  0.066.

The current null preserves marginal edge spectra while breaking shared-node
realizability. The paper therefore does not conclude that the observed graphs
are coherent or that graph consistency caused the negative direction.
Graph-constrained nulls, held-out runs, a predeclared reverse-direction rule,
positive controls, and sensitivity checks belong to a separate prospective
Gate12C-2 plan.

The manuscript is grounded in these tracked records:

- workstream/235_GATE12C1_FIRST_EMPIRICAL_EXECUTION_PLAN.md
- workstream/236_GATE12C1_FIRST_EMPIRICAL_RESULT_MEMO.md
- workstream/240_GATE12C1_POST_HOC_DESCRIPTIVE_REPORTING_ADDENDUM.md
- tools/run_gate12c_compressed_overlap_associator.py
- tools/summarize_gate12c1_first_empirical_grid.py

The archival capsule at https://doi.org/10.5281/zenodo.21355572 packages the
immutable case manifest, endpoint and block-level score files, checksums,
runner outputs, software-environment receipt, manuscript source, and final PDF.
The matching repository snapshot is identified by the immutable tag
`gate12c1-parenthesization-defects-v1.0.0`.

## Licensing

The manuscript and archived non-code publication materials are licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0). Source code in the
archival capsule remains licensed under the Mozilla Public License 2.0
(MPL 2.0). The file-level mapping is recorded in the capsule's `LICENSES.txt`.

Build target, once a local LaTeX installation is available: pdflatex main.tex.
