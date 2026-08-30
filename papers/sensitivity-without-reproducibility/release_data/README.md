# Public Stage M release data

This directory contains a model-forward-free public projection of the frozen
Stage M result used by *Sensitivity Without Reproducibility*.

The release records expose all 60 layer-block values required to audit the
reported qualification boundary:

- split-half singular-spectrum floor `f`;
- exact-square responses for both nuisance halves;
- broken-square responses for both nuisance halves;
- frozen broken-sensitivity threshold `theta`;
- layer and block qualification states;
- edge/path conditioning summaries and ranks;
- both cross-fitted return-energy components;
- packet-disagreement diagnostics; and
- map-derived competence, with the primary amplitude present only for blocks
  that passed the frozen all-three-layer gate.

The source is the immutable Stage M result with SHA-256
`8c56c308856d486b77356a42fe54ef0bea8f91e486dc5dd3820c3a60db5be772`.
The extractor refuses any other source and verifies the paper's 20-block,
60-layer-block, 59/60 sensitivity, 35/60 reproducibility, and 5/20 joint-pass
aggregates before writing output.

`stage_m_release_data_manifest.json` records the execution and retrieval
bindings. It also closes the historical Qwen3.6-27B root-manifest receipt:
the root manifest retained the pre-operator `execution_claim.json` SHA, while
the final claim appended the later operator terminal stage. The retrieved and
remote-redownloaded final object are byte-identical. No response, activation,
metric, or scientific state was affected, and the historical manifest remains
unchanged.

Stage E remained unopened. These files contain no behavioral outcome, no
representation--function association, and no post-hoc rescue analysis.

## Regeneration

Run the extractor with the frozen Stage M source JSON and this directory as the two
arguments. It uses only the Python standard library and performs no model or
network operation.
