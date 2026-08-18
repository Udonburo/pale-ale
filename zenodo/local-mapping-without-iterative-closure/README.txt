Local Mapping Without Iterative Closure
=======================================

Full title:

  Local Mapping Without Iterative Closure: Zero-Shot and Input-Output-Only
  In-Context Graph-XOR Capability Boundaries in Qwen3

Author: Aoi Kawasaki
Date: 2026-08-18
DOI: 10.5281/zenodo.21992852
Resource type: Publication / Technical note
Review status: Not peer reviewed
Package status: ZENODO_PUBLISHED_MINOR_FILE_CORRECTION

Primary result
--------------

Correct input-output demonstrations made a two-input XOR mapping behaviorally
available in Qwen3-4B and Qwen3-8B, but no correct-demonstration P3 cell met
the predeclared score-signal criterion on the frozen ledgers at 4, 16, or 64
demonstrations.

Correct-demonstration P2 cells passed behaviorally at 16 and 64 shots in 4B
and at 4, 16, and 64 shots in 8B. Under the joint correct-versus-shuffled
formation rule, the earliest qualifying shot count was 16 for 4B and 4 for
8B; the corresponding 16- and 64-shot correct-versus-shuffled contrasts in
8B did not satisfy the joint formation criterion.

The result does not establish a reusable XOR algorithm, a hidden iterative
state, a global-obstruction representation, or absence of hidden information.
In the ICL branches, P5 remained unopened; the zero-shot scout separately
measured P4/P5 and found no detected signal. A2-M, decorated-theta B,
activation extraction, naturality, and causal intervention remained unopened
across every branch.

Files
-----

  local-mapping-without-iterative-closure.pdf
    Seven-page DOI-bearing technical report.

  local-mapping-without-iterative-closure.tex
    LaTeX source for the DOI-bearing report.

  capability-matrix.png
    Integrated publication figure spanning the zero-shot and ICL branches.

  reproducibility-capsule.zip
    Public-safe plans, runners, exact result summaries, cell-level metrics,
    matrix source data, model/runtime bindings, report LaTeX source, licenses,
    and internal SHA-256 inventory.

  zenodo-description.md
    Suggested public description for the Zenodo record.

  zenodo-metadata.json
    Human-reviewable metadata binding. It is not a publication receipt.

  release_manifest.json
    Public release identity, scope, licensing, and result summary.

  pdf_build_receipt.json
    PDF build, page geometry, checksum, and visual-QA record.

  github-release-body.md
    Concise companion release text for the public repository.

  LICENSES.txt
    Licensing split between the report/figure and bundled code.

  CHECKSUMS-SHA256.txt
    SHA-256 inventory for the top-level publication package.

Publication status
------------------

The package is bound to the published Zenodo DOI above. This corrected public
export removes an internal project codename without changing the scientific
results, frozen branch artifacts, model outputs, or claim boundary.

Study position
--------------

This is a standalone graph-XOR capability-boundary study. It is related to,
but is not a version or continuation of, the four replay/artifact-structure
reports listed in the Zenodo metadata.
