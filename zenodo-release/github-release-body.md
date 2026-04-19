## Paper

Structural Replay in Dense Transformers Under a Frozen FP32 Regime: Evidence
from Closed 3B/4B Models and Boundary Results

Author: Aoi Kawasaki  
Date: 2026-04-09

## Zenodo DOI

https://doi.org/10.5281/zenodo.19483162

This DOI is the frozen paper release DOI for the April 2026 Gate12A technical
report surface. The associated transport-first mathematical telemetry note is a
separate citable surface at https://doi.org/10.5281/zenodo.19569052. The earlier
replication checkpoint / prior release surface remains available at
https://doi.org/10.5281/zenodo.19340221.

## Included assets

- `paper.pdf`
- `paper-source.zip`
- `selected-manifests.zip`

## Summary

This release accompanies the frozen technical report describing structural
replay under a frozen FP32 dense-transformer regime. It is the paper-facing
release bundle for the current Gate12A report surface, with selected manifests
and checksums included for provenance and boundary-condition inspection.

The downstream Gate12A cross-model replay summaries and structural quartet
verdicts are frozen under:

`084eb7878d8cb016243950e1cf4b4bd7379daaba`

One upstream Gate8 candidate-execution artifact exception:

`Qwen/Qwen2.5-3B-Instruct` transcript bundle under
`58d06742f23a0bc7ba25c6ecde790e2e03b4324e`

Closed-mainline Gate8 execution is mixed CPU/CUDA:

- CPU for 8/9 runs
- CUDA for 1/9 run

The release also includes the LaTeX source bundle used to build the frozen
report PDF.
