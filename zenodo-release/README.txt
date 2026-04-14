Title: Structural Replay in Dense Transformers Under a Frozen FP32 Regime:
Evidence from Closed 3B/4B Models and Boundary Results

Author: Aoi Kawasaki
Release date: 2026-04-09

Contents:
- paper.pdf
  Frozen technical report PDF for the public release.
- paper-source.zip
  LaTeX source bundle used to build the report PDF.
- selected-manifests.zip
  Selected archived manifests and status files referenced by the report for
  provenance and boundary-condition checks.

Mainline replay summaries and structural quartet verdicts are frozen under:
084eb7878d8cb016243950e1cf4b4bd7379daaba

One upstream Gate8 exception:
The Qwen/Qwen2.5-3B-Instruct transcript candidate-execution artifact was
generated under:
58d06742f23a0bc7ba25c6ecde790e2e03b4324e

Execution regime summary:
- FP32 execution
- Gate8 candidate-execution manifests: CPU for 8/9 closed-mainline runs,
  CUDA for 1/9 run
- Frozen protocol defined at the level of precision, observation surface,
  artifact format, and replay rules

Repository:
https://github.com/Udonburo/pale-ale
