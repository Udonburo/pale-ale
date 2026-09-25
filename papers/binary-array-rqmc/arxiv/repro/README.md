# Binary Array-RQMC reproduction companion

This directory accompanies *Exact Binary Projections for Array-RQMC: Joint Laws and Pathwise-Preserving Execution* by Aoi Kawasaki.

It provides a small, self-contained way to inspect and test the two execution rewrites. No private workspace, pretrained model, GPU, or external dataset is required. The source kernels retain the computational function bodies of the measured implementations. Imports and module layout have been simplified; unused experiment drivers and method-selection machinery are omitted.

## What can be reproduced here?

- The full consumed-vector law for small scrambles, checked by exact finite enumeration.
- Its support size, covariance, fixed-linear-readout variance, and an equal-covariance/different-nonlinear-variance witness, checked using exact integer and rational arithmetic.
- The SciPy 1.15.3 direction/scrambling convention, including injected common scrambles.
- Exact label ordering, absorption, float64 ties, full trajectories, and random-generator end states against full-sort references.
- A small three-implementation timing comparison on your own CPU.
- Both manuscript figures and Tables 2–3 and C1–C2, regenerated from the bundled **saved** numerical inputs.

The revised companion additionally includes all 720 original timing blocks
from the order-maintenance study and a script that recomputes Table 3's
point estimates and paired-round confidence intervals. It also includes the
Appendix B stopped-chain example, independently enumerated with rational state
propagation and integer arithmetic with fixed labels.

This is not the full original experimental archive. Per-replicate calibration
and validation datasets and baseline orchestration are not included. Table 2's
intervals and Appendix C's precision results are retained from saved analyses;
their regeneration is not a rerun of those experiments. The CRN and
antithetic-tape results are not newly benchmarked by this companion.

## Availability and citation

The version-specific archival identifier for the original v1.0.0 manuscript and companion is [doi:10.5281/zenodo.22728405](https://doi.org/10.5281/zenodo.22728405), version 1.0.0. The new stopped-chain and timing-reanalysis files in this revision are not in that deposit. For these additions, retain this directory from the exact Git commit cited by the revised manuscript. The development repository's moving branch is not an exact source reference. The companion source revision and the manuscript revision are recorded separately; the commands below do not require a manuscript build.

## Environment and commands

The reference scientific environment is Python 3.11.2, NumPy 1.26.4, SciPy 1.15.3, and Numba 0.61.2. Figure generation additionally uses Matplotlib 3.10.8. The version assertion in the projection test is intentional: the statement concerns a specified Sobol implementation, not every version or construction.

From this directory, install in a separate environment:

~~~text
python -m venv .venv
~~~

Activate it using the command appropriate to your shell, then:

~~~text
python -m pip install -r requirements.txt
python -m unittest discover -v
python run_smoke.py --sizes 512 4096 --repeats 3
python render_results.py
python reanalyze_timing.py
python stopped_witness.py
~~~

The first test invocation compiles small Numba helpers. Tests and the smoke driver do not overwrite saved study results. Numba may create a local cache. The renderer writes only `TABLES.md` and the two PNG/SVG figure pairs under `../figures/`.

For a quicker installation check:

~~~text
python run_smoke.py --sizes 32 64 --repeats 2
~~~

All simulation here is a consistency check or engineering smoke test. The smoke driver prints new local timings to standard output; it does not replace manuscript numbers, perform MSE calibration, or claim independent confirmation of the original performance estimates. Its warm kernel comparison omits the original multi-block calibration workflow, so it should not reproduce acquisition-inclusive costs.

## Files

| File | Role |
|---|---|
| `reference.py` | Original library-array transition loop, chain parameters, and independent readout noise. |
| `projection.py` | Exact rank-bit generator and a full-sort array execution. |
| `ordered.py` | Incremental branch merge, tie repair, and absorbed-state merge. |
| `test_projection.py` | Finite-law, SciPy-convention, and common-sign reference tests. |
| `test_projection_structure.py` | Standalone exact support, covariance, linear-variance, and nonlinear-witness checks for Corollary 1. |
| `test_ordered.py` | Exhaustive and adversarial ordering tests and complete-path comparisons. |
| `run_smoke.py` | Small CPU timing driver for library, direct, and order-maintained arrays. |
| `data/cases.json` | The 16 execution configurations representing eight model laws. |
| `data/saved_results.json` | Saved profile observations, aggregate timing statistics, calibration costs, and target-validation summaries. |
| `data/provenance.json` | Source paths/hashes and export field mappings; source paths are not dependencies. |
| `render_results.py` | Saved-data checks, table regeneration, and scientific figure generation. |
| `stopped_witness.py` | Two exact enumerations of the complete stopped-chain payoff distribution in Appendix B. |
| `data/stopped_witness_results.json` | Exact means, variances, and enumeration sizes for the added mathematical example. |
| `data/ordering_timing_rows.csv` | All 720 original timing blocks from the second study; block times are in seconds. |
| `reanalyze_timing.py` | Recomputes Table 3 from the original rows, including the paired-round bootstrap. |
| `test_revision_evidence.py` | Distribution, covariance, and timing-reanalysis checks for the additions. |
| `test_saved_results.py` | Grain, arithmetic, and integer crossover checks. |

The three array kernels return labeled noisy outputs, signal outputs, and execution statistics. Their `record_trace=True` option retains states for testing only and should not be used for speed comparisons. The tests run against functionally separate full-sort references rather than merely comparing two summaries from one reducer.

## Interpretation

The library and direct binary generators have the same **joint consumed-vector law under the ideal randomization model**. They do not share the same finite integer-seed-to-output map. The direct and order-maintained arrays do share that map under the stated arithmetic and transition conditions. The tests keep these two guarantees separate.

The suite contains 22 tests. The four Corollary 1 checks require only the Python
standard library and can be run separately with
`python -m unittest test_projection_structure -v`. They enumerate mathematical
laws, not new simulator observations. Their nonlinear witness is not an
additional benchmark payoff or a claimed advantage over every antithetic law.

The default smoke workload is the quadratic case with horizon 60 and barrier 5.1. `N` is the number of paths per process; one block uses three arrays of length `N`. A manuscript estimator evaluation averages `k` independent blocks. The smoke driver uses `k=1` and a few repetitions; these are timing repetitions, not selected accuracy-target configurations.

The warm CRN and antithetic-tape entries in the saved target table include initialization of input-encoding masks in the measured baseline implementations. Their fused transition loops do not need the mask values. This included cost is not an unavoidable cost of those methods. No retrospective adjustment has been applied.

For Figure 1, the first batch has eight instrumented profiles per configuration and implementation, the second three. Component sums exclude untimed work and must not be substituted for separately measured end-to-end times.

For Figure C1, the model is `total cost = setup cost + reuse count × warm evaluation cost`. Setup includes previously collected calibration data in the main plot. The 2,799-evaluation quadratic crossover is therefore conditional on that accounting and unchanged workload. Treating prior variance acquisition as sunk gives 476 instead.

## Source and license

Computational definitions are extracted from the study sources listed in `data/provenance.json`; display identifiers and import paths are adapted. The source paths record lineage and do not imply that the omitted original datasets are bundled. Saved interval endpoints are retained as reported, with arithmetic checks where the compact export suffices.

Code retains the repository's Mozilla Public License 2.0; the accompanying `LICENSE` is the full license text. The manuscript, figures, documentation, and bundled result data are CC BY 4.0. See the root `LICENSES.txt` for the per-file scope. Original dependencies retain their respective licenses.

## Original timing export

`ordering_timing_rows.csv` exports the original `rows` arrays from the 16 files
`workstream/local/stochastic_execution_rewriting/binary_array_ordering/timing/law{0..7}_{recent,distant}.json`.
Their paths record provenance, not runtime dependencies. `source_condition`
retains the original name; `condition` maps law 0 to case 01, etc.
`condition_order` follows the original study configuration, and `source_row`
is the zero-based position within a condition's JSON array. Thus it preserves
within-condition execution order; it is not a reconstructed global timestamp.
The numeric fields (`n`, `round`, `reps`, `wall`, `cpu`, `calls`, `transitions`)
are direct exports, with no imputed observations. Source JSON file identities
are already recorded in `data/provenance.json`.

Wall and CPU times are for the whole block of `reps` estimator evaluations.
Each fixed-size evaluation uses `k=1`; repetitions time separate estimates.
`reanalyze_timing.py` divides by that count, forms each condition's ratio of
mean times, and takes the geometric mean of 16 ratios. It resamples the five
rounds jointly for the two array methods, independently within each fixed
condition, using 2,000 replicates and the original NumPy generator seed 2091221.
It verifies both ratios and interval endpoints against the stored results.
The case set is held fixed: these intervals measure timing variability on the
recorded machine, not uncertainty across a population of models.
