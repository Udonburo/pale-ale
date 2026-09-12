# Exact Binary Projections for Array-RQMC: Joint Laws and Pathwise-Preserving Execution

Aoi Kawasaki. Preprint; not peer reviewed.

## Contribution

For a specified two-dimensional Sobol net with linear matrix scrambling and a
digital shift, the complete rank-ordered binary innovation vector is uniform
over exactly N outcomes at N = 2^m paths. An odd mask and a fair bit describe the
law, requiring m independent fair bits per step under the stated ideal
randomization model. Independent adjacent antithetic pairs have the same
covariance but need not have the same higher-order law; exact nonlinear
witnesses give variances 0 versus 2 and 4 versus 2.

This characterization enables direct generation without generating unused
coordinates or sorting the point set. A second rewrite maintains exactly the
reference state order for stopped scalar binary chains with monotone branches,
including absorption and floating-point ties.

## Measured results and limits

Across eight fixed benchmark cases in two input encodings, state-order
maintenance reduced warm execution time by 37.2% at N = 512 and 67.5% at
N = 4,096 relative to direct binary generation with full state sorting.
The encodings do not create sixteen independent model laws. The measurements
are from one CPU/software stack, not a held-out hardware or dynamics study.

The joint-law equivalence of the library and direct generators is an ideal-law
statement, not equality under the same finite integer seed. Direct and
order-maintained execution do preserve the same finite-seed trajectories and
outputs under the stated arithmetic conditions. No new convergence rate or
universal variance advantage is claimed.

Comparisons with the measured CRN and antithetic implementations remain
workload- and reuse-dependent. Avoidable mask initialization was included in
the measured baselines; it is not an intrinsic CRN cost. Startup and calibration
can outweigh warm-time savings. The publication retains those recorded costs.

## Included materials and reproducibility scope

The record contains the manuscript PDF and a self-contained reproducibility
capsule with manuscript/build sources, three array kernels, 19 tests, saved
table/figure inputs, two figures, dependency versions, and SHA-256 inventories.
The tests include exact finite enumeration, within-project separate reference
implementations, absorption and floating-point tie checks, and saved-data
arithmetic. They are not third-party peer review.

The capsule does not include every raw calibration/validation observation,
bootstrap sample, timing round, or baseline/calibration driver. Regenerating
saved tables and figures is not a rerun of the full original experiment.
No GPU, pretrained model, private workspace, or external dataset is required.

## Rights and disclosure

Publication materials and saved data: CC BY 4.0. Code: MPL-2.0, with per-file
scope in LICENSES.txt. Generative-AI assistance is disclosed in the manuscript.
The version-specific archival identifier is doi:10.5281/zenodo.22728405.
