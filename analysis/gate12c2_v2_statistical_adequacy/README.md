# Gate12C-2 v2 statistical adequacy audit

This directory contains a read-only diagnostic of the consumed N1 locked
synthetic result. It does not amend, rerun, or supersede the historical
`RETIRE_OR_DEMOTE` decision.

The audit separates three questions:

1. finite null-draw Monte Carlo adequacy;
2. N1 product/cross-edge geometry adequacy beyond edge marginals; and
3. descriptive reconstruction at the independent-dataset grain.

Run from the repository root, using a new output directory outside the locked
result root:

```powershell
python analysis/gate12c2_v2_statistical_adequacy/audit.py `
  --input $env:GATE12C2_LOCKED_RESULT_ROOT `
  --output $env:GATE12C2_AUDIT_OUTPUT
```

The script verifies every manifest-bound source file before analysis. Generated
CSV, JSON, and figure files are development evidence only.

## Completed audit

The 2026-08-12 run reproduced the locked p95 shift exactly and classified the
failure as:

```text
FINITE_DRAW_PRECISION_DOMINANT / N1_GEOMETRY_NOT_EXONERATED
```

The historical `RETIRE_OR_DEMOTE` decision remains unchanged. The audit supports
one bounded balanced/sequential N1 development prototype; it does not authorize
a new locked suite or a real held-out surface.

The executed companion notebook is
`gate12c2_v2_statistical_adequacy_audit.ipynb`. Targeted tests are in
`test_audit.py` and include reconstruction checks against the frozen endpoint
implementation and generator derangements.
