# IMPLEMENTATION_CHECKLIST.phase4.gate3.md

**Spec:** `SPEC.phase4.gate3.md v4.2.0-draft.1`  
**Scope in this file:** PR6a (compute-only)

## PR6a -- Gate3 compute-only telemetry

- [x] Add `crates/diagnose/src/metrics_common.rs` and share Gate2/Gate3 metric primitives.
- [x] Refactor Gate2 to use shared `projective_chordal_distance` and `higher_grade_energy_ratio`.
- [x] Add `crates/diagnose/src/gate3_telemetry.rs` compute-only module.
- [x] Implement `compute_gate3_telemetry` with closed `Gate3MissingReason`.
- [x] Reuse Gate2 rotor construction path semantics (normalize vec8 -> simple rotor -> embed).
- [x] Enforce `P_t = R_{t+1} * reverse(R_t)` order for local torsion.
- [x] Implement deterministic nearest-rank p90 with `total_cmp`.
- [x] Add tests:
  - [x] `TooFewSteps`
  - [x] `InvalidVec8`
  - [x] antipodal missing-step handling
  - [x] smooth vs kink sensitivity
  - [x] bitwise determinism (`to_bits`)
- [x] Re-export minimal Gate3 compute API from `crates/diagnose/src/lib.rs`.
