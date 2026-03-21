# IMPLEMENTATION_CHECKLIST.phase4.gate3.md

**Spec:** `SPEC.phase4.gate3.md v4.2.0-draft.1`  
**Scope in this file:** PR6a + PR6b

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

## PR6b -- Gate3 run orchestration + artifacts + CLI

- [x] Add `crates/diagnose/src/gate3.rs` with compute orchestration entrypoint:
  - [x] `run_gate3_and_write(...)`
  - [x] deterministic sample processing order (`sample_id ASC`, then input order)
- [x] Implement Gate3 machine artifacts:
  - [x] `manifest.json` (deterministic key order)
  - [x] `summary.csv` (single row, fixed schema)
  - [x] `samples.csv` (per sample, fixed schema)
- [x] Add lightweight Gate3 manifest validator:
  - [x] required key checks
  - [x] fixed ID checks
  - [x] sci_17e float string checks
  - [x] forbidden token checks (`NaN` / `Inf`)
- [x] Re-export Gate3 run API from `crates/diagnose/src/lib.rs`.
- [x] Add CLI wiring (`pale-ale gate3 run`) in `crates/cli/src/main.rs`:
  - [x] required identity flags
  - [x] thin wrapper over diagnose orchestrator
  - [x] JSON envelope + non-JSON summary output
- [x] Add smoke tooling:
  - [x] `tools/run_gate3_smoke.ps1`
  - [x] `tools/README_smoke_gate3.md`
  - [x] determinism hash checks (A/B)
  - [x] smooth vs kink comparison output
- [x] Tests/quality gates:
  - [x] `cargo test -p pale-ale-diagnose`
  - [x] `cargo test -p pale-ale-cli`
  - [x] `cargo test --workspace`
  - [x] `cargo clippy --workspace --all-targets -- -D warnings`
