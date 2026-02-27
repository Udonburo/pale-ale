# IMPLEMENTATION_CHECKLIST.phase4.gate2.md - Gate 2 Telemetry Plan (v4.1.0-ssot.3)

**Spec:** `SPEC.phase4.gate2.md v4.1.0-ssot.3`  
**Goal:** Even128 algebra + H1/H2/H3 telemetry (no thresholds)

---

## PR5a - `pale-ale-rotor` (leaf math)

### 5a.1 Basis map

- [x] index<->mask for even masks (grade asc, mask asc)
- [x] tests for grade counts 1/28/70/28/1
- [x] implement `grade2_index_of_mask(m)` per `#2.2` (grade-2 masks in ascending mask order)
- [x] implement Gate1 lex index -> mask -> Even128 grade-2 index remapping
- [x] test: all 28 `(i,j)` pairs map to the correct Even128 grade-2 index via mask lookup
- [x] test: first divergence is fixed (`Gate1 k=2` is pair `(0,3)` / mask `9`, while Even128 grade-2 index `2` is pair `(1,2)` / mask `6`)

### 5a.2 sign(A,B)

- [x] implement `#3.2` exactly
- [x] tests: anti-commutation, `(e_i e_j)^2 = -1`, etc.

### 5a.3 mul_even128

- [x] deterministic nested loop order
- [x] associativity tests on many basis blades / random sparse inputs
- [x] composition time-arrow correctness tests for `left_fold_mul_time_reversed_normalize_once` (time-reversed left fold vs forward-order non-equivalence)

### 5a.4 reverse + n2 + normalize

- [x] reverse operator per grade
- [x] define `n2(A) = scalar_part(A * ~A)`
- [x] **MUST satisfy lemma:** `n2(A) == Sum a_i^2` for this representation  
      (add explicit test on random Even128 values comparing both paths)
- [x] normalization uses `sqrt(n2)`; reject non-finite or <=0

**Embedded Gate1 rotor test (corrected expectation):**

- [x] Take Gate1 D1 rotor (`s=sqrt(0.5)`, one bivector component=`sqrt(0.5)`), embed -> Even128
- [x] Verify `n2 == 1.0` (within epsilon), because coefficients are L2-normalized and lemma holds.
- [x] **H3 initialization test:** Verified in Gate2 telemetry tests (`h3_total_ratio_is_zero_for_single_embedded_rotor_chain`)

### 5a.5 inner(A,B)

- [x] implement `inner = scalar_part(~A * B)`
- [x] verify lemma: equals coefficient dot (within epsilon) on random inputs

### 5a.6 left-fold normalize-once

- [x] strict left fold for products
- [x] normalize exactly once at end (no intermediate normalize)

---

## PR5b - `pale-ale-diagnose` telemetry

- [x] reuse Gate1 SimpleRotor29 construction, embed to Even128
- [x] H1-B / H2 / H3 as per SPEC
- [x] explicit missing counts; no silent drops

---

## PR5c - `pale-ale-diagnose` orchestrator + `pale-ale-cli` wiring

- [x] Gate2 JSON v1 input contract implemented (`run_id`, `samples`, optional labels/lengths, strict Vec8 validation)
- [x] diagnose orchestrator implemented (`run_gate2_and_write`) with deterministic per-sample PR5b execution
- [x] telemetry-only run aggregation implemented (no threshold-based invalidation)
- [x] artifacts implemented: `manifest.json`, `summary.csv` (single row), `samples.csv` (stable order)
- [x] lightweight Gate2 manifest validator implemented (`validate_gate2_manifest_json`)
- [x] CLI subcommand implemented: `pale-ale gate2 run` (thin call to diagnose orchestrator)
- [x] determinism tests added (byte-identical artifacts on repeated runs)
- [x] verification passed:
  - `cargo test -p pale-ale-diagnose`
  - `cargo test -p pale-ale-cli`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
