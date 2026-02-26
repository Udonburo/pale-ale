# IMPLEMENTATION_CHECKLIST.phase4.gate2.md - Gate 2 Telemetry Plan (v4.1.0-ssot.3)

**Spec:** `SPEC.phase4.gate2.md v4.1.0-ssot.3`  
**Goal:** Even128 algebra + H1/H2/H3 telemetry (no thresholds)

---

## PR5a - `pale-ale-rotor` (leaf math)

### 5a.1 Basis map

- [ ] index<->mask for even masks (grade asc, mask asc)
- [ ] tests for grade counts 1/28/70/28/1
- [ ] implement `grade2_index_of_mask(m)` per `#2.2` (grade-2 masks in ascending mask order)
- [ ] implement Gate1 lex index -> mask -> Even128 grade-2 index remapping
- [ ] test: all 28 `(i,j)` pairs map to the correct Even128 grade-2 index via mask lookup
- [ ] test: first divergence is fixed (`Gate1 k=2` is pair `(0,3)` / mask `9`, while Even128 grade-2 index `2` is pair `(1,2)` / mask `6`)

### 5a.2 sign(A,B)

- [ ] implement `#3.2` exactly
- [ ] tests: anti-commutation, `(e_i e_j)^2 = -1`, etc.

### 5a.3 mul_even128

- [ ] deterministic nested loop order
- [ ] associativity tests on many basis blades / random sparse inputs
- [ ] **Time Arrow Test:** Verify `x2' = (R12 * R01) * x0 * ~(R12 * R01)` and `x2 = R12 * (R01 * x0 * ~R01) * ~R12` match within epsilon.

### 5a.4 reverse + n2 + normalize

- [ ] reverse operator per grade
- [ ] define `n2(A) = scalar_part(A * ~A)`
- [ ] **MUST satisfy lemma:** `n2(A) == Sum a_i^2` for this representation  
      (add explicit test on random Even128 values comparing both paths)
- [ ] normalization uses `sqrt(n2)`; reject non-finite or <=0

**Embedded Gate1 rotor test (corrected expectation):**

- [ ] Take Gate1 D1 rotor (`s=sqrt(0.5)`, one bivector component=`sqrt(0.5)`), embed -> Even128
- [ ] Verify `n2 == 1.0` (within epsilon), because coefficients are L2-normalized and lemma holds.
- [ ] **H3 initialization test:** Verify `higher_grade_energy_ratio == 0.0` immediately after embedding a single Gate1 rotor.

### 5a.5 inner(A,B)

- [ ] implement `inner = scalar_part(~A * B)`
- [ ] verify lemma: equals coefficient dot (within epsilon) on random inputs

### 5a.6 left-fold normalize-once

- [ ] strict left fold for products
- [ ] normalize exactly once at end (no intermediate normalize)

---

## PR5b - `pale-ale-diagnose` telemetry

- [ ] reuse Gate1 SimpleRotor29 construction, embed to Even128
- [ ] H1-B / H2 / H3 as per SPEC
- [ ] explicit missing counts; no silent drops
