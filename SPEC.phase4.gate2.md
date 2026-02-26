# SPEC.phase4.gate2.md - Phase 4 Gate 2 Telemetry (v4.1.x)

**Closed Algebra (Cl+(8)) -> Holonomy / Closure -> Higher-Grade Energy**  
**SSOT Status:** **DRAFT-FROZEN CANDIDATE** (Telemetry Protocol; no thresholding in v4.1.0)  
**Spec Version:** `v4.1.0-ssot.3`  
**Relation:** Gate1 SSOT remains frozen at `SPEC.phase4.md v4.0.0-ssot.9`  
**Non-Negotiables:** determinism, auditability, no silent failure, strict gate separation

---

## 0. Purpose and Boundary

Gate 2 enables **composition-dependent observables** by upgrading representation to a **closed even subalgebra** `Cl+(8)` (128D). Gate 2 is **telemetry only** in v4.1.0: no thresholds, no run invalidation. Thresholding is reserved for v4.1.1+ (`#14`).

---

## 0.1. Forbidden (v4.1.0) (Must)

To preserve Gate 2 as a **telemetry protocol** (not a gate), the following are hard-banned in `v4.1.0`:

1. **Threshold-based invalidation**
   - No new run invalidation rules may be introduced in Gate 2.
   - Gate 2 outputs metrics only; interpretation and thresholding are reserved for `v4.1.1+` (`#14`).

2. **Intermediate normalization inside composition chains**
   - Composition MUST be strict left-fold without intermediate normalization.
   - Normalization is permitted **exactly once at the end** (`strict_left_fold_time_reversed_normalize_once_v1`).

3. **H2 loop shortcut**
   - In triangle holonomy, it is forbidden to set `R20 = ~R02` or derive `R20` from an existing rotor by reuse.
   - `R20` MUST be constructed directly as `even_rotor(ans_{i+2} -> ans_i)` as specified in `#8.2`.

---

## 1. Observables (Gate 2)

- **H1-B (AnsWalk Closure Error)**: composed adjacent ans->ans rotors vs endpoint rotor.
- **H2 (Triangle Loop Holonomy)**: 3-step loop closure.
- **H3 (Higher-Grade Energy Ratio)**: grade >=4 energy share after composition.

Interpretation deferred.

---

## 2. Representation: `Even128` in `Cl+(8)` (Must)

### 2.1 Blade encoding

Basis vectors `{e0..e7}`. Blade mask `m in [0,255]` with bit `i` indicating inclusion.

### 2.2 Even basis ordering (Must)

Even masks only, ordered by:

1. grade (popcount) ascending among `{0,2,4,6,8}`
2. mask ascending within grade

This ordering is canonical.

### 2.3 Dimension check (Must)

`1 + 28 + 70 + 28 + 1 = 128`.

---

## 3. Algebra (Must): deterministic blade multiplication

### 3.1 Blade product rule (Euclidean signature)

For masks `A,B`:

`e_A * e_B = sign(A,B) * e_(A xor B)`

### 3.2 Sign algorithm (Must; fully fixed)

`sign(A,B) = (-1)^S(A,B)`

`S(A,B) = Sum_{i in set_bits(A)} popcount( B & ((1<<i) - 1) )`

No alternate "swap-count" is permitted.

### 3.3 Even128 geometric product (Must)

Deterministic accumulation order:
`for i in 0..128 { for j in 0..128 { ... } }`

---

## 4. Involutions and scalar projections (Must)

### 4.1 Reverse operator `~` (Must)

For grade-k basis blade, reverse sign:
`rev_sign(k) = (-1)^(k(k-1)/2)`.

Reverse applies per-grade sign to coefficients (consistent with standard GA reverse).

### 4.2 Scalar part

`scalar_part(X)` means coefficient of mask `0` in canonical basis.

---

## 5. Norm and inner product (Must; fixes SSOT ambiguity)

### 5.1 Norm^2 definition (Must)

`n2(A) := scalar_part( A * ~A )`

**Key SSOT lemma (Must; fixes prior confusion):**  
In this representation (orthonormal blade basis, Euclidean signature, reverse as defined above),

`scalar_part( A * ~A ) = Sum_{i=0..127} (a_i)^2` (exact arithmetic)

Reason: for any basis blade `E`, `E * ~E = 1` (scalar), and scalar cross-terms vanish in an orthonormal blade basis.

**Implementation permission (Must):**  
Implementations MAY compute `n2(A)` as `Sum a_i^2` directly (fast-path), but MUST treat it as **exactly equivalent** to `scalar_part(A * ~A)` under this SSOT.  
(Do not invent alternating-sign `E0-E2+...` substitutes; that quantity corresponds to a different bilinear form.)

### 5.2 Normalization (Must)

`normalize(A) := A / sqrt(n2(A))`

If `n2` non-finite or `n2 <= 0` -> invalid (explicitly recorded).

### 5.3 Scalar inner product used by distances (Must)

Define:

`inner(A,B) := scalar_part( ~A * B )`

**Key SSOT lemma (Must):**  
Under the same basis, `inner(A,B) = Sum a_i b_i` (coefficient dot product) in exact arithmetic.

So you may compute it either as:

- algebraic path: `scalar_part(~A * B)` (definition), or
- coefficient dot fast-path: `Sum a_i b_i` (equivalent)

This resolves the `#3`/`#5` mismatch: the **definition is unique**, and the dot-product is an allowed optimized implementation.

---

## 6. Composition order (Must)

### 6.1 Strict left-fold time-reversed composition (Must)

To correctly apply a sequence of rotors `R_0, R_1, ..., R_{T-1}` to a vector via the sandwich product (e.g. `R_1(R_0(v)) = (R_1 * R_0) * v * ~(R_1 * R_0)`), older rotors MUST be on the right. Accumulation is strictly left-folded over the **time-reversed** list of rotors:

`P_raw = (((R_{T-1} * R_{T-2}) * ...) * R_0)`

### 6.2 Normalization timing (Must)

**Normalize exactly once at the end:** `P = normalize(P_raw)`  
**No intermediate normalization** in `v4.1.0-ssot.3`.

---

## 7. Gate2 rotor construction path (Must)

### 7.1 Vec8 contract

Reuse Gate1 Vec8 normalization/finiteness rules.

### 7.2 SimpleRotor29 construction (Must)

Reuse Gate1 `simple_rotor29_uv_v1` contract (atan2 theta, sqrt half-angle, antipodal split policy).

### 7.3 Embed SimpleRotor29 -> Even128 (Must)

`embed_simple29_to_even128_v1`:

- scalar -> grade0 slot (index 0 / mask 0)
- bivector coefficient at Gate1 lex index `k` corresponds to pair `(i,j)` with `i<j`; map via
  `mask = (1<<i) | (1<<j)`, then set the Even128 grade-2 slot by `grade2_index_of_mask(mask)` under `#2.2` ordering.
  Gate1 lex ordering and grade-2 mask-ascending ordering are **not equivalent**; explicit index remapping is mandatory.
- grades 4/6/8 -> 0

### 7.4 Antipodal angle-only

No Even128 rotor is fabricated. Step is `missing_even_rotor_step=true`.

---

## 8. Metrics (v4.1.0 telemetry)

### 8.1 H1-B (AnsWalk closure error) - Must

Construct adjacent ans->ans rotors:

- `R_t = even_rotor(ans_t -> ans_{t+1})` for `t=0..N-2`
- Missing/invalid steps -> H1-B missing.

Compose (time-reversed left fold, older rotors on the right):

- `R_total_raw = left_fold_mul([R_{N-2}, R_{N-3}, ..., R_0])`
- `R_total = normalize(R_total_raw)` (once)

Direct endpoint rotor:

- `R_direct = even_rotor(ans_0 -> ans_{N-1})`

Distance (projective chordal, consistent with Gate1):

- `inn = inner(R_total, R_direct)` (definition `#5.3`)
- `a = min(1.0, abs(inn))`
- `d = sqrt(max(0, 2*(1 - a)))`

### 8.2 H2 (triangle loop holonomy) - Must

For each `i` with `i+2 < N` where all required rotors exist:

- `R01 = even_rotor(ans_i -> ans_{i+1})`
- `R12 = even_rotor(ans_{i+1} -> ans_{i+2})`
- `R20 = even_rotor(ans_{i+2} -> ans_i)` **constructed directly** (no `~R02` shortcut)

`L_raw = (R20 * R12) * R01` (strict left association, older rotors on the right)  
`L = normalize(L_raw)` (once)

Distance to identity:

- `I = [1,0,...,0]`
- `inn = inner(L, I) = scalar_part(~L)` (definition `#5.3`)
- `d = sqrt(max(0, 2*(1 - min(1, abs(inn)))))`

Aggregate per sample (nearest-rank, total_cmp):

- `h2_loop_max`, `h2_loop_mean`, `h2_loop_p90`

### 8.3 H3 (higher-grade energy ratio) - Must

For raw products (`R_total_raw`, `L_raw`):

- compute grade energies `E0/E2/E4/E6/E8` as sum of squares in each block
- `E_total = Sum Ek`
- `higher_grade_energy_ratio = (E4+E6+E8)/E_total` (if `E_total>0` else `0`)

Emit:

- `h3_ratio_total_product` (from `R_total_raw`)
- `h3_ratio_triangle_loop_*` stats (from `L_raw`)

---

## 11. Known Traps (Must)

- **Flat-space holonomy caveat:** In flat Euclidean `R^8`, triangle holonomy under exact transport is identically zero.  
  Non-zero values indicate numerical drift and/or non-planar composition effects from estimated local rotors. Interpretation deferred.
- Composition is order-sensitive in floating point; strict left-fold is mandatory.
- No intermediate normalization (reserved for version bump).

---

## 12. Determinism & Required IDs (Must)

Gate 2 is implementation-constraining. Every run / artifact that claims Gate 2 telemetry MUST declare the following IDs exactly.

### 12.1 Fixed IDs (Gate 2 core)

- `spec_version = v4.1.0-ssot.3`
- `method_id = rotor_holonomy_telemetry_v1`
- `algebra_id = cl8_even128_mask_grade_order_v1`
- `blade_sign_id = swapcount_popcount_v1`
- `reverse_id = reverse_grade_sign_v1`
- `normalize_id = scalar_part_a_mul_rev_a_v1`
- `composition_id = strict_left_fold_time_reversed_normalize_once_v1`
- `embed_id = embed_simple29_to_even128_v1`
- `h3_name_id = higher_grade_energy_ratio_v1`

### 12.2 Gate 1 reused IDs (declared as-is)

Gate 2 reuses Gate 1 construction and Vec8 contracts. The following IDs MUST be carried through unchanged (values exactly as in Gate 1 SSOT), and MUST NOT be redefined here:

- `rotor_construction_id = simple_rotor29_uv_v1`
- `theta_source_id = theta_uv_atan2_v1`
- `bivector_basis_id = lex_i_lt_j_v1`
- `antipodal_policy_id = antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)`
- (Any additional Gate 1 IDs referenced by name in this document must keep the Gate 1 SSOT values.)

### 12.3 Determinism constraints (Gate 2)

- Multiplication accumulation order is fixed (`#3.3`).
- Composition order is strict left-fold over a **time-reversed** sequence (`#6.1`).
- No intermediate normalization (`#6.2`).
- Any future changes require spec bump + new IDs.

---

## 14. Reserved: Future Thresholding (v4.1.1+)

Reserved section for future gates; any threshold introduction requires spec bump + new ids.
