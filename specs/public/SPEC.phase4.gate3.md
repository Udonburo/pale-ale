# SPEC.phase4.gate3.md -- Phase 4 Gate 3 Local Rotor Geometry (v4.2.x)

**SSOT Status:** DRAFT
**Spec Version:** `v4.2.0-draft.1`
**Relation:**
- Gate1 SSOT: `SPEC.phase4.md v4.0.0-ssot.9` (frozen)
- Gate2 SSOT: `SPEC.phase4.gate2.md v4.1.0-ssot.3` (frozen; telemetry-only)
**Non-Negotiables:** determinism, auditability, no silent failure, strict gate separation

---

## 0. Purpose and Boundary

Gate2 measures global composition observables (closure/holonomy/grade leak) over a full trajectory.
Gate3 measures local geometry of adjacent step rotors in `Cl+(8)`.

Gate3 in `v4.2.0-draft.1` is telemetry-only:
- no threshold-based invalidation,
- no modification to Gate1/Gate2 semantics or artifacts,
- explicit missing reasons (no silent drops).

### 0.1 Forbidden (Must)

1. Threshold-based run invalidation in Gate3.
2. Fabricating Even128 rotors for missing transitions.
3. Replacing Gate2 rotor construction path with ad-hoc formulas.
4. Defining step rotor as `normalize(u_{t+1} * reverse(u_t))` directly from raw vectors.
   Gate3 MUST use the same step-construction path as Gate2.

---

## 1. Input Contract

Gate3 run input is JSON v1:

```rust
pub struct Gate3RunInputV1 {
    pub run_id: String,                            // default: "gate3_run"
    pub explicitly_unrelated_sample_ids: Vec<u64>,// default: [] (reserved; non-gating)
    pub samples: Vec<Gate3SampleInputV1>,
}

pub struct Gate3SampleInputV1 {
    pub sample_id: u64,
    pub ans_vec8: Vec<Vec<f64>>,                  // (steps, 8), finite
    pub sample_label: Option<u8>,                 // 0 | 1 | null
    pub answer_length: Option<usize>,
}
```

Rules:
- each `ans_vec8[row]` must have exactly 8 floats,
- all values must be finite,
- `sample_label` when present must be 0 or 1.

---

## 2. Rotor Sequence Construction (Must)

For sample trajectory `[u_0, u_1, ..., u_{T-1}]`:

1. For each adjacent pair `(u_t, u_{t+1})`, first normalize both vectors via Gate2/Gate1 Vec8 contract:
   - `normalize_vec8(u_t)`
   - `normalize_vec8(u_{t+1})`
2. Build simple step rotor using Gate2 path:
   - `simple_rotor29_doc_to_ans(from_norm, to_norm, RotorConfig::default())`
3. Map outcome:
   - `RotorStep::Materialized { r29, .. }` -> `R_t = embed_simple29_to_even128(&r29)`
   - `RotorStep::AntipodalAngleOnly { .. }` -> missing step (`missing_even_rotor_step`)
   - construction error -> missing step (`missing_even_rotor_step`)

No Even128 fabrication is allowed for missing steps.

Rotor sequence is `[R_0, R_1, ..., R_{T-2}]` with length `T-1`.

---

## 3. Gate3 Observables

Let valid adjacent rotor pairs be indices where both `R_t` and `R_{t+1}` are materialized.

### 3.1 L1 -- Local Rotor Curvature

For each valid adjacent pair:

```text
kappa_t = d_proj(R_t, R_{t+1})
```

with Gate2-consistent projective chordal distance:

```text
d_proj(A, B) = sqrt(max(0, 2 * (1 - min(1, abs(inner(A, B))))))
```

### 3.2 L2 -- Local Grade Torsion

Order is fixed and non-commutative:

```text
P_t = R_{t+1} * reverse(R_t)
tau_t = higher_grade_energy_ratio(P_t)
```

`higher_grade_energy_ratio(X)` is the grade 4/6/8 energy share over total energy.

### 3.3 L3/L4 Per-Sample Statistics

From `kappa_t` sequence:
- `l3_kappa_max`
- `l3_kappa_mean`
- `l3_kappa_std` (population std)
- `l3_kappa_ratio = l3_kappa_max / max(l3_kappa_mean, eps_ratio)`

From `tau_t` sequence:
- `l4_tau_max`
- `l4_tau_mean`
- `l4_tau_std` (population std)
- `l4_tau_p90` (nearest-rank p90, sort with `total_cmp`)

`eps_ratio = 1e-12`.

---

## 4. Missing Semantics (Closed Enum)

Gate3 sample missing reasons are closed:

- `too_few_steps` : `T < 4`
- `invalid_vec8` : row length != 8 or non-finite value
- `all_steps_missing` : `count_rotors_valid == 0`
- `insufficient_adjacent_rotors` : no valid adjacent rotor pair for kappa/tau

Gate3 telemetry-only policy:
- these reasons are sample-level telemetry outcomes,
- they do not trigger Gate3 run invalidation in `v4.2.0-draft.1`.

---

## 5. Run-Level Aggregation (PR6b contract)

Deterministic processing order:
- sort samples by `sample_id ASC`, tie-break by original input order.

Run counts:
- `n_samples_total`
- `n_samples_valid`
- `n_samples_missing`
- `kappa_count_total` (sum over samples)
- `tau_count_total` (sum over samples)
- `missing_even_rotor_steps_total` (sum over samples)

Global aggregates are computed over valid samples only.
Representative sample values are:
- kappa channel: `l3_kappa_mean`
- tau channel: `l4_tau_mean`

Then compute for each channel:
- global mean
- global p90 (nearest-rank, `total_cmp`)
- global max

If no valid sample exists in a channel, global stats are null/empty in artifacts.

---

## 6. Artifacts

Gate3 run emits 3 machine artifacts:
- `manifest.json`
- `summary.csv` (single row)
- `samples.csv` (per sample)

Encoding/format guarantees:
- UTF-8, LF
- deterministic key/column/row ordering
- float strings in `sci_17e_v1` format (`{:.17e}`)
- no `NaN` / `Inf`
- `samples.csv` sorted by `sample_id ASC`

### 6.1 summary.csv columns (fixed order)

```text
run_id,
n_samples_total,
n_samples_valid,
n_samples_missing,
kappa_count_total,
tau_count_total,
kappa_global_mean,
kappa_global_p90,
kappa_global_max,
tau_global_mean,
tau_global_p90,
tau_global_max
```

### 6.2 samples.csv columns (fixed order)

```text
sample_id,
sample_label,
answer_length,
steps_total,
rotors_total,
rotors_valid,
missing_even_rotor_steps,
kappa_count,
tau_count,
l3_kappa_max,
l3_kappa_mean,
l3_kappa_std,
l3_kappa_ratio,
l4_tau_max,
l4_tau_mean,
l4_tau_std,
l4_tau_p90,
missing_reason
```

---

## 7. Determinism and Required IDs (Must)

### 7.1 Fixed Gate3 IDs

```text
spec_version = v4.2.0-draft.1
method_id = rotor_local_geometry_telemetry_v1
curvature_id = local_curvature_projective_chordal_v1
torsion_id = local_torsion_higher_grade_ratio_v1
```

### 7.2 Reused IDs (must remain unchanged)

```text
algebra_id = cl8_even128_mask_grade_order_v1
blade_sign_id = swapcount_popcount_v1
reverse_id = reverse_grade_sign_v1
normalize_id = scalar_part_a_mul_rev_a_v1
composition_id = strict_left_fold_time_reversed_normalize_once_v1
embed_id = embed_simple29_to_even128_v1
rotor_construction_id = simple_rotor29_uv_v1
theta_source_id = theta_uv_atan2_v1
bivector_basis_id = lex_i_lt_j_v1
antipodal_policy_id = antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)
```

### 7.3 Artifact IDs and constants

```text
spec_hash_raw_input_id = spec_text_raw_utf8_v1
spec_hash_input_id = spec_text_utf8_lf_v1
float_format_id = sci_17e_v1
summary_schema_id = gate3_summary_csv_v1
samples_schema_id = gate3_samples_csv_v1
eps_ratio = 1.00000000000000000e-12
```

Any semantic change to formulas, enums, ordering, or IDs requires spec bump + new IDs.

---

## 8. NOT in Gate3

- Threshold-based invalidation (reserved for `v4.2.1+`)
- Gate4 conformal fusion
- Persistent homology topology channel (future method_id)
- Any Gate1/Gate2 behavior change

---

## 9. Roadmap

- `v4.2.0-draft.1`: local geometry telemetry contract, IDs, artifact schema
- `v4.2.0`: freeze Gate3 SSOT after review/field validation
- `v4.2.1+`: optional thresholding layer (if introduced, must be versioned)
- `v4.3+`: Gate4 conformal fusion and advanced topology streams
