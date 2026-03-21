---

# SPEC.phase4.md — pale_ale Phase 4 Constitution (v4.x)

**Rotor Field → Closed Algebra → Directed Topology → Conformal Trust**  
**SSOT Status:** **READY** (Gate 1 is decision-complete and implementation-constraining)  
**Spec Version:** `v4.0.0-ssot.9`  
**Primary Deliverable:** Gate 1 (`rotor_diagnostics_v1`)  
**Non-Negotiables:** determinism, auditability, _no silent failure_, strict gate separation

---

## 0. Context (Phase 3 is a hard constraint)

Phase 3 (Q2/Q2b) established that **low-dim text embeddings + scalar distance** are structurally unreliable for hallucination detection under realistic conditions:

- signal dilution (document-level),
- unit-level AUROC ≈ 0.5 (chance),
- strong fluency/length confounding,
- GO is not seed-robust.

**Phase 4 changes the observable (What):**  
We do **not** treat “text embeddings as points”. We treat **semantic transitions / rotor trajectories** as the primary measurable object.

---

## 1. Constitution: What Phase 4 measures (Observables)

Phase 4 is a three-layer observation stack:

**(A) Rotor Field (Transition Field):** local geometric transition objects (rotors) computed from aligned unit vectors.  
**(B) Closed Algebra (Holonomy/Closure):** composition requires algebraic closure (forbidden in Gate 1).  
**(C) Directed Topology:** topology with time-arrow (directed flag complexes / zigzag persistence).

Final horizon: **Topological Conformal Trust**  
Use topological anomalies (β₀/β₁ lifetimes, holonomy spikes, etc.) as penalties inside conformal prediction nonconformity.

---

## 2. Gate Strategy: two-stage separation is constitutional

### 2.1 Gate 1 (v4.0.0) — First Blood

**Goal:** prove _signal existence_ using rotor-only, composition-free diagnostics.  
**Primary success criterion:** **AUROC ≥ 0.55** on the **primary metric** (defined in §8) under the Gate 1 protocol.

**Forbidden in Gate 1 (hard ban):**

- rotor composition / geometric product on rotors (closure error / holonomy),
- CGA,
- persistent homology / VR construction,
- any path-dependent “closure error” derived from rotor multiplication,
- reusing Indexer embeddings as rotor vectors.

### 2.2 Gate 2 (v4.1+) — Closed algebra + holonomy

Enable closure/holonomy **only** after upgrading representation to a closed algebra (e.g. `Cl⁺(8)`).

### 2.3 Gate 3 (v4.2+) — Directed topology + conformal fusion

Directed β₁ / zigzag persistence + conformal fusion pipeline.

---

## 3. Data Model (Rust): canonical forms + safety rules

### 3.1 Gate 1 canonical rotor: `SimpleRotor29` (NOT closed under multiplication)

**Purpose:** stable _local_ geometry + trajectory statistics **without composition**.

- Dim: 29 = 1 scalar + 28 bivector coefficients (8D bivector basis)
- Canonical gauge: enforce `scalar ≥ 0` (projective: r ≡ −r)
- Degeneracy: strict two-type handling (near_collinear vs near_antipodal)
- Silent failure forbidden: every fallback/drop/abort is counted + surfaced.

#### 3.1.0 Bivector basis ordering (Must): fixed 28D index map

To prevent implementation drift, the 28 bivector coefficients `b` are indexed in fixed lexicographic order with `i < j`:

`[(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]`

Coefficient definition is fixed:

`w_ij = u_i v_j - u_j v_i`

Any equivalent implementation must preserve this exact ordering in serialized and in-memory 29D rotor vectors.

#### 3.1.1 Rotor construction (Must): fixed formulas for `(s,b)`

Input is normalized finite `u,v ∈ R^8` (§4). Use this exact order:

1. `dot_raw = <u,v>`
2. `dot = clamp(dot_raw, -1, 1)`
3. build wedge coefficients `w_ij` using §3.1.0 ordering
4. `wedge_norm = ||wedge||`
5. `theta_uv = atan2(wedge_norm, dot)` (primary angle source, `[0, π]`)
6. half-angle components (trig function calls forbidden for this step):
   - `s = sqrt(max(0, (1 + dot)/2))`
   - `sin_half = sqrt(max(0, (1 - dot)/2))`
7. normal rotor case (only when not in degeneracy branches from §3.1.3):
   - `b = (wedge / wedge_norm) * sin_half`
   - rotor vector `r_pre = [s, b...]`

Steps 6-7 are executed only when rotor vectors are materialized (i.e., not `antipodal_angle_only`).

No `acos` anywhere in Gate 1. `sin`/`cos` are also forbidden for half-angle construction.

#### 3.1.2 θ definition (Must): uv-sourced + audit consistency

Gate 1 angle is fixed to:

**`theta := theta_uv = atan2(wedge_norm, dot)`**

For steps with a materialized rotor vector, an optional audit check may compute:

**`theta_rotor := 2 · atan2(||b||, s)`**

If both are computed, implementations should log their absolute gap statistics (e.g., `max_theta_source_gap`) for diagnostics.

#### 3.1.3 Degeneracy purge (Must): split policy + fixed decision order

**Parameters (defaults; any change must be explicit and logged):**

- `eps_norm = 1e-6` (Vec8 norm tolerance)
- `eps_dot = 1e-6` (dot clamp margin)
- `eps_dist = 1e-9` (denominator-zero tolerance for wandering_ratio)
- `tau_wedge = 1e-6` (wedge_norm threshold)
- `tau_plane = 1e-5` (plane-valid threshold for bhat metrics)
- `tau_antipodal_dot = 1.0 - 1e-6` (antipodal dot threshold)
- `max_antipodal_drop_rate = 0.20` (per sample)
- `min_rotors = 3` (post-filter rotor count validity)
- `min_planes = 2` (post-filter plane count validity)

**Decision order and comparisons (Must):**

0. Vec8 non-finite / norm failure handling in §4 is evaluated first and aborts before this decision tree.
1. if `dot <= -tau_antipodal_dot`:
   - Action A (default): `antipodal_angle_only=true`
     - keep `theta` contribution for **M1 max_theta** only
     - exclude this step from plane metrics, rotor-vector distance metrics, and rotor trajectories
     - **do not materialize rotor vectors** for this step (`s,b,r_pre` are not constructed)
     - record: increment `count_antipodal_angle_only`
   - Action B (hard failure path): if `theta_uv` is non-finite, **DROP this step**
     - `excluded_step=true`, record `count_antipodal_drop += 1`
2. else if `wedge_norm <= tau_wedge` and `dot >= 0`:
   - near_collinear identity fallback: `(s=1, b=0)`, `is_collinear=true`
   - record: increment `count_collinear`
3. else:
   - normal rotor from §3.1.1

`eps_dot` is a numerical logging margin; branch decisions are governed by `tau_antipodal_dot`, `tau_wedge`, and the fixed comparison operators above.

Fail-fast per sample: if `count_antipodal_drop / steps_total > max_antipodal_drop_rate` → **ABORT SAMPLE** (linking/representation failure).

Rationale: antipodal plane direction is unstable and can create fake β₁-like holes; angle-only keeps Primary signal while preventing topology contamination.

---

## 4. Gate 1 Input Contract (Must): Vec8 normalization + finiteness checks

Gate 1 core input is **Vec8 unit vectors**. This is not “recommended”; it is a **contract**.

### 4.1 Vec8 acceptance rules

For every vector `x ∈ R^8`:

- If any component is NaN/Inf → **ABORT SAMPLE** (`excluded_reason=non_finite_vec8`)
- Compute `norm = ||x||`
  - If `norm == 0` or non-finite → **ABORT SAMPLE** (`excluded_reason=zero_or_nonfinite_norm`)
  - Else normalize: `x ← x / norm`
- Record:
  - `normalized_count += 1` if `|norm - 1| > eps_norm`
  - `max_norm_err = max(max_norm_err, |norm - 1|)`

`vec8_total` is fixed as:

- `vec8_total = doc_unit_count + ans_unit_count` per sample
- denominator uses all input Vec8 units exactly once (no dedup, no link-based filtering)
- this definition is used for `normalized_rate = normalized_count / vec8_total`

This ensures `dot≈±1` has the correct geometric meaning.

### 4.2 Unitization and unit counts (Must)

Gate 1 must bind inputs to an explicit unitization contract:

- `doc_unit_count = len(doc_vec8_sample)`
- `ans_unit_count = len(ans_vec8_sample)`
- `links_topk` must reference these counts for id-range validation (§6.6)
- unitization/segmentation strategy must be declared as `unitization_id` in manifest

---

## 5. Gate 1 Distance (ε): Projective Chordal (fixed)

Gate 1 distance is projective (r ≡ −r) and branchless.

Let `r` be the **29D rotor vector after renormalization** (§8.4). Define:

- `inner = <r1, r2>`
- `a = min(1.0, |inner|)`
- `d²(r1,r2) = max(0, 2(1 - a))`
- `d(r1,r2) = sqrt(d²)`

Clamping and `max(0, ...)` are mandatory to prevent negative-roundoff `d²` and NaN drift across implementations.

**SSOT decision:** Gate 1 uses `d` (with sqrt). Any `d²` fast-path is a **versioned change** (`proj_chordal_v2`) and is not allowed in v4.0.0.

---

## 6. Linking (Top-K): measurement precondition + noise containment

Linking is a **measurement gate**. Bad links destroy AUROC by producing artificial “acrobatics” even for truthful samples.

### 6.1 Two-encoder separation (constitutional)

- **Indexer Encoder**: candidate retrieval only (MiniLM allowed)
- **Rotor Encoder**: produces Vec8 unit vectors (MiniLM forbidden)

### 6.2 Rust API contract (Must): prevent Indexer leakage at the type level

Gate 1 core accepts **only**:

- `doc_vec8: Vec<Vec<[f64; 8]>>` (per sample: list of doc units)
- `ans_vec8: Vec<Vec<[f64; 8]>>` (per sample: list of answer units)
- `links_topk: Vec<Vec<Link>>` (per sample: per answer unit: top-k doc unit ids + rank)

Where `Link` contains:

- `ans_unit_id: u32`
- `doc_unit_id: u32`
- `rank: u16` (1..k)

Gate 1 core must **not** expose any API that accepts Indexer embeddings as rotor vectors.

### 6.3 Indexer score semantics (Must): rank-first, score-secondary

SSOT policy:

- **Primary linking stability is rank-based.**
- `rank` is the canonical ordering key.
- Scores may be stored for diagnostics, but gates must not depend solely on raw score thresholds unless score is explicitly normalized to cosine in `[-1,1]`.

**Required link artifacts:**

- `link_topk.csv`: includes `(sample_id, ans_unit_id, doc_unit_id, rank, indexer_score_optional)`
- `link_sanity.md`: human-readable list for K fixed samples

### 6.4 Link sanity protocol (Must): quantified PASS/FAIL + manifest record

**K fixed samples:** `K=16` (fixed seed), each sample contributes exactly one sanity judgment.

Sampling reproducibility contract (Must):

- fixed fields in manifest:
  - `link_sanity_rng_id`
  - `link_sanity_seed`
  - `link_sanity_sampling_id`
- selected sample ids must be emitted in both:
  - `link_sanity.md` header
  - `manifest.json` as `link_sanity_selected_sample_ids`

Sanity judgment unit (fixed):

- for each sanity sample, select representative answer unit as the minimum `ans_unit_id` that has at least one available top-1 candidate after §6.6 canonicalization (`rank == 1`)
- if none exists, assign category `NO_LINK` for that sample and count it as `unrelated`
- record selected unit id and selected top-1 doc id (or `NO_LINK`) in `link_sanity.md`

Human sanity rule:

- Count `unrelated` judgments across exactly `K=16` sanity judgments.
- **FAIL if unrelated > 6** (strict numeric threshold).
- Save: `link_sanity_pass = true/false` in `manifest.json`.
- If FAIL → **RUN INVALID** (`run_invalid_reason=link_sanity_fail`, do not report AUROC as a model result).

Additional automated collapse gates (non-score-dependent):

- Random-like collapse (fixed):
  - compute category distribution `p_j` over the same `K=16` sanity judgments (categories are `doc_unit_id` plus optional `NO_LINK`)
  - entropy `H = -Σ p_j ln p_j`, `H_max = ln(M)` where `M` is number of observed categories
  - normalized entropy `H_norm = H / H_max` for `M>1`, else `H_norm = 0`
  - if `H_norm > 0.95` → invalid (`run_invalid_reason=random_like_link_collapse`)
- Dominance collapse (fixed):
  - let `max_share = max_j p_j`
  - if `max_share > 0.50` → invalid (`run_invalid_reason=dominant_link_collapse`)

### 6.5 Step accounting + missing-link policy (Must)

To make rates and exclusions comparable across implementations:

- `steps_total = #answer_units_in_sample` (independent of link availability)
- if `steps_total == 0`, sample is invalid for all metrics (`excluded_reason=no_answer_units`)
- if an answer unit has `available_links == 0`:
  - mark step as `missing_link_step=true`
  - increment `count_missing_link_steps`
  - no rotor vector is created for that step (Top-1 and Trimmed-Best)
  - no `theta` is created for that step (excluded from M1)
- Top-1 strictness:
  - Top-1 step uses link with `rank == 1` only
  - if canonicalized `available_links > 0` but `rank == 1` is absent, mark `missing_top1_step=true` for Top-1 track
  - increment `count_missing_top1_steps`
  - treat as missing step in Top-1 track (same exclusion behavior as missing_link_step for Top-1)
  - Trimmed-Best track may continue with available links (`rank >= 2` allowed)
- define `missing_link_step_rate = count_missing_link_steps / steps_total`
- define `missing_top1_step_rate = count_missing_top1_steps / steps_total`
- threshold: `max_missing_link_step_rate = 0.20`
  - if `missing_link_step_rate > max_missing_link_step_rate`, rotor-vector metrics are missing (`metric_missing_reason=too_many_missing_link_steps`)

### 6.6 `links_topk` canonicalization (Must): sort/dedup/rank validation

For each sample and `ans_unit_id`, core processing uses a canonicalized link list:

- sort by `(rank asc, doc_unit_id asc)` using deterministic integer ordering
- invalid rank links are dropped: `rank == 0` or `rank > k` (where `k` is from `trimmed_best_v1`)
- invalid id links are dropped:
  - `ans_unit_id` outside `[0, ans_unit_count - 1]`
  - `doc_unit_id` outside `[0, doc_unit_count - 1]`
- duplicate `(ans_unit_id, doc_unit_id)` entries keep only the smallest rank; remaining duplicates are dropped
- `available_links` is computed **after** this canonicalization
- if multiple `rank == 1` links remain, Top-1 uses the first canonicalized entry (equivalently: minimum `doc_unit_id` among `rank == 1`)

Required diagnostics:

- `count_invalid_rank_links`
- `count_invalid_ans_unit_id`
- `count_invalid_doc_unit_id`
- `count_link_dedup`
- `count_multi_rank1_links` (optional)
- `count_unsorted_link_rows` (optional if core always canonicalizes before any use)

If canonicalized `available_links == 0`, the step follows `missing_link_step` handling from §6.5.

### 6.7 Rotor encoder provenance (Must): representation identity in manifest

To prevent silent representation drift, every run must declare:

- `rotor_encoder_id` (model + revision)
- `rotor_encoder_preproc_id` (text preprocessing)
- `vec8_postproc_id` (pool/projection/post-normalization pipeline)

---

## 7. Top-K rotor aggregation: fixed algorithm (Must)

We retain top-k for audit, but we compute two tracks deterministically.

### 7.1 Track A: Top-1 (baseline)

For each answer unit, use only canonicalized `rank=1` link.

If `rank=1` is absent for that answer unit, Top-1 marks `missing_top1_step=true` and does not substitute `rank=2+`.
If multiple `rank=1` links exist, Top-1 selects the canonicalized first one (minimum `doc_unit_id`).

### 7.2 Track B: Trimmed-Best (Must, fixed definition)

**ID:** `trimmed_best_v1(k=8, p=0.5, key=rank, tie=doc_unit_id)`

- Let `k_eff = min(k, available_links)`
- Let `m = ceil(k_eff * p)` (with p=0.5, m is top half)
- Select links by increasing `rank`; ties resolved by increasing `doc_unit_id`
- For each selected link, compute the rotor representation (or its 29D vector form)
- Compute arithmetic mean of the selected 29D rotor vectors: `r̄ = mean(r_i)`
- **Renormalize**: `r_trim = r̄ / ||r̄||` (if `||r̄|| == 0` or non-finite → invalid, `excluded_reason=trimmed_best_zero_or_nonfinite_norm`)
- Use `r_trim` as the representative per answer unit step.

**SSOT decision:** No alternative aggregation methods exist in v4.0.0.

---

## 8. Gate 1 Diagnostics (`rotor_diagnostics_v1`): composition-free metrics

### 8.1 Primary task (Must): label definition + evaluation unit fixed

**Primary task:** **sample-level hallucination detection**.

- Input may include unit labels.
- Sample label is defined as:

**`halluc_any(sample) = any(halluc_unit == 1)`**

Label missing policy (Must):

- if any `halluc_unit` is missing within a sample, that sample label is treated as missing
- samples with missing sample label are excluded from supervised AUROC metrics (`excluded_reason=missing_halluc_unit_label`)
- unsupervised diagnostics may still include those samples

Evaluation mode contract (Must):

- `evaluation_mode_id` must be explicitly set to `supervised_v1` or `unsupervised_v1`
- `missing_halluc_unit_label` exclusions apply only when `evaluation_mode_id=supervised_v1`
- when `evaluation_mode_id=unsupervised_v1`, label-based exclusions are disabled

If unit labels are unavailable, Gate 1 cannot claim hallucination detection and must be run in “unsupervised diagnostics mode” only.

### 8.2 Primary metric for First Blood (Should→fixed for public-facing discipline)

To avoid p-hacking: Gate 1 **predeclares** a primary metric.

**Primary metric:** `max_theta` on **Top-1 track**  
All other metrics are still computed and reported, but “First Blood” is declared only on this primary metric unless the spec is version-bumped.

### 8.3 Rotor construction: mapping direction fixed (Should→fixed)

Gate 1 computes rotors for **doc → ans** transitions:

- For each answer unit `a_i`, use linked doc unit `d_j` and compute rotor from `u=d_j_vec8` to `v=a_i_vec8`.

No ans→ans adjacent transition rotors in v4.0.0 (reserved for `v4.0.1+`).

### 8.4 Rotor29 normalization procedure (Must)

After constructing `(s,b)` for steps that materialize rotor vectors:

- Build 29D vector `r = [s, b...]`
- **Renormalize**: `r ← r / ||r||`
  - identity fallback already has norm 1, but renorm is still applied (no-op)
- Record:
  - `renorm_count`
  - `max_rotor_norm_err_pre` (max `| ||r_pre|| - 1 |`)

If renorm fails (`||r|| == 0` or non-finite) → invalid (`excluded_reason=rotor_renorm_failure`).

### 8.5 Metrics: all are “suspicion scores” (Should→fixed)

All reported metrics must follow:

- **higher_is_more_suspicious = true**
- If an internal quantity is inverted, it must be transformed so that “higher = worse”.

### 8.6 Metric definitions (v1)

Compute per sample, per track (Top-1 and Trimmed-Best). `antipodal_angle_only` steps contribute to M1 only; dropped steps, `missing_link_step`, and `missing_top1_step` are excluded from step-wise rotor-vector computations.

**(M1) max_theta (Primary)**

- `theta_t = theta_uv = atan2(wedge_norm_t, dot_t)` for steps with finite `theta_uv` (including `antipodal_angle_only`)
- `max_theta = max(theta_t)`
- Validity for M1: `n_theta_valid >= 1`; otherwise M1 is missing (reason resolution follows §8.7 validity rules)

**(M2) plane_turn_rate (β₁ precursor, composition-free)**  
Define `bhat_t = b_t / ||b_t||` for rotor-vector steps where `||b_t|| > tau_plane`.  
Construct compressed plane sequence by preserving original step order and dropping non-plane steps.  
For adjacent pairs in this compressed sequence:

- `d_plane(t,t+1) = 1 - |<bhat_t, bhat_{t+1}>|`  
  Report:
- `plane_turn_mean`, `plane_turn_max`, `plane_turn_var`  
  Validity requires `count_planes >= min_planes`; otherwise metric is missing (`metric_missing_reason=missing_planes`).

**(M3) alignment_mean / alignment_var**  
Define a global plane `b_global`:

- Prefer start→end rotor plane if both endpoints have valid `bhat`.
- Else fallback: normalized mean of all valid `bhat_t` (deterministic).  
  Compute:
- `a_t = 1 - |<bhat_t, b_global>|`  
  Report mean/var.  
  Requires `count_planes >= min_planes`; otherwise metric is missing (`metric_missing_reason=missing_planes`).

**(M4) wandering_ratio (with denominator-zero rule + dual AUC)**  
Using Gate 1 distance `d`:

- `R_start` = first valid rotor-vector step in sequence (after exclusions)
- `R_end` = last valid rotor-vector step in sequence (after exclusions)
- If fewer than 2 rotor-vector steps exist, metric is missing (`metric_missing_reason=too_few_rotor_steps_for_wandering`)
- Construct compressed rotor-vector sequence by preserving original step order and dropping non-rotor steps
- `L = Σ d(R_t, R_{t+1})` over adjacent pairs in this compressed sequence
- `D = d(R_start, R_end)`  
  If `D < eps_dist`:
- `wandering_ratio = 0.0`
- `degenerate_path = true`  
  Else:
- `wandering_ratio = L / D`

**(M5) degeneracy / drop rates (also valid as suspicion features)**  
Report:

- `rate_collinear = count_collinear / steps_total`
- `rate_antipodal_angle_only = count_antipodal_angle_only / steps_total`
- `rate_antipodal_drop = count_antipodal_drop / steps_total`
- `rate_missing_link_steps = count_missing_link_steps / steps_total`
- `rate_missing_top1_steps = count_missing_top1_steps / steps_total`
- `normalized_rate = normalized_count / vec8_total`  
  These rates are themselves candidate suspicion scores (higher = more suspicious), but are not primary.

### 8.7 Validity thresholds: min steps split (Should→fixed)

Define counts **after applying near_antipodal split policy**:

- `n_theta_valid` = number of steps with finite `theta_uv` (includes `antipodal_angle_only`, excludes `missing_link_step` and `missing_top1_step`)
- `n_rotors_valid` = number of steps with materialized rotor vectors (includes identity fallbacks, excludes `antipodal_angle_only`, `missing_link_step`, and `missing_top1_step`)
- `n_planes_valid` = number of rotor-vector steps with `||b|| > tau_plane`

Validity rules:

- If `n_theta_valid < 1`:
  - if `count_missing_top1_steps > 0`, M1 is missing (`metric_missing_reason=missing_top1_link`)
  - else if `count_missing_link_steps > 0`, M1 is missing (`metric_missing_reason=missing_links_for_theta`)
  - else M1 is missing (`metric_missing_reason=missing_theta`)
- If `n_rotors_valid < min_rotors` → rotor-vector metrics are missing (`metric_missing_reason=too_few_rotors`)
- If `missing_link_step_rate > max_missing_link_step_rate` → rotor-vector metrics are missing (`metric_missing_reason=too_many_missing_link_steps`)
- If a metric requires planes and `n_planes_valid < min_planes` → that metric is missing (`metric_missing_reason=missing_planes`)

---

## 9. Exclusions, aborts, and run validity (Must)

Ambiguity here causes “AUC inflation by dropping inconvenient samples”. This spec forbids that.

### 9.1 Sample states

A sample (for a given metric) is one of:

- **used**: included in AUROC computation for a given metric
- **excluded**: removed from AUROC computation due to abort/invalid conditions
- **metric_missing**: sample is present, but a metric value is undefined for rule-based reasons (`metric_missing_reason`)

**Abort (fail-fast triggered):**

- near_antipodal **drop** rate exceeds threshold (`excluded_reason=excess_antipodal_drop_rate`),
- non-finite vec8 (`excluded_reason=non_finite_vec8`),
- normalization failure (`excluded_reason=zero_or_nonfinite_norm`),
- linking sanity failure for that sample if defined per-sample (`excluded_reason=link_sanity_sample_fail`).

**Invalid:**

- no answer units (`steps_total == 0`),
- missing sample label in supervised AUROC mode only (`excluded_reason=missing_halluc_unit_label`, does not block unsupervised diagnostics),
- rotor renorm failure (`excluded_reason=rotor_renorm_failure`),
- trimmed_best representative norm is zero / non-finite (`excluded_reason=trimmed_best_zero_or_nonfinite_norm`, track-scoped to Trimmed-Best only).

### 9.2 Exclusion handling (Must)

- `abort` samples are **excluded** from evaluation sets.
- `invalid` sample/metric entries are **excluded** from that metric’s evaluation set.
- `metric_missing` entries are excluded from that metric only and must be counted separately from `excluded_reason`.
- Track-scoped invalidation is mandatory: failures in Trimmed-Best must not invalidate Top-1 entries unless an explicit run-level invalid reason is triggered.
- Every exclusion must write:
  - `excluded_reason`
  - counts per reason
  - exclusion rate per metric
- Every metric-missing entry must write:
  - `metric_missing_reason`
  - counts per reason
  - metric-missing rate per metric

### 9.3 Run-level exclusion ceiling (Must)

To prevent “good AUC from massive exclusions”:

- Define supervised primary accounting:
- `n_supervised_eligible` = samples with non-missing sample label
- `n_supervised_excluded_primary` = supervised-eligible samples excluded from primary metric (abort/invalid + primary metric missing)
- `n_supervised_used_primary` = `n_supervised_eligible - n_supervised_excluded_primary`
- `primary_exclusion_rate = n_supervised_excluded_primary / n_supervised_eligible` (if `n_supervised_eligible > 0`)
- `label_missing_rate = n_label_missing / total_samples` must be reported separately and does not trigger exclusion-ceiling invalidation.
- all four accounting fields above plus `label_missing_rate` must be written to manifest and summary artifacts
- If `n_supervised_eligible == 0`, run is invalid (`run_invalid_reason=no_supervised_eligible_samples`).
- If `primary_exclusion_rate > 0.10` for the **primary metric**, then:
  - **RUN INVALID** (`run_invalid_reason=excess_exclusions_primary`)
  - AUROC is not declared as First Blood.

This ceiling can be adjusted only via spec version bump.

### 9.4 Reason IDs (Must): closed enums for audit fields

To prevent aggregation drift, reason fields must use only these IDs.

`excluded_reason` (sample/metric level):

- `non_finite_vec8`
- `zero_or_nonfinite_norm`
- `no_answer_units`
- `missing_halluc_unit_label`
- `excess_antipodal_drop_rate`
- `rotor_renorm_failure`
- `trimmed_best_zero_or_nonfinite_norm`
- `link_sanity_sample_fail`

`run_invalid_reason` (run level):

- `link_sanity_fail`
- `random_like_link_collapse`
- `dominant_link_collapse`
- `excess_exclusions_primary`
- `empty_quantile_population_primary`
- `no_supervised_eligible_samples`

`run_warning` (run level):

- `antipodal_angle_only_high`

`auc_undefined_reason` (metric level):

- `single_class_after_exclusions`

`metric_missing_reason` (metric level):

- `missing_top1_link`
- `missing_links_for_theta`
- `missing_theta`
- `missing_planes`
- `too_few_rotors`
- `too_many_missing_link_steps`
- `too_few_rotor_steps_for_wandering`

---

## 10. AUROC computation rules (Must): ties, missing, class absence

AUROC must be deterministic and comparable across implementations.

### 10.1 Missing data

- Excluded samples are removed before AUROC computation.
- For a metric, samples where that metric is missing are excluded **for that metric** and counted under `metric_missing_reason` (not `excluded_reason`).

### 10.2 Ties (Must)

AUROC ranking must use **average rank** for ties.
- This must be stated in the `manifest.json` (`auc_ties = average_rank`).
- Rank direction is fixed: `rank 1 = smallest score`, `rank n = largest score` (higher suspicion score must receive larger rank).
- Ranking method is fixed: sort scores with `total_cmp`; each tie group receives the mean of its 1-indexed rank positions.

### 10.2b AUROC algorithm (Must): Mann–Whitney rank-sum

AUROC is computed with rank-sum over suspicion scores (higher score = more suspicious):

- let `n_pos` and `n_neg` be class counts after exclusions
- let `sum_rank_pos` be sum of average ranks for positive class
- `auc = (sum_rank_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)`

Implementations must use this formula (equivalent ROC integration variants are out of scope for Gate 1 SSOT).

### 10.3 Class absence (Must)

If after exclusions:

- only one class remains (all positives or all negatives),  
  then AUROC is **undefined** and the run is **invalid for that metric**.  
  Record:
- `auc_undefined_reason = single_class_after_exclusions`  
  Do not emit `NaN` and quietly continue.

### 10.4 Dual AUROC for wandering_ratio (Must)

For `wandering_ratio`, output:

- `auc_include_deg_path` (include degenerate_path as score=0)
- `auc_exclude_deg_path` (exclude degenerate_path samples)

Both must include their own exclusion counts.

### 10.5 Optional statistical confidence outputs (Non-gating)

To improve interpretability of Primary AUC without changing gate decisions:

- optional outputs:
  - `auc_primary_bootstrap_ci95_low`
  - `auc_primary_bootstrap_ci95_high`
  - `auc_primary_permutation_p`
- optional method id:
  - `stats_id = bootstrap_ci_v1+permutation_test_v1`
- if omitted, gating behavior is unchanged and `stats_id = none`.

---

## 11. τ monitoring + representation collapse gates (Must)

τ parameters only make sense if their induced regimes are observable. Therefore:

### 11.0 Quantile definition (Must): fixed scope + fixed population + fixed algorithm

Quantiles used in gates/logging must be computed with this exact contract.

Scope:

- run-level collapse gates use **Top-1 track quantiles only**
- Trimmed-Best quantiles are optional diagnostics and never used for run invalidation in v4.0.0
- if Trimmed-Best quantiles are omitted, set `quantiles_missing_ok=true` in manifest
- gate order is fixed:
  - evaluate link-sanity invalidation first (`link_sanity_fail`, `random_like_link_collapse`, `dominant_link_collapse`)
  - evaluate quantile-based collapse gates only if run is still valid
  - if already invalid from link sanity, quantiles may be logged as reference only (`quantile_reference_only=true`)

Population (per run, Top-1 track):

- `dot` and `wedge_norm` samples are collected from Top-1 steps where canonicalized `rank==1` exists and normalized finite `u,v` exist
- steps with `missing_top1_step` or `missing_link_step` are excluded
- population includes collinear and antipodal-angle-only steps
- supervised/unsupervised label availability does not affect quantile population membership

Algorithm:

- sort values ascending using `total_cmp`
- nearest-rank index for quantile `p`: `idx = ceil(p * n) - 1`, clamped to `[0, n-1]`
- quantile value is `sorted[idx]`
- fixed `p` set for Gate 1 logging: `p ∈ {0.01, 0.50, 0.90, 0.99}`

Empty population handling:

- if `n = 0` for Top-1 quantile population, mark run invalid (`run_invalid_reason=empty_quantile_population_primary`)
- if Trimmed-Best quantiles are enabled and its population is empty, log as missing (no run invalidation)

### 11.1 Required distribution logging (Must)

For each run, record in `manifest.json` and `summary.csv`:

- `wedge_norm_p50/p90/p99`
- `dot_p1/p50/p90/p99`
- `rate_collinear`, `rate_antipodal_angle_only`, `rate_antipodal_drop`, `rate_missing_link_steps`, `rate_missing_top1_steps`, `normalized_rate`
- `degenerate_path_rate` per track:
  - numerator: count of samples where `degenerate_path=true` for that track
  - denominator: count of samples where `wandering_ratio` is **not** `metric_missing` for that track
  - computed separately for Top-1 and Trimmed-Best tracks
  - field names: `degenerate_path_rate` (Top-1), `trimmed_degenerate_path_rate` (Trimmed-Best, optional in v4.0.0)

Field naming scope:

- unprefixed quantile fields above refer to Top-1 track only
- optional Trimmed-Best quantiles, if emitted, must use `trimmed_` prefix (e.g., `trimmed_wedge_norm_p99`)

### 11.2 Collapse gates (Must)

Abort run if any of these hold for **Top-1 primary track**:

- `rate_collinear > 0.80` (representation collapsed / overly degenerate)
- `rate_antipodal_drop > 0.20` on average (linking/representation failure)
- `wedge_norm_p99 < tau_wedge` (almost everything degenerate)
- `wedge_norm_p50 >> expected` is not strictly gateable without a prior baseline; record only (v4.0.0)

These thresholds are SSOT; changes require spec version bump.

### 11.3 Antipodal angle-only warning channel (Should→fixed warning policy)

To detect runs where Primary may be dominated by angle-only antipodal steps:

- required logs: `rate_antipodal_angle_only_p50`, `rate_antipodal_angle_only_p90`
- warning quantile population is fixed to per-sample `rate_antipodal_angle_only` values for samples with `steps_total > 0` and without abort
- compute `share_samples_antipodal_angle_only_gt_0_50`
- if `share_samples_antipodal_angle_only_gt_0_50 > 0.20`, set `run_warning=antipodal_angle_only_high`

This is a warning channel only; it does not invalidate the run in v4.0.0.

### 11.4 Representation robustness diagnostics (Should→mandatory reporting)

To detect failure modes without changing gates:

- effective dimension diagnostic:
  - compute covariance eigenvalues `λ_i` over run-level Vec8 population (doc+ans units after Vec8 normalization)
  - `vec8_eff_dim_pr = (Σ λ_i)^2 / Σ (λ_i^2)` (participation ratio)
- Trimmed-Best stability diagnostics:
  - `trimmed_rbar_norm_pre_p50`
  - `trimmed_rbar_norm_pre_p10`
  - `trimmed_rbar_norm_pre_p01`
  - `trimmed_failure_rate` (rate of `trimmed_best_zero_or_nonfinite_norm`)

These are non-gating diagnostics in Gate 1.

---

## 12. Confound quarantine (Should→adopted as mandatory reporting)

To prevent repeating Phase 3 failure modes, Gate 1 must report minimal confound diagnostics:

For primary metric (Top-1 max_theta) and for each track:

- Spearman correlation with `answer_len` (or unit count proxy if answer_len unavailable)
- Stratified AUROC by answer length tertiles (short/medium/long), if answer_len available
- Exclusion rate by answer length tertiles (`exclusion_rate_short/medium/long`); if `answer_len` unavailable, use unit-count tertiles as proxy
- `length_confound_warning` (non-gating flag):
  - set `true` if `abs(rho_len_max_theta) > 0.70`
  - else `false`

Spearman implementation contract (Must for reproducibility):

- rank ties use `average_rank`
- missing/excluded rows are removed before ranking
- `rho` is computed as Pearson correlation on ranked variables

These do **not** change gating; they are mandatory outputs for interpretation.

---

## 13. Determinism & audit (Must)

### 13.0 Run identity and provenance (Must)

Every run must carry reproducibility identity fields:

- spec identity:
  - `spec_hash_raw_blake3`
  - `spec_hash_raw_input_id = spec_text_raw_utf8_v1`
  - `spec_hash_blake3`
  - `spec_hash_input_id = spec_text_utf8_lf_v1`
- dataset identity:
  - `dataset_revision_id`
  - `dataset_hash_blake3`
- code/build identity:
  - `code_git_commit`
  - `build_target_triple`
  - `rustc_version`

Hash procedure definitions (Must):

- **raw** (`spec_text_raw_utf8_v1`): hash the file byte sequence exactly as read from disk. No line-ending conversion, no trailing-whitespace trimming, no BOM stripping — input bytes are hashed verbatim.
- **lf** (`spec_text_utf8_lf_v1`): convert all `CRLF` (`\r\n`) sequences to `LF` (`\n`), then hash the resulting byte sequence. No other transformations (no trailing-whitespace trimming, no BOM stripping).

### 13.1 Deterministic parallelism

- Parallelism is **sample-level only**.
- Within a sample: fixed order, single thread.
- Determinism scope is fixed to: **same binary + same target triple + same dataset revision**.
- Cross-CPU/OS bitwise identity is out of scope for Gate 1.

### 13.2 Sorting / comparisons

- float comparisons use `total_cmp`
- reductions are stable (fixed order)

### 13.3 Numeric output encoding (Must)

To ensure machine-comparable artifacts across implementations:

- all floating-point fields in `manifest.json` must be serialized as strings using `"{:.17e}"` formatting
- all floating-point fields in `summary.csv` must use `"{:.17e}"` formatting
- NaN/Inf outputs are forbidden; use explicit reason fields instead
- optional human-facing reports may use different formatting, but gating reads only machine-formatted artifacts

### 13.4 Artifact ordering and schema stability (Must)

To make diffs and audits stable:

- `link_topk.csv` row order is fixed: `(sample_id, ans_unit_id, rank, doc_unit_id)` ascending
- `summary.csv` column order must follow a fixed schema id
- `manifest.json` key order should be stable (deterministic serializer; e.g., lexicographic/BTreeMap order)

### 13.5 Required method IDs in manifest (Must)

Every run must include:

- `spec_version = v4.0.0-ssot.9`
- `spec_hash_raw_blake3`
- `spec_hash_raw_input_id = spec_text_raw_utf8_v1`
- `spec_hash_blake3`
- `spec_hash_input_id = spec_text_utf8_lf_v1`
- `dataset_revision_id`
- `dataset_hash_blake3`
- `code_git_commit`
- `build_target_triple`
- `rustc_version`
- `method_id = rotor_diagnostics_v1`
- `distance_id = proj_chordal_v1`
- `trimmed_best_id = trimmed_best_v1(k=8,p=0.5,key=rank,tie=doc_unit_id)`
- `evaluation_mode_id = supervised_v1|unsupervised_v1`
- `unitization_id = <unitization_scheme_id>`
- `bivector_basis_id = lex_i_lt_j_v1`
- `antipodal_policy_id = antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)`
- `rotor_construction_id = simple_rotor29_uv_v1`
- `theta_source_id = theta_uv_atan2_v1`
- `rotor_encoder_id = <encoder+revision>`
- `rotor_encoder_preproc_id = <preproc_id>`
- `vec8_postproc_id = <vec8_postproc_id>`
- `quantile_id = nearest_rank_total_cmp_v1`
- `quantile_track_scope = top1_only_primary_v1`
- `top1_policy_id = strict_rank1_or_missing_v1`
- `quantiles_missing_ok = true|false` (required; true only when optional Trimmed-Best quantiles are omitted)
- `quantile_reference_only = true|false` (required; true when quantiles are logged but collapse gates are skipped due to prior run invalidation)
- `link_sanity_id = sanity16_single_judgment_v1`
- `link_sanity_rng_id`
- `link_sanity_seed`
- `link_sanity_sampling_id`
- `link_sanity_selected_sample_ids`
- `links_topk_canonicalization_id = link_rank_doc_canon_v1`
- `auc_algorithm_id = mann_whitney_rank_sum_v1`
- `rank_method_id = average_rank_total_cmp_v1`
- `stats_id = bootstrap_ci_v1+permutation_test_v1|none`
- `label_missing_policy_id = exclude_sample_on_missing_halluc_unit_v1`
- `reason_enum_id = reason_ids_v1`
- `metric_missing_enum_id = metric_missing_reason_ids_v1`
- `float_format_id = sci_17e_v1`
- `summary_schema_id = summary_csv_schema_v1`
- `link_topk_schema_id = link_topk_csv_schema_v1`
- `determinism_scope = same_binary_same_target_same_dataset`
- all τ/ε thresholds
- link sanity PASS/FAIL + unrelated count
- entropy collapse stats (`H_norm`, `max_share`)
- supervised exclusion accounting (`n_supervised_eligible`, `n_supervised_used_primary`, `n_supervised_excluded_primary`, `primary_exclusion_rate`, `label_missing_rate`)
- representation diagnostics (`vec8_eff_dim_pr`, `trimmed_rbar_norm_pre_p50/p10/p01`, `trimmed_failure_rate`, `degenerate_path_rate`)
- confound warning field (`length_confound_warning`)
- exclusion counts + exclusion ceiling result
- warning channel outputs (including `run_warning` when raised)

Silent fallback is forbidden: any fallback must be explicitly logged.

---

## 14. Known traps (must remain visible)

1. **SimpleRotor29 is not closed** → closure/holonomy metrics forbidden in Gate 1.
2. **Indexer leakage** → prevented by Vec8-only core API.
3. **near_antipodal** → must be angle-only for M1 and excluded from plane/path metrics; hard-drop only on non-finite angle path.
4. **Bad linking** → must pass quantified sanity; otherwise AUROC is meaningless.
5. **High exclusion rate** → invalidates run; “AUC by dropping” is forbidden.
6. **Confounds (length/fluency)** → must be reported for quarantine.
7. **Missing-link drift** → `steps_total` and `missing_link_step` policy must stay fixed, or rate metrics become incomparable.
8. **Quantile drift** → gate thresholds depend on fixed nearest-rank quantiles and fixed populations.
9. **Top-1 drift** → Top-1 must not substitute `rank>1` when `rank==1` is missing.
10. **Small sanity slice risk** → `K=16` can be underpowered; treat sanity outcome with logged slice ids and deterministic sampler metadata.

---

## 15. Gate 2+ roadmap (non-binding details, binding boundary)

### 15.1 Gate 2 (v4.1+): closed algebra requirement

Closure error / holonomy requires a closed representation, e.g. `Cl⁺(8)` (128D).  
Only Gate 2 may define rotor composition and closure.

### 15.2 Gate 3 (v4.2+): directed topology + conformal fusion

Directed complexes + zigzag persistence + conformal fusion of topological anomalies.

---

# Appendix A — Minimal Rust Core I/F (contract-level)

Gate 1 core must expose an interface accepting **only**:

- Vec8 arrays (doc and answer unit vectors),
- link IDs + rank,
- config thresholds.

It must not accept Indexer embeddings as rotor vectors.

---

# Appendix B — Required artifacts (Gate 1 run)

Minimum output set:

- `manifest.json` (with all required IDs/thresholds/quantiles/exclusions)
- `summary.csv` (AUROC for all metrics + dual wandering AUC + exclusions)
- `link_topk.csv`
- `link_sanity.md` (+ PASS/FAIL, unrelated count)
- optional but recommended: histograms for dot/wedge_norm

---

# Appendix C — Glossary (terms used consistently)

- **abort**: fail-fast triggered; sample excluded with reason
- **invalid**: insufficient data post-filter; sample excluded with reason
- **metric missing**: metric undefined for a sample; excluded for that metric with `metric_missing_reason`
- **run invalid**: violates run-level gates (link sanity fail, exclusion ceiling, collapse gates)

---

# Appendix D — Golden Test Vectors (Gate 1)

Use these vectors as reference sanity checks for `simple_rotor29_uv_v1` and `theta_uv_atan2_v1`.

Conventions:

- `e0=[1,0,0,0,0,0,0,0]`, `e1=[0,1,0,0,0,0,0,0]`
- `theta_uv = atan2(wedge_norm, dot)`
- bivector index `(0,1)` is the first basis coordinate from §3.1.0

Case D1: orthogonal axis

- input: `u=e0`, `v=e1`
- expected: `dot=0`, `wedge_norm=1`, `theta_uv=π/2`
- rotor (normal branch): `s=sqrt(1/2)`, `sin_half=sqrt(1/2)`, `b_(0,1)=sqrt(1/2)`, others `0`

Case D2: collinear identity fallback

- input: `u=e0`, `v=e0`
- expected: `dot=1`, `wedge_norm=0`, `theta_uv=0`
- branch: near_collinear identity fallback `(s=1,b=0)`

Case D3: antipodal angle-only

- input: `u=e0`, `v=-e0`
- expected: `dot=-1`, `wedge_norm=0`, `theta_uv=π` (finite)
- branch: `antipodal_angle_only=true`, rotor vector materialization forbidden

Case D4: 45-degree in `(e0,e1)` plane

- input: `u=e0`, `v=(e0+e1)/sqrt(2)`
- expected: `dot=1/sqrt(2)`, `wedge_norm=1/sqrt(2)`, `theta_uv=π/4`
- rotor (normal branch): `s=sqrt((1+1/sqrt(2))/2)`, `sin_half=sqrt((1-1/sqrt(2))/2)`, `b_(0,1)=sin_half`, others `0`

---

## Implementation note (non-spec)

Everything above is **implementation-constraining**. If a future change is desired (e.g., `d²` accumulation, alternative aggregation, different primary label), it must be done via:

- spec version bump,
- new method_id and distance_id/aggregate_id,
- explicit changelog.

---

### What this SSOT now guarantees

If an implementer follows this document literally, then:

- aggregation cannot drift,
- evaluation cannot be gamed via exclusions,
- AUROC cannot vary due to ties/missing/class absence choices,
- linking noise cannot silently poison the rotor field,
- antipodal noise cannot masquerade as topology,
- Phase 3 regressions (Indexer leakage) are physically blocked by the API.

---
