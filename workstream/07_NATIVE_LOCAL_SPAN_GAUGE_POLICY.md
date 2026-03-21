# Native Local Span Gauge Policy

Status: Draft
Role: Tracked RFC / implementation-binding design for Gate6-A
Date: 2026-03-17

## 0. Purpose

This document defines how Gate6-A constructs a deterministic native local span object from full hidden-space observables.

The goal is to:

- preserve local token-step geometry
- avoid FWHT-style global compression
- control gauge drift
- produce a transport-ready local object
- remain reproducible

## 1. Raw Inputs

At token step `t`, assume the following full hidden-space observables are available from the same model, same layer, and same token position convention:

- `V_t_raw in R^d`
- `Splus_t_raw in R^d`
- `Sminus_t_raw in R^d`

These are the only required raw vectors for Gate6-A v0.

No future-step vector is required in Gate6-A construction.
`V_{t+1}` is reserved for downstream motif definitions.

## 2. Normalization Policy

Define:

- `v = V_t_raw / ||V_t_raw||`
- `p = Splus_t_raw / ||Splus_t_raw||`
- `m = Sminus_t_raw / ||Sminus_t_raw||`

Required checks:

- all entries finite
- norm greater than `tau_norm_abs`
- normalization performed in `float64` during construction
- storage may downcast later only if explicitly allowed

Record original norms separately:

- `norm_V_raw`
- `norm_Splus_raw`
- `norm_Sminus_raw`

These are metadata, not primary geometry.

## 3. Construction Matrix

Gate6-A uses an anchor-and-relation construction instead of raw stacking alone.

Define:

- `d_plus = p - v`
- `d_minus = m - v`

Construct the local matrix:

`X_t = [v, d_plus, d_minus] in R^(d x 3)`

Rationale:

- `v` preserves the current anchor direction
- `d_plus` and `d_minus` encode local relational deformation relative to the current state
- this avoids collapsing the geometry into a purely centered simplex
- seam-like joint drift should often move all three vectors coherently, whereas contradiction-like defect should distort relational directions

For diagnostics only, also record the raw triple Gram matrix `G_t` of the normalized raw observables `(v, p, m)`.

## 4. Local Span Extraction

Compute thin SVD:

`X_t = U_t Sigma_t W_t^T`

Where:

- `U_t in R^(d x 3)`
- `Sigma_t = diag(sigma_1, sigma_2, sigma_3)` with `sigma_1 >= sigma_2 >= sigma_3 >= 0`
- `W_t in R^(3 x 3)`

Effective rank:

`r_t = count_i[sigma_i >= max(tau_rank_abs, tau_rank_rel * sigma_1)]`

Recommended defaults:

- `tau_norm_abs = 1e-12`
- `tau_rank_abs = 1e-10`
- `tau_rank_rel = 1e-6`
- `tau_sign_tie_abs = 1e-15`

The deterministic local orthonormal basis is:

`B_t = U_t[:, :r_t]`

The canonical projector is:

`P_t = B_t @ B_t.T`

Important:

- `P_t` is the primary invariant object
- `B_t` is only a gauge-fixed realization

## 5. Gauge Discipline

### 5.1 Basis ordering

Basis columns are ordered by descending singular value.
This is fixed by SVD ordering.

### 5.2 Sign convention

Each basis column `b_i` is sign-fixed as follows:

- compute `max_abs_i = max_k abs(b_i[k])`
- form the candidate set `J_i = {k : abs(abs(b_i[k]) - max_abs_i) <= tau_sign_tie_abs}`
- choose `j = min(J_i)`
- if `b_i[j] < 0`, replace `b_i <- -b_i`

This removes the most common sign ambiguity while remaining deterministic under exact or near-exact magnitude ties.

### 5.3 Rank-drop policy

If `r_t < 3`, do not artificially inflate rank.
Gate6-A must preserve actual local rank.

Flags:

- `rank_drop_to_2`
- `rank_drop_to_1`

### 5.4 Near-collinear policy

Near-collinearity is not a failure.
It is a geometric fact and should appear as reduced rank, not as an exception.

### 5.5 Projector primacy

Any downstream comparison across token steps that can be expressed in projector form should prefer `P_t` over raw `B_t`.

This is the central anti-jitter rule.

## 6. Local Coordinates

Project the normalized raw observables into the local frame:

`C_t = B_t^T [v, p, m]`

`C_t` has shape `r_t x 3`.

This is the compact native coordinate chart for the current token step.

For fixed-width storage, define `C_t_padded` with shape `3 x 3`:

- first `r_t` rows contain `C_t`
- remaining rows are zero

This local coordinate chart is essential because:

- it preserves the within-step geometry exactly up to numerical precision
- it allows a compatibility embedding without recomputing from full hidden space

## 7. Compatibility Embedding For Gate6-A Benchmarking

Gate6-A first benchmark must keep the Gate5 triad-loop motif unchanged.
Therefore a deterministic compatibility view is required.

Define canonical local-8 embedding:

`E8(x_1, ..., x_r) = (x_1, ..., x_r, 0, ..., 0) in R^8`

Apply this to the three local coordinate vectors:

- `V_local8`
- `Splus_local8`
- `Sminus_local8`

These are not the native object.
They are a compatibility lane for direct comparison with Gate5 transport logic.

This preserves the experimental discipline:

- boundary changes
- motif stays fixed

## 8. Required Flags

Per token step, record at least:

- `all_finite`
- `rank_local`
- `rank_drop_to_2`
- `rank_drop_to_1`
- `sigma_1`
- `sigma_2`
- `sigma_3`
- `used_sign_fix`
- `norm_V_raw`
- `norm_Splus_raw`
- `norm_Sminus_raw`

Optional but recommended:

- `cond_like = sigma_1 / max(sigma_r, tau_rank_abs)`
- `near_degenerate = (sigma_3 / sigma_1 < tau_warn_rel)`

## 9. Invariants vs Auxiliary Representations

Primary invariant:

- `P_t`

Stable compact summaries:

- `Sigma_t`
- `G_t`
- `C_t_padded`

Auxiliary realization:

- `B_t`

Transport-ready compatibility view:

- `V_local8`
- `Splus_local8`
- `Sminus_local8`

This hierarchy must remain explicit in all docs and code.

## 10. Non-Goals

Gate6-A does not yet do:

- motif proliferation
- defect field aggregation
- verdicting
- nonassociative transport
- projector-valued transport law on its own

It only builds the native local object and the compatibility embedding needed to benchmark the same triad-loop motif under a new boundary.
