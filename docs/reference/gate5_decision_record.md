## Confirmed input boundary

- Gate5's canonical computational input boundary is `Gate4RunInputV1`, serialized on disk as `gate4_input.json`.
- This is confirmed by `crates/diagnose/src/gate4.rs` (`Gate4RunInputV1`, `Gate4SampleInputV1`, `Gate4TokenStepInputV1`), `tools/build_gate4_input.py`, and `tools/README_cfa.md`.
- The required proxy observables already exist on each token step as `V_8d`, `Splus_8d`, and `Sminus_8d`, alongside `label_token`, `baseline_logprob`, and `baseline_entropy`.
- `gate4_token_features.csv` is a downstream Gate4 artifact, not the primary Gate5 computational boundary.

## Confirmed rotor/algebra reuse path

- The reusable edge-construction path already exists in Rust via `simple_rotor29_doc_to_ans` in `crates/rotor/src/lib.rs`.
- Gate5 v0 shall freeze the edge-construction thresholds to the current code values and pass them explicitly:
  - `tau_wedge = 1e-6`
  - `tau_antipodal_dot = 1.0 - 1e-6`
- Gate5 v0 shall not rely on `RotorConfig::default()` implicitly, because a future change to `Default` would silently drift Gate5 behavior.
- Branch order is fixed by current code: antipodal -> collinear -> normal.
- Same-direction collinear inputs currently materialize as identity-like `RotorStep::Materialized { is_collinear: true, r29[0] = 1.0 }`; they are not missing.
- Antipodal inputs currently return `RotorStep::AntipodalAngleOnly { theta }`; there is no materialized rotor for that branch in the current code.
- The reusable composition path already exists in Rust via `embed_simple29_to_even128`, `left_fold_mul_time_reversed_normalize_once`, `Even128::identity`, `inner`, `scalar_part`, and `normalize` in `crates/rotor/src/even128.rs`.
- Current composition order is fixed by code and tests: `left_fold_mul_time_reversed_normalize_once(&[R1, R2, R3]) = normalize(R3 * R2 * R1)`.

## Confirmed residual family

- The default Gate5 loop residual is `rotor_loop_chordal_v1`, compared against identity in the same algebra space as `R_loop`.
- Under the current reusable path, the comparison space is normalized `Even128`: materialize edge rotors in simple29, embed to `Even128`, compose there, and compare `R_loop` against `Even128::identity()`.
- The Gate5 v0 residual formulas are fixed as:
  - `R_loop = left_fold_mul_time_reversed_normalize_once([R1, R2, R3])`
  - equivalently, `R_loop = normalize(R3 * R2 * R1)` in `Even128`
  - `I = Even128::identity()`
  - `a = min(1.0, abs(inner(R_loop, I)))`
  - `rotor_loop_chordal_v1 = sqrt(max(0.0, 2.0 * (1.0 - a)))`
  - `rotor_loop_identity_gap_v1 = 1.0 - a`
  - `rotor_loop_nonscalar_norm_v1 = sqrt(sum_{k=1..127} R_loop.coeffs[k]^2)`
- `rotor_loop_identity_gap_v1` is redundant in this path. For normalized `R_loop`, `rotor_loop_chordal_v1 = sqrt(2.0 * rotor_loop_identity_gap_v1)`, so they are monotone-equivalent.
- `rotor_loop_identity_gap_v1` therefore SHALL NOT be emitted as a headline Gate5 v0 metric.
- `rotor_loop_nonscalar_norm_v1` is also likely to be strongly redundant with `rotor_loop_chordal_v1` under normalized `Even128`, because it reduces to a monotone function of the scalar coefficient magnitude in the current path.
- `rotor_loop_nonscalar_norm_v1` may remain as telemetry-only output in v0, but SHALL NOT be treated as a second headline metric or as independent evidence of ranking lift.
- A high-grade residual metric is a post-v0 candidate only. It is not frozen for Gate5 v0 because no code-backed implementation path has yet been fixed in the repo.

## Confirmed edge/loop missing enums

- Current Rust behavior maps cleanly onto a closed Gate5 `edge_outcome` enum:
  - `materialized` for `RotorStep::Materialized { is_collinear: false, .. }`
  - `collinear_identity` for `RotorStep::Materialized { is_collinear: true, .. }`
  - `antipodal_angle_only` for `RotorStep::AntipodalAngleOnly { .. }`
  - `vec8_nonfinite_component` for `Vec8Error::NonFiniteComponent`
  - `vec8_zero_or_nonfinite_norm` for `Vec8Error::ZeroOrNonFiniteNorm`
  - `rotor_nonfinite_theta` for `RotorError::NonFiniteTheta`
  - `rotor_renorm_failure` for `RotorError::RenormFailure`
- Gate4's existing `transition_missing_reason` enum only covers `none|final_step_no_successor`; it is not sufficient for Gate5 edge/loop telemetry.
- Gate5 therefore needs a separate closed `loop_outcome` enum:
  - `none`
  - `partial_loop_missing`
  - `invalid_loop_product`
- If any required edge is unusable, the loop residual is missing with `loop_outcome=partial_loop_missing`. Gate5 v0 should not fabricate angle-only loop fallbacks.

## Confirmed alignment policy for F/E comparison

- Existing `score_F_loop` is token-step aligned in both `tools/eval_triality_token.py` and `crates/diagnose/src/gate4.rs`.
- Existing `score_E_v_sminus_vnext` is transition-aligned and undefined on the final step; Gate4 serializes that missing final transition as `transition_missing_reason=final_step_no_successor`.
- Gate5 may compare directly against `F` in token-step space.
- Gate5 may report `E` only in a separate transition-aligned section with an explicit alignment caveat. It must not fabricate a fake token-aligned `E`.

## Confirmed Seam v0 perturbation families

- The current repo does not already contain a Seam Challenge generator. The only seam-related repo evidence found during inspection is interpretive text in `SPEC.internal.draft.md` about tokenizer/subword seam spikes.
- Minimal codebase-aligned Gate5 v0 scope is therefore to introduce a new deterministic seam generator with only:
  - punctuation injection
  - casing perturbation
  - spacing perturbation
  - harmless fragmentation triggers
- Synonym substitution is excluded from v0.
- The generator should record `seed`, `perturbation_family`, and perturbation spans in JSONL plus companion metadata JSON.

## Confirmed implementation path

- Gate5 v0 SHALL reuse the existing Rust rotor/algebra path via a minimal Rust helper or diagnose-side compute-only entrypoint.
- Pure Python rotor/algebra reimplementation is out of scope for v0.
- Minimal Gate5 v0 path:
  - read `Gate4RunInputV1` / `gate4_input.json`
  - construct per-edge rotors from `V_8d`, `Splus_8d`, and `Sminus_8d` with `simple_rotor29_doc_to_ans`
  - embed materialized edges to `Even128`
  - compose `[R1, R2, R3]` with `left_fold_mul_time_reversed_normalize_once`
  - emit sign-invariant loop residual telemetry and closed missing enums
  - keep orchestration, aggregation, and reporting in Python
- Gate4 logic, existing Gate4 score families, and the Gate4 CSV schema remain unchanged in this scope.
- Prior expectation is modest or null effect size. Gate5 is a falsifiable telemetry experiment, not a presumed uplift layer.
