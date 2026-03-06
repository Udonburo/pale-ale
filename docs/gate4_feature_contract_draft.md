# Gate4 Feature Contract Draft

Status: Draft (Python-prototyped, Rust implementation pending)

## Purpose
Define a stable, implementation-ready feature schema for local token/span diagnostics derived from Triality telemetry and baselines.

## Dataset Unit
- One row per token step (`step`) in a single sample trajectory.
- Optional companion transition table for pairwise features (`step -> step+1`).

## Required Identity Columns
- `run_id` (string)
- `sample_id` (u64)
- `variant` (`consistent|frustrated|unknown`)
- `world_type` (string, optional)
- `step` (u32 local index)
- `absolute_pos` (u32 original token position in model sequence)

## Required Label / Coverage Columns
- `label_token` (`0|1`) token-level defect label
- `label_transition` (`0|1`) transition label (`max(label_t,label_t+1)` by default)
- `defect_span_id` (string or null)
- `label_coverage_ratio` (f64 in [0,1])
- `exact_token_match_ratio` (f64 in [0,1+], expected >= 0.98)

## Required Score Columns
- Baselines:
  - `score_A_logprob` = `-logprob_t`
  - `score_B_entropy` = `entropy_t`
- Geometry:
  - `score_C_v_curvature` = `d_proj(V_t, V_t+1)`
  - `score_D_v_splus_vnext` = `d_proj(V_t,Splus_t)+d_proj(Splus_t,V_t+1)`
  - `score_E_v_sminus_vnext` = `d_proj(V_t,Sminus_t)+d_proj(Sminus_t,V_t+1)`
  - `score_F_loop` = `d_proj(V_t,Splus_t)+d_proj(Splus_t,Sminus_t)+d_proj(Sminus_t,V_t)`

## Optional Normalized / Rank Columns
- `z_A`, `z_B`, `z_C`, `z_D`, `z_E`, `z_F`
- `rank_E_desc` (1 = highest)
- `is_topk_E` (`0|1`, configurable K)

## Required Aggregate Columns (sample-level companion file)
- `auprc_A`, `auprc_B`, `auprc_C`, `auprc_D`, `auprc_E`, `auprc_F`
- `best_baseline_name` (`A|B`)
- `delta_auprc_E_vs_best_baseline`
- `hit_at_10_E`
- `first_hit_distance_E_p90`
- `first_hit_after_defect_distance_E_p90`

## Provenance Columns
- `model_id`, `model_revision`
- `seed`, `perm_r`, `primary_score`
- `triplets_sha256`, `labels_sha256`, `feature_table_sha256`
- `script_sha256_extract`, `script_sha256_eval`, `script_sha256_featuregen`

## File Contract (Draft)
- `gate4_token_features.csv` (token-step rows)
- `gate4_sample_summary.csv` (sample-level rows)
- UTF-8 + LF
- Deterministic row ordering:
  - token table: `(sample_id ASC, step ASC)`
  - summary: `(sample_id ASC)`

## Rust Mapping Notes (Future)
- `Gate4FeatureRow` struct should use `Option<f64>` for scores undefined on final step transitions.
- Deterministic CSV writer with fixed float formatting and stable header order.
