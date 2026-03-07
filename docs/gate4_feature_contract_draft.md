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
- `label_coverage_ratio` (f64 in [0,1])
- `exact_token_match_ratio` (f64 in [0,1+], expected >= 0.98)

## Optional Label Provenance Columns
- `defect_span_id` (string or null)

## Required Missing-State Columns
- `transition_missing_reason` (`none|final_step_no_successor`)

On-wire encoding rule for transition-aligned scores undefined on the final step:
- `score_C_v_curvature`, `score_D_v_splus_vnext`, and `score_E_v_sminus_vnext` serialize as empty string in CSV when `transition_missing_reason=final_step_no_successor`
- otherwise `transition_missing_reason=none`

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

## Optional Support Metrics (sample-level companion file)
- `first_hit_distance_E_p90`
- `first_hit_after_defect_distance_E_p90`

## Provenance Columns
- `model_id`, `model_revision`
- `seed`, `perm_r`, `primary_score`
- `dataset_revision_id`, `dataset_hash_blake3`
- `spec_hash_raw_blake3`, `spec_hash_blake3`
- `triplets_sha256`, `labels_sha256`, `feature_table_sha256`
- `script_sha256_extract`, `script_sha256_eval`, `script_sha256_featuregen`

Canonical manifest identity uses `*_blake3` fields for dataset/spec identity. Prototype-era SHA-256 artifact hashes may be carried as auxiliary provenance fields, but they are not canonical identity keys.

## File Contract (Draft)
- `gate4_token_features.csv` (token-step rows)
- `gate4_sample_summary.csv` (sample-level rows)
- `manifest.json` (run-level provenance / identity)
- UTF-8 + LF
- Deterministic row ordering:
  - token table: `(sample_id ASC, step ASC)`
  - summary: `(sample_id ASC)`

## Rust Mapping Notes (Future)
- `Gate4FeatureRow` struct should use `Option<f64>` for scores undefined on final step transitions.
- Deterministic CSV writer with fixed float formatting and stable header order.
- Machine-format floats inherit `sci_17e_v1`.
