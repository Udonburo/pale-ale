# pale-ale Internal SSOT

## 0. Scope Guard (`FROZEN`)

This document is the internal SSOT for pale-ale identity, the Gate4 contract, and research positioning.

This document does not replace and does not supersede:

- `SPEC.public.md`
- `SPEC.phase4.md`
- `SPEC.phase4.gate2.md`

This document exists in parallel with the specifications above. The public specification and the Phase 4 constitutional rules remain in force.

Gate 1 / Gate 2 cross-gate design rules, namely determinism, closed enums, exclusion patterns, and float formatting, are inherited from `SPEC.phase4.md`. Any explicit override for Gate4 requires a version bump and a migration note. `SPEC.phase4.gate2.md` is inherited as a reinforcing source for composition-dependent telemetry boundary rules.

## 1. Header

| key | value |
|---|---|
| Title | `pale-ale internal SSOT: identity, Gate4 contract, research positioning` |
| Status | `DRAFT` |
| Date | `2026-03-06` |
| SSOT Version | `v0.1.0-ssot.draft.0` |

Non-negotiables:

- determinism
- reproducibility
- strict provenance
- no silent failure
- closed enums

Commitment Levels:

- `FROZEN`: must not change without a version bump and a migration note.
- `PROVISIONAL`: may change in the future, but the reason and diff must be recorded.
- `RESEARCH`: interpretive umbrella only; not implementation-binding.

Source resolution order:

- Layer 1, Gate4-specific definitions: `tools/eval_triality_token.py` > `docs/gate4_feature_contract_draft.md` > `tools/extract_triality_triplets.py` > `tools/README_cfa.md`
- Layer 2, cross-gate rules: `SPEC.phase4.md` > `SPEC.phase4.gate2.md` > Layer 1

Evidence / attestation files are fact sources, not definition sources.

## 2. Identity & Scope (`FROZEN`)

pale-ale is a `locally fluent but globally non-integrable trajectory defect telemetry engine`.

In this document, Gate4 is a feature sink that collects proxy observables and fixes features and provenance in a stable format at token / transition / sample / run granularity. Even if the term `triality` remains in source filenames or legacy scripts, this document uses the implementation-facing umbrella term `proxy observables`. Source: `tools/extract_triality_triplets.py`, `tools/eval_triality_token.py`

pale-ale is not:

- a general hallucination detector
- an E8 / holonomy / sheaf prover
- a benchmark leaderboard claim

Within the current evidence scope, the operational definition of `defect` is:

- In a CFA row, `defect_spans` are character-span ground truth on the answer string. Source: `tools/README_cfa.md`, `tools/labels_from_cfa_spans.py`
- In the current CFA batch path, `label_source = cfa_defect_spans_v1`. Source: `tools/labels_from_cfa_spans.py`, `tools/run_cfa_batch_primaryE.py`
- A token step is positive when its `answer_char_start` and `answer_char_end` are fully contained in a defect span. The exact predicate is `ts >= ds and te <= de`. Source: `tools/labels_from_cfa_spans.py`
- In the current implementation, the transition label uses `TRANSITION_LABEL_MODE = "max_pair"`, so the transition label is `max(label_t, label_t+1)`. Source: `tools/eval_triality_token.py`

Scope guard:

- current defect labels refer primarily to constructed contradiction spans in `CFA`
- this SSOT is evidence-bound to current benchmarks and does NOT imply open-world truth claims

## 3. Minimal Math of the Signal

### 3.1 Extraction Scope and Method IDs (`FROZEN`)

In the current CFA prereg / case-study evidence path, the extraction mode is `teacher_forcing_forward_v1`. Source: `tools/extract_triality_triplets.py`, `tools/run_cfa_batch_primaryE.py`

`tools/extract_triality_triplets.py` also contains `autoregressive_generate_v1`. However, in the current evidence-bound Gate4 contract, the CFA teacher-forcing path is canonical. Gate4 sink integration for the autoregressive path remains `PROVISIONAL-TODO`.

Current method IDs:

- `proj_id = fwht_pad_pow2_take8_v1`
- `splus_def_id = attn_lastlayer_weighted_hidden_v1`
- `sminus_def_id = lm_head_row_expectation_topk{topk}_v1`
- `alignment_method = offset_overlap_v1`
- `label_mapping_mode = triplet_char_offsets_v1` in the current batch path

Source: `tools/extract_triality_triplets.py`, `tools/labels_from_cfa_spans.py`, `tools/run_cfa_batch_primaryE.py`

### 3.2 Primitive Proxy Observables (`FROZEN` definitions, `PROVISIONAL` interpretation)

The current canonical extraction path encodes `full_text = prompt + target_answer` with a fast tokenizer and `add_special_tokens=False`. Target tokens are selected from the offset mapping by taking token indices satisfying `end > answer_char_start`. Empty-offset special tokens are excluded. Source: `tools/extract_triality_triplets.py`

If the first target index is `0` in teacher forcing, `bos_token_id` is prepended to the sequence and each subsequent target index is shifted by `+1`. Source: `tools/extract_triality_triplets.py`

The raw observables in the teacher-forcing path are:

- `V_t`: `hidden_last[t, :]`
- `Splus_t`: `compute_splus_from_past(attn_to_past=out.attentions[-1][0, :, t, :t], hidden_past=hidden_last[:t, :], hidden_dim=int(hidden_last.shape[1]))`
- `Sminus_t`: `top_probs @ lm_weight[top_idx]`, where `top_idx, top_probs = topk_probs_and_entropy(logits_prev, topk=effective_topk)`

Source: `tools/extract_triality_triplets.py`

`compute_splus_from_past` has signature `compute_splus_from_past(attn_to_past: torch.Tensor, hidden_past: torch.Tensor, hidden_dim: int) -> torch.Tensor`. Shape note:

- `attn_to_past`: shape `[num_heads, t]`, taken from `out.attentions[-1][0, :, t, :t]`; the batch dimension is hard-indexed at `0`
- `hidden_past`: shape `[t, hidden]`

`compute_splus_from_past` computes `weights = attn_to_past.mean(dim=0)` and returns `torch.matmul(weights, hidden_past)`. If `hidden_past.shape[0] <= 0`, it returns a zero vector. Source: `tools/extract_triality_triplets.py`

The canonical fields written into each teacher-forcing row are:

- `step`
- `absolute_pos`
- `answer_char_start`
- `answer_char_end`
- `token_id`
- `token_str`
- `V_8d`
- `Splus_8d`
- `Sminus_8d`
- `baseline_logprob`
- `baseline_entropy`

Source: `tools/extract_triality_triplets.py`

### 3.3 Projection, Normalization, NaN / Inf Handling (`FROZEN`)

The projection function is `project_fwht_to_8(values)`. The behavior is copied directly from the current implementation.

1. Convert each input element to `float`.
2. If any element is non-finite, hard-fail via `ensure_finite_vector(..., "project_input")`.
3. Compute `target_len = next_pow2(length)`.
4. If `target_len > length`, zero-pad.
5. Apply `fwht_inplace` to the padded vector.
6. Take the first 8 coefficients.
7. If the result is shorter than 8, zero-pad.
8. Apply `normalize8`.

Source: `tools/extract_triality_triplets.py`

Behavior of `normalize8(values)`:

- Hard-fail unless length is exactly 8.
- `norm = sqrt(sum(v_i^2))`
- If `norm` is non-finite or `< 1e-12`, use deterministic fallback `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Otherwise, return `v / norm`
- Hard-fail if any output entry is non-finite

Source: `tools/extract_triality_triplets.py`

Non-finite policy on the serialization side:

- `write_ndjson` uses `json.dumps(..., allow_nan=False, separators=(",", ":"))`
- `write_meta_json` also uses `allow_nan=False`
- On the evaluator side, `parse_float` maps non-finite values to `None`
- `d_proj` hard-fails if any vector entry is non-finite

Source: `tools/extract_triality_triplets.py`, `tools/eval_triality_token.py`

### 3.4 Baseline Features (`FROZEN`)

Current baseline features are:

- `baseline_logprob`
- `baseline_entropy`

In the teacher-forcing path, `baseline_logprob = log_softmax(logits_prev)[actual_token_id]`. In the autoregressive path, `baseline_logprob = log_softmax(logits_next)[next_token_id]`. Source: `tools/extract_triality_triplets.py`

`baseline_entropy` is not full-vocab entropy. It is top-k renormalized entropy. Exact rule:

- `k = min(max(1, topk), vocab)`
- `top_vals, top_idx = torch.topk(logits, k)`
- `top_probs = torch.softmax(top_vals, dim=0)`
- `entropy = float((-(top_probs * torch.log(top_probs.clamp_min(1e-30))).sum()).item())`

Source: `tools/extract_triality_triplets.py`

Baseline definitions as score inputs:

- `A = -baseline_logprob`
- `B = baseline_entropy`

Source: `tools/eval_triality_token.py`

### 3.5 Local Geometric Distance (`FROZEN`)

The current local geometric distance is `d_proj`.

`dot_abs_clamped(x, y)`:

- `x` and `y` must both be length 8
- Each entry must be finite under `parse_float`
- `acc = ?_i x_i y_i`
- `inner = min(1.0, abs(acc))`

`d_proj(x, y)`:

- `inner = dot_abs_clamped(x, y)`
- `d_proj(x, y) = sqrt(max(0.0, 2.0 * (1.0 - inner)))`

Source: `tools/eval_triality_token.py`

### 3.6 Score Definitions `A`-`F` (`PROVISIONAL`)

Score names, formulas, and inputs follow the current implementation exactly.

| key | canonical report name | formula | alignment |
|---|---|---|---|
| `A` | `A:-logprob` | `-logprob_t` | token-step, length `N` |
| `B` | `B:entropy` | `entropy_t` | token-step, length `N` |
| `C` | `C:V_curvature` | `d_proj(V_t, V_t+1)` | transition, length `N-1` |
| `D` | `D:V_Splus_Vnext` | `d_proj(V_t, Splus_t) + d_proj(Splus_t, V_t+1)` | transition, length `N-1` |
| `E` | `E:V_Sminus_Vnext` | `d_proj(V_t, Sminus_t) + d_proj(Sminus_t, V_t+1)` | transition, length `N-1` |
| `F` | `F:loop_V_Splus_Sminus_V` | `d_proj(V_t, Splus_t) + d_proj(Splus_t, Sminus_t) + d_proj(Sminus_t, V_t)` | token-step, length `N` |

Source: `tools/eval_triality_token.py`, `docs/gate4_feature_contract_draft.md`

AUPRC determinism:

- `average_precision(labels, scores)` uses `indexed.sort(key=lambda i: (-float(scores[i]), i))`
- Therefore the AUPRC tie-break rule is fixed as `(-score, index ASC)`

Source: `tools/eval_triality_token.py`

Label alignment:

- `labels_token = labels_step`
- `labels_trans[t] = y_t`
- In the current implementation, `y_t = max(label_t, label_t+1)` because `TRANSITION_LABEL_MODE = "max_pair"`

Source: `tools/eval_triality_token.py`

`labels_step` load behavior:

- The evaluator initializes all steps to `0` by default
- A point label row is accepted as `{"step": int, "label": 0|1, ...}`
- A range label row is accepted as `{"step_start": int, "step_end": int, "label": 0|1}`
- Invalid rows are ignored

Source: `tools/eval_triality_token.py`

Reason for marking score definitions as `PROVISIONAL`:

- The current authoritative source is Python implementation
- Edge-case behavior during porting, especially NaN / Inf, empty spans, missing labels, threshold ties, and final-step missing encoding, may require revision in Rust

However, the fact that CFA prereg used the current definitions is `FROZEN`. Source: `tools/README_cfa.md`, `tools/run_cfa_batch_primaryE.py`, `attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt`

### 3.7 Primary Score Status (`FROZEN` for CFA prereg scope only)

`Primary score = E` is `FROZEN` only for the `CFA prereg evaluation scope`. It is not a universal default.

Short rationale:

- `tools/README_cfa.md` prereg section states `Primary score is fixed to E`
- `tools/run_cfa_batch_primaryE.py` hard-locks `PRIMARY_SCORE = "E"`
- The batch report records `primary_score=E`
- Representative case studies are also organized around `CFA Primary=E`

By contrast, the general evaluator CLI default is `--primary-score F`. Source: `tools/eval_triality_token.py`

Current GO thresholds in the token-eval harness are:

- `GO_MAX_P_EMP = 0.05`
- `GO_MIN_DELTA_AUPRC = 0.02`
- `GO_MIN_PRIMARY_AUPRC = 0.15`

In the current evaluator, GO is emitted only if all of the following hold:

- `primary_auprc is not None`
- `p_emp is not None`
- `best_baseline_auprc is not None`
- `primary_auprc >= GO_MIN_PRIMARY_AUPRC`
- `primary_delta is not None`
- `primary_delta >= GO_MIN_DELTA_AUPRC`
- `p_emp <= GO_MAX_P_EMP`
- `coverage_gate_pass`

Source: `tools/eval_triality_token.py`

## 4. Evidence Map (`FROZEN`, facts only)

### 4.1 What Is Evidenced

Regarding determinism / artifact stability, the following facts are evidenced in the current repo:

- The extractor fixes the seed in `random` and `torch`; when CUDA is used, it also applies `manual_seed_all` and disables TF32. Under `--deterministic`, it additionally sets `torch.use_deterministic_algorithms(True, warn_only=True)` and `cudnn.deterministic=True`, `cudnn.benchmark=False`. Source: `tools/extract_triality_triplets.py`
- Triplet NDJSON is written as UTF-8 + LF, with `allow_nan=False` and payload SHA-256. Source: `tools/extract_triality_triplets.py`
- The evaluator loads NDJSON in ascending `step` order and fixes the AUPRC tie break to `(-score, index)`. Source: `tools/eval_triality_token.py`
- The case-study index and batch report record script SHA256 and model revision. Source: `attestations/triality/case_study/index.md`, `attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt`

Regarding alignment / coverage, the following facts are evidenced in the current repo:

- The teacher-forcing path requires fast-tokenizer offset mapping. Source: `tools/extract_triality_triplets.py`
- The extractor writes `answer_char_start` / `answer_char_end` per step. Source: `tools/extract_triality_triplets.py`
- The current batch path uses `triplet_char_offsets_v1` and sets coverage to `1.0`. Source: `tools/labels_from_cfa_spans.py`, `tools/run_cfa_batch_primaryE.py`
- The current teacher-forcing extractor records `exact_token_match_ratio = extracted_target_count / expected_target_count` and hard-fails if the ratio is `< 0.98`. Source: `tools/extract_triality_triplets.py`
- The prereg batch skip rules are `exact_token_match_ratio < 0.98` and `coverage < 0.30`. Source: `tools/README_cfa.md`, `tools/run_cfa_batch_primaryE.py`

CFA prereg batch result:

- `total_rows=200`, `status_ok=200`
- `consistent=100`, `frustrated=100`
- `frustrated_median_auprc_e=4.18695021254729349e-01`
- `frustrated_median_best_baseline_auprc=2.31062877842769160e-01`
- `frustrated_median_delta_auprc=1.79877666081394716e-01`
- `empirical_p_value=4.99750124937531218e-04`
- `VERDICT=GO`

Source: `attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt`

Case-study behavior:

- The representative set splits 15 frustrated samples into `top`, `median`, and `bottom` buckets. Source: `attestations/triality/case_study/index.md`, `tools/README_cfa.md`
- The case-study local metric command uses `--topk 10 --percentile 0.90`. Source: `tools/README_cfa.md`, `tools/eval_local_span.py`
- `hit_at_10` is the positive-token count inside the top-10 `score_E` tokens. `first_hit_distance_signed = first_hit_step - defect_start_step`. Source: `tools/eval_local_span.py`
- All 15 rows in the representative-set summary have `coverage=1.0` and `exact_match=1.0`. Source: `attestations/triality/case_study/representative_set_summary.md`
- In the top bucket, `hit@10` is `6,4,6,6,4`, and `first_hit_distance` is `0,0,5,1,0`. Source: `attestations/triality/case_study/representative_set_summary.md`
- In the median bucket, `hit@10` is `3,3,3,2,2`, and `first_hit_distance` is `4,3,-5,2,2`. Source: `attestations/triality/case_study/representative_set_summary.md`
- In the bottom bucket, `hit@10` is `2,4,2,3,3`, and `first_hit_distance` is `9,2,11,3,3`. Source: `attestations/triality/case_study/representative_set_summary.md`

Negative stability check (consistent-only diagnostics):

- Population: `consistent_samples=100`, `frustrated_samples=100`, `consistent_transitions=2287`
- Consistent run-level `score_E`: `mean=2.243e+00`, `p90=2.679e+00`, `max=2.815e+00`
- Reference: `median_frustrated_max_E=2.755e+00`
- Stability criterion: share of consistent samples with `max_E < median_frustrated_max_E` = `0.48`
- Verdict: `Red` (threshold was `>= 0.90` for Green)
- Pooled Spearman: `rho(E, A)=-0.127`, `rho(E, B)=-0.137` (weak negative; spikes are not baseline copies)
- Strongest consistent-side spikes are dominated by tokenizer/subword seam patterns (`"trans"->"itivity"`, `"ability"->"."`, subword-split proper nouns). This is an observation, not a proven causal claim.
- CFA GO remains valid: GO is AUPRC-based (positive enrichment), not threshold-based (absolute separation)

Source: `attestations/triality/negative_stability/consistent100_scoreE_report.txt`, `tools/check_cfa_negative_stability.py`

### 4.2 What Is Not Proven

The following are not proven by the current repo evidence:

- open-world hallucination detection
- generalization across models
- robustness to tokenizer / model revision changes
- any theorem-level claim about holonomy / sheaf / proxy-observable theory / E8
- false positive rate stability on ground-truth negative distributions
- `score_E` absolute threshold separation between consistent and frustrated samples (negative stability check returned `Red`; `max(score_E)` distributions overlap at 48% vs 90% Green threshold)

## 5. Gate4 Frozen Contract: Feature Sink

### 5.1 Boundary (`FROZEN`)

Gate4 is collection / aggregation only.

Gate4 may:

- collect observables and scores already defined by the current extractor / labeler / evaluator
- sink features and provenance into a fixed schema at token / transition / sample / run granularity
- perform aggregation for prereg evaluation reports and case-study outputs

Gate4 must not:

- introduce new math
- introduce learned fusion
- introduce thresholding
- emit verdicts beyond prereg evaluation reports
- invent benchmark-specific scores inside Gate4

### 5.2 Inputs (`PROVISIONAL`)

#### 5.2.1 Required token-step NDJSON schema for current CFA path

In the current CFA teacher-forcing path, Gate4 sink must accept the following row schema. Source: `tools/extract_triality_triplets.py`

| field | type | requirement | note |
|---|---|---|---|
| `step` | int | required | current writer emits `0..N-1`; duplicate-step behavior is `UNSPECIFIED`. |
| `absolute_pos` | int | required | absolute position in the teacher-forcing tensor. |
| `answer_char_start` | int | required for CFA char-offset labels | if absent, `triplet_char_offsets_v1` cannot be used. |
| `answer_char_end` | int | required for CFA char-offset labels | if absent, `triplet_char_offsets_v1` cannot be used. |
| `token_id` | int | required | target token id. |
| `token_str` | string | required | raw token string. |
| `V_8d` | array[8] of finite float | required | after `fwht_pad_pow2_take8_v1`. |
| `Splus_8d` | array[8] of finite float | required | after `attn_lastlayer_weighted_hidden_v1` + projection. |
| `Sminus_8d` | array[8] of finite float | required | after `lm_head_row_expectation_topk{topk}_v1` + projection. |
| `baseline_logprob` | finite float | required | for the teacher-forced actual token. |
| `baseline_entropy` | finite float | required | top-k renormalized entropy. |

Ordering constraints:

- semantic order is `step ASC`
- the evaluator sorts by `step ASC` on load
- field order on wire depends on current Python dict insertion order, but the formal schema order is `UNSPECIFIED`

#### 5.2.2 Required extraction metadata / provenance fields

The current extractor `meta.json` explicitly defines these provenance fields:

- `model_id`
- `model_revision`
- `transformers_version`
- `torch_version`
- `seed`
- `topk_requested`
- `topk_effective`
- `max_new_tokens`
- `proj_id`
- `splus_def_id`
- `sminus_def_id`
- `prompt_sha256`
- `target_answer_sha256`
- `output_ndjson_sha256`
- `output_ndjson_path`
- `device`
- `dtype`
- `deterministic_requested`
- `n_steps_written`
- `extraction_mode`
- `alignment_method`
- `target_token_count_expected`
- `target_token_count_extracted`
- `exact_token_match_ratio`
- `bos_prepended_for_teacher_forcing`
- `answer_char_start`
- `target_token_indices_count`
- `target_only_token_count`
- `boundary_merge_token_delta`

Source: `tools/extract_triality_triplets.py`

`topk_requested` and `topk_effective` are extractor provenance. The current evaluator does not reuse these values to recompute baselines. The evaluator reads `baseline_entropy` directly from the NDJSON. Therefore `topk_*` is carried as score provenance, not as an eval-side recomputation parameter. Source: `tools/extract_triality_triplets.py`, `tools/eval_triality_token.py`

#### 5.2.3 Required label schema

The evaluator accepts the following label row schema. Source: `tools/eval_triality_token.py`

- point row: `{"step": int, "label": 0|1, ...}`
- range row: `{"step_start": int, "step_end": int, "label": 0|1}`

The current CFA label writer emits the following row schema. Source: `tools/labels_from_cfa_spans.py`

- `step`
- `label`
- `token_id`

Current labels meta fields:

- `label_source`
- `cfa_jsonl`
- `sample_id`
- `variant`
- `world_type`
- `has_defect` in standalone label tool only
- `triplets_path`
- `tokenizer_model` and `tokenizer_local_files_only` in standalone label tool only
- `label_mapping_mode`
- `n_triplet_steps`
- `n_answer_tokens` in standalone label tool only
- `n_defect_spans`
- `answer_positive_token_count` in standalone label tool only
- `mapped_positive_tokens`
- `total_positive_tokens`
- `equal_blocks`
- `final_alignment_coverage_ratio`
- `min_coverage_threshold`
- `fail_below_coverage`
- `final_positive_steps`
- `final_negative_steps`
- `labels_out`

Source: `tools/labels_from_cfa_spans.py`, `tools/run_cfa_batch_primaryE.py`

#### 5.2.4 Span-to-token mapping constraints

Canonical mapping constraints in the current CFA path:

- if the extractor row has `answer_char_start` / `answer_char_end`, use `triplet_char_offsets_v1`
- the positive-token predicate is `ts >= ds and te <= de`
- the direct char-offset path defines `coverage = 1.0`
- fallback `token_id_sequence_alignment_v1` exists only if the direct char-offset path cannot be used
- fallback uses equal-block mapping from `difflib.SequenceMatcher(..., autojunk=False)`

Source: `tools/labels_from_cfa_spans.py`

### 5.3 Outputs (`PROVISIONAL`)

#### 5.3.1 Per-token table contract

The normative target artifact name is `gate4_token_features.csv`. Source: `docs/gate4_feature_contract_draft.md`

Required columns:

- identity: `run_id`, `sample_id`, `variant`, `world_type`, `step`, `absolute_pos`
- label / coverage: `label_token`, `label_transition`, `label_coverage_ratio`, `exact_token_match_ratio`
- missing-state: `transition_missing_reason`
- score: `score_A_logprob`, `score_B_entropy`, `score_C_v_curvature`, `score_D_v_splus_vnext`, `score_E_v_sminus_vnext`, `score_F_loop`

Optional columns:

- `defect_span_id`
- `z_A`, `z_B`, `z_C`, `z_D`, `z_E`, `z_F`
- `rank_E_desc`
- `is_topk_E`

Source: `docs/gate4_feature_contract_draft.md`

Final-step undefined rule:

- `docs/gate4_feature_contract_draft.md` explicitly uses `Option<f64>` in its Rust mapping note for `scores undefined on final step transitions`
- therefore, under the current canonical interpretation, transition-aligned scores `C/D/E` are undefined at the final step in the token table
- the on-wire CSV encoding is now fixed: `score_C_v_curvature`, `score_D_v_splus_vnext`, and `score_E_v_sminus_vnext` serialize as empty string on the final step
- the companion closed enum is `transition_missing_reason = none|final_step_no_successor`

Source: `docs/gate4_feature_contract_draft.md`

The current auxiliary case-study token table emits a narrower schema. It is not the canonical Gate4 sink; it is a case-study support artifact. Current columns:

- `step`
- `absolute_pos`
- `token_text`
- `label`
- `is_defect_token`
- `score_A_logprob`
- `score_B_entropy`
- `score_E`
- `z_A`
- `z_B`
- `z_E`
- `rank_E_desc`
- `in_defect_span`

Source: `tools/plot_cfa_case_pair.py`

#### 5.3.2 Per-sample summary contract

The normative target artifact name is `gate4_sample_summary.csv`. Source: `docs/gate4_feature_contract_draft.md`

Required columns:

- `auprc_A`, `auprc_B`, `auprc_C`, `auprc_D`, `auprc_E`, `auprc_F`
- `best_baseline_name`
- `delta_auprc_E_vs_best_baseline`
- `hit_at_10_E`

Source: `docs/gate4_feature_contract_draft.md`

Optional support metrics:

- `first_hit_distance_E_p90`
- `first_hit_after_defect_distance_E_p90`

Source: `docs/gate4_feature_contract_draft.md`

The currently implemented batch summary row schema is `results.jsonl`. Current keys:

- `sample_id`
- `variant`
- `world_type`
- `primary_score`
- `seed`
- `perm_r`
- `status`
- `reason`
- `exact_token_match_ratio`
- `coverage`
- `AUPRC_E`
- `AUPRC_best_baseline`
- `delta`
- `p_emp`
- `no_positive_imputed`
- `sample_dir`
- `eval_report`
- `model_id`
- `model_revision`
- `eval_verdict` on successful rows

Source: `tools/run_cfa_batch_primaryE.py`

The current case-study per-sample meta JSON stores subset metrics and artifact SHA values. Source: `tools/plot_cfa_case_pair.py`

`first_hit_distance_E_p90` and `first_hit_after_defect_distance_E_p90` do not have a fixed percentile population anywhere in the current repo. `eval_local_span.py` only defines single-sample scalars `first_hit_distance_signed` / `first_hit_after_defect_distance`. Therefore these two columns are optional support metrics, and their exact computation contract remains `PROVISIONAL-TODO`. Source: `docs/gate4_feature_contract_draft.md`, `tools/eval_local_span.py`

#### 5.3.3 Run-level manifest contract

There is no canonical Gate4 `manifest.json` yet in the current prototype. The current authoritative run-level artifact is the prereg batch report text. Current fields:

- `date`
- `experiment`
- `primary_score`
- `seed`
- `perm_r`
- `cfa_jsonl`
- `results_jsonl`
- `cfa_sha256`
- `results_sha256`
- `status_counts`
- `class_valid_counts`
- `class_descriptive_stats`
- `group_contrast_on_delta_auprc`
- `preregistered_gate`
- `model_details`
- `script_sha256`

Source: `tools/aggregate_cfa_batch.py`, `attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt`

At minimum, the feature sink must carry the following provenance from current code:

- `model_id`
- `model_revision`
- `seed`
- `perm_r` when evaluation is run
- `primary_score` when evaluation is run
- `dataset_revision_id`
- `dataset_hash_blake3`
- `spec_hash_raw_blake3`
- `spec_hash_blake3`
- `proj_id`
- `splus_def_id`
- `sminus_def_id`
- `triplets_sha256`
- `labels_sha256`
- `feature_table_sha256`
- `script_sha256_extract`
- `script_sha256_eval`
- `script_sha256_featuregen`
- auxiliary SHA-256 artifact hashes when emitted by prototype tools

Canonical Gate4 manifest identity now follows the Phase4 naming pattern for dataset/spec identity:

- `dataset_revision_id`
- `dataset_hash_blake3`
- `spec_hash_raw_blake3`
- `spec_hash_blake3`

Prototype-era SHA-256 content hashes such as `cfa_sha256`, `results_sha256`, `prompt_sha256`, and `target_answer_sha256` remain auxiliary provenance only. They are not canonical identity keys. Source: `docs/gate4_feature_contract_draft.md`, `tools/extract_triality_triplets.py`, `tools/aggregate_cfa_batch.py`, `tools/plot_cfa_case_pair.py`, `SPEC.phase4.md`

### 5.4 Determinism Rules (`FROZEN`, inherited)

Sorting:

- token table row order: `(sample_id ASC, step ASC)` if multi-sample table. Source: `docs/gate4_feature_contract_draft.md`
- summary row order: `(sample_id ASC)`. Source: `docs/gate4_feature_contract_draft.md`
- NDJSON logical row order: `step ASC`. Source: `tools/eval_triality_token.py`
- AP ranking tie break: `(-score, index)`. Source: `tools/eval_triality_token.py`
- inherit the `total_cmp` principle for cross-gate float comparisons. Source: `SPEC.phase4.md`

Stable hashing:

- current prototypes use SHA-256 file-content hashes for artifacts and scripts. Source: `tools/extract_triality_triplets.py`, `tools/aggregate_cfa_batch.py`, `tools/plot_cfa_case_pair.py`
- Phase4 manifest identity treats `spec_hash_*_blake3` and `dataset_hash_blake3` as canonical. Source: `SPEC.phase4.md`
- integrated Gate4 manifest hashing follows the Conflict Register

Closed enums:

- the closed-enum requirement itself is inherited as `FROZEN`. Source: `SPEC.phase4.md`
- Gate4 v1 fixes the token-table missing enum as `transition_missing_reason = none|final_step_no_successor`
- prototype batch `status = ok|error|skip_empty_text|skip_token_match|skip_coverage` plus free-text `reason` remains auxiliary and non-canonical for Gate4 sink output. Source: `tools/run_cfa_batch_primaryE.py`

Float formatting:

- inherit `float_format_id = sci_17e_v1` as the cross-gate machine-artifact rule. Source: `SPEC.phase4.md`
- current prototype report / CSV writers use `"{:.17e}"`, but missing floats are encoded inconsistently as `NA` or empty string depending on artifact. Source: `tools/eval_triality_token.py`, `tools/plot_cfa_case_pair.py`, `tools/aggregate_cfa_batch.py`
- Gate4 token-table numeric missing encoding is fixed to empty string for final-step transition-aligned score fields `C/D/E`

### 5.5 Explicit Non-Goals (`FROZEN`)

- no new math
- no learned fusion
- no thresholding
- no verdict output beyond prereg evaluation reports
- no benchmark-specific score invention inside Gate4

## 6. Conflict Register (`FROZEN`)

| id | conflict | winner by priority | SSOT choice | migration rule |
|---|---|---|---|---|
| `C1` | general evaluator default primary score is `F`, while CFA prereg fixes primary to `E` | scoped split; general rule from `tools/eval_triality_token.py`, prereg scope from `tools/README_cfa.md` plus batch code | general evaluator default remains `F`; CFA prereg scope freezes `E` only | doc / UI must always label whether a statement is ?general evaluator default? or ?CFA prereg endpoint? |
| `C2` | Gate4 docs did not previously declare an exact float-format id, and missing values diverged across Python artifacts as `NA` or empty string | Layer 2 wins: `SPEC.phase4.md` plus current Gate4 draft patch | machine-format floats inherit `sci_17e_v1`; Gate4 token-table missing encoding is fixed to empty string for final-step transition-aligned scores `C/D/E` | keep prototype-era artifact-specific missing encodings outside canonical Gate4 token-table schema; align future writers to the fixed token-table rule |
| `C3` | current Gate4 prototypes hash artifacts with SHA-256, while Phase4 manifest identity uses BLAKE3 fields and dataset identity naming was previously unresolved | Layer 2 wins for integrated manifest | canonical Gate4 manifest identity uses `dataset_revision_id`, `dataset_hash_blake3`, `spec_hash_raw_blake3`, and `spec_hash_blake3`; current SHA-256 usage is auxiliary provenance only | migration scope includes at least `prompt_sha256`, `target_answer_sha256`, `output_ndjson_sha256`, `cfa_sha256`, `results_sha256`, `script_sha256`, and case-study artifact SHA fields. When Gate4 gets canonical manifest support, emit both migration-era SHA-256 artifact hashes and Phase4 BLAKE3 identity fields until deprecation is documented |
| `C4` | coverage naming diverges: `label_coverage_ratio` in feature-contract draft, `final_alignment_coverage_ratio` in labels meta, `coverage` in batch results | artifact-specific names coexist; no single higher-priority artifact covers all three contexts | canonical token-table column name stays `label_coverage_ratio`; canonical labels-meta field stays `final_alignment_coverage_ratio`; batch `coverage` is auxiliary | future Gate4 sink should either normalize all three under a shared glossary or emit explicit aliases |
| `C5` | best-baseline naming diverges: `best_baseline_name (A|B)` in draft / case-study code, `best_baseline = A:-logprob|B:entropy` in evaluator report; evaluator also hard-resolves exact ties in favor of `A:-logprob` | context split; score-name authority from `tools/eval_triality_token.py`, compact CSV field from `docs/gate4_feature_contract_draft.md` | score identities are `A:-logprob` and `B:entropy`; compact summary shorthand may use `A|B` only when schema explicitly says so; exact AUPRC ties choose `A:-logprob` in the current evaluator | future canonical summary should carry both `best_baseline_key` and `best_baseline_name_full`; if tie semantics matter outside evaluator reports, freeze them explicitly |
| `C6` | draft previously required `first_hit_distance_E_p90` and `first_hit_after_defect_distance_E_p90`, but current code only defines single-sample distances | resolved by Gate4 draft patch | both fields move out of the required summary contract and remain optional support metrics with `PROVISIONAL-TODO` computation semantics | define percentile scope, aggregation population, and tie handling before promoting them back into any frozen summary schema |
| `C7` | Layer 2 requires closed enums, but current Gate4 batch results use ad hoc `status` values plus free-text `reason` | Layer 2 wins on principle; current Gate4 draft patch fixes the token-table missing enum only | Gate4 v1 fixes `transition_missing_reason = none|final_step_no_successor`; prototype batch `status/reason` remain auxiliary and non-canonical | introduce any additional Gate4-specific `excluded_reason` / `metric_missing_reason` enums only when a canonical Gate4 writer actually needs them |
| `C8` | gate4 feature draft expects full A-F sink columns, while current case-study token table emits only A/B/E | Layer 1 winner for canonical sink schema is `docs/gate4_feature_contract_draft.md`; case-study output is auxiliary evidence artifact | canonical Gate4 sink targets full A-F contract; current case-study table is not canonical | extend feature writer or keep case-study table explicitly non-canonical in future docs |

## 7. Roadmap & Freeze Triggers (`PROVISIONAL`)

The following are required to promote this document from `DRAFT` to `FROZEN`:

- at least one non-synthetic benchmark beyond CFA OR a stricter CFA v2
- an expanded baseline set, for example windowed surprisal, that is still beaten
- replication across 2 model variants in the same family
- demonstrated false positive rate stability on a ground-truth negative set
- one end-to-end Gate4 implementation exercising the contract

Items that remain `RESEARCH` only:

- holonomy framing
- sheaf obstruction framing
- triality as Lie-theoretic framing
- E8 narratives

In summary: within the current repo, the items that can be fixed as implementation-binding are the extraction / projection / score definitions for proxy observables, the endpoint choice for CFA prereg scope, and the boundary that limits Gate4 to a feature sink. Theoretical narratives beyond that boundary are not frozen by this document.
