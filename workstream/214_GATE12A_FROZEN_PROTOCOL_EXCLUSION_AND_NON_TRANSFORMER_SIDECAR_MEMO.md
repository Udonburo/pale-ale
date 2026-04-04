# Gate12A Frozen-Protocol Exclusion And Non-Transformer Sidecar Memo

Status: empirical sidecar memo draft
Role: narrow sidecar memo recording one frozen-protocol exclusion (`Gemma 4 E2B`), one dependency-polluted exclusion (`RWKV v6 Finch 1.6B`), and one completed non-transformer sidecar transcript run (`state-spaces/mamba-2.8b-hf`) under an explicitly changed observation surface, not a dense-transformer checkpoint rewrite, not a non-transformer portability claim, and not a Gate12B promotion
Date: 2026-04-05

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md`

## 0. Scope

This memo asks only:

- `what happened when the current single-GPU FP32 line was pushed beyond the closed dense-transformer mainline into one frozen-protocol exclusion path and one non-transformer sidecar path?`

Its subject is fixed to:

- `google/gemma-4-e2b-it`
- `RWKV/v6-Finch-1B6-HF`
- `state-spaces/mamba-2.8b-hf`
- the current `transformers==4.57.3` frozen mainline environment
- the current single-GPU FP32 lane on one `L4`

It does:

- record why `Gemma 4 E2B` stayed out of the main evidence set
- record why `RWKV v6 Finch 1.6B` stayed out of the main evidence set
- record one completed `Mamba 2.8B` transcript sidecar handoff plus local Gate12A replay
- keep the sidecar/mainline boundary explicit

It does not:

- promote any non-transformer result into the dense-transformer mainline
- claim that the `Mamba` fallback observation is equivalent to the main attention-weighted observation
- claim non-transformer structural portability
- revise the current checkpoint boundary beyond `202`

## 1. Admission Rule

For this line, mainline admission is stricter than simple loader success.

A candidate belongs to the current dense-transformer main evidence set only if all of the following remain unchanged:

- `transformers==4.57.3`
- no extractor branch specific to the model
- single-GPU FP32 execution
- the existing Gate8 handoff and Gate12A replay protocol

So `loader succeeded` is necessary but not sufficient.

This matters because the present paper is stronger if it keeps the controlled regime narrow and explicit.

## 2. Gemma 4 E2B Was Excluded By Frozen-Protocol Failure

The attempted candidate was:

- `google/gemma-4-e2b-it`

The attempted frozen-protocol smoke failed at configuration load under:

- `transformers==4.57.3`

The concrete failure was:

- `AutoConfig.from_pretrained(...)` raised `ValueError` because the checkpoint declared `model_type = gemma4` and the frozen library did not recognize that architecture

So the exclusion reason is not:

- a negative empirical Gate12A result

It is:

- a frozen-protocol failure at the library boundary

This exclusion therefore strengthens the current controlled regime rather than weakening it:

- `Gemma 4 E2B` was not omitted because it was inconvenient
- it was omitted because the mainline environment was held fixed

## 3. RWKV v6 Finch 1.6B Was Excluded By Dependency And Loader Pollution

The attempted candidate was:

- `RWKV/v6-Finch-1B6-HF`

The loader smoke only succeeded after crossing a boundary the current mainline does not allow:

- `trust_remote_code=True`

and then failed again because the dynamically loaded model required:

- `bitsandbytes`

So the exclusion reason is not:

- a negative Gate12A replay result

It is:

- remote-code admission plus extra dependency pollution beyond the frozen dense-transformer line

This keeps the current mainline sentence clean:

- `RWKV` was tested for feasibility
- but it did not satisfy the current controlled-regime admission rule

## 4. Mamba 2.8B Completed Gate8 As A Sidecar Only

The accepted sidecar candidate was:

- `state-spaces/mamba-2.8b-hf`

Unlike `Gemma 4` and `RWKV`, `Mamba 2.8B` did load and run on the current single-GPU FP32 lane.

But it did not preserve the main observation surface.

The current extractor uses:

- `attn_lastlayer_weighted_hidden_v1`

and directly reads `out.attentions`.

`Mamba` does not return that surface.
So the extractor was extended with an explicit opt-in fallback:

- `prefix_mean_hidden_v1`

The sidecar handoff is now fixed locally at:

- `runs/remote_gpu_imports/mamba28b_transcript_sidecar_gate8_handoff_2026-04-04/`

The extracted candidate-execution handoff is:

- `runs/remote_gpu_imports/mamba28b_transcript_sidecar_gate8_handoff_2026-04-04/extracted/runs/gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_candidate_execution/`

The sample-level sidecar metadata records:

- `model_id = state-spaces/mamba-2.8b-hf`
- `splus_def_id = prefix_mean_hidden_v1`
- `dtype = float32`
- `family = transcript_v1`
- `samples = 128`

So the current `Mamba` run is a valid sidecar transcript run.
It is not a valid mainline dense-transformer replication.

## 5. Mamba Sidecar Diverged Upstream Before The Terminal Gate12A Readout

The local Gate12A replay surfaces are:

- `runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_gate9k/`
- `runs/gate12a_calibration_seed_audit_recheck_from_gate12a_upstream_gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_gate9k/`
- `runs/gate12a_triangle_text_surface_audit_recheck_from_gate12a_upstream_gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_gate9k/`
- `runs/gate12a_triangle_reading_queue_recheck_from_gate12a_upstream_gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_gate9k/`
- `runs/gate12a_triangle_reading_packet_balanced_recheck_from_gate12a_upstream_gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_gate9k/`
- `runs/gate12a_triangle_phenotype_tag_prep_recheck_from_gate12a_upstream_gate8cm_state_spaces_mamba_2_8b_hf_transcript_128r_gate9k/`

The terminal Gate12A discrete read is not, by itself, decisive here.

Its terminal counts match the positive dense-transformer transcript pattern at the same surface shape:

- `defined_triangle_holonomy_count = 320`
- `defined_triangle_holonomy_within_threshold_count = 0`
- `zero_overlap_count = 0`
- `packet_row_count = 12`

So the negative sidecar reading is anchored earlier in the path.

At `Gate9J`, the current `Mamba` transcript line binds as:

- `distributed_underactivation_status = clear`
- `distributed_consistent_branch_status = clear`
- `direct_baseline_answer_suppression_status = clear`
- `gap_loss_explained_as_token_only_status = not_yet_denied`

where the current dense-transformer transcript lines instead bind with the stronger underactivation/suppression pattern, e.g. `Qwen3-4B` records:

- `distributed_underactivation_status = triggered`
- `distributed_consistent_branch_status = underactivated`
- `direct_baseline_answer_suppression_status = triggered`
- `gap_loss_explained_as_token_only_status = denied`

That divergence persists at `Gate9K`.

`Mamba 2.8B` binds there as:

- `distributed_underactivation_status_at_bind = clear`
- `distributed_consistent_branch_status_at_bind = clear`

while the current dense-transformer transcript lines bind the same location with:

- `distributed_underactivation_status_at_bind = triggered`
- `distributed_consistent_branch_status_at_bind = underactivated`

So the current narrow negative sentence is not:

- `the terminal Gate12A discrete count alone proved failure`

It is:

- `under prefix_mean_hidden_v1, the Mamba sidecar can be carried through Gate8 and Gate12A, but it does not preserve the current dense-transformer transcript path signature through Gate9J/Gate9K`

So the current narrow sidecar sentence is:

- `Mamba 2.8B can be carried through Gate8 and Gate12A under a prefix-mean hidden fallback surface, but that sidecar surface does not reproduce the current dense-transformer transcript path signature.`

## 6. What This Memo Earns

This memo earns only the following narrow sentence:

- `the current main dense-transformer evidence set remains clean because Gemma 4 and RWKV were excluded by frozen-protocol and dependency-admission rules, while Mamba 2.8B completed only as a sidecar under prefix_mean_hidden_v1 and did not preserve the current dense-transformer transcript path signature through Gate9J/Gate9K.`

This is enough to say:

- the current admission rule is real rather than rhetorical
- not every promising contemporary model was quietly folded into the main evidence set
- the first non-transformer sidecar did produce evidence
- that evidence is negative with respect to the current dense-transformer transcript path signature claim

This is not enough to say:

- that attention is uniquely responsible for the structural replay
- that all non-transformers will fail the same way
- that `Mamba` phenotype should now be compared directly to the dense-transformer first-pass line

## 7. Release Boundary Note

This memo should be treated as:

- a tracked post-checkpoint sidecar memo

and not as:

- a checkpoint rewrite
- a new release boundary

The current published checkpoint remains `202`.
The dense-transformer technical-report line in `213` remains the mainline statement.
This sidecar memo only records the first controlled attempt to move beyond it.

## 8. Next Honest Move

The next honest move is:

1. keep `Gemma 4 E2B` recorded as a frozen-protocol exclusion
2. keep `RWKV v6 Finch 1.6B` recorded as a dependency-polluted exclusion
3. treat `Mamba 2.8B` as a negative sidecar result unless a new observation surface is independently justified
4. avoid mixing `prefix_mean_hidden_v1` evidence into the current dense-transformer mainline claim

That leaves the current paper in a strong position:

- dense-transformer mainline remains controlled
- sidecar probes are real
- negative sidecar evidence is kept explicit rather than hidden
