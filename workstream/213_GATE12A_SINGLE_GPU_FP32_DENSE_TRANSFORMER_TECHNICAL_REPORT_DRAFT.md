# Gate12A Single-GPU FP32 Dense-Transformer Technical Report Draft

Status: technical report draft
Role: first concise synthesis over the current Gate12A post-checkpoint dense-transformer evidence, limited to a controlled single-GPU FP32 regime, recording structural replay, phenotype non-portability, and the now-tested cloud-to-local handoff pipeline, not a transformer-universal law claim, not a multi-GPU scale claim, and not a Gate12B promotion
Date: 2026-04-04

This draft proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md`
- `206_GATE12A_LLAMA_3_2_1B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`
- `212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`

## 0. Scope

This draft asks only:

- `what can now be said, narrowly and defensibly, about Gate12A under the current single-GPU FP32 dense-transformer regime?`

Its subject is fixed to:

- the current committed Gate12A observable surface
- dense-transformer instruction models only
- single-GPU FP32 candidate generation
- local replay and first-pass attachment
- the current transcript_v1 / briefing_v1 / archive_v1 family set where available

It does:

- synthesize the current full-family-set dense-transformer evidence
- separate structural replay from phenotype behavior
- record that the current cloud-to-local handoff pipeline now works
- state a narrow technical claim suitable for a first paper or whitepaper draft

It does not:

- claim a transformer-universal law
- claim scale invariance above the currently observed 3B/4B regime
- claim non-transformer portability
- claim that phenotype behavior is model-invariant
- revise the published checkpoint boundary beyond `202`

## 1. Controlled Regime

The current report is intentionally narrow.

The operative regime is:

- single-GPU FP32 candidate generation on cloud hardware
- replayable `gate8cm_*_candidate_execution` handoff bundles
- local `run_gate12a_family_replay.py`
- local first-pass attachment and summarize-only refresh

The practical lesson is now explicit:

- `slim core` import bundles are sufficient for memo/provenance use
- they are not sufficient for local Gate12A replay
- replayable handoff requires full `candidate_execution`-equivalent evidence
- the current `run_gate8_candidate_finalize.py` helper closes late-stage Gate8 handoffs into replayable bundles

So the present report is also an operational result:

- `Gate8 in cloud, Gate12A local` is no longer a proposal; it has now run successfully on current local evidence

## 2. Evidence Base

The main full-family-set evidence in this draft is:

- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

represented by:

- `runs/gate12a_cross_model_replay_qwen_qwen2_5_3b_instruct/`
- `runs/gate12a_cross_model_replay_meta_llama_llama_3_2_3b_instruct/`
- `runs/gate12a_cross_model_replay_qwen_qwen3_4b/`

with attached first-pass surfaces for all three families on each model.

Supporting but secondary context remains available from:

- `Qwen/Qwen2.5-0.5B`
- `meta-llama/Llama-3.2-1B-Instruct`
- transcript-only imported follow-ups such as `Qwen/Qwen2.5-1.5B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`

Those secondary lines help interpret the current results, but the present report does not need them to establish its main narrow claim.

## 3. Structural Result

Across the current 3B/4B full-family-set evidence, the machine-side structural checkpoints stay live.

For all nine family runs in the current main evidence set:

- `zero_overlap_clear = True`
- `all_defined_triangles_anchor_rich = True`
- `trusted_tree_gt_residual_chord = True`
- `plain_gt_anchor_qualified = True`

So the strongest current structural sentence is:

- under the current single-GPU FP32 dense-transformer regime, Gate12A structural replay survives across the tested 3B/4B family sets

This is already stronger than a one-family anecdote.
It survives:

- multiple vendors
- multiple model lines
- multiple family prompts

while keeping the observable surface fixed.

## 4. Phenotype Result

The phenotype layer does not collapse with the structure.

The current evidence says:

- structural replay is portable
- phenotype replay is not simply portable

More concretely:

- `Qwen2.5-3B-Instruct` preserves the structural layer but shows family-specific mixed phenotype surfaces rather than a single canonical rule
- `Llama-3.2-3B-Instruct` preserves the structural layer while shifting much more aggressively at briefing/archive flat-band surface
- `Qwen3-4B` preserves the structural layer but does not simply inherit the `Qwen2.5-3B-Instruct` phenotype surface

So the present phenotype result is:

- family conditioning persists
- vendor or line identity still matters
- scale-up or vendor-upgrade does not produce phenotype invariance

That is not a weak result.
It is the main reason the report can stay narrow without collapsing into a universal-law claim.

## 5. Current Technical Claim

The current first-paper claim should be kept to this level:

- within the current single-GPU FP32 dense-transformer regime, Gate12A structural replay replicates across the tested 3B/4B family sets, while the first-pass phenotype layer remains family-conditioned and model-specific

That sentence is strong enough to matter, but narrow enough to defend.

## 6. Why The Current 3B/4B Regime Is Enough For A First Paper

This report is not trying to win a scale contest.
It is trying to establish a controlled empirical surface.

The present regime is useful because it keeps three things fixed enough to interpret:

- precision: current runs stay in FP32 rather than collapsing into quantized noise
- transport surface: the same Gate12A observable is reused across models and families
- operational reproducibility: the cloud-to-local handoff is now tested rather than hypothetical

So the current 3B/4B lines are not merely a fallback.
They are the presently controlled regime.

## 7. Not Claimed

This report does not claim:

- that every transformer family will reproduce the same result
- that 7B, 70B, or larger models will preserve the same phenotype surface
- that non-transformer lines such as Mamba or RWKV already fall under the same statement
- that the current result should be published as a Zenodo-bound law claim

Those are future extensions, not current conclusions.

## 8. Future Work

The obvious next extensions are:

- close `mistralai/Mistral-7B-Instruct-v0.3` as a full family set
- add non-transformer sidecar checks such as Mamba or RWKV
- test whether larger multi-GPU FP32 regimes preserve the same structural result
- determine whether phenotype non-portability narrows, widens, or phase-shifts beyond the present 3B/4B envelope

For now, those belong in future work rather than in the main claim.

## 9. Bottom Line

The first concise paper can already say something real:

- the current Gate12A structural layer is reproducible across multiple 3B/4B dense-transformer family sets under one controlled FP32 regime
- the phenotype layer is not eliminated by that structural replay
- the cloud-to-local Gate8-to-Gate12A pipeline is now operational

That is enough for a first technical report.
