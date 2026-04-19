# Gate12A Evidence Atlas

This atlas is a map of the current frozen Gate12A protocol surface. It is not a
leaderboard, a threshold doctrine, or a universal-law claim.

The atlas separates three things that should stay separate:

- machine-side structural replay under the frozen single-GPU FP32 Gate12A
  observable surface
- packet-local or memo-local first-pass phenotype readings
- admission boundaries, sidecar-only runs, and excluded surfaces

Evidence is limited to the tracked Gate12A memo line. This document does not
read `runs/` as a primary source and does not add new execution.

## Evidence Boundary

The rows below are grounded in these tracked memos only:

- [`200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md`](../workstream/200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md)
- [`201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`](../workstream/201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md)
- [`202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md`](../workstream/202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md)
- [`206_GATE12A_LLAMA_3_2_1B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/206_GATE12A_LLAMA_3_2_1B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
- [`207_GATE12A_QWEN_2_5_1_5B_INSTRUCT_TRANSCRIPT_V1_GPU_IMPORT_REPLICATION_MEMO.md`](../workstream/207_GATE12A_QWEN_2_5_1_5B_INSTRUCT_TRANSCRIPT_V1_GPU_IMPORT_REPLICATION_MEMO.md)
- [`210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
- [`211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
- [`212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
- [`214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md`](../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md)
- [`215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md)

Where a memo describes a local provisional, packet-local, or memo-local
first-pass read, this atlas preserves that limitation.

## Summary Matrix

| Model | Scale | Family coverage | Structural replay status | First-pass phenotype readout | Tracked memo pointer |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Mainline dense-transformer fixed family set: `transcript_v1 / briefing_v1 / archive_v1` | Held across the full family set at machine scope under the unchanged Gate12A observable surface. | Family-conditioned. Transcript remains scale-sensitive; briefing preserves a Qwen-style high-tension split while shifting flat-band minority; archive remains break-bearing but mixed. | [`210`](../workstream/210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md) |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Mainline dense-transformer fixed family set: `transcript_v1 / briefing_v1 / archive_v1` | Held across the full family set at machine scope under the unchanged Gate12A observable surface. | Family-conditioned. The first-pass surface does not simply reproduce the smaller Llama line or the Qwen2.5-3B line; archive remains break-bearing at high tension while flat becomes surface-noise-only in the memo read. | [`211`](../workstream/211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md) |
| `Qwen/Qwen3-4B` | 4B | Mainline dense-transformer fixed family set: `transcript_v1 / briefing_v1 / archive_v1` | Held across the full family set at machine scope under the unchanged Gate12A observable surface. | Family-conditioned. The vendor-upgrade probe does not simply inherit the Qwen2.5-3B first-pass surface; transcript shifts surface-noise-heavy, briefing keeps the Qwen2.5 flat split while high tension shifts, and archive hardens to a break-candidate packet. | [`212`](../workstream/212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md) |
| `Qwen/Qwen2.5-0.5B` | 0.5B | Tracked post-checkpoint lower-bound dense-transformer fixed family set: `transcript_v1 / briefing_v1 / archive_v1` | Held across the full family set at machine scope under the same frozen Gate12A observable surface. | Packet-local / memo-local only. The memo records surface-noise-only transcript high tension, mixed briefing packets, and conflict-adopted archive high tension; it explicitly avoids scale-monotonicity or release-bound claims. | [`215`](../workstream/215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md) |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Tracked post-checkpoint dense-transformer fixed family set: `transcript_v1 / briefing_v1 / archive_v1` | Held across the full family set at machine scope under the unchanged Gate12A observable surface. | Family-conditioned. Transcript and archive matched the then-current Qwen-like first-pass splits, while briefing remained the family-conditioned instability point. | [`206`](../workstream/206_GATE12A_LLAMA_3_2_1B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md) |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Tracked post-checkpoint follow-up: imported GPU `transcript_v1` only | Held on the imported transcript-only surface at machine scope. No family-set completion is claimed. | Transcript flat-band alignment remained, but high-tension behavior shifted toward `hypothetical_rejection`; the memo treats this as scale-sensitive follow-up evidence, not a wider Qwen-family law. | [`207`](../workstream/207_GATE12A_QWEN_2_5_1_5B_INSTRUCT_TRANSCRIPT_V1_GPU_IMPORT_REPLICATION_MEMO.md) |
| Non-mainline / admission boundary: `google/gemma-4-e2b-it`, `RWKV/v6-Finch-1B6-HF`, `state-spaces/mamba-2.8b-hf` | Boundary / sidecar | Admission-boundary row: Gemma 4 E2B excluded by frozen-protocol failure; RWKV excluded by remote-code/dependency pollution; Mamba 2.8B transcript completed sidecar-only under `prefix_mean_hidden_v1` | Not part of the dense-transformer mainline. Mamba's terminal Gate12A shape is not treated as dense-transformer structural portability because the sidecar diverged upstream through Gate9J/Gate9K. | Sidecar-only / excluded. The memo records an admission boundary, not a first-pass phenotype comparison against the dense-transformer mainline. | [`214`](../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md) |

## Mainline Dense-Transformer Closures

The current 3B/4B mainline rows are:

- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

The tracked memos record full `transcript_v1 / briefing_v1 / archive_v1`
family-set closures at machine scope for all three. Across those rows, the
structural replay status is stronger than the first-pass phenotype portability:
the machine-side Gate12A structural checks remain live, while the packet-local
first-pass reads remain family-conditioned and model-specific.

## Post-Checkpoint Lower-Bound / Follow-Up Closures

The post-checkpoint follow-up rows should not be promoted into a new release
boundary by this atlas.

- `Qwen/Qwen2.5-0.5B` is a tracked lower-bound dense-transformer family-set
  completion under the frozen Gate12A surface.
- `meta-llama/Llama-3.2-1B-Instruct` is a tracked full family-set completion
  with local provisional first-pass reads.
- `Qwen/Qwen2.5-1.5B-Instruct` is transcript-only imported GPU follow-up
  evidence, not a fixed family-set completion.

These rows help map where the frozen observable surface keeps functioning, but
they do not revise the published checkpoint boundary beyond `202`.

## Admission Boundaries / Sidecar-Only Surfaces

Memo `214` keeps the admission boundary explicit:

- `google/gemma-4-e2b-it` is excluded by frozen-protocol failure at the
  `transformers==4.57.3` configuration boundary.
- `RWKV/v6-Finch-1B6-HF` is excluded by remote-code admission and dependency
  pollution.
- `state-spaces/mamba-2.8b-hf` is sidecar-only under `prefix_mean_hidden_v1`;
  it is not a dense-transformer mainline replication and does not preserve the
  current dense-transformer transcript path signature through Gate9J/Gate9K.

The sidecar row is an admission-boundary map, not a claim about all
non-transformer systems.

## Reading Rule

Read this atlas conservatively:

- structural replay and first-pass phenotype portability are different columns
- packet-local first-pass reads are not family-wide phenotype laws
- sidecar-only surfaces are not dense-transformer mainline evidence
- tracked post-checkpoint follow-up memory is not a new Zenodo or release claim
