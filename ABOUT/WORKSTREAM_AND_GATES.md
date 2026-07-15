# Workstream And Gates

This repo uses two pieces of project vocabulary that are unusual outside the repo itself: **Gate** and **Workstream**.

## What A Gate Is

A Gate is a named research checkpoint.

Each Gate exists to answer some bounded question honestly and to freeze what that answer does and does not earn. A Gate is not just "the next experiment." It is a checkpoint where the repo decides:

- what is now established
- what remains open
- what is denied
- what would count as a falsifier for the current line

That is why Gate files often use names like:

- constitution
- contract
- audit
- memo
- smoke
- closeout
- freeze

Those names are deliberate. They indicate the role the file plays in the line.

## What Workstream Means

The `workstream/` directory is the numbered tracked research memory for the repo.

Each file is a memory unit in reading order. The number is part of the meaning. Later files extend the line, but they do not silently erase what earlier checkpoints did or did not earn.

In practice, that means:

- the numbering should be preserved
- checkpoint boundaries matter
- "latest" does not always mean "replace everything before it"
- public reading order should follow the established checkpoint line, not random file timestamps

## How To Read It Without Getting Lost

If you are new, do not start by reading every file in numeric order.

Use this route:

1. Read [`../README.md`](../README.md) and [`README.md`](README.md).
2. Read the current summary and closeout surfaces:
   - [`../workstream/25_GATE8_MAINLINE_SUMMARY.md`](../workstream/25_GATE8_MAINLINE_SUMMARY.md)
   - [`../workstream/26_GATE9_GRAPH_GAUGE_CONSTITUTION.md`](../workstream/26_GATE9_GRAPH_GAUGE_CONSTITUTION.md)
   - [`../workstream/62_GATE9_CLOSEOUT.md`](../workstream/62_GATE9_CLOSEOUT.md)
   - [`../workstream/76_GATE10_CLOSEOUT.md`](../workstream/76_GATE10_CLOSEOUT.md)
   - [`../workstream/77_GATE6_TO_GATE10_MAINLINE_OVERVIEW.md`](../workstream/77_GATE6_TO_GATE10_MAINLINE_OVERVIEW.md)
3. Read the Gate11 membrane and the Gate12/Gate12A line:
   - [`../workstream/195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`](../workstream/195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md)
   - [`../workstream/196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`](../workstream/196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md)
   - [`../workstream/197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`](../workstream/197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md)
   - [`../workstream/198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`](../workstream/198_GATE12A_DISCRETE_CONNECTION_AUDIT.md)
4. Then read the current report-facing memos:
   - [`../workstream/199_GATE12A_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`](../workstream/199_GATE12A_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md)
   - [`../workstream/200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md`](../workstream/200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md)
   - [`../workstream/201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`](../workstream/201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md)
   - [`../workstream/202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md`](../workstream/202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md)
   - [`../workstream/203_GATE12A_ARCHIVE_V1_ANCHOR_RICH_CLOSURE_TENSION_FIRST_PASS_BREAK_CANDIDATE_MEMO.md`](../workstream/203_GATE12A_ARCHIVE_V1_ANCHOR_RICH_CLOSURE_TENSION_FIRST_PASS_BREAK_CANDIDATE_MEMO.md)
   - [`../workstream/210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
   - [`../workstream/211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
   - [`../workstream/212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md`](../workstream/212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md)
   - [`../workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md`](../workstream/213_GATE12A_SINGLE_GPU_FP32_DENSE_TRANSFORMER_TECHNICAL_REPORT_DRAFT.md)
   - [`../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md`](../workstream/214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md)

## The Current Gate Map

This is the shortest plain-language map to the current line:

| Gate | Role in the current public line |
| --- | --- |
| Gate8 | earlier mainline summary and bridge point |
| Gate9 | graph-gauge constitution and blocker/object discipline |
| Gate10 | trusted-tree settlement court and bounded broader-pattern judgment |
| Gate11 | admissibility membrane; boundary on what can be reopened |
| Gate12 | discrete-connection constitution |
| Gate12A | implementation, replay audit surface, family replication line, and frozen base paper-facing evidence |
| Gate12B | read-only observer-relative secondary audit over Gate12A artifacts |
| Gate12C-1 | predeclared compression-interleaved parenthesization null test |

## Why Workstream Stays Visible

The point of keeping Workstream explicit is not aesthetics. It is control.

The repo uses Workstream to keep:

- reading order visible
- checkpoint claims recoverable
- later expansions from pretending to have been present earlier
- paper-facing summaries tied back to the tracked memory that earned them

That is why `workstream/` remains central even when a cleaner `ABOUT/` surface is added on top.
