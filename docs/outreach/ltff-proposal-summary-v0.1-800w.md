# Structural Telemetry for Agentic AI Artifact Chains

LTFF proposal summary
Version: v0.1

## Summary / Ask

I am seeking support for a 3–6 month independent research project on pale-ale, open-source structural telemetry for LLM and agent evaluation artifacts. The project frames pale-ale as pre-scalar oversight infrastructure for agentic AI artifact chains: prompts, plans, tool calls, retrieved context, intermediate files, evaluator notes, decisions, and final actions.

The practical output is not a model score. It is a reproducible shortlist of artifacts or trace rows for human-review prioritization before an evaluation run is reduced to scalar scores or pass/fail labels. Support would fund researcher time, compute/API usage, reproducibility infrastructure, and, where feasible, light annotation or advisory review.

## Problem: agentic systems and artifact-chain failures

Current evaluation pipelines often compress rich AI system behavior into final-output scores, pass/fail labels, preference judgments, or aggregate metrics. These views are useful, but they are lossy. For increasingly agentic systems, the safety-relevant failure may not be localized in the final answer. It may occur across a chain of actions, retrieved context, intermediate files, tool calls, observations, and decisions.

This creates a local plausibility / global non-coherence problem. Individual steps can appear reasonable while the overall artifact chain stops preserving a required relation, constraint, or source alignment. If inspected only after scalar compression, reviewers may know that a run failed without knowing which part of the trace deserves attention. Final-output scores may be too late and too lossy for scalable oversight.

## Approach: pre-scalar structural telemetry

pale-ale treats generated outputs and evaluation traces as artifact graphs. It reads already materialized records rather than changing model weights or rerunning inference as part of the audit layer. The goal is to preserve machine-checkable structural signals that remain visible before a trace is collapsed into a final label or score.

The current implementation focuses on directed relations between artifacts, relation preservation, explicit cycle non-closure, source anchors, candidate rows, and source-facing audit rows joined back to prompts, answers, and retrieved or referenced material. The core primitive is a structural candidate surface: a way to identify parts of a trace that may deserve closer human review. It supports oversight by making review targets reproducible, inspectable, and comparable across artifact formats.

## Current status and bounded evidence

I have released a public implementation and three bounded technical reports: Structural Replay in Dense Transformers Under a Frozen FP32 Regime; Transport-First Defect Telemetry on Replay Artifact Graphs; and Gate12B Observer-Relative Closure Signatures on Replay Artifact Graphs.

The current evidence is intentionally bounded. A recent Gate12B audit found a small but suggestive source-facing alignment pattern in one archive-family queue. I treat this as preliminary bounded artifact-level evidence, not as a general-purpose monitor or a claim about model internals. The line has moved toward executable validation work, but the central question is still empirical: whether the primitive remains useful beyond the current bounded artifact setting.

## 3–6 month validation plan

The next phase is validation and scaling, not theoretical expansion. Planned work:

- define a minimal agent-artifact graph schema covering prompts, retrieved sources, generated answers, tool calls, evaluator annotations, intermediate files, source anchors, and source-facing review rows;
- run structural telemetry over a larger and more diverse set of evaluation or red-team style traces with explicit manifests;
- test rendering / artifact-format ablations across archive, transcript, briefing, and related views;
- produce source-facing audit rows and, where feasible, run multi-annotator review checks;
- compare structural-telemetry candidate rows against simple final-output or heuristic triage baselines;
- extend the harness toward more agentic traces involving tool calls, retrieved context, observations, and decisions;
- release 1–2 short reports documenting methods, limitations, failure cases, and open questions.

## Why this may matter for long-term AI safety

Scalable oversight may require more than better labels on final outputs. As systems become more agentic, oversight may also depend on whether artifact chains preserve required relations, constraints, and source alignments across steps. Human reviewers cannot inspect every intermediate object in every long trace, but scalar summaries may erase the evidence needed to locate failures.

If pale-ale succeeds in this bounded form, it could contribute to evaluation and red-team triage, monitoring of multi-step agent workflows, reproducible safety-case evidence packages, comparison of artifact-level failure modes across protocols, and earlier identification of traces where local plausibility hides global non-coherence. The claim is modest: test whether structural telemetry provides useful human-review prioritization signal over materialized evaluation artifacts.

## Support requested

I am seeking a 3–6 month independent research grant to test whether pale-ale can generalize from bounded replay-artifact studies toward a useful pre-scalar oversight layer for LLM and agent evaluation traces. The main open question is no longer just whether the primitive can be formalized or implemented; it is whether it remains useful beyond the current bounded artifact setting. That is a validation and scaling problem, and that is what this grant would fund.
