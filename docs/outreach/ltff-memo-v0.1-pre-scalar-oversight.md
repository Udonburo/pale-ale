# Structural Telemetry for Agentic AI Artifact Chains

A pre-scalar oversight proposal for LTFF
Version: v0.1

Aoi Kawasaki
Independent researcher / research engineer, Japan
GitHub: https://github.com/Udonburo/pale-ale

## Summary

I am seeking support for a 3–6 month independent research project on pale-ale, an open-source structural telemetry layer for inspecting LLM and agent evaluation artifacts before they are collapsed into scalar scores or pass/fail labels.

The motivation is that increasingly agentic AI systems will produce rich chains of artifacts: prompts, plans, tool calls, retrieved context, intermediate files, evaluator notes, decisions, and final actions. Failures may not appear as single obviously bad outputs. Individual steps can look locally plausible while the overall artifact chain fails to preserve required relations, constraints, or source alignment.

Final-output scores may be too late and too lossy for scalable oversight.

pale-ale investigates structural telemetry over these artifact chains as a pre-scalar oversight layer: a way to identify where a chain may stop preserving required relations before the result is summarized as pass/fail or a scalar score.

The practical output is not a model score, not a benchmark, and not a correctness classifier. It is a reproducible shortlist of artifacts or trace rows that may deserve closer human inspection.

## Problem

Current evaluation pipelines often compress rich AI system behavior into final-output scores, pass/fail labels, preference judgments, or benchmark aggregates. These views are useful, but they are lossy.

For future agentic systems, the safety-relevant failure may not be localized in the final answer. It may occur across a chain of actions, retrieved sources, intermediate files, tool calls, observations, and decisions. A system may appear locally plausible at each step while globally failing to preserve a required constraint, source alignment, or relation across the trace.

This creates a scalable oversight problem.

If the artifact chain is inspected only after scalar compression, the evaluation pipeline may preserve the final judgment while losing the structural evidence needed to understand where the chain broke. A monitor or evaluator may know that a run failed without knowing which part of the trace deserves human review.

pale-ale starts from a narrower, testable question:

Can structural telemetry over materialized evaluation artifacts help identify where locally plausible traces stop remaining globally coherent before they are collapsed into scalar outcomes?

## Approach

pale-ale treats generated outputs and evaluation traces as artifact graphs. It reads already materialized records rather than changing model weights or rerunning inference as part of the audit layer.

The current implementation focuses on machine-checkable structural signals over:

- materialized artifact records;
- directed relations between artifacts;
- explicit cycle and non-closure structure;
- source anchors and candidate rows;
- source-facing audit rows joined back to prompts, answers, and retrieved or referenced material.

The core primitive is a structural candidate surface. It is not a model score. It is not intended to replace human judgment. It is intended to help decide where human judgment should be spent.

In practical terms, pale-ale asks whether an evaluation trace contains structural signals that remain visible before the trace is reduced to a scalar metric or pass/fail label.

## Current status

I have released a public implementation and three bounded technical reports:

1. Structural Replay in Dense Transformers Under a Frozen FP32 Regime

A controlled replay-analysis report under a frozen dense-transformer protocol.

2. Transport-First Defect Telemetry on Replay Artifact Graphs

A minimal artifact-graph formulation of closure inconsistency over directed relations and explicit cycles.

3. Gate12B Observer-Relative Closure Signatures on Replay Artifact Graphs

A read-only secondary audit over existing LLM replay artifacts.

The current evidence is intentionally bounded. A recent Gate12B audit found a small but suggestive source-facing alignment pattern in one archive-family queue. I treat this as preliminary artifact-level evidence, not as a detector, benchmark, universal interpretability law, or mechanistic claim about model weights.

The line has also reached a practical execution stage. A current smoke validation lane exists for fixed rendering-sensitivity checks across transcript, briefing, and archive-style artifact views. I do not treat this as a new public claim surface. I treat it as evidence that the project has moved from pure ideation toward executable validation work.

## Why this may matter for long-term AI safety

A technical part of AI safety is not exhausted by teaching models preferred behavior at the output level. For increasingly agentic systems, safety may also depend on whether the system’s artifact chain preserves required relations, constraints, and source alignments across steps.

This is especially relevant for scalable oversight. Human reviewers cannot inspect every intermediate step of every long agent trace. But final-output scores alone may be too compressed to show where a failure occurred. A useful oversight layer should help identify which parts of a trace deserve inspection before the full chain is collapsed into a final judgment.

pale-ale is aimed at this intermediate object: the evaluation artifact chain.

If successful, it could contribute to:

- triage of evaluation and red-team traces;
- monitoring of multi-step agent workflows;
- reproducible safety-case evidence packages;
- comparison of artifact-level failure modes across evaluation protocols;
- earlier identification of traces where local plausibility may hide global non-coherence.

The claim is intentionally modest. I am not claiming that pale-ale detects deception, proves semantic inconsistency, or reveals model-internal mechanisms. The proposal is to test whether structural telemetry can provide useful human-review prioritization signal over materialized evaluation artifacts.

## Research plan

The current line has reached the point where the main bottleneck is validation, not ideation.

The next phase would test whether this primitive generalizes beyond the current bounded protocol. The goal is to test scale, rendering sensitivity, annotator agreement, and triage utility rather than to claim a general detector.

Over 3–6 months, I would use grant support to move pale-ale from bounded replay-artifact studies toward a more general pre-scalar telemetry harness for LLM and agent evaluation traces.

Planned work:

1. Agent-artifact graph schema

Define a minimal schema for materialized evaluation artifacts, including prompts, retrieved sources, generated answers, tool calls, evaluator annotations, intermediate files, source anchors, and source-facing review rows.

2. Expanded evidence package

Run structural telemetry over a larger and more diverse set of evaluation or red-team style traces, using reproducible artifacts and explicit manifests rather than informal examples.

3. Rendering / artifact-format ablations

Test sensitivity across archive / transcript / briefing-style views and other artifact-format choices where feasible. The goal is to determine whether candidate surfaces are stable, rendering-sensitive, or artifact-format dependent.

4. Source-facing review checks

Produce human-reviewable audit rows and, where feasible, run multi-annotator checks to estimate whether candidate rows are interpretable and review-worthy beyond my own labeling.

5. Baseline comparison

Compare structural-telemetry candidate rows against simple final-output or heuristic triage baselines, to test whether the method provides useful additional prioritization signal.

6. Agent-trace extension

Extend the artifact graph schema and telemetry harness toward more agentic traces involving tool calls, retrieved context, intermediate files, observations, and decisions.

7. Reporting

Release 1–2 short technical reports documenting the method, evidence, limitations, failure cases, and remaining open questions.

## Expected outputs

By the end of the project, I expect to produce:

- a public agent-artifact graph schema;
- open-source tooling for structural telemetry over evaluation artifacts;
- an expanded reproducible evidence package with manifests;
- rendering / artifact-format ablations;
- source-facing audit rows suitable for human review;
- preliminary multi-annotator or review-agreement results where feasible;
- baseline comparisons against simple heuristic triage;
- 1–2 public technical reports documenting method, limits, and failure cases.

## Longer-term research direction

Longer term, pale-ale is motivated by the hypothesis that scalable oversight may require structural invariants over artifact chains, not only better labels on final outputs.

The present project does not claim to solve this theoretical problem. It tests a narrow precursor: whether relation preservation, cycle non-closure, rendering sensitivity, and source-facing review rows provide useful telemetry for identifying where agentic traces stop remaining globally coherent.

If this succeeds, later work could study observer-relative and scale-sensitive invariants over learned-system artifacts more formally.

## Why support is needed

This line cannot move much faster without dedicated time and compute.

The main open question is not whether the primitive is coherent—the formalization and implementation are in place—but whether it generalizes beyond the current bounded artifact setting. That is a validation and scaling problem, and that is what this grant would fund.

Support would be used for:

- researcher time;
- compute and API usage;
- storage and reproducibility infrastructure;
- possible annotation or review support;
- preparation of public reports and artifacts.

## Ask

I am seeking a 3–6 month independent research grant to test whether pale-ale’s structural telemetry primitive can generalize from bounded replay-artifact studies toward a useful pre-scalar oversight layer for LLM and agent evaluation traces.

I would especially value support for compute/API usage, researcher time, and light advisory feedback from evaluation or AI safety researchers.

## Links

GitHub: https://github.com/Udonburo/pale-ale
Gate12B report: https://doi.org/10.5281/zenodo.20080003
Structural Replay report: https://doi.org/10.5281/zenodo.19483162
Transport-First Defect Telemetry report: https://doi.org/10.5281/zenodo.19569052
