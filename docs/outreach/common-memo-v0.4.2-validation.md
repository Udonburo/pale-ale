# Structural Telemetry for LLM and Agent Artifact Graphs

A one-page research memo
Version: v0.4.2 validation patch

Aoi Kawasaki
Independent researcher / research engineer, Japan
GitHub: https://github.com/Udonburo/pale-ale

## Summary

I am building pale-ale, a structural telemetry layer for inspecting LLM and agent evaluation artifacts before they are reduced to scalar scores or pass/fail labels.

The near-term research question is whether structural telemetry can help evaluation, red-team, and agent-monitoring researchers identify which artifacts or trace rows deserve human review, beyond final-output scores or labels alone.

The practical output is not a model score, but a shortlist of artifacts or trace rows that may deserve closer human inspection.

Here, “structural telemetry” means machine-checkable signals over evaluation artifacts—directed relations, cycles, source anchors, and candidate rows—kept visible before an evaluation pipeline collapses them into aggregate metrics.

pale-ale is not a new benchmark, not a correctness classifier, and not a mechanistic claim about model weights. It is a reproducible artifact-level audit layer for preserving structural signals in evaluation traces.

## Problem

Most LLM evaluation pipelines eventually compress model behavior into final-output scores, pass/fail labels, preference judgments, or benchmark aggregates. These views are useful, but they can erase structure that is present in the materialized artifacts themselves.

In retrieval-heavy, document-grounded, or agentic workflows, the failure of interest is often not only whether the final answer was correct. It is also where the artifact chain stopped preserving a required relation, constraint, or source alignment.

For example, a generated answer may sound plausible while following a conflicting or unsupported source anchor. A replay graph may contain cycle-level non-closure that is lost once the workflow is collapsed into a scalar score.

## Approach

pale-ale treats generated outputs and evaluation traces as artifact graphs. It reads already materialized records rather than changing model weights or rerunning inference as part of the audit layer.

The implementation is Rust-first and public. It focuses on:

- materialized artifact records and directed relations;
- explicit cycle / non-closure structure;
- relation-patterned closure signatures;
- source-facing audit rows joined back to prompts, answers, and source anchors.

It is intended to support human judgment by making review targets reproducible and inspectable.

## Evidence so far

I have released three public technical reports and a corresponding public codebase:

- Structural Replay: controlled replay under a frozen FP32 dense-transformer protocol.
- Transport-First Defect Telemetry: artifact-level closure inconsistency over directed relations, explicit cycles, and bounded closure-defect checks.
- Gate12B Observer-Relative Closure Signatures: a read-only secondary audit over existing LLM replay artifacts.

In one bounded archive source-facing queue from Gate12B, selected high-side candidates aligned with conflict-following rows in 8/8 cases, while selected flat-side candidates aligned with support-following or non-gluing rows in 8/8 cases.

I report this as a bounded artifact result, not as a universal interpretability law, correctness classifier, model-quality benchmark, or weight-level causal claim.

## Why this may matter

As LLM systems become more agentic, evaluation artifacts will become richer: plans, tool calls, retrieved context, intermediate files, evaluator notes, observations, decisions, and final actions.

pale-ale is aimed at this regime. Potential uses include:

- identifying artifacts or trace rows for human review;
- evaluation-log and red-team triage;
- monitoring multi-step agent traces;
- producing reproducible evidence packages for safety cases;
- auditing document-grounded workflows where local plausibility may hide non-local inconsistency.

The core bet is that future evaluations will need telemetry over artifact chains, not only final answers.

## Next 3–6 months

I want to extend pale-ale from bounded replay-artifact studies toward a more general structural telemetry harness for LLM and agent evaluation traces.

The next phase is designed to test scale, rendering sensitivity, annotator agreement, and triage utility rather than to claim a general detector.

Planned deliverables:

- a minimal agent-artifact graph schema for retrieval, tool-call, answer, evaluator, and intermediate-file artifacts;
- an expanded reproducible evidence package over evaluation or red-team style traces;
- rendering / artifact-format ablations, including archive / transcript / briefing-style views;
- source-facing review checks, including multi-annotator review where feasible;
- comparison against simple final-output or heuristic triage baselines;
- 1–2 short reports documenting method, limits, and failure cases.

## Ask

I am looking for:

- technical feedback from evaluation, red-team, or agent-monitoring researchers;
- access to suitable evaluation or agent-trace artifacts under an appropriate agreement;
- compute/API support for controlled runs;
- a 3–6 month independent research grant;
- contractor-style research engineering collaboration with an evaluation, red-team, or agent-monitoring group.

## Links

GitHub: https://github.com/Udonburo/pale-ale
Gate12B report: https://doi.org/10.5281/zenodo.20080003
Structural Replay report: https://doi.org/10.5281/zenodo.19483162
Transport-First Defect Telemetry report: https://doi.org/10.5281/zenodo.19569052
