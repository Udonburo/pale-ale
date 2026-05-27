# pale-ale Trace Triage

This directory contains a first-contact demo concept for pale-ale Trace Triage.
It is separate from RANVIER and is not a RANVIER redesign.

Trace Triage is for long LLM or agent evaluation traces where a reviewer has limited attention and needs to know which rows to inspect first. The demo uses a synthetic policy/RAG/evaluation trace to show how scalar-only evaluation can pass while hiding where a source constraint changed inside the trace.

Core message:

> Scalar-only evaluation says pass. pale-ale shows the 3 trace rows a human should inspect first.

pale-ale does not decide correctness. It tells a reviewer where to look first.

## Rendered demo

The static UI lives in [`../../../apps/trace-triage-demo/`](../../../apps/trace-triage-demo/README.md).

That app renders this directory's storyboard and synthetic trace fixture as a
one-glance external-review demo for evaluation, red-team, agent-monitoring, AI
safety, grant, and pilot-collaboration contexts.

## Files

- `storyboard.md`: seven-section first-contact storyboard for the demo.
- `synthetic-trace.json`: structured synthetic trace data for the storyboard.
- `README.md`: this orientation note.

## Relationship to RANVIER

RANVIER remains a document-grounded constraint audit sidecar.

pale-ale Trace Triage is a separate demo concept for human-review prioritization over long LLM, agent, RAG, or evaluation traces. Its output is a shortlist of rows for human review, not a benchmark result, model-quality score, or correctness classifier.

## Intended audience

The first screen is written for a busy evaluation, red-team, agent-monitoring, or AI safety researcher. In roughly 30 seconds, the viewer should understand:

- scalar-only evaluation can pass while hiding where a long trace broke;
- pale-ale does not score the model;
- pale-ale shortlists artifacts or trace rows for human review;
- this is review triage, not a benchmark or correctness classifier.

## Example scope

The trace is synthetic and illustrative. The same-review-budget comparison is labeled as an example, not a benchmark result. Claims should stay tied to the storyboard and the bounded evidence surfaced by the trace rows.
