# Structural Telemetry for Agentic AI Artifact Chains

LTFF EOI draft
Version: v0.3

I am seeking a 3–6 month independent research grant to develop pale-ale, an open-source structural telemetry layer for inspecting LLM and agent evaluation artifacts before they are collapsed into scalar scores or pass/fail labels.

The motivation is that increasingly agentic AI systems will produce rich chains of artifacts: prompts, plans, tool calls, retrieved context, intermediate files, evaluator notes, decisions, and final actions. Failures may not appear as single obviously bad outputs. Individual steps can look locally plausible while the overall artifact chain fails to preserve required relations, constraints, or source alignment. Final-output scores may be too late and too lossy for scalable oversight.

pale-ale treats generated outputs and evaluation traces as artifact graphs. It computes machine-checkable structural signals over directed relations, explicit cycles, source anchors, and candidate rows. The practical output is not a model score or correctness classifier, but a reproducible shortlist of artifacts or trace rows that may deserve human inspection.

I have released a public implementation and three bounded technical reports. A recent Gate12B audit found a small but suggestive source-facing alignment pattern in one archive-family queue. I treat this as preliminary artifact-level evidence, not as a detector, benchmark, or mechanistic claim.

The current line has reached the point where the main bottleneck is validation, not ideation. The next phase would test whether this primitive generalizes beyond the current bounded protocol: larger and more diverse evaluation traces, rendering and artifact-format ablations, multi-annotator review, comparison against simple final-output or heuristic triage baselines, and extension toward agent-trace workflows.

This line cannot move much faster without dedicated time and compute. The main open question is no longer just whether the primitive can be formalized or implemented; it is whether it remains useful beyond the current bounded artifact setting. That is a validation and scaling problem, and that is what this grant would fund.
