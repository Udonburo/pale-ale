# Workstream

This directory is the numbered research history. For current development,
start with the relevant implementation and study protocol. For
published findings, use the [publication catalog](../publications/README.md).
Local studies are under ignored `local/`; use its `README.md` when present.

| Files | Historical subject |
| --- | --- |
| 06–77 | Gate6–Gate10 experiments and summaries |
| 78–195 | Closed Gate11 operator-admissibility work |
| 196–216 | Gate12A replay, replication, and report |
| 217–230, 239 | Gate12B observer-relative analysis and manuscript |
| 231–238 | Gate12C feasibility and Gate12C-1 experiment |

The numbers preserve historical reading order. Read the notes relevant to the
question at hand; the full sequence is not a prerequisite for new work.
Old “not authorized,” “freeze,” “fresh review,” and “next required state” text
records the circumstances of those tasks, not present-day working instructions.

The 58 Gate11 audit generators and their tests have been removed from `tools/`.
They recursively inspected prior operational status rather than providing an
active experiment. Their source remains recoverable from Git history, and
the numbered notes retain the historical findings. Do not regenerate that chain
as a prerequisite for current research.

For the retired tools, use the complete
[pre-retirement checkout](https://github.com/Udonburo/pale-ale/tree/f16bcdd/tools),
including its original dependencies, rather than copying an isolated generator
into the current tree.

Later completed studies are summarized separately in the
[CRD technical report](../analysis/crd/TECHNICAL_REPORT.md); they do not extend
the Gate11 operational chain.

`*.local.md` and `workstream/local/` are ignored by Git and protected by the
repo-local pre-push hook against accidental publication.
