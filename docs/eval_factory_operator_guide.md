# Eval Factory Operator Guide

This guide is for operating `tools/run_eval_checks.py` from the repository
root. It documents the standing eval-factory tiers and the status-source
discipline around them. It does not add execution, widen Gate12A, or promote
local `runs/` material into tracked public evidence.

## Status Sources

The eval factory uses two different status sources. Keep them separate when
reading output or writing follow-up notes.

| Source | Meaning | Operator use |
| --- | --- | --- |
| Tracked memo status | Status grounded in tracked repository documents, especially the Gate12A memo line and the Evidence Atlas. This is the public memo-facing surface. | Use for public-facing Gate12A status, with the same frozen-regime limits and memo-local wording already present in the tracked docs. |
| Runs-derived materialized status | Status parsed from already-materialized local outputs, such as `runs/gate12a_cross_model_replay_*/cross_model_family_summary.csv` and nearby manifests when present. | Use only as a read-only local artifact rollup. It can report what is materialized in this checkout, but it does not override or extend tracked memo status. |

`runs/` remains local/generated working output. Its summaries can help an
operator see what has been materialized, but they are not the tracked public
evidence source.

## Current Tier Posture

| Tier | Current posture | What the entrypoint does now | What it does not do |
| --- | --- | --- | --- |
| `cpu-nightly` | Lightweight validation. | Checks expected repo files, tracked memo presence, tier shape, L4 weekly exclusions, and shallow existing summary/manifests when `runs/` is present. | Does not invoke GPU/model execution or refresh Gate12A evidence. |
| `summarize-existing` | Read-only rollup. | Reports memo-facing surfaces, tracked memo model surfaces, and runs-derived materialized cross-model summaries from existing local files. | Does not generate new runs or turn `runs/` status into public memo status. |
| `l4-smoke` | Narrow execution lane for the 0.5B boundary set. | Prints a dry-run by default; with `--execute` and an explicit `--out-dir`, runs the fixed 0.5B lane. | Does not expand beyond the fixed 0.5B family set, and does not promote a new checkpoint. |
| `l4-weekly` | Bounded mainline standing lane. | Prints the standing plan by default; when bounded execution support is present, preflights or executes one current 3B/4B target at a time. | Does not include 7B FP32, sidecar candidates, quantized candidates, protocol-expanding candidates, a new checkpoint, memo promotion, or Gate12B promotion. |

The L4 tiers describe operational lanes. They are not claim surfaces by
themselves. Treat their output as planning or local status text unless a
tracked memo or release document separately records a result.

## L4-smoke Runbook

Use [`l4_smoke_runbook.md`](l4_smoke_runbook.md) before operating the
`l4-smoke` lane. Local Windows use is for dry-run and lightweight inspection;
real `--execute` posture is expected on the GCP L4 VM. Do not confuse local
Windows Python with the VM interpreter.

## L4-weekly Escalation

Use [`l4_weekly_escalation_guide.md`](l4_weekly_escalation_guide.md) before
moving from the smoke lane into weekly planning. The weekly lane is bounded to
the current 3B/4B dense-transformer mainline and remains plan-only by default.
Use [`l4_weekly_execute_runbook.md`](l4_weekly_execute_runbook.md) for bounded
one-target-at-a-time weekly preflight or execution once support is present.

## Artifact Reference

Use [`eval_factory_artifact_reference.md`](eval_factory_artifact_reference.md)
when interpreting eval-factory outputs. The table below maps each tier to its
primary output class and interpretation boundary.

| Tier | Primary artifact/output | Source class | Interpretation boundary |
| --- | --- | --- | --- |
| `cpu-nightly` | stdout validation report | Checkout validation status | Validates expected files and shallow local shapes; does not create empirical evidence. |
| `summarize-existing` | stdout rollup of tracked docs and local summaries | Tracked memo status plus runs-derived materialized status | Separates memo-facing status from local `runs/` materialization; does not let `runs/` override tracked memos. |
| `l4-smoke` dry-run | stdout dry-run plan | Operational plan | Shows fixed target set and command shape; does not execute a model. |
| `l4-smoke --preflight-only` | stdout diagnostics and optional `eval_factory_l4_smoke_preflight.json` | Eval-factory preflight artifact | Checks interpreter/GPU posture; does not run the replay or update memo status. |
| `l4-smoke --execute` | stdout, `eval_factory_l4_smoke_preflight.json`, and possibly `eval_factory_l4_smoke_status.json` | Eval-factory preflight artifact plus execute/status artifact | Records one operator execution attempt for the fixed 0.5B lane; does not create a new checkpoint by itself. |
| `l4-weekly` | stdout plan and optional `eval_factory_l4_weekly_plan.json` | Eval-factory weekly plan artifact | Compiles the bounded current 3B/4B mainline plan; does not execute a model or imply 7B FP32, sidecar, protocol expansion, quantized candidates, or Gate12B. |
| `l4-weekly --preflight-only --target ...` | stdout diagnostics and a weekly preflight artifact when supported | Eval-factory weekly preflight artifact | Checks one target's VM/interpreter/GPU posture; does not run replay or update memo status. |
| `l4-weekly --execute --target ...` | stdout, weekly preflight artifact, weekly execute/status artifact, and downstream materialized outputs when supported | Eval-factory weekly execute/status artifact plus runs-derived materialized status | Records one bounded target execution attempt; does not create a checkpoint, memo claim, phenotype readout, or Gate12B signal by itself. |

## Example Commands

Run commands from the repository root.

```powershell
python tools/run_eval_checks.py --tier summarize-existing
python tools/run_eval_checks.py --tier cpu-nightly
python tools/run_eval_checks.py --tier l4-smoke
python tools/run_eval_checks.py --tier l4-weekly
```

## Reading Output

For `summarize-existing`, read these sections separately:

- `memo-facing surfaces`: whether expected tracked docs and release-adjacent
  files are present.
- `tracked memo model surfaces`: model/memo mappings grounded in tracked
  workstream documents.
- `runs-derived materialized cross-model summaries`: local summary CSV and
  manifest status from existing `runs/` directories.

For `cpu-nightly`, `PASS`, `WARN`, and `FAIL` are validation statuses for the
checkout. A passing check is not a new empirical result. A warning about a
missing local summary does not remove a tracked memo result.

For `l4-smoke`, dry-run output is a plan, while `--execute` output is
runs-derived operator status for the fixed smoke lane. For `l4-weekly`, default
output is a plan. Bounded weekly preflight and execute artifacts, when present,
are operator status for one target at a time. Do not read plan text as proof
that a run has been executed.

## Interpretation discipline

- Do not confuse tracked memo status with runs CSV status.
- Do not treat the Evidence Atlas as a leaderboard.
- Do not widen the dense-transformer mainline with sidecar or
  admission-boundary rows.
- Do not infer Gate12B from eval-factory tiers.
- Do not read weekly structural pass status as a phenotype readout unless a
  separate phenotype read was performed and recorded.
- Preserve packet-local, memo-local, and family-conditioned wording where the
  tracked memo line uses it.

## Operator Checklist

Before citing or summarizing eval-factory output:

- Identify whether the line came from tracked memo status or runs-derived
  materialized status.
- Use the Evidence Atlas for the public map of the frozen Gate12A observable
  surface.
- Use `summarize-existing` to inspect local materialized summaries without
  running new jobs.
- Use `cpu-nightly` to catch lightweight repository or manifest-shape issues.
- Treat L4 tier output as lane posture unless a separate tracked memo records
  completed evidence.
