# L4-weekly Escalation Guide

This guide explains how to move from local inspection and the fixed
`l4-smoke` lane into `l4-weekly` planning. It is operator guidance only. It
does not add a checkpoint, widen the dense-transformer mainline, or revise the
Gate12A memo surface.

## What This Lane Is

`l4-weekly` is a bounded weekly planning lane for the current 3B/4B
dense-transformer mainline under the frozen Gate12A observable surface.

It is limited to:

- current 3B/4B dense-transformer mainline targets
- `transcript_v1 / briefing_v1 / archive_v1` expected family coverage
- plan compilation and operator scheduling posture

It is not:

- 7B FP32
- sidecar
- protocol-expanding
- quantized
- Gate12B
- a new checkpoint or release surface

Current `tools/run_eval_checks.py --tier l4-weekly` output is `plan-only`.
With `--out-dir`, it can write `eval_factory_l4_weekly_plan.json`. That file
is a planning artifact, not an execution artifact.

## Escalation Discipline

Move through the lanes as operator posture, not as claim escalation.

| Step | Lane | Use it for | Interpretation boundary |
| --- | --- | --- | --- |
| 1 | Local dry-run | Inspect command shape, fixed targets, and docs/tools from a local checkout. | No model execution and no empirical status change. |
| 2 | Remote `l4-smoke` | Run or preflight the fixed `Qwen/Qwen2.5-0.5B` smoke lane on the GCP L4 VM posture. | Operator/runs-derived status only unless later recorded by a tracked memo. |
| 3 | `l4-weekly` planning | Compile the bounded weekly plan for current 3B/4B dense-transformer mainline targets. | Weekly planning is not weekly execution and does not imply a new checkpoint. |

Do not treat success at one step as a scientific promotion into the next step.
The move from smoke to weekly is an operator scheduling move, not a doctrinal
move.

## Command Shapes

Plan only, no artifact:

```powershell
python tools/run_eval_checks.py --tier l4-weekly
```

Plan only, with a local planning artifact:

```powershell
$outDir = "runs\l4-weekly-plan-20260420T120000Z"
python tools\run_eval_checks.py --tier l4-weekly --out-dir $outDir
```

On a Linux VM or shell:

```bash
OUT_DIR="runs/l4-weekly-plan-$(date -u +%Y%m%dT%H%M%SZ)"
python3 tools/run_eval_checks.py --tier l4-weekly --out-dir "$OUT_DIR"
```

The expected artifact path is:

```text
$OUT_DIR/eval_factory_l4_weekly_plan.json
```

This artifact has `mode: plan-only` and `result: plan-only`. It does not run a
subprocess, GPU job, or model replay.

## Weekly Target Matrix

| Model | Expected family coverage |
| --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | `transcript_v1 / briefing_v1 / archive_v1` |
| `meta-llama/Llama-3.2-3B-Instruct` | `transcript_v1 / briefing_v1 / archive_v1` |
| `Qwen/Qwen3-4B` | `transcript_v1 / briefing_v1 / archive_v1` |

This matrix is a planning target list. It is not a ranking and does not add a
fresh structural or first-pass read.

## Reading The Weekly Plan

Read stdout first:

- `tier`
- `mode`
- `intent`
- `expected resource posture`
- `weekly target matrix`
- `per-model planned families`
- `planned entrypoints`
- `exclusions`
- `artifact`
- `execution`
- `final result`

If `--out-dir` was provided, inspect:

```text
$OUT_DIR/eval_factory_l4_weekly_plan.json
```

Typical fields to inspect:

- `schema_id`
- `tier`
- `mode`
- `resource_posture`
- `weekly_target_matrix`
- `planned_entrypoints`
- `exclusions`
- `result`

Keep `weekly_target_matrix` and `exclusions` together. The target list is only
valid with the exclusions still in force.

## Do Not Over-read This

- Weekly planning != weekly execution.
- Weekly execution != new checkpoint.
- Mainline exclusions remain in force.
- Evidence Atlas is not a leaderboard.
- `eval_factory_l4_weekly_plan.json` is a planning artifact, not a tracked
  memo.
- Sidecar or admission-boundary rows do not widen the dense-transformer
  mainline.
- `l4-weekly` does not imply 7B FP32, quantized candidates, protocol expansion,
  or Gate12B.

## Closeout Rule

When reporting a weekly planning pass, include only:

- command used
- branch or commit posture
- whether an artifact was written
- artifact path, if present
- target matrix
- exclusions
- `result: plan-only`

Do not summarize the weekly plan as an executed empirical result.
