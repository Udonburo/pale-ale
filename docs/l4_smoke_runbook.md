# L4-smoke Remote Operator Runbook

This runbook documents the practical `l4-smoke` lane for
`tools/run_eval_checks.py`. It is operational guidance only. It does not widen
Gate12A, promote a new checkpoint, or treat local `runs/` material as tracked
public evidence. For the source-class map, use
[`eval_factory_artifact_reference.md`](eval_factory_artifact_reference.md).
For escalation beyond the smoke lane, use
[`l4_weekly_escalation_guide.md`](l4_weekly_escalation_guide.md).

## What This Lane Is

`l4-smoke` is one narrow real execution lane.

- Model is fixed to `Qwen/Qwen2.5-0.5B`.
- Families are fixed to `transcript_v1 / briefing_v1 / archive_v1`.
- The lane uses the committed Gate12A cross-model replay entrypoint through
  `tools/run_eval_checks.py`.
- The lane is not 1B, 1.5B, 3B, 4B, 7B FP32, sidecar, quantized,
  protocol-expanding, or Gate12B work.

Read the output as operator status for a fixed smoke lane. Do not read it as a
new public claim surface unless a separate tracked memo or release document
records that status.

## Posture Split

Use local Windows for dry-run and lightweight inspection:

- confirm the branch and docs/tools are present
- inspect the fixed target set and planned command shape
- verify that you are not about to paste placeholder syntax into PowerShell

Use the GCP L4 VM posture for real `--execute` runs:

- run from the VM checkout, not from the local Windows checkout
- use the VM interpreter, not local Windows Python
- verify CUDA from the same interpreter that will run the command
- choose an output directory on the VM filesystem

The local command prompt and the VM shell can both say `python`, but they are
different interpreters in different environments. Always print
`sys.executable` when there is any doubt.

## Exact Command Shapes

### Local Windows PowerShell Dry-run

From the repository root on Windows:

```powershell
git fetch origin
git status --short --branch
python .\tools\run_eval_checks.py --tier l4-smoke
```

This is the appropriate local Windows posture. It prints the fixed target set,
selected entrypoints, planned command, and `result: dry-run`. It does not run a
model.

Do not paste this into PowerShell:

```powershell
python .\tools\run_eval_checks.py --tier l4-smoke --execute --out-dir <PATH>
```

`<PATH>` is placeholder notation in prose. In PowerShell, angle brackets are
parsed as shell syntax. Use a quoted string or variable when you need a real
path, and reserve `--execute` for the VM posture:

```powershell
$outDir = "runs\l4-smoke-qwen05b-20260420T120000Z"
Write-Host "Chosen remote output root would be: $outDir"
```

### Remote Linux / GCP L4 VM Execute

From the repository root on the GCP L4 VM:

```bash
git fetch origin
git switch main
git pull --ff-only origin main

python3 -c "import sys; print(sys.executable)"
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

OUT_DIR="runs/l4-smoke-qwen05b-$(date -u +%Y%m%dT%H%M%SZ)"
python3 tools/run_eval_checks.py --tier l4-smoke --preflight-only --out-dir "$OUT_DIR"
python3 tools/run_eval_checks.py --tier l4-smoke --execute --out-dir "$OUT_DIR"
```

The `OUT_DIR` value is a concrete directory on the VM. Keep it under `runs/`
or another clearly local artifact root unless a separate archival procedure has
been chosen.

## Choosing `--out-dir`

Choose an output directory before running `--execute`.

- Use a fresh path for each run.
- Keep the path local to the machine that performs execution.
- Prefer a name that records the lane and UTC time, such as
  `runs/l4-smoke-qwen05b-20260420T120000Z`.
- Do not point `--out-dir` at a file.
- Do not commit generated contents from `runs/` as public evidence.

## Preflight / Pre-run Checks

Before a real VM execution:

- Confirm the VM checkout is on the intended branch, normally updated `main`:
  `git status --short --branch`.
- Confirm the code includes the expected l4-smoke execution lane from PR #79.
- Confirm the interpreter: `python3 -c "import sys; print(sys.executable)"`.
- Confirm the GPU is visible: `nvidia-smi`.
- Confirm CUDA through the same interpreter:
  `python3 -c "import torch; print(torch.cuda.is_available())"`.
- Confirm expected entrypoints are present:
  `tools/run_eval_checks.py`,
  `tools/run_gate12a_cross_model_replay.py`,
  `tools/run_gate8_scaleup.py`, and
  `tools/run_gate12a_family_replay.py`.
- Choose a fresh `--out-dir`.

If any of these checks fail, stop and fix the execution posture before running
`--execute`.

## Post-run Inspection

Inspect stdout first. The l4-smoke runner prints these sections:

- `tier`, `mode`, and `fixed target set`
- `actual entrypoints selected`
- `out-dir`
- `planned command`
- `out of scope`
- `environment diagnostics`
- `posture classification`
- `preflight result`
- `per-family dispatch/result summary`
- `notes`, when present
- `final pass/fail summary`

Then inspect artifacts in this order:

1. Preflight artifact:

```bash
cat "$OUT_DIR/eval_factory_l4_smoke_preflight.json"
```

2. Execute/status artifact, when downstream execution was attempted:

```bash
cat "$OUT_DIR/eval_factory_l4_smoke_status.json"
```

The preflight artifact records interpreter, OS, CUDA, GPU, `nvidia-smi`,
posture, errors, and remediation hints. The status artifact records the tier,
mode, model id, family set, entrypoint, executed command, output directory,
subprocess return code, embedded preflight, per-family results, and notes. Both
are operator artifacts for the run.

The cross-model summary, when produced, is expected under:

```text
$OUT_DIR/gate12a_cross_model_replay_qwen_qwen2_5_0_5b/cross_model_family_summary.csv
```

Keep these status layers separate:

- Tracked memo status is the public memo-facing surface recorded in tracked
  workstream documents and summarized by the Evidence Atlas.
- Runs-derived materialized status is what an existing local run directory and
  summary CSV say.
- The preflight artifact is an environment receipt, not a replay result.
- The l4-smoke status artifact is one run's operator record. It can support
  local inspection, but it does not rewrite tracked memo status by itself.

### Failure Reading

Precondition failure appears before meaningful preflight. Examples include
missing `--out-dir`, an output path that points to a file, or missing
entrypoints. Read the command line and stdout first.

Preflight failure means the environment was inspected and was not classified as
`remote_cuda_ready`. In `--execute`, downstream model execution is not invoked
when preflight fails. Inspect `eval_factory_l4_smoke_preflight.json`,
especially `posture_classification`, `preflight_ok`, `errors`, and
`remediation_hints`.

Downstream execution failure happens after preflight passes and the replay
subprocess is attempted. Inspect `eval_factory_l4_smoke_status.json`,
especially `returncode`, `family_results`, and `notes`.

## Troubleshooting

| Symptom | Likely cause | Operator response |
| --- | --- | --- |
| PowerShell reports a parse error around `<PATH>`. | Placeholder notation was pasted into PowerShell. | Do not use angle brackets as a real argument. Use a quoted path or variable such as `$outDir = "runs\l4-smoke-qwen05b-20260420T120000Z"`. Keep Windows local work to dry-run unless you are deliberately testing a VM-equivalent environment. |
| `--out-dir is required for --tier l4-smoke --execute`. | `--execute` was supplied without an output root. | Choose a fresh concrete output directory, then rerun on the VM with `--out-dir "$OUT_DIR"`. |
| `posture_classification` is `local_windows_no_cuda`, `python_missing_torch`, `cuda_unavailable`, or `unknown_posture`. | The preflight did not see the expected GCP L4 CUDA-ready posture. | Treat this as preflight failure. Read `errors` and `remediation_hints`; do not expect `eval_factory_l4_smoke_status.json` if downstream execution was blocked. |
| `--device cuda requested but CUDA is unavailable`. | The command is running on a non-GPU machine, the wrong interpreter, or a VM environment where CUDA/PyTorch is not visible. | Treat this as downstream execution failure if preflight passed, or as posture failure if preflight did not pass. Check `nvidia-smi`, `sys.executable`, and `torch.cuda.is_available()` on the VM. Do not reinterpret this as a Gate12A result. |
| The command is using local Windows Python. | The shell is local, or the SSH/VM session is not active. | Run the command from the GCP L4 VM shell and confirm `sys.executable` points to the VM environment. |
| `eval_factory_l4_smoke_preflight.json` is missing. | `--preflight-only` was run without `--out-dir`, execution stopped before artifact writing, or the wrong output directory is being inspected. | Check stdout and the exact `out-dir` line. Re-run preflight on the VM with `--preflight-only --out-dir "$OUT_DIR"` when an artifact receipt is needed. |
| `eval_factory_l4_smoke_status.json` is missing or malformed. | Preflight blocked downstream execution, the runner failed before writing the status artifact, the process was interrupted, or the wrong output directory is being inspected. | Check stdout, preflight result, `final pass/fail summary`, and the exact `out-dir` line. Rerun only after fixing the underlying posture or path issue. |
| A smoke run succeeds and someone treats it as a new checkpoint or Gate12B. | Runs-derived operator status is being confused with tracked memo status. | Keep the result as local materialized status unless a separate tracked memo or release process records it. The l4-smoke lane does not widen Gate12A or imply Gate12B. |

## Do Not Over-read This

- Successful smoke run != new checkpoint.
- Receipt/status artifact != tracked memo.
- Atlas != leaderboard.
- Sidecar/admission-boundary rows do not widen mainline.
- Local `runs/` materialization does not change the release surface by itself.

## Closeout Rule

After a run, summarize only what the operator surface actually says:

- command used
- VM/interpreter posture
- output directory
- preflight artifact path
- status artifact path
- final pass/fail summary
- any notes emitted by the runner

Do not add model-family, checkpoint, release, or Gate12B claims from the runbook
alone.
