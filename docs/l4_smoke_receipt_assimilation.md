# L4-smoke Receipt Assimilation

This guide explains how to assimilate a successful VM `l4-smoke --execute`
receipt into an operator status note. It is not a workstream memo update, a
checkpoint declaration, a release note, or a Gate12B signal.

Use this only after the run has a concrete output directory on the VM and the
operator has preserved whatever local artifact bundle they intend to keep.
Generated `runs/` material remains local operator evidence unless a separate
tracked memo or release process records it.

## Source Class

Name the source class before writing the status note:

```text
source class: eval-factory execute/status artifact
```

The supporting files have different boundaries:

| File | Source class | Use | Boundary |
| --- | --- | --- | --- |
| `eval_factory_l4_smoke_preflight.json` | Eval-factory preflight artifact | Confirms interpreter, CUDA, GPU, and posture classification. | Not evidence that replay ran. |
| `eval_factory_l4_smoke_status.json` | Eval-factory execute/status artifact | Records one `l4-smoke --execute` attempt, embedded preflight, dispatch summary, family rows, and notes. | Not a tracked memo, checkpoint, release surface, or Gate12B signal. |
| `eval_factory_l4_smoke_execute.log` | Operator stdout/stderr capture | Helps reconstruct command flow and runner sections. | JSON artifacts remain the contract source for fields. |
| `cross_model_family_summary.csv` | Runs-derived materialized status | Shows local rows produced under the run output directory. | Does not override tracked memo status or widen the public claim surface. |

## Read Order

Read the receipt in this order:

1. `eval_factory_l4_smoke_preflight.json`
2. `eval_factory_l4_smoke_status.json`
3. `eval_factory_l4_smoke_execute.log`
4. `gate12a_cross_model_replay_qwen_qwen2_5_0_5b/cross_model_family_summary.csv`

Do not start from the CSV. The CSV is useful only after the preflight and
status artifacts establish the lane, target set, command, and execution result.

## What To Inspect

### 1. Preflight JSON

Inspect these fields first:

- `schema_id`
- `schema_version`
- `tier`
- `mode`
- `fixed_target_set`
- `sys_executable`
- `torch_importable`
- `torch_cuda_available`
- `gpu_count`
- `gpu_names`
- `nvidia_smi_available`
- `posture_classification`
- `preflight_ok`
- `errors`
- `remediation_hints`
- `result`

For a successful VM receipt, the posture fields should support
`posture_classification: remote_cuda_ready`, `preflight_ok: true`, and
`result: pass`. If they do not, stop and treat the receipt as a preflight or
posture failure.

### 2. Status JSON

Then inspect:

- `schema_id`
- `schema_version`
- `tier`
- `mode`
- `fixed_target_set`
- `model_id`
- `families`
- `entrypoint`
- `command`
- `out_dir`
- `returncode`
- `preflight`
- `downstream_dispatch_summary`
- `family_results`
- `notes`
- `result`

For a successful fixed-lane receipt, the status artifact should show
`mode: execute`, `returncode: 0`, `downstream_dispatch_summary.result: pass`,
the expected family count, and a `family_results` row for each fixed family.

### 3. Execute Log

Use the log to check what the operator saw:

- command line
- `out-dir`
- selected entrypoints
- environment diagnostics
- posture classification
- preflight result
- per-family dispatch/result summary
- final pass/fail summary

Do not quote log-only prose as a stronger status than the JSON artifacts
support.

### 4. Cross-model Summary CSV

Use the CSV as local materialized status under the run output directory. Inspect
only fields that are actually present in the generated file. Typical checks are:

- family rows present for `transcript_v1`, `briefing_v1`, and `archive_v1`
- structural replay flags reported by the summary
- score columns or comparison columns needed for operator troubleshooting
- first-pass status columns, when present

The CSV is not tracked memo status. It should not be cited as public evidence
unless a separate tracked process promotes that evidence.

## Assimilated Status Shape

A narrow closeout note should look like this:

```text
source class: eval-factory execute/status artifact
lane: l4-smoke
result: pass
posture: remote_cuda_ready
fixed target: Qwen/Qwen2.5-0.5B
families: transcript_v1 / briefing_v1 / archive_v1
structural replay receipt: status artifact reports pass for the fixed family set
phenotype readout: pending_local_read
interpretation boundary: operator receipt only; not tracked memo status, not a new checkpoint, not Gate12B
```

Keep the wording about structure and phenotype separate. The receipt can say
that the fixed-lane execution reported structural pass status. It cannot turn
`pending_local_read` into a phenotype conclusion.

## Phenotype Boundary

Preserve `runs_first_pass_status: pending_local_read` when that is what the
status artifact reports.

This means the VM receipt did not add a fresh phenotype readout. Do not infer a
packet-local first-pass phenotype result from structural pass status, from the
CSV alone, or from the fact that the VM execution succeeded.

If phenotype reading is performed later, record it as a separate operator step
with its own source class and do not backfill it into this receipt.

## Troubleshooting Note

A prior failed attempt caused by missing Python packages, such as
`transformers`, is a dependency/setup failure. Treat it only as troubleshooting
context.

Do not assimilate a dependency failure as:

- a Gate12A structural failure
- a model-family result
- a checkpoint boundary
- a Gate12B signal
- a reason to widen or narrow the public memo surface

If a later retry succeeds after dependency installation, summarize the failed
attempt separately and keep the successful receipt tied to its own output
directory.

## Do Not Over-read This

- Successful VM `l4-smoke --execute` receipt != new checkpoint.
- Execute/status artifact != tracked memo.
- `runs_first_pass_status: pending_local_read` != phenotype readout.
- Cross-summary CSV != public release evidence by itself.
- Dependency setup failure != research result.
- This receipt does not imply Gate12B.

## Closeout Checklist

Before writing the final operator note, confirm:

- the output directory is concrete and preserved as needed
- preflight JSON is present and valid enough to read
- status JSON is present and valid enough to read
- execute log is present when stdout/stderr capture is part of the retained
  bundle
- cross-model summary path is recorded only as local materialized status
- the summary names `eval-factory execute/status artifact` as the source class
- phenotype remains `pending_local_read` unless a separate readout step exists
