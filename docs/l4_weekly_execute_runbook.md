# L4-weekly Execute Runbook

This runbook documents the bounded `l4-weekly` execution lane for operators
once a checkout exposes weekly `--target` preflight and execute support. It is
operational guidance only. It does not update the Gate12A memo line, create a
new checkpoint, widen the dense-transformer mainline, or imply Gate12B.

If the local checkout only supports `l4-weekly` plan output, use
[`l4_weekly_escalation_guide.md`](l4_weekly_escalation_guide.md) and do not
synthesize preflight or execute artifacts.

## What This Lane Is

`l4-weekly` remains plan-only by default. Bounded execution support adds two
operator modes:

- `--preflight-only`: inspect the VM/interpreter/GPU posture for one weekly
  target.
- `--execute`: run one bounded weekly target after preflight.

Use one target at a time. Do not run the whole weekly matrix as one opaque job.

Current target aliases are:

| Target alias | Model | Expected family coverage |
| --- | --- | --- |
| `qwen2_5_3b` | `Qwen/Qwen2.5-3B-Instruct` | `transcript_v1 / briefing_v1 / archive_v1` |
| `llama3_2_3b` | `meta-llama/Llama-3.2-3B-Instruct` | `transcript_v1 / briefing_v1 / archive_v1` |
| `qwen3_4b` | `Qwen/Qwen3-4B` | `transcript_v1 / briefing_v1 / archive_v1` |

The lane is still bounded to the current 3B/4B dense-transformer mainline under
the frozen Gate12A observable surface.

## What This Lane Is Not

`l4-weekly` execution is not:

- 7B FP32
- sidecar work
- protocol-expanding work
- quantized work
- a new checkpoint
- a memo claim by itself
- a Gate12B signal

It is an operator execution lane. Treat its artifacts as local operator status
unless a separate tracked memo or release process records them.

## Command Shapes

Run commands from the repository root. Confirm the checkout supports the
bounded weekly target flags before using the preflight or execute shapes.

Plan-only default:

```bash
OUT_DIR="runs/l4-weekly-plan-$(date -u +%Y%m%dT%H%M%SZ)"
python3 tools/run_eval_checks.py --tier l4-weekly --out-dir "$OUT_DIR"
```

Preflight one target:

```bash
OUT_DIR="runs/l4-weekly-qwen2_5_3b-$(date -u +%Y%m%dT%H%M%SZ)"
python3 tools/run_eval_checks.py --tier l4-weekly --preflight-only --target qwen2_5_3b --out-dir "$OUT_DIR"
```

Execute one target:

```bash
OUT_DIR="runs/l4-weekly-qwen2_5_3b-$(date -u +%Y%m%dT%H%M%SZ)"
python3 tools/run_eval_checks.py --tier l4-weekly --execute --target qwen2_5_3b --out-dir "$OUT_DIR"
```

Rotate the target alias deliberately:

```bash
python3 tools/run_eval_checks.py --tier l4-weekly --preflight-only --target llama3_2_3b --out-dir "$OUT_DIR"
python3 tools/run_eval_checks.py --tier l4-weekly --preflight-only --target qwen3_4b --out-dir "$OUT_DIR"
```

Do not omit `--target` for bounded execution. Do not treat plan-only output as
proof that execution happened.

## Pre-run Checks

Before a real VM execution:

- Confirm the checkout is updated to the intended branch or commit.
- Confirm `tools/run_eval_checks.py` exposes the bounded weekly target flags.
- Confirm the selected target alias is one of `qwen2_5_3b`,
  `llama3_2_3b`, or `qwen3_4b`.
- Confirm the VM interpreter: `python3 -c "import sys; print(sys.executable)"`.
- Confirm the GPU is visible: `nvidia-smi`.
- Confirm CUDA through the same interpreter:
  `python3 -c "import torch; print(torch.cuda.is_available())"`.
- Choose a fresh `--out-dir`.

If preflight fails, stop. Do not continue to execute.

## Reading Order

Read weekly execution surfaces in this order:

1. Weekly preflight artifact.
2. Weekly execute/status artifact.
3. Downstream materialized outputs under the chosen `--out-dir`.

Use stdout to find the exact artifact paths for the checkout that produced the
run. Do not substitute `l4-smoke` artifact names for weekly artifacts unless the
implementation explicitly writes them.

## First Successful Weekly Pilot

The first successful bounded weekly pilot is the `qwen2_5_3b` VM run under:

```text
runs/eval_factory_l4_weekly_qwen2_5_3b_vm_20260422T040509Z
```

Use it as an operator/status surface only. It is not a new checkpoint, not a
memo claim, not a release surface, and not a Gate12B signal.

The execute command shape was:

```bash
python3 tools/run_eval_checks.py --tier l4-weekly --execute --target qwen2_5_3b --out-dir runs/eval_factory_l4_weekly_qwen2_5_3b_vm_20260422T040509Z
```

Read the pilot in this order:

1. `eval_factory_l4_weekly_preflight.json`
2. `eval_factory_l4_weekly_status.json`
3. `gate12a_cross_model_replay_qwen_qwen2_5_3b_instruct/cross_model_family_summary.csv`

If the weekly status artifact reports `result: pass`, `families_expected: 3`,
`families_reported: 3`, and `fail: 0`, that supports a narrow operator
closeout for the pilot. It does not turn structural pass status into a
phenotype readout.

If family rows report `runs_first_pass_status: pending_local_read`, preserve:

```text
phenotype readout: pending_local_read
```

`pending_local_read` remains not-read.

### Weekly Preflight Artifact

Use the preflight artifact to answer only environment and posture questions:

- selected target alias
- model id and expected families
- interpreter path
- OS/platform
- PyTorch import status
- CUDA visibility
- GPU name/count
- `nvidia-smi` visibility
- posture classification
- preflight result
- errors and remediation hints

Preflight does not prove that replay ran.

### Weekly Execute/status Artifact

Use the execute/status artifact to answer only operator execution questions:

- selected target alias
- model id and expected families
- command
- output directory
- embedded preflight
- downstream return code
- family rows or dispatch summary
- notes
- final result

An execute/status artifact is not a tracked memo, not a release surface, and
not a checkpoint.

### Downstream Materialized Outputs

Read downstream outputs only after the preflight and status artifacts establish
the lane, target, command, and result. Treat downstream CSVs, logs, and local
summaries as runs-derived materialized status.

Do not let downstream materialization override tracked memo/workstream evidence.

## Interpretation Boundary

Keep structural result and phenotype readout separate.

A successful weekly execute artifact may support an operator statement that a
bounded target reported structural pass status, if the artifact and downstream
summary actually say that. It does not create a phenotype claim by itself.

If phenotype fields remain `pending_local_read`, preserve that wording:

```text
phenotype readout: pending_local_read
```

`pending_local_read` means not read. Do not infer first-pass phenotype from a
structural result, a successful process return code, or a complete family set.

## Closeout Shape

A narrow weekly closeout note should include:

- source class
- command used
- target alias
- model id
- family coverage
- VM/interpreter posture
- output directory
- preflight artifact path
- execute/status artifact path
- downstream materialized output path, if present
- structural result exactly as reported
- phenotype readout, if separately recorded
- interpretation boundary

Use wording like:

```text
source class: eval-factory weekly execute/status artifact
lane: l4-weekly
target: qwen2_5_3b
model: Qwen/Qwen2.5-3B-Instruct
families: transcript_v1 / briefing_v1 / archive_v1
structural result: as reported by the weekly execute/status artifact
phenotype readout: pending_local_read
interpretation boundary: operator status only; not tracked memo status, not a new checkpoint, not Gate12B
```

## Do Not Over-read This

- Weekly plan != weekly execution.
- Weekly preflight != weekly execution.
- Weekly execute/status artifact != tracked memo.
- Successful weekly execution != new checkpoint.
- Structural pass != phenotype read.
- `pending_local_read` remains not-read.
- The weekly lane does not include 7B FP32, sidecar, protocol-expanding, or
  quantized work.
- The weekly lane does not imply Gate12B.
