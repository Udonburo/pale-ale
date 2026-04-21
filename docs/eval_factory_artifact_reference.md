# Eval-factory Artifact Reference

This reference explains how to read eval-factory status sources and artifacts.
It is for operator interpretation only. It does not add empirical claims,
revise the Gate12A memo line, or make `runs/` tracked public evidence.

For exact schema ids, required field groups, and JSON examples, see
[`eval_factory_json_contracts.md`](eval_factory_json_contracts.md).
For successful VM receipt closeout, see
[`l4_smoke_receipt_assimilation.md`](l4_smoke_receipt_assimilation.md).

## Source Classes

| Source class | What it is | Comes from | Can be used to conclude | Cannot be used to conclude | Typical fields to inspect |
| --- | --- | --- | --- | --- | --- |
| Tracked memo status | Public memo-facing status recorded in tracked repository documents. | The numbered Gate12A workstream memos and public docs such as the Evidence Atlas. | What the tracked memo line says under its frozen-regime wording. | That a local `runs/` directory exists, that a new run happened, or that a new checkpoint was created. | Memo id, model, scale, family coverage, structural replay status, packet-local or memo-local first-pass wording, admission-boundary notes. |
| Runs-derived materialized status | Status parsed from already-materialized local outputs. | Local generated artifacts such as `runs/gate12a_cross_model_replay_*/cross_model_family_summary.csv` and nearby manifests, usually surfaced through `summarize-existing`. | What this checkout can currently read from local materialized files. | That the public memo surface has changed, that a release claim has changed, or that sidecar/admission-boundary rows widen mainline evidence. | Run id, summary path, model id, families, row count, structural flag counts, first-pass status counts, manifest notes. |
| Eval-factory preflight artifact | Environment and GPU posture receipt for the `l4-smoke` lane. | `eval_factory_l4_smoke_preflight.json`, written under `--out-dir` when `--preflight-only` is run with an output root, and before `--execute` dispatch. | Whether the selected interpreter and host posture look ready for the GCP L4 execution lane. | That model execution happened, that the smoke lane passed, or that tracked memo status changed. | `sys_executable`, `os_name`, `torch_importable`, `torch_cuda_available`, `gpu_count`, `gpu_names`, `nvidia_smi_available`, `posture_classification`, `preflight_ok`, `errors`, `remediation_hints`. |
| Eval-factory execute/status artifact | Operator status artifact for one `l4-smoke --execute` attempt. | `eval_factory_l4_smoke_status.json`, written under `--out-dir` after downstream execution is attempted. | The command, return code, embedded preflight, family-result rows, and notes for that execution attempt. | That a new checkpoint exists, that Gate12B is implied, or that a receipt/status artifact is a tracked memo. | `tier`, `mode`, `model_id`, `families`, `entrypoint`, `command`, `out_dir`, `returncode`, `preflight`, `family_results`, `notes`. |

## Reading Order

1. Start with the source class. Decide whether the line is tracked memo status,
   runs-derived materialized status, a preflight receipt, or an execute/status
   artifact.
2. Check the interpretation boundary for that source class before writing any
   summary.
3. Use exact fields from the artifact or memo. Do not convert receipt fields
   into research claims.

## L4-smoke Artifact Paths

For the fixed `l4-smoke` lane, choose a concrete output root such as:

```text
runs/l4-smoke-qwen05b-20260420T120000Z
```

The preflight artifact is:

```text
$OUT_DIR/eval_factory_l4_smoke_preflight.json
```

The execute/status artifact is:

```text
$OUT_DIR/eval_factory_l4_smoke_status.json
```

The cross-model summary, when downstream execution produces it, is:

```text
$OUT_DIR/gate12a_cross_model_replay_qwen_qwen2_5_0_5b/cross_model_family_summary.csv
```

## Failure Boundaries

Precondition failure happens before the preflight is meaningful. Examples:
missing `--out-dir`, an output path that points to a file, or missing committed
entrypoints. In this case, inspect stdout and the command line first.

Preflight failure means the runner inspected the environment and did not see
the expected remote CUDA-ready posture. In `--execute`, downstream model
execution is not invoked when preflight fails. Inspect
`posture_classification`, `preflight_ok`, `errors`, and
`remediation_hints`.

Downstream execution failure happens after preflight passes and the replay
subprocess is attempted. Inspect `returncode`, `family_results`, and `notes`
in `eval_factory_l4_smoke_status.json`, plus stdout/stderr snippets when the
runner reports them.

## Do Not Over-read This

- Successful smoke run != new checkpoint.
- Receipt/status artifact != tracked memo.
- Atlas != leaderboard.
- Sidecar/admission-boundary rows do not widen mainline.
- `runs/` materialization != release or Zenodo evidence.

When writing a status note, preserve the current frozen-regime wording and say
which source class you used.
