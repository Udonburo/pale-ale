# Eval-factory JSON Contract Reference

This reference describes the current eval-factory JSON artifact contracts for
operators and implementers. It is a contract and interpretation guide only. It
does not turn local artifacts into tracked workstream evidence, widen Gate12A,
or imply a new checkpoint.

## Source-class Boundary

Keep these source classes separate before reading any field:

| Source class | Meaning | Boundary |
| --- | --- | --- |
| Tracked memo status | Status recorded in tracked workstream memos and public docs such as the Evidence Atlas. | Public memo-facing evidence surface. |
| Runs-derived materialized status | Status parsed from local generated outputs, such as `runs/gate12a_cross_model_replay_*/cross_model_family_summary.csv`. | Local materialization only; does not override tracked memos. |
| Eval-factory preflight artifact | Environment/GPU posture receipt for `l4-smoke`. | Shows readiness posture; does not show that replay ran. |
| Eval-factory execute/status artifact | Operator status for one `l4-smoke --execute` attempt. | Runs-derived operator record; not a tracked memo. |
| Eval-factory weekly plan artifact | Plan-only artifact for bounded `l4-weekly` targets. | Planning surface only; not weekly execution. |

All schema versions below use `schema_version: 1`.

## Contract Overview

| Artifact | Schema id | Source class | Written by | Primary use |
| --- | --- | --- | --- | --- |
| `eval_factory_l4_smoke_preflight.json` | `pale-ale.eval_factory.l4_smoke.preflight.v1` | Eval-factory preflight artifact | `--tier l4-smoke --preflight-only --out-dir ...`, and `--tier l4-smoke --execute --out-dir ...` before dispatch | Record interpreter, CUDA, GPU, and posture classification. |
| `eval_factory_l4_smoke_status.json` | `pale-ale.eval_factory.l4_smoke.status.v1` | Eval-factory execute/status artifact | `--tier l4-smoke --execute --out-dir ...` after downstream dispatch is attempted | Record command, embedded preflight, dispatch summary, family rows, and notes. |
| `eval_factory_l4_weekly_plan.json` | `pale-ale.eval_factory.l4_weekly.plan.v1` | Eval-factory weekly plan artifact | `--tier l4-weekly --out-dir ...` | Record the current bounded 3B/4B weekly planning matrix and exclusions. |

## `eval_factory_l4_smoke_preflight.json`

**Where it comes from:** written under the chosen `--out-dir` by the l4-smoke
preflight path. During `--execute`, it is written before downstream replay is
allowed to run.

**What it is for:** confirming whether the selected interpreter and host are in
the expected posture for the GCP L4 execution lane.

**What it is not for:** proving that model replay happened, proving that a
smoke run passed, creating a tracked memo, or changing the Gate12A claim
surface.

### Required field groups

| Group | Required fields |
| --- | --- |
| Identity / schema | `schema_id`, `schema_version`, `created_at`, `tier`, `mode`, `result` |
| Fixed target set | `fixed_target_set.boundary`, `fixed_target_set.model_id`, `fixed_target_set.model_label`, `fixed_target_set.families`, `fixed_target_set.device` |
| Posture / environment | `sys_executable`, `python_version`, `cwd`, `platform`, `os_name`, `torch_importable`, `torch_version`, `torch_cuda_available`, `torch_cuda_version`, `gpu_count`, `gpu_names`, `nvidia_smi_available`, `nvidia_smi_path`, `nvidia_smi_summary`, `nvidia_smi_error`, `posture_classification`, `preflight_ok` |
| Result / notes | `remediation_hints`, `errors` |

`mode` is one of `preflight-only` or `execute`. `result` is `pass` or `fail`.
`posture_classification` is one of `remote_cuda_ready`,
`local_windows_no_cuda`, `python_missing_torch`, `cuda_unavailable`, or
`unknown_posture`.

### Partial snippet

```json
{
  "schema_id": "pale-ale.eval_factory.l4_smoke.preflight.v1",
  "schema_version": 1,
  "created_at": "2026-04-20T00:00:00Z",
  "tier": "l4-smoke",
  "mode": "preflight-only",
  "fixed_target_set": {
    "boundary": "0.5B fixed family boundary set: transcript_v1, briefing_v1, archive_v1",
    "model_id": "Qwen/Qwen2.5-0.5B",
    "model_label": "qwen_qwen2_5_0_5b",
    "families": ["transcript_v1", "briefing_v1", "archive_v1"],
    "device": "cuda"
  },
  "sys_executable": "/opt/venv/bin/python3",
  "os_name": "Linux",
  "torch_importable": true,
  "torch_cuda_available": true,
  "gpu_count": 1,
  "gpu_names": ["NVIDIA L4"],
  "nvidia_smi_available": true,
  "posture_classification": "remote_cuda_ready",
  "preflight_ok": true,
  "remediation_hints": [],
  "errors": [],
  "result": "pass"
}
```

This snippet is illustrative and not a complete artifact; the full contract
also includes the remaining posture/environment strings listed above.

### Do not over-read this artifact

- Preflight artifact != successful run.
- `remote_cuda_ready` != downstream replay completion.
- A failed preflight is an environment/posture record, not a Gate12A result.
- This artifact does not imply a new checkpoint or Gate12B.

## `eval_factory_l4_smoke_status.json`

**Where it comes from:** written under the chosen `--out-dir` after
`--tier l4-smoke --execute` attempts downstream dispatch.

**What it is for:** recording one fixed-lane execution attempt, including the
command, embedded preflight, downstream summary, family-result rows, and notes.

**What it is not for:** replacing tracked workstream memos, promoting a smoke
run into a checkpoint, widening the mainline, or implying Gate12B.

### Required field groups

| Group | Required fields |
| --- | --- |
| Identity / schema | `schema_id`, `schema_version`, `created_at`, `tier`, `mode`, `result` |
| Fixed target set | `fixed_target_set.boundary`, `fixed_target_set.model_id`, `fixed_target_set.model_label`, `fixed_target_set.families`, `fixed_target_set.device` |
| Execution / dispatch | `entrypoint`, `command`, `out_dir`, `returncode`, `downstream_dispatch_summary.subprocess_returncode`, `downstream_dispatch_summary.families_expected`, `downstream_dispatch_summary.families_reported`, `downstream_dispatch_summary.fail`, `downstream_dispatch_summary.result` |
| Embedded preflight | `preflight` using the `pale-ale.eval_factory.l4_smoke.preflight.v1` contract with `mode: execute` |
| Result / notes | `family_results`, `notes` |

The writer also emits top-level `model_id`, `model_label`, and `families` as
convenience fields that mirror the fixed target set.

Generated `family_results` rows currently use fields such as `family`,
`dispatch`, `structural_flags_all_true`, and `runs_first_pass_status`.

### Partial snippet

```json
{
  "schema_id": "pale-ale.eval_factory.l4_smoke.status.v1",
  "schema_version": 1,
  "created_at": "2026-04-20T00:00:00Z",
  "tier": "l4-smoke",
  "mode": "execute",
  "fixed_target_set": {
    "boundary": "0.5B fixed family boundary set: transcript_v1, briefing_v1, archive_v1",
    "model_id": "Qwen/Qwen2.5-0.5B",
    "model_label": "qwen_qwen2_5_0_5b",
    "families": ["transcript_v1", "briefing_v1", "archive_v1"],
    "device": "cuda"
  },
  "entrypoint": "tools/run_gate12a_cross_model_replay.py",
  "out_dir": "runs/l4-smoke-qwen05b-20260420T000000Z",
  "returncode": 0,
  "preflight": {
    "schema_id": "pale-ale.eval_factory.l4_smoke.preflight.v1",
    "schema_version": 1,
    "created_at": "2026-04-20T00:00:00Z",
    "tier": "l4-smoke",
    "mode": "execute",
    "result": "pass"
  },
  "downstream_dispatch_summary": {
    "subprocess_returncode": 0,
    "families_expected": 3,
    "families_reported": 3,
    "fail": 0,
    "result": "pass"
  },
  "result": "pass",
  "family_results": [
    {
      "family": "transcript_v1",
      "dispatch": "completed",
      "structural_flags_all_true": "True",
      "runs_first_pass_status": "pending_local_read"
    }
  ],
  "notes": []
}
```

This snippet is illustrative and partial. A full valid status artifact includes
the complete `command`, `model_id`, `model_label`, `families`, and the complete
embedded preflight contract.

### Do not over-read this artifact

- Execute/status artifact != tracked memo.
- A `pass` status is one operator execution record, not a release surface.
- Runs-derived family rows do not override packet-local or memo-local wording.
- This artifact does not imply a new checkpoint or Gate12B.

## `eval_factory_l4_weekly_plan.json`

**Where it comes from:** written under the chosen `--out-dir` by
`--tier l4-weekly --out-dir ...`.

**What it is for:** recording the current bounded weekly planning matrix,
planned entrypoints, and exclusions.

**What it is not for:** weekly execution, model dispatch, tracked memo status,
or any expansion beyond the current 3B/4B dense-transformer mainline.

### Required field groups

| Group | Required fields |
| --- | --- |
| Identity / schema | `schema_id`, `schema_version`, `created_at`, `tier`, `mode`, `result` |
| Planning posture | `resource_posture` |
| Target set | `weekly_target_matrix[].model_id`, `weekly_target_matrix[].model_label`, `weekly_target_matrix[].families` |
| Planned dispatch surface | `planned_entrypoints` |
| Exclusions | `exclusions` |

`mode` and `result` are both `plan-only`. The current `weekly_target_matrix`
is exactly:

| Model id | Model label | Families |
| --- | --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | `qwen_qwen2_5_3b_instruct` | `transcript_v1 / briefing_v1 / archive_v1` |
| `meta-llama/Llama-3.2-3B-Instruct` | `meta_llama_llama_3_2_3b_instruct` | `transcript_v1 / briefing_v1 / archive_v1` |
| `Qwen/Qwen3-4B` | `qwen_qwen3_4b` | `transcript_v1 / briefing_v1 / archive_v1` |

The current `exclusions` are `7B FP32`,
`protocol-expanding candidates`, `quantized candidates`, `sidecar candidates`,
and `Gate12B promotion`.

### Partial snippet

```json
{
  "schema_id": "pale-ale.eval_factory.l4_weekly.plan.v1",
  "schema_version": 1,
  "created_at": "2026-04-20T00:00:00Z",
  "tier": "l4-weekly",
  "mode": "plan-only",
  "resource_posture": "single L4 posture for planned weekly work; excludes 7B FP32",
  "weekly_target_matrix": [
    {
      "model_id": "Qwen/Qwen2.5-3B-Instruct",
      "model_label": "qwen_qwen2_5_3b_instruct",
      "families": ["transcript_v1", "briefing_v1", "archive_v1"]
    },
    {
      "model_id": "meta-llama/Llama-3.2-3B-Instruct",
      "model_label": "meta_llama_llama_3_2_3b_instruct",
      "families": ["transcript_v1", "briefing_v1", "archive_v1"]
    },
    {
      "model_id": "Qwen/Qwen3-4B",
      "model_label": "qwen_qwen3_4b",
      "families": ["transcript_v1", "briefing_v1", "archive_v1"]
    }
  ],
  "planned_entrypoints": [
    "tools/run_gate12a_cross_model_replay.py",
    "tools/run_gate8_scaleup.py",
    "tools/run_gate12a_family_replay.py"
  ],
  "exclusions": [
    "7B FP32",
    "protocol-expanding candidates",
    "quantized candidates",
    "sidecar candidates",
    "Gate12B promotion"
  ],
  "result": "plan-only"
}
```

### Do not over-read this artifact

- Weekly plan artifact != weekly execution.
- Plan-only target matrix != new empirical result.
- Mainline exclusions remain part of the contract.
- This artifact does not imply a new checkpoint or Gate12B.

## Operator Reading Rule

When summarizing any artifact, name the source class first, then quote only the
fields that support the operator statement. Do not convert JSON receipt fields
into tracked workstream claims.
