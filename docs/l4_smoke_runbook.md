# L4 smoke runbook

The smoke lane runs the fixed Qwen2.5-0.5B family set on a CUDA-ready Linux
host. Run commands from the repository root. On a local Windows machine,
`python tools/run_eval_checks.py --tier l4-smoke` shows the command without
executing a model.

## Execute

Use the intended checkout and interpreter. Check GPU availability with
`nvidia-smi` and `python3 -c "import torch; print(torch.cuda.is_available())"`
when setting up a host. The runner performs preflight before dispatching.

```bash
OUT_DIR="runs/l4-smoke-qwen05b-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"
set -o pipefail
python3 tools/run_eval_checks.py --tier l4-smoke --execute --out-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/eval_factory_l4_smoke_execute.log"
```

Use a fresh output directory for a new experiment. The log is ordinary
troubleshooting output. To inspect only the environment, use
`--preflight-only` instead of `--execute`.

## Read the result

Start with `eval_factory_l4_smoke_status.json`: `returncode`,
`downstream_dispatch_summary`, `family_results`, and `notes` describe the run.
It also embeds the preflight information. If dispatch never started, inspect
`eval_factory_l4_smoke_preflight.json` and the log for the setup error.

Measured rows are in
`gate12a_cross_model_replay_qwen_qwen2_5_0_5b/cross_model_family_summary.csv`
beneath the output directory. Report the observed values and their scope.
Structural success does not turn `pending_local_read` into a phenotype result.

| Problem | Action |
| --- | --- |
| Missing packages or CUDA unavailable | Fix the interpreter/environment, then retry the affected operation |
| Missing or malformed status | Check the log, process exit, and selected output directory |
| PowerShell rejects `<PATH>` | Substitute a real quoted path such as `"runs/my-smoke"` |
| Storage write fails | Preserve computed scientific data and repair persistence without changing seeds, queries, or the estimator |

A setup or storage problem is not itself a scientific failure. Follow the
study's actual scientific conditions when resuming; ordinary repairs need no
new authorization, freeze, or review documents.

## Share only when needed

Local work needs the result files and, when useful, a short summary in the
conversation. When transferring a copy, use
`python3 tools/package_eval_factory_receipt.py --run-dir "$OUT_DIR"`.
It copies four selected files and one manifest. `--tarball` optionally adds the
full run archive. See [result inspection and export](l4_smoke_receipt_assimilation.md).
