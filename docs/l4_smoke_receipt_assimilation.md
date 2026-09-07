# Inspect or export an L4 result

A normal local run is complete when its status and result files have been
written. Read those files directly. Packaging, an operator note, and checksum
sidecars are not required steps.

| File | What to read |
| --- | --- |
| `eval_factory_l4_smoke_status.json` | Command, return code, embedded preflight, family results, and errors |
| `eval_factory_l4_smoke_preflight.json` | Interpreter, CUDA/GPU availability, and setup errors when troubleshooting |
| `eval_factory_l4_smoke_execute.log` | Captured stdout/stderr when needed to diagnose the run |
| `gate12a_cross_model_replay_qwen_qwen2_5_0_5b/cross_model_family_summary.csv` | Measured family rows and structural flags |

Report the output path and observed result. A setup or persistence error is an
operational failure; fix it and retry the affected operation within the study's
scientific conditions. Structural pass does not fill in an unmeasured phenotype:
keep `pending_local_read` when that is what the data says.

## Optional export

When another person or machine needs a copy, export the selected files:

```bash
python3 tools/package_eval_factory_receipt.py --run-dir "$OUT_DIR"
```

The helper copies the four files above and writes one
`operator_receipt_manifest.json` containing their hashes and run metadata under
`runs/receipt_bundles/`. It also supports the weekly lane and its corresponding
filenames. Capture the execution log during the run if an export will be needed;
see the [runbook](l4_smoke_runbook.md).

Use `--tarball` only when the recipient needs the entire run directory. Its
hash goes in the same manifest. Use `--inspect-only` to check an export without
writing it. No checksum-of-checksum files or follow-up receipts are generated.
Existing exports that declare legacy checksum sidecars remain readable.

`cpu-nightly` does not require or verify transfer bundles. `summarize-existing`
reads their metadata without rehashing their contents and labels it
`metadata-valid`. To verify a received copy, select that export explicitly:

```bash
python3 tools/package_eval_factory_receipt.py --verify-export path/to/operator_receipt_manifest.json
```

This checks the copied files against the manifest and exits nonzero on damage.
Published studies retain their result provenance; an export does not change a
published claim.
