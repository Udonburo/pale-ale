# Triality Attestations

This folder contains commit-ready artifacts for teacher-forcing token/span evaluation.

Key files:
- `YYYY-MM-DD_hf_sample0_raw.json`: raw HF sample (index 0) used for first end-to-end run.
- `YYYY-MM-DD_eval_hf0_teacher_forcing.txt`: sample0 eval with default primary score.
- `YYYY-MM-DD_eval_hf0_teacher_forcing_primaryE.txt`: sample0 eval with locked primary score E.
- `YYYY-MM-DD_holdout_primaryE_results.jsonl`: reconstructed holdout results for indices `1..50`.
- `YYYY-MM-DD_holdout_primaryE_summary.txt`: aggregate stats over holdout results.
- `batch/YYYY-MM-DD_eval_hf{idx}_primaryE.txt`: per-index holdout eval reports.

Notes:
- `runs/` is gitignored; commit-ready summaries are copied here.
- Reports are normalized to repo-relative paths (no local absolute user paths).
