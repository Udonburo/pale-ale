# Triality Attestations

This folder contains commit-ready artifacts for teacher-forcing token/span evaluation.

Key files:
- `YYYY-MM-DD_hf_sample0_raw.json`: raw HF sample (index 0) used for first end-to-end run.
- `YYYY-MM-DD_eval_hf0_teacher_forcing.txt`: sample0 eval with default primary score.
- `YYYY-MM-DD_eval_hf0_teacher_forcing_primaryE.txt`: sample0 eval with locked primary score E.
- `YYYY-MM-DD_holdout_primaryE_results.jsonl`: reconstructed holdout results for indices `1..50`.
- `YYYY-MM-DD_holdout_primaryE_summary.txt`: aggregate stats over holdout results.
- `batch/YYYY-MM-DD_eval_hf{idx}_primaryE.txt`: per-index holdout eval reports.
- `YYYY-MM-DD_manifest_index.md`: commit-scope manifest with commands, SHA256s, and validation summary.
- `YYYY-MM-DD_cfa_batch_primaryE_report.txt`: CFA preregistered batch aggregate report.
- `case_study/index.md`: representative-set case-study manifest.
- `case_study/representative_set_summary.md`: top/median/bottom local-metric summary.

Notes:
- `runs/` is gitignored; commit-ready summaries are copied here.
- Reports are normalized to repo-relative paths (no local absolute user paths).
- `cfa_batch/` per-sample CFA eval reports are treated as local verbose artifacts and are ignored from git; keep the aggregate report instead.
