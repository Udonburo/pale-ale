# Attestations

This directory stores reproducibility evidence (hashes/results), not raw run artifacts.

Rules:
- Keep machine-generated artifacts in `runs/` (ignored by git).
- Commit only compact evidence summaries here (SHA-256, command, result).
- Use UTF-8 + LF.
- Use date-prefixed filenames: `YYYY-MM-DD_<purpose>.txt`.

Layout:
- `attestations/smoke/gate2/` : Gate2 smoke evidence
- `attestations/smoke/gate3/` : Gate3 smoke evidence

