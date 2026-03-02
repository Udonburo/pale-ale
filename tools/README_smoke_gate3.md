# Gate3 Smoke Test

This smoke test runs Gate3 telemetry end-to-end on synthetic trajectories.

It performs three runs:
- `smooth` -> `runs/gate3_smoke_A`
- `smooth` (same input) -> `runs/gate3_smoke_B` for determinism
- `kink` -> `runs/gate3_smoke_kink` for sensitivity

## Run

```powershell
pwsh tools/run_gate3_smoke.ps1
```

## Expected

- Determinism check prints `PASS` with identical SHA-256 for:
  - `manifest.json`
  - `summary.csv`
  - `samples.csv`
- Smooth vs kink comparison table shows visible numeric differences in tau-related fields.
