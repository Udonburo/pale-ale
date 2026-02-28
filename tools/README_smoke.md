# Gate2 Smoke Test

This smoke test generates synthetic Gate2 inputs and runs `pale-ale gate2 run` in three passes:

- Run A: smooth trajectories
- Run B: same smooth input again (determinism check)
- Run C: kink trajectories (sensitivity check)

## Run

```powershell
pwsh tools/run_gate2_smoke.ps1
```

## What it verifies

- Determinism: `manifest.json`, `summary.csv`, `samples.csv` from A/B are SHA-256 identical.
- Sensitivity: `summary.csv` values for smooth and kink are printed side-by-side for manual inspection.

## Generated paths

- Inputs: `data/gate2/smooth_4x24.json`, `data/gate2/kink_4x24.json`
- Outputs: `runs/gate2_smoke_A`, `runs/gate2_smoke_B`, `runs/gate2_smoke_kink`

All generated JSON is UTF-8 with LF and validated before write:
`sample_id` int, each `ans_vec8` row length 8, all values finite.
