# Gate7 Progression Leak Workflow

Gate7 starts from the smallest object-native dynamic motif:

- current projector `P_t`
- next-step anchor direction `v_{t+1}`

The primary score is:

- `progression_leak_v1 = 1 - ||P_t v_{t+1}||^2 / ||v_{t+1}||^2`

This workflow does not use `compat_local8`.
It consumes the canonical Gate6 native object artifacts directly.

Coverage note:

- the terminal row of each sample has no successor by construction
- it is recorded as `final_step_no_successor`
- it is not counted as true missing transition coverage

## 1) Run the minimal progression consumer

For CFA:

```powershell
python tools/run_gate7_progression_leak_consumer.py --gate6-dir runs/gate6_cfa_full --out-dir runs/gate7a_cfa_full --run-id gate7a_cfa_full
```

For Seam:

```powershell
python tools/run_gate7_progression_leak_consumer.py --gate6-dir runs/gate6_seam_full --out-dir runs/gate7a_seam_full --run-id gate7a_seam_full
```

Outputs:

- `manifest.json`
- `gate7a_token_telemetry.csv`
- `gate7a_sample_summary.csv`
- `gate7a_aggregate_summary.md`
- `checksums.json`

## 2) Evaluate Seam pairs with Gate5-style quietness vocabulary

```powershell
python tools/evaluate_gate6_native_object_seam_pairs.py --token-csv runs/gate7a_seam_full/gate7a_token_telemetry.csv --seam-jsonl runs/gate5_seam_64e2e/seam_v0.jsonl --out-dir runs/gate7a_seam_pairs_full --run-id gate7a_seam_pairs_full --primary-metric progression_leak_v1 --guardrail-metric score_F_gram_loop_v1 --artifact-prefix gate7a_seam
```

Outputs:

- `manifest.json`
- `gate7a_seam_pair_summary.csv`
- `gate7a_seam_family_summary.csv`
- `gate7a_seam_report.md`
- `checksums.json`

## 3) Expected reading

This first Gate7 unit is only a smoke.

It is meant to answer:

- does projector-native progression leakage carry signal at all?
- does it deserve one more narrow dynamic iterate before field aggregation?

It is not meant to settle:

- persistent topology
- burst aggregation
- retrieval conflict
- benchmark policy

## 4) Run the two-step projector closure iterate

This keeps the same datasets and evaluation surface, but changes the law to:

- `progression_closure_v2 = 1 - ||P_{t+1} P_t v_{t+1}||^2 / ||v_{t+1}||^2`

For CFA:

```powershell
python tools/run_gate7_progression_closure_consumer_v2.py --gate6-dir runs/gate6_cfa_full --out-dir runs/gate7b_cfa_full --run-id gate7b_cfa_full
```

For Seam:

```powershell
python tools/run_gate7_progression_closure_consumer_v2.py --gate6-dir runs/gate6_seam_full --out-dir runs/gate7b_seam_full --run-id gate7b_seam_full
python tools/evaluate_gate6_native_object_seam_pairs.py --token-csv runs/gate7b_seam_full/gate7b_token_telemetry.csv --seam-jsonl runs/gate5_seam_64e2e/seam_v0.jsonl --out-dir runs/gate7b_seam_pairs_full --run-id gate7b_seam_pairs_full --primary-metric progression_closure_v2 --guardrail-metric score_F_gram_loop_v1 --artifact-prefix gate7b_seam
```

## 5) Run the anisotropic-projector fallback

This is the single allowed `Sigma_t` operatorization fallback:

- `A_t = B_t diag(sigma_i / sigma_1) B_t^T`
- `progression_anisotropic_closure_v3 = 1 - ||A_{t+1} A_t v_{t+1}||^2 / ||v_{t+1}||^2`

For CFA:

```powershell
python tools/run_gate7_progression_anisotropic_consumer_v3.py --gate6-dir runs/gate6_cfa_full --out-dir runs/gate7c_cfa_full --run-id gate7c_cfa_full
```

For Seam:

```powershell
python tools/run_gate7_progression_anisotropic_consumer_v3.py --gate6-dir runs/gate6_seam_full --out-dir runs/gate7c_seam_full --run-id gate7c_seam_full
python tools/evaluate_gate6_native_object_seam_pairs.py --token-csv runs/gate7c_seam_full/gate7c_token_telemetry.csv --seam-jsonl runs/gate5_seam_64e2e/seam_v0.jsonl --out-dir runs/gate7c_seam_pairs_full --run-id gate7c_seam_pairs_full --primary-metric progression_anisotropic_closure_v3 --guardrail-metric score_F_gram_loop_v1 --artifact-prefix gate7c_seam
```

## 6) Gate7 decision rule

Gate7 remains a motif bake-off only.

That means:

- do not add field aggregation in this phase
- do not rescue mixed signal with burst logic
- compare dynamic laws directly on CFA plus Seam pairs

If no dynamic law clears the seam-tail discipline, Gate7 closes without unlocking field aggregation.
