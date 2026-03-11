# Gate5 Workflow

Gate5 consumes the existing `Gate4RunInputV1` boundary from `gate4_input.json`.

It does not change Gate4. It reads the same proxy observables and emits transport-loop telemetry.

## 1) Prepare or reuse `gate4_input.json`

For CFA, reuse the existing Gate4 batch ingestion flow:

```powershell
python tools/run_gate4_batch_ingestion.py --samples-root runs/cfa_batch_primaryE/samples --all-samples --cfa-jsonl data/cfa/cfa_v1.jsonl --out-dir runs/gate5_cfa_ingestion --dataset-revision-id cfa_v1_gate5_v0
```

This writes:

- `runs/gate5_cfa_ingestion/gate4_input.json`

## 2) Run Gate5 spike on CFA

```powershell
python tools/run_gate5_spike.py --input runs/gate5_cfa_ingestion/gate4_input.json --out-dir runs/gate5_cfa_spike --run-id gate5_cfa_spike --dataset-revision-id cfa_v1_gate5_v0 --cfa-jsonl data/cfa/cfa_v1.jsonl
```

Outputs:

- `runs/gate5_cfa_spike/manifest.json`
- `runs/gate5_cfa_spike/gate5_token_telemetry.csv`
- `runs/gate5_cfa_spike/gate5_sample_summary.csv`
- `runs/gate5_cfa_spike/gate5_attestation_report.txt`
- `runs/gate5_cfa_spike/gate5_aggregate_report.md`

## 3) Generate Seam Challenge Set v0

```powershell
python tools/generate_seam_challenge.py --out data/seam/seam_v0.jsonl --meta-out data/seam/seam_v0_meta.json --n-pairs 64 --seed 7
```

The generator emits paired rows:

- `clean_consistent`
- `seam_perturbed_consistent`

Each row carries:

- `pair_id`
- `perturbation_family`
- `perturbation_spans`

## 4) Run Gate5 spike on Seam

Gate5 still consumes `gate4_input.json`, so the Seam side must first be extracted and packed through the same teacher-forcing path as CFA.

Use the dedicated Seam builder:

```powershell
python tools/build_seam_gate4_input.py --seam-jsonl data/seam/seam_v0.jsonl --out-dir runs/seam_gate5 --device auto
```

This writes:

- `runs/seam_gate5/gate4_input.json`
- `runs/seam_gate5/samples/sample_*/`
- `runs/seam_gate5/seam_selection_manifest.json`
- `runs/seam_gate5/seam_build_manifest.json`

Notes:

- By default the builder enforces the paired quietness contract and fails if a subset drops either the clean or perturbed side of a `pair_id`.
- Use `--allow-incomplete-pairs` only for debugging or partial extraction, not for the canonical Seam quietness run.

Once a Seam `gate4_input.json` exists:

```powershell
python tools/run_gate5_spike.py --input runs/seam_gate5/gate4_input.json --out-dir runs/gate5_seam_spike --run-id gate5_seam_spike --dataset-revision-id seam_v0_gate5 --seam-jsonl data/seam/seam_v0.jsonl --evaluation-mode-id unsupervised_v1
```

Notes:

- `aggregate_gate5_spike.py` uses the Seam JSONL as a sidecar for pair linkage and perturbation spans.
- Seam aggregation now also emits structured sidecars next to the markdown report:
  - `*_seam_pair_summary.csv`
  - `*_seam_family_summary.csv`
- If no canonical dataset hash path is available, the attestation report will note a local fallback identity source instead of claiming full attestation.

## 5) Re-aggregate from existing Gate5 artifacts

```powershell
python tools/aggregate_gate5_spike.py --gate5-out-dir runs/gate5_cfa_spike --out runs/gate5_cfa_spike/gate5_aggregate_report.md --cfa-jsonl data/cfa/cfa_v1.jsonl
```

or

```powershell
python tools/aggregate_gate5_spike.py --gate5-out-dir runs/gate5_seam_spike --out runs/gate5_seam_spike/gate5_aggregate_report.md --surface seam --seam-jsonl data/seam/seam_v0.jsonl
```
