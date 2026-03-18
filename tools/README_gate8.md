# Gate8 Benchmark Constitution Workflow

Gate8 begins with benchmark constitution, not immediate full data generation.

The purpose of this stage is to freeze:

- candidate set
- evaluator vocabulary
- conflict taxonomy
- label and provenance contract

before any large benchmark batch is generated.

## 1) Generate the Gate8 skeleton scaffold

```powershell
python tools/generate_gate8_semiclosed_conflict.py --out-dir runs/gate8_constitution_skeleton --run-id gate8_constitution_skeleton --samples-per-cell 8
```

Outputs:

- `manifest.json`
- `conflict_plan.json`
- `label_contract.json`
- `world_plan.json`
- `rendering_plan.json`
- `target_plan.json`
- `sample_index.jsonl`
- `checksums.json`

## 2) What the skeleton does

The skeleton does not yet generate final retrieval passages or answer strings.

It only emits a deterministic scaffold for:

- the four-cell conflict taxonomy
- the fixed comparison set
- planned sample rows with stable IDs
- label/provenance contract binding
- constitution-only placeholder bindings for world / rendering / target plans

## 3) What the skeleton does not do

It does not yet:

- render natural-language retrieval chunks
- create final answer targets
- create token-level labels
- run any Gate6 or Gate7 candidate

Those belong to the next generation stage, after the constitution is accepted.
