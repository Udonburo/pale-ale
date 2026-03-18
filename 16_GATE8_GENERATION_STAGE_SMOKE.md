# Gate8 Generation Stage Smoke

Status: Draft
Role: Tracked RFC / first materialized benchmark stage
Date: 2026-03-19

## 0. Scope

This file records the first Gate8 generation-stage smoke.

It does not evaluate candidates yet.

It only proves that the Gate8 constitution can be materialized into explicit:

- world truth
- retrieval rendering
- answer target
- token/span labels

without collapsing the four-layer contract.

## 1. Why This Stage Exists

Gate8 constitution alone is not enough.

The benchmark must also show that:

- the taxonomy can be rendered deterministically
- provenance remains layered
- answer targets are fixed before candidate outputs
- token and span labels can be explained from world truth plus rendering

This stage is the minimum proof of that.

## 2. Inputs

The generation-stage smoke consumes the accepted Gate8 constitution scaffold:

- `manifest.json`
- `conflict_plan.json`
- `label_contract.json`
- `sample_index.jsonl`

from the constitution run.

## 3. Outputs

The materialized smoke emits:

- `manifest.json`
- `conflict_plan.json`
- `label_contract.json`
- `world_plan.json`
- `rendering_plan.json`
- `target_plan.json`
- `sample_index.jsonl`
- `world_truth.jsonl`
- `retrieval_renderings.jsonl`
- `answer_targets.jsonl`
- `benchmark_rows.jsonl`
- `checksums.json`

## 4. What Is Frozen

This stage does not reopen:

- candidate set
- evaluator vocabulary
- aggregation ban
- four-cell taxonomy

Those remain inherited from the constitution.

## 5. What Is New

This stage makes the benchmark concrete enough to inspect:

- which world truth underlies each sample
- which retrieval chunks are support or conflict-bearing
- which answer target is intended
- where the support / conflict / defect spans live in the answer

## 6. Interpretation

This stage is not a benchmark win.

It is a contract win if:

- the four cells remain distinct
- the provenance chain stays auditable
- the answer-target regime stays fixed before scoring
- the labels can be reconstructed from the artifacts alone

If those fail, the benchmark is not ready for candidate execution.
