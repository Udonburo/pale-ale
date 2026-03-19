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
- stable world/rendering slots that can be shared across target variants
- label/provenance contract binding
- constitution-only placeholder bindings for world / rendering / target plans

## 3) What the skeleton does not do

It does not yet:

- render natural-language retrieval chunks
- create final answer targets
- create token-level labels
- run any Gate6 or Gate7 candidate

Those belong to the next generation stage, after the constitution is accepted.

## 4) Materialize the first Gate8 generation-stage smoke

```powershell
python tools/materialize_gate8_semiclosed_conflict.py --constitution-dir runs/gate8_constitution_skeleton --out-dir runs/gate8a_generation_smoke --run-id gate8a_generation_smoke
```

Outputs:

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

## 5) What the generation-stage smoke does

The generation-stage smoke still does not run candidate metrics.

It only materializes the four benchmark layers:

- world truth
- retrieval rendering
- answer target
- token/span labels

under the fixed Gate8 constitution.

## 6) Run the fixed Gate8 candidate batch

```powershell
python tools/run_gate8_candidate_batch.py --benchmark-dir runs/gate8a_generation_smoke --out-dir runs/gate8a_candidate_execution --device auto
```

This execution stage keeps the Gate8 constitution frozen:

- candidate set remains `F / gate6f / gate6h / gate7c`
- evaluator vocabulary remains:
  - `global_auprc`
  - `mean_sample_auprc`
  - `hit@10`
  - `first_hit_distance`
  - `mean_delta_max`
  - `mean_delta_p90`
  - `mean_iqr_normalized_delta_max`
  - `mean_top10_inflation`
- aggregation remains banned

The runner performs:

1. teacher-forcing extraction for each Gate8 benchmark row
2. defect-span label materialization into `labels.jsonl`
3. Gate6 native local span build
4. fixed-candidate consumers:
   - `gate6f`
   - `gate6h`
   - `gate7c`
5. Gate8 standing evaluation for:
   - `score_F_gram_loop_v1`
   - `sigma_gap_tailkeep_weighted_gram_loop_v2`
   - `sigma_sqrtgap_tailkeep_object_v2`
   - `progression_anisotropic_closure_v3`

The fixed court is also emitted with explicit label-granularity provenance:

- `F`, `gate6f`, and `gate6h` use `label_token` / `token`
- `gate7c` uses `label_transition` / `transition`
- execution artifacts must state that this is regime-consistent but not same-granularity

Quietness pairing is fixed deterministically as:

- `clean_support` <-> `surface_noisy_clean`
- matched by shared `world_id`
- with distinct `rendering_id` values inside that same world control

Artifacts include:

- `sample_registry.jsonl`
- `quietness_pairs.jsonl`
- `candidate_summary.csv`
- `gate8a_standing_summary.md`
- per-candidate evaluation reports under `evaluations/`

`candidate_summary.csv`, execution `manifest.json`, and per-candidate evaluation manifests now carry
candidate-level `label_key` / `label_granularity` metadata so the Gate8 standing court cannot silently
forget the `transition vs token` caveat.

## 7) One-shot Gate8 scale-up

If the constitution and fixed-set execution path are already accepted, use the one-shot runner:

```powershell
python tools/run_gate8_scaleup.py --run-prefix gate8b_128r --samples-per-cell 32 --device cpu --model-id Qwen/Qwen2.5-0.5B
```

This produces:

- `runs/gate8b_128r_constitution/`
- `runs/gate8b_128r_benchmark/`
- `runs/gate8b_128r_candidate_execution/`

With `samples-per-cell=32`, Gate8 materializes:

- `4 cells`
- `32 rows per cell`
- `128 total benchmark rows`

Use `--skip-execution` when only the larger benchmark generation is needed:

```powershell
python tools/run_gate8_scaleup.py --run-prefix gate8b_128r --samples-per-cell 32 --skip-execution
```
