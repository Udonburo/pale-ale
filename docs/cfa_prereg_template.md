# Triality CFA Preregistration Template

## Metadata
- Experiment ID:
- Date:
- Commit SHA:
- Dataset ID: `cfa_v1`
- Model ID:
- Seed:

## Hypothesis
Primary claim: token/span-level triality score detects constructed defect spans better than Shannon-only baselines under closed-world contradiction injection.

## Dataset Protocol
- Worlds: `genealogy`, `temporal`, `reachability`
- Balanced variants: `consistent` / `frustrated`
- Defect labels: `defect_spans` from generator output (SSOT)
- Split policy: fixed deterministic split by `sample_id`

## Fixed Endpoints
- Primary endpoint: choose exactly one (`E` or `F`) before running.
- Baselines: `A:-logprob`, `B:entropy`
- Secondary ablations: `C`, `D`, and whichever of `E/F` is not primary.

## Fixed Evaluation Settings
- `eval_triality_token.py --primary-score <locked>`
- `perm-R=2000`
- `seed=7`
- `min-label-coverage=0.30`
- Transition label mode: `max_pair` (in script)

## Go / No-Go Criteria
- `primary_auprc >= 0.15`
- `delta_auprc_primary_vs_best_baseline >= 0.02`
- `perm_p_empirical <= 0.05`
- Coverage gate pass for every evaluated sample.

## Exclusions (must be declared before run)
- Samples with extraction failure (model/tokenizer unavailable)
- Samples with label coverage below threshold

## Reporting
- Per-sample report files under `attestations/triality/`
- Aggregate summary must include:
  - counts by status (`ok`, `skip_coverage`, `error`)
  - primary AUPRC median/mean
  - delta median/mean
  - min/median `p_emp`
