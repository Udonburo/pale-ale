# Gate8B Execution And Portability

Status: tracked first-pass portability snapshot
Role: tracked Gate8B execution read under one controlled rendering-family shift
Date: 2026-03-21

## 0. Scope

This file records the first executed Gate8B portability read.

It is not:

- a bridge file
- a candidate-promotion file
- a field-aggregation file

It is:

- the tracked read for one frozen-court regime shift

The active evidence package is now:

- `gate8o_128r_briefing_candidate_execution`
- `gate8p_200r_briefing_candidate_execution`

These runs reuse the same fixed candidate court:

- `F` / `score_F_gram_loop_v1`
- `gate6f` / `sigma_gap_tailkeep_weighted_gram_loop_v2`
- `gate6h` / `sigma_sqrtgap_tailkeep_object_v2`
- `gate7c` / `progression_anisotropic_closure_v3`

They also preserve:

- standing evaluator freeze
- aggregation ban
- diagnostic-only bridge status

The only new regime axis in this workstream is:

- `rendering_family_id = briefing_v1`

## 1. Evidence Package

The tracked execution artifacts are:

- `runs/gate8o_128r_briefing_constitution/manifest.json`
- `runs/gate8o_128r_briefing_benchmark/manifest.json`
- `runs/gate8o_128r_briefing_candidate_execution/manifest.json`
- `runs/gate8o_128r_briefing_candidate_execution/candidate_summary.csv`
- `runs/gate8p_200r_briefing_constitution/manifest.json`
- `runs/gate8p_200r_briefing_benchmark/manifest.json`
- `runs/gate8p_200r_briefing_candidate_execution/manifest.json`
- `runs/gate8p_200r_briefing_candidate_execution/candidate_summary.csv`

The execution artifacts now carry:

- top-level `rendering_family_id`
- execution report `rendering_family_id`
- quietness handoff `rendering_family_id`

Both tracked runs bind:

- `code_git_commit = 7567e2d`

## 2. Correct Read

The Gate8B first-pass question was:

- does `gate7c revival` survive regime shift?

On the `briefing_v1` family, the answer is:

- yes, for this first controlled shift

At 128 rows:

- `gate7c direct_global_auprc = 0.204235 > F = 0.185301`
- `gate7c distributed_global_auprc = 0.196939 > F = 0.188251`
- `gate7c quiet_mean_delta_p90 = -0.004292 < F = 0.009822`
- `gate7c quiet_mean_top10_inflation = 2.468750 < F = 2.781250`

At 200 rows:

- `gate7c direct_global_auprc = 0.203008 > F = 0.183225`
- `gate7c distributed_global_auprc = 0.185937 > F = 0.174681`
- `gate7c quiet_mean_delta_p90 = -0.004840 < F = 0.011861`
- `gate7c quiet_mean_top10_inflation = 2.360000 < F = 2.940000`

`gate6f` and `gate6h` remain clearly behind the top pair.

## 3. Earned Statement

The earned first-pass Gate8B sentence is:

- the current `gate7c revival` survives one controlled rendering-family shift under a frozen court

This is stronger than the old Gate8A sentence in one specific way:

- the revival is no longer only attached to the original Gate8 rendering package

It is not stronger in the following ways:

- it does not explain the revival
- it does not erase the bridge failures
- it does not promote `gate7c`
- it does not replace `F`
- it does not prove regime invariance beyond this first family shift

## 4. Relationship To Gate8A

Gate8A and Gate8B now have distinct roles.

Gate8A remains:

- corrected-court execution snapshot
- scale-up and bridge-failure ledger
- `standing strong, bridge unresolved`

Gate8B first pass now adds:

- one executed portability read under `briefing_v1`

So the combined project memory is:

- Gate8A: standing survived court repair and bridge failure
- Gate8B first pass: that standing also survives one controlled rendering-family shift

## 5. Caveats

The following caveats remain active.

- `gate7c` still uses transition labels while `F`, `gate6f`, and `gate6h` use token labels
- this is still a frozen-court comparison, not same-granularity comparison
- only one new rendering family has been tested
- diagnostic bridges remain diagnostic-only and unresolved
- no field aggregation or promotion claim is earned here

## 6. Next Boundary

The next honest move is not:

- bridge rescue
- candidate promotion
- field aggregation

The next honest move is:

- decide whether one more tightly-controlled regime shift is worth the cost

If no further shift is taken immediately, this file is already enough to close Gate8B first pass cleanly.
