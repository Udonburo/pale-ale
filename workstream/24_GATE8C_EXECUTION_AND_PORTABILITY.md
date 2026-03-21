# Gate8C Execution And Portability

Status: tracked second-family portability boundary snapshot
Role: tracked Gate8C execution read under one additional controlled rendering-family shift
Date: 2026-03-21

## 0. Scope

This file records the second executed Gate8 portability read.

It is not:

- a bridge file
- a candidate-promotion file
- a field-aggregation file

It is:

- the tracked read for a second frozen-court regime shift

The active evidence package is now:

- `gate8q_128r_transcript_candidate_execution`
- `gate8r_200r_transcript_candidate_execution`

These runs preserve:

- candidates: `F`, `gate6f`, `gate6h`, `gate7c`
- standing evaluator freeze
- aggregation ban
- diagnostic-only bridge status
- quietness pairing rule `shared_world_id_v1`

The only new regime axis in this workstream is:

- `rendering_family_id = transcript_v1`

## 1. Evidence Package

The tracked execution artifacts are:

- `runs/gate8q_128r_transcript_constitution/manifest.json`
- `runs/gate8q_128r_transcript_benchmark/manifest.json`
- `runs/gate8q_128r_transcript_candidate_execution/manifest.json`
- `runs/gate8q_128r_transcript_candidate_execution/candidate_summary.csv`
- `runs/gate8r_200r_transcript_constitution/manifest.json`
- `runs/gate8r_200r_transcript_benchmark/manifest.json`
- `runs/gate8r_200r_transcript_candidate_execution/manifest.json`
- `runs/gate8r_200r_transcript_candidate_execution/candidate_summary.csv`

Both tracked runs bind:

- `rendering_family_id = transcript_v1`
- `code_git_commit = 744bc7a`

## 2. Correct Read

The Gate8C question was:

- does `gate7c revival` survive a second regime shift?

The answer is mixed.

Conflict-side standing survives, but quietness does not.

At 128 rows:

- `gate7c direct_global_auprc = 0.235397 > F = 0.216630`
- `gate7c distributed_global_auprc = 0.178819 > F = 0.172439`
- `F quiet_mean_delta_p90 = -0.011465 < gate7c = 0.002489`
- `F quiet_mean_top10_inflation = 2.375000 < gate7c = 2.812500`

At 200 rows:

- `gate7c direct_global_auprc = 0.238740 > F = 0.214732`
- `gate7c distributed_global_auprc = 0.171347 > F = 0.163006`
- `F quiet_mean_delta_p90 = -0.012448 < gate7c = 0.002495`
- `F quiet_mean_top10_inflation = 2.420000 < gate7c = 2.840000`

`gate6f` and `gate6h` remain behind the top pair, but that is not the main story of this pass.

## 3. Earned Statement

The earned Gate8C sentence is:

- the conflict-side revival survives a second controlled rendering-family shift, but quietness collapses there

This is not a clean second portability win.

It is a boundary read.

What it earns is:

- evidence that portability is not simply monotone across rendering-family shifts
- a concrete regime boundary where the conflict-side revival and quietness no longer travel together

What it does not earn is:

- bridge rescue
- field aggregation
- candidate promotion
- replacement of `F`
- a claim that `gate7c` is robust across arbitrary rhetorical packaging

## 4. Relationship To Gate8B

Gate8B and Gate8C now play different roles.

Gate8B established:

- one clean portability success under `briefing_v1`

Gate8C now establishes:

- the next rendering-family shift preserves the conflict-side revival
- but it breaks the quietness side

So the combined memory is:

- the revival is not local to the original Gate8 rendering family
- but its portability boundary appears sooner than full regime-invariance language would allow

## 5. Caveats

The following caveats remain active.

- `gate7c` still uses transition labels while `F`, `gate6f`, and `gate6h` use token labels
- this remains a frozen-court comparison, not same-granularity comparison
- bridge diagnostics remain diagnostic-only and unresolved
- this is still one family at a time, not broad benchmark expansion

## 6. Next Boundary

The next honest move is still not:

- bridge rescue
- field aggregation
- broader benchmark expansion

The next honest options are only:

- stop here and preserve Gate8C as the portability-boundary read
- or, if another shift is truly worth the cost, spec one more rendering family under the same frozen court
