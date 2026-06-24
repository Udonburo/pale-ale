# Gate12C First Real-Artifact Feasibility Census Memo

Status: bounded empirical census memo
Role: first real Gate12C-0 artifact-surface census over the canonical twelve Gate12A discrete-connection inputs, not a Gate12C-1 implementation, not compressed associator evidence, not a Gate12B overlay, and not a physical nonassociativity claim
Date: 2026-06-25

This memo records the first real-artifact execution of:

- `231_GATE12C_ASSOCIATOR_FEASIBILITY_AND_EQUAL_RANK_ALPHA_CONTRACT.md`
- `232_GATE12C_FIRST_REAL_ARTIFACT_FEASIBILITY_CENSUS_PLAN.md`
- `tools/inspect_gate12c_associator_feasibility.py`

## 0. Execution Boundary

This run executed Gate12C-0 only.

It did not:

- implement Gate12C-1
- calculate compressed associator values
- add rectangular rank-mismatch support
- change Gate12A or Gate12B semantics
- consume Gate12B high/flat candidates
- run model inference
- regenerate Gate8 or Gate12A artifacts
- tune thresholds after seeing results
- commit generated `runs/` outputs

The only tracked implementation change before census execution was report-boundary wording in the generated Gate12C-0 read file.

## 1. Builder Revision

The census builder was committed before real-artifact execution.

```text
builder_commit = c1186eb58e9baad01e5a20cfc1d1283e9016d91a
builder_commit_subject = Harden Gate12C-0 real-run report boundary
builder_script = tools/inspect_gate12c_associator_feasibility.py
builder_script_sha256 = af03afc8d320c9d86f1c35128b19ff0a22d9d860490c7befcfac4ea5e4be8c92
```

The wording-only change did not alter schema version, method id, calculations, tolerances, eligibility, status rules, output field names, or artifact structure.

### 1.1 Integration Provenance Note

PR `#113` was integrated to `main` through a squash merge after the census completed.

```text
execution_builder_commit = c1186eb58e9baad01e5a20cfc1d1283e9016d91a
integration_commit = 655116f7246ebf3051b0a78404353d0f9072b678
integration_mode = squash
builder_script_sha256 = af03afc8d320c9d86f1c35128b19ff0a22d9d860490c7befcfac4ea5e4be8c92
```

This does not change the census result. It records that the execution builder commit is preserved through PR provenance rather than as a first-parent `main` ancestor.

## 2. Frozen Settings

The run used the merged Gate12C-0 defaults:

| Setting | Value |
| --- | ---: |
| `tau_overlap_sv_min` | `1e-8` |
| `tau_overlap_singular_value_abs_error` | `1e-8` |
| `tau_transport_reconstruction_fro` | `1e-8` |
| `tau_ordinary_associator_fro` | `1e-10` |
| `tau_split_rel` | `1e-3` |
| `epsilon` | `1e-12` |
| `min_eligible_cycles` | `1` |

`min_eligible_cycles = 1` was an existence-only threshold. It was not a publication threshold, a statistical sufficiency threshold, a theory-selection threshold, or a Gate12C-1 success criterion.

## 3. Inventory

All twelve intended canonical Gate12A source directories were found. All had the seven required Gate12C-0 input files, and source `checksums.json` verification passed for those required files.

| Case | Model | Family | Source path | Source run id | Source checksums |
| --- | --- | --- | --- | --- | --- |
| case_01 | `qwen_qwen2_5_0_5b` | `transcript_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k` | `pass` |
| case_02 | `qwen_qwen2_5_0_5b` | `briefing_200r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k` | `pass` |
| case_03 | `qwen_qwen2_5_0_5b` | `archive_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k` | `pass` |
| case_04 | `qwen_qwen2_5_3b_instruct` | `transcript_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k` | `pass` |
| case_05 | `qwen_qwen2_5_3b_instruct` | `briefing_200r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k` | `pass` |
| case_06 | `qwen_qwen2_5_3b_instruct` | `archive_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k` | `pass` |
| case_07 | `meta_llama_llama_3_2_3b_instruct` | `transcript_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k` | `pass` |
| case_08 | `meta_llama_llama_3_2_3b_instruct` | `briefing_200r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k` | `pass` |
| case_09 | `meta_llama_llama_3_2_3b_instruct` | `archive_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k` | `pass` |
| case_10 | `qwen_qwen3_4b` | `transcript_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k` | `pass` |
| case_11 | `qwen_qwen3_4b` | `briefing_200r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k` | `pass` |
| case_12 | `qwen_qwen3_4b` | `archive_128r` | `C:\Users\aoika\Documents\GitHub\pale-ale\runs\gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k` | `gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k` | `pass` |

No intended case was missing or incomplete.

## 4. Per-Case Census

All rows used `min_eligible_cycles = 1`.

| Case | Contract | Empirical | Residual triangles | Equal-rank | Rank 1 | Rank 2 | Rank 3 | Rank >=4 | Eligible cycles | Probes | Stable both | Near left | Near right | Near both | Stable cycles | Immutability | Output checksums |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| case_01 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_02 | `pass` | `pass_declared_minimum` | 500 | 500 | 0 | 0 | 500 | 0 | 500 | 3000 | 3000 | 0 | 0 | 0 | 500 | `pass` | `pass` |
| case_03 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_04 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_05 | `pass` | `pass_declared_minimum` | 500 | 500 | 0 | 0 | 500 | 0 | 500 | 3000 | 3000 | 0 | 0 | 0 | 500 | `pass` | `pass` |
| case_06 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_07 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_08 | `pass` | `pass_declared_minimum` | 500 | 500 | 0 | 0 | 500 | 0 | 500 | 3000 | 3000 | 0 | 0 | 0 | 500 | `pass` | `pass` |
| case_09 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_10 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |
| case_11 | `pass` | `pass_declared_minimum` | 500 | 500 | 0 | 0 | 500 | 0 | 500 | 3000 | 3000 | 0 | 0 | 0 | 500 | `pass` | `pass` |
| case_12 | `pass` | `pass_declared_minimum` | 320 | 320 | 0 | 0 | 320 | 0 | 320 | 1920 | 1920 | 0 | 0 | 0 | 320 | `pass` | `pass` |

## 5. Reconstruction and Associativity Nulls

| Case | Reconstructed edges | Failed edge reconstruction | SV max abs err | Transport max Fro err | Ordinary assoc max Fro | Ordinary assoc failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case_01 | 4205 | 0 | 0.0 | 0.0 | 3.3518004512569147e-16 | 0 |
| case_02 | 6266 | 0 | 0.0 | 0.0 | 3.521787733787174e-16 | 0 |
| case_03 | 3981 | 0 | 0.0 | 0.0 | 2.9317339971313326e-16 | 0 |
| case_04 | 4205 | 0 | 0.0 | 0.0 | 3.526409525466914e-16 | 0 |
| case_05 | 6266 | 0 | 0.0 | 0.0 | 3.5633745601369095e-16 | 0 |
| case_06 | 3981 | 0 | 0.0 | 0.0 | 3.55114462729738e-16 | 0 |
| case_07 | 4205 | 0 | 0.0 | 0.0 | 3.3972246307450016e-16 | 0 |
| case_08 | 6266 | 0 | 0.0 | 0.0 | 3.1585784838245976e-16 | 0 |
| case_09 | 3981 | 0 | 0.0 | 0.0 | 3.4359788356585895e-16 | 0 |
| case_10 | 4205 | 0 | 0.0 | 0.0 | 2.9010883917888854e-16 | 0 |
| case_11 | 6266 | 0 | 0.0 | 0.0 | 3.525203960930295e-16 | 0 |
| case_12 | 3981 | 0 | 0.0 | 0.0 | 2.871268686172127e-16 | 0 |

Every case passed the contract boundary:

- `failed_edge_reconstruction_count = 0` for all cases
- `overlap_singular_value_max_abs_error = 0.0` for all cases
- `transport_reconstruction_max_fro_error = 0.0` for all cases
- `ordinary_associator_failed_count = 0` for all cases

## 6. Aggregate Census

| Quantity | Value |
| --- | ---: |
| Intended cases | 12 |
| Found complete cases | 12 |
| Executed or verified output cases | 12 |
| Contract pass cases | 12 |
| Empirical pass declared-minimum cases | 12 |
| Total residual-bearing explicit triangles | 4,560 |
| Total defined equal-rank triangles | 4,560 |
| Total rank-1 triangles | 0 |
| Total rank-2 triangles | 0 |
| Total rank-3 triangles | 4,560 |
| Total rank >=4 triangles | 0 |
| Total eligible equal-rank common-rank >=2 cycles | 4,560 |
| Total probe configurations | 27,360 |
| Total stable-both inner cuts | 27,360 |
| Total near-degenerate left cuts | 0 |
| Total near-degenerate right cuts | 0 |
| Total near-degenerate both cuts | 0 |
| Total cycles with at least one stable q | 4,560 |

The model/family breadth is broad: all four predeclared model lines and all three predeclared rendering families passed the declared existence threshold under the frozen Gate12C-0 settings.

## 7. Generated Local Outputs

Generated Gate12C outputs were written under the isolated census worktree and are intentionally not tracked:

```text
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen2_5_0_5b_transcript_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen2_5_0_5b_briefing_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen2_5_0_5b_archive_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen2_5_3b_instruct_transcript_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen2_5_3b_instruct_briefing_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen2_5_3b_instruct_archive_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_llama3_2_3b_instruct_transcript_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_llama3_2_3b_instruct_briefing_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_llama3_2_3b_instruct_archive_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen3_4b_transcript_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen3_4b_briefing_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_qwen3_4b_archive_min1
C:\Users\aoika\Documents\GitHub\pale-ale-gate12c-census\runs\gate12c_associator_feasibility_canonical_grid_min1
```

The local cross-run summary directory contains:

```text
manifest.json
gate12c_canonical_grid_inventory.csv
gate12c_canonical_grid_summary.csv
gate12c_canonical_grid_summary.json
gate12c_canonical_grid_read.md
checksums.json
```

## 8. Interpretation

This census establishes Outcome A from `232`: eligible surface exists broadly.

The canonical real Gate12A grid exposes a stable equal-rank, common-rank `r = 3` surface under the frozen Gate12C-0 settings. Every predeclared model/family case has at least one eligible cycle with stable nontrivial cuts, and in this census all probe configurations were stable under `tau_split_rel = 1e-3`.

This does not establish a nonzero compressed associator. Gate12C-0 only counted artifact-surface eligibility and stable SVD cuts. It did not compute parenthesized compressed overlap compositions.

## 9. Explicit Non-Claims

This memo does not claim:

- Type-III evidence
- nonassociative physics
- compressed associator magnitude
- Gate12C-1 implementation readiness without a separate implementation contract
- rectangular rank-mismatch feasibility
- Gate12B high/flat overlay relevance
- source-facing interpretation
- model-quality ranking
- threshold-robustness beyond the frozen settings listed above

Stable-cut availability is a domain-admissibility result, not associator evidence.

## 10. Bounded Next-Step Decision

The data support drafting a Gate12C-1 equal-rank alpha implementation scope over the predeclared broad canonical surface.

That next scope should remain equal-rank only, common rank `r = 3` for this artifact family, and should preserve the controls in `231`: ordinary associativity null, no-compression null, gauge covariance, spectrum-preserving orientation null, separation from Gate12A holonomy residuals, and separation from pairwise edge defects.

Rectangular rank-mismatch support remains deferred.
