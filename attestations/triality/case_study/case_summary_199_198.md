# CFA Case Study: sample 199 vs 198

## Pair
- Frustrated sample_id: `199`
- Consistent contrast_sample_id: `198`
- World type: `genealogy`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Solis is the parent of Zorin.
2) Zorin is the parent of Frost.
Question: Which ancestor relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, Frost is an ancestor of Solis. This follows by transitivity of the parent relation.
```
- Defect spans: `[{"start": 19, "end": 48, "label": "frustration_defect_v1", "text": "Frost is an ancestor of Solis"}]`

### Consistent Answer
```text
Given these facts, Solis is an ancestor of Frost. This follows by transitivity of the parent relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 199 (frustrated) | 2.01739444130748485e-01 | 1.80014845200267226e-01 | 3.75340136054421736e-01 | A | 1.73600691923673250e-01 |
| 198 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 16 | `Ġtrans` | 2.62559709804156238e+00 | 0 |
| 2 | 10 | `ĠSol` | 2.53651369312696540e+00 | 1 |
| 3 | 13 | `ĠThis` | 2.52379212166020794e+00 | 0 |
| 4 | 8 | `Ġancestor` | 2.50879007884791294e+00 | 1 |
| 5 | 15 | `Ġby` | 2.49422300138095299e+00 | 0 |
| 6 | 17 | `itivity` | 2.41607767223354397e+00 | 0 |
| 7 | 1 | `iven` | 2.38109642114086739e+00 | 0 |
| 8 | 2 | `Ġthese` | 2.31163137271963759e+00 | 0 |
| 9 | 19 | `Ġthe` | 2.25379980982445405e+00 | 0 |
| 10 | 7 | `Ġan` | 2.22040236086905107e+00 | 1 |

- Overlap top-10 with defect-transition labels: `3/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample199_vs_198/token_table_199.csv`
- token_table consistent: `runs/cfa_case_study/sample199_vs_198/token_table_198.csv`
- pair overlay CSV: `runs/cfa_case_study/sample199_vs_198/pair_overlay_199_vs_198.csv`
- plot frustrated: `runs/cfa_case_study/sample199_vs_198/plot_case_199.png`
- plot pair: `runs/cfa_case_study/sample199_vs_198/plot_pair_compare_199_198.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
