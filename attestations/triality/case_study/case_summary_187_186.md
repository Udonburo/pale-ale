# CFA Case Study: sample 187 vs 186

## Pair
- Frustrated sample_id: `187`
- Consistent contrast_sample_id: `186`
- World type: `genealogy`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Aster is the parent of Haven.
2) Haven is the parent of Noble.
Question: Which ancestor relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, Noble is an ancestor of Aster. This follows by transitivity of the parent relation.
```
- Defect spans: `[{"start": 19, "end": 48, "label": "frustration_defect_v1", "text": "Noble is an ancestor of Aster"}]`

### Consistent Answer
```text
Given these facts, Aster is an ancestor of Noble. This follows by transitivity of the parent relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 187 (frustrated) | 1.65922381711855388e-01 | 1.59081996434937617e-01 | 3.51010101010100994e-01 | A | 1.85087719298245607e-01 |
| 186 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 15 | `Ġtrans` | 2.68686431245973800e+00 | 0 |
| 2 | 16 | `itivity` | 2.53268974802492863e+00 | 0 |
| 3 | 8 | `Ġancestor` | 2.52208043495455048e+00 | 1 |
| 4 | 7 | `Ġan` | 2.46953505675891982e+00 | 1 |
| 5 | 11 | `.` | 2.45489369769943488e+00 | 0 |
| 6 | 14 | `Ġby` | 2.43732178251632581e+00 | 0 |
| 7 | 2 | `Ġthese` | 2.35476711282583828e+00 | 0 |
| 8 | 3 | `Ġfacts` | 2.34765049187069064e+00 | 0 |
| 9 | 12 | `ĠThis` | 2.32546696903439010e+00 | 0 |
| 10 | 18 | `Ġthe` | 2.31152563212883866e+00 | 0 |

- Overlap top-10 with defect-transition labels: `2/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample187_vs_186/token_table_187.csv`
- token_table consistent: `runs/cfa_case_study/sample187_vs_186/token_table_186.csv`
- pair overlay CSV: `runs/cfa_case_study/sample187_vs_186/pair_overlay_187_vs_186.csv`
- plot frustrated: `runs/cfa_case_study/sample187_vs_186/plot_case_187.png`
- plot pair: `runs/cfa_case_study/sample187_vs_186/plot_pair_compare_187_186.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
