# CFA Case Study: sample 109 vs 108

## Pair
- Frustrated sample_id: `109`
- Consistent contrast_sample_id: `108`
- World type: `genealogy`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Noble is the parent of Umber.
2) Umber is the parent of Aster.
Question: Which ancestor relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, Aster is an ancestor of Noble. This follows by transitivity of the parent relation.
```
- Defect spans: `[{"start": 19, "end": 48, "label": "frustration_defect_v1", "text": "Aster is an ancestor of Noble"}]`

### Consistent Answer
```text
Given these facts, Noble is an ancestor of Aster. This follows by transitivity of the parent relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 109 (frustrated) | 2.21795397584871268e-01 | 3.47853926259499069e-01 | 3.95039682539682580e-01 | B | 4.71857562801835106e-02 |
| 108 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 15 | `Ġtrans` | 2.75397586576977949e+00 | 0 |
| 2 | 14 | `Ġby` | 2.51797508835977002e+00 | 0 |
| 3 | 8 | `Ġancestor` | 2.50835886678605569e+00 | 1 |
| 4 | 10 | `ĠNoble` | 2.47345937821044970e+00 | 1 |
| 5 | 16 | `itivity` | 2.46906509813365060e+00 | 0 |
| 6 | 12 | `ĠThis` | 2.40030231348962353e+00 | 0 |
| 7 | 7 | `Ġan` | 2.36278318703086221e+00 | 1 |
| 8 | 11 | `.` | 2.36209139146409131e+00 | 0 |
| 9 | 18 | `Ġthe` | 2.28346138429508505e+00 | 0 |
| 10 | 9 | `Ġof` | 2.25854246336566522e+00 | 1 |

- Overlap top-10 with defect-transition labels: `4/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample109_vs_108/token_table_109.csv`
- token_table consistent: `runs/cfa_case_study/sample109_vs_108/token_table_108.csv`
- pair overlay CSV: `runs/cfa_case_study/sample109_vs_108/pair_overlay_109_vs_108.csv`
- plot frustrated: `runs/cfa_case_study/sample109_vs_108/plot_case_109.png`
- plot pair: `runs/cfa_case_study/sample109_vs_108/plot_pair_compare_109_108.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
