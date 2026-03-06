# CFA Case Study: sample 139 vs 138

## Pair
- Frustrated sample_id: `139`
- Consistent contrast_sample_id: `138`
- World type: `genealogy`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Grove is the parent of Noble.
2) Noble is the parent of Thorn.
Question: Which ancestor relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, Thorn is an ancestor of Grove. This follows by transitivity of the parent relation.
```
- Defect spans: `[{"start": 19, "end": 48, "label": "frustration_defect_v1", "text": "Thorn is an ancestor of Grove"}]`

### Consistent Answer
```text
Given these facts, Grove is an ancestor of Thorn. This follows by transitivity of the parent relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 139 (frustrated) | 1.75128730918204578e-01 | 1.59796282149223312e-01 | 3.56837606837606847e-01 | A | 1.81708875919402268e-01 |
| 138 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 1 | `iven` | 2.67853480611771477e+00 | 0 |
| 2 | 15 | `Ġtrans` | 2.65505946688594374e+00 | 0 |
| 3 | 10 | `ĠGrove` | 2.62695537178452465e+00 | 1 |
| 4 | 8 | `Ġancestor` | 2.57952570386174163e+00 | 1 |
| 5 | 14 | `Ġby` | 2.55362665596032290e+00 | 0 |
| 6 | 12 | `ĠThis` | 2.46364374774753347e+00 | 0 |
| 7 | 16 | `itivity` | 2.41440139694650480e+00 | 0 |
| 8 | 2 | `Ġthese` | 2.26753918088905770e+00 | 0 |
| 9 | 7 | `Ġan` | 2.26460382848926489e+00 | 1 |
| 10 | 18 | `Ġthe` | 2.21358564279831160e+00 | 0 |

- Overlap top-10 with defect-transition labels: `3/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample139_vs_138/token_table_139.csv`
- token_table consistent: `runs/cfa_case_study/sample139_vs_138/token_table_138.csv`
- pair overlay CSV: `runs/cfa_case_study/sample139_vs_138/pair_overlay_139_vs_138.csv`
- plot frustrated: `runs/cfa_case_study/sample139_vs_138/plot_case_139.png`
- plot pair: `runs/cfa_case_study/sample139_vs_138/plot_pair_compare_139_138.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
