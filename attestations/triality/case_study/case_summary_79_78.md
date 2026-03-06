# CFA Case Study: sample 79 vs 78

## Pair
- Frustrated sample_id: `79`
- Consistent contrast_sample_id: `78`
- World type: `genealogy`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Umber is the parent of Beryl.
2) Beryl is the parent of Haven.
Question: Which ancestor relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, Haven is an ancestor of Umber. This follows by transitivity of the parent relation.
```
- Defect spans: `[{"start": 19, "end": 48, "label": "frustration_defect_v1", "text": "Haven is an ancestor of Umber"}]`

### Consistent Answer
```text
Given these facts, Umber is an ancestor of Haven. This follows by transitivity of the parent relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 79 (frustrated) | 2.13103080494384867e-01 | 1.82672838222037309e-01 | 3.91149536737772030e-01 | A | 1.78046456243387163e-01 |
| 78 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 15 | `Ġby` | 2.55051388822339486e+00 | 0 |
| 2 | 16 | `Ġtrans` | 2.54167337174843055e+00 | 0 |
| 3 | 9 | `Ġof` | 2.50574018561621648e+00 | 1 |
| 4 | 13 | `ĠThis` | 2.49026528904596134e+00 | 0 |
| 5 | 8 | `Ġancestor` | 2.44503374826648523e+00 | 1 |
| 6 | 10 | `ĠUm` | 2.34128436990805966e+00 | 1 |
| 7 | 1 | `iven` | 2.32814163071259372e+00 | 0 |
| 8 | 20 | `Ġparent` | 2.29839376390853722e+00 | 0 |
| 9 | 12 | `.` | 2.28759956496829897e+00 | 0 |
| 10 | 19 | `Ġthe` | 2.28579808145099506e+00 | 0 |

- Overlap top-10 with defect-transition labels: `3/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample79_vs_78/token_table_79.csv`
- token_table consistent: `runs/cfa_case_study/sample79_vs_78/token_table_78.csv`
- pair overlay CSV: `runs/cfa_case_study/sample79_vs_78/pair_overlay_79_vs_78.csv`
- plot frustrated: `runs/cfa_case_study/sample79_vs_78/plot_case_79.png`
- plot pair: `runs/cfa_case_study/sample79_vs_78/plot_pair_compare_79_78.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
