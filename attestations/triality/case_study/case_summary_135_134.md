# CFA Case Study: sample 135 vs 134

## Pair
- Frustrated sample_id: `135`
- Consistent contrast_sample_id: `134`
- World type: `temporal`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Event Aster happened before event Haven.
2) Event Haven happened before event Noble.
Question: Which order relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, event Noble happened before event Aster. This follows by transitivity of the before relation.
```
- Defect spans: `[{"start": 19, "end": 58, "label": "frustration_defect_v1", "text": "event Noble happened before event Aster"}]`

### Consistent Answer
```text
Given these facts, event Aster happened before event Noble. This follows by transitivity of the before relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 135 (frustrated) | 1.92357609710550881e-01 | 2.58073038073038052e-01 | 2.89711019974177864e-01 | B | 3.16379819011398111e-02 |
| 134 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 15 | `Ġtrans` | 2.72169621559501262e+00 | 0 |
| 2 | 18 | `Ġthe` | 2.70058305473520655e+00 | 0 |
| 3 | 19 | `Ġbefore` | 2.67415638326700478e+00 | 0 |
| 4 | 7 | `Ġhappened` | 2.67241837947389582e+00 | 1 |
| 5 | 14 | `Ġby` | 2.62902134835396772e+00 | 0 |
| 6 | 6 | `ĠNoble` | 2.56589294870831042e+00 | 1 |
| 7 | 3 | `Ġfacts` | 2.53385910732399822e+00 | 0 |
| 8 | 16 | `itivity` | 2.52623364311077170e+00 | 0 |
| 9 | 11 | `.` | 2.36245527052433069e+00 | 0 |
| 10 | 12 | `ĠThis` | 2.34431164376433010e+00 | 0 |

- Overlap top-10 with defect-transition labels: `2/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample135_vs_134/token_table_135.csv`
- token_table consistent: `runs/cfa_case_study/sample135_vs_134/token_table_134.csv`
- pair overlay CSV: `runs/cfa_case_study/sample135_vs_134/pair_overlay_135_vs_134.csv`
- plot frustrated: `runs/cfa_case_study/sample135_vs_134/plot_case_135.png`
- plot pair: `runs/cfa_case_study/sample135_vs_134/plot_pair_compare_135_134.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
