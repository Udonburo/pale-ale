# CFA Case Study: sample 141 vs 140

## Pair
- Frustrated sample_id: `141`
- Consistent contrast_sample_id: `140`
- World type: `temporal`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Event Jasper happened before event Quill.
2) Event Quill happened before event Waltz.
Question: Which order relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, event Waltz happened before event Jasper. This follows by transitivity of the before relation.
```
- Defect spans: `[{"start": 19, "end": 59, "label": "frustration_defect_v1", "text": "event Waltz happened before event Jasper"}]`

### Consistent Answer
```text
Given these facts, event Jasper happened before event Waltz. This follows by transitivity of the before relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 141 (frustrated) | 1.88261690947112953e-01 | 2.04523234414538790e-01 | 6.75703614779245010e-01 | B | 4.71180380364706219e-01 |
| 140 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 7 | `z` | 2.78571784383798748e+00 | 1 |
| 2 | 8 | `Ġhappened` | 2.78488539546775460e+00 | 1 |
| 3 | 6 | `ĠWalt` | 2.73250368785327424e+00 | 1 |
| 4 | 16 | `Ġtrans` | 2.68147278625115026e+00 | 0 |
| 5 | 20 | `Ġbefore` | 2.66864125793128082e+00 | 0 |
| 6 | 15 | `Ġby` | 2.65308035600930925e+00 | 0 |
| 7 | 11 | `ĠJasper` | 2.62504222146457877e+00 | 1 |
| 8 | 17 | `itivity` | 2.62492154628981122e+00 | 0 |
| 9 | 19 | `Ġthe` | 2.60202731555743938e+00 | 0 |
| 10 | 3 | `Ġfacts` | 2.51506539614267410e+00 | 0 |

- Overlap top-10 with defect-transition labels: `4/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample141_vs_140/token_table_141.csv`
- token_table consistent: `runs/cfa_case_study/sample141_vs_140/token_table_140.csv`
- pair overlay CSV: `runs/cfa_case_study/sample141_vs_140/pair_overlay_141_vs_140.csv`
- plot frustrated: `runs/cfa_case_study/sample141_vs_140/plot_case_141.png`
- plot pair: `runs/cfa_case_study/sample141_vs_140/plot_pair_compare_141_140.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
