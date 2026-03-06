# CFA Case Study: sample 75 vs 74

## Pair
- Frustrated sample_id: `75`
- Consistent contrast_sample_id: `74`
- World type: `temporal`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Event Opalx happened before event Vivid.
2) Event Vivid happened before event Beryl.
Question: Which order relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, event Beryl happened before event Opalx. This follows by transitivity of the before relation.
```
- Defect spans: `[{"start": 19, "end": 58, "label": "frustration_defect_v1", "text": "event Beryl happened before event Opalx"}]`

### Consistent Answer
```text
Given these facts, event Opalx happened before event Beryl. This follows by transitivity of the before relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 75 (frustrated) | 2.17784964923837948e-01 | 2.31496682665160936e-01 | 6.47912127814088690e-01 | B | 4.16415445148927754e-01 |
| 74 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 7 | `eryl` | 2.75514914393417376e+00 | 1 |
| 2 | 8 | `Ġhappened` | 2.71340730964012122e+00 | 1 |
| 3 | 18 | `Ġtrans` | 2.70196030821569266e+00 | 0 |
| 4 | 11 | `ĠOp` | 2.69629885616344467e+00 | 1 |
| 5 | 22 | `Ġbefore` | 2.67031968856448554e+00 | 0 |
| 6 | 17 | `Ġby` | 2.63489977694735122e+00 | 0 |
| 7 | 12 | `al` | 2.63478436013281136e+00 | 1 |
| 8 | 6 | `ĠB` | 2.54123904500014941e+00 | 1 |
| 9 | 21 | `Ġthe` | 2.53730257379651025e+00 | 0 |
| 10 | 13 | `x` | 2.52441330290456456e+00 | 1 |

- Overlap top-10 with defect-transition labels: `6/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample75_vs_74/token_table_75.csv`
- token_table consistent: `runs/cfa_case_study/sample75_vs_74/token_table_74.csv`
- pair overlay CSV: `runs/cfa_case_study/sample75_vs_74/pair_overlay_75_vs_74.csv`
- plot frustrated: `runs/cfa_case_study/sample75_vs_74/plot_case_75.png`
- plot pair: `runs/cfa_case_study/sample75_vs_74/plot_pair_compare_75_74.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
