# CFA Case Study: sample 153 vs 152

## Pair
- Frustrated sample_id: `153`
- Consistent contrast_sample_id: `152`
- World type: `temporal`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Event Beryl happened before event Ivory.
2) Event Ivory happened before event Opalx.
Question: Which order relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, event Opalx happened before event Beryl. This follows by transitivity of the before relation.
```
- Defect spans: `[{"start": 19, "end": 58, "label": "frustration_defect_v1", "text": "event Opalx happened before event Beryl"}]`

### Consistent Answer
```text
Given these facts, event Beryl happened before event Opalx. This follows by transitivity of the before relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 153 (frustrated) | 2.18107385558758582e-01 | 2.28248980075353103e-01 | 6.53511303511303598e-01 | B | 4.25262323435950496e-01 |
| 152 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 11 | `Ġevent` | 2.71215564532689024e+00 | 1 |
| 2 | 22 | `Ġbefore` | 2.68372506909541464e+00 | 0 |
| 3 | 13 | `eryl` | 2.68261475438829855e+00 | 1 |
| 4 | 8 | `x` | 2.67755415023189691e+00 | 1 |
| 5 | 9 | `Ġhappened` | 2.66227306619031268e+00 | 1 |
| 6 | 18 | `Ġtrans` | 2.63819956357090746e+00 | 0 |
| 7 | 17 | `Ġby` | 2.62438095741900135e+00 | 0 |
| 8 | 21 | `Ġthe` | 2.58478461104847224e+00 | 0 |
| 9 | 7 | `al` | 2.51856323673224036e+00 | 1 |
| 10 | 12 | `ĠB` | 2.48979653567203574e+00 | 1 |

- Overlap top-10 with defect-transition labels: `6/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample153_vs_152/token_table_153.csv`
- token_table consistent: `runs/cfa_case_study/sample153_vs_152/token_table_152.csv`
- pair overlay CSV: `runs/cfa_case_study/sample153_vs_152/pair_overlay_153_vs_152.csv`
- plot frustrated: `runs/cfa_case_study/sample153_vs_152/plot_case_153.png`
- plot pair: `runs/cfa_case_study/sample153_vs_152/plot_pair_compare_153_152.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
