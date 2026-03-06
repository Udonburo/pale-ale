# CFA Case Study: sample 117 vs 116

## Pair
- Frustrated sample_id: `117`
- Consistent contrast_sample_id: `116`
- World type: `temporal`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Event Zorin happened before event Grove.
2) Event Grove happened before event Mirth.
Question: Which order relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, event Mirth happened before event Zorin. This follows by transitivity of the before relation.
```
- Defect spans: `[{"start": 19, "end": 58, "label": "frustration_defect_v1", "text": "event Mirth happened before event Zorin"}]`

### Consistent Answer
```text
Given these facts, event Zorin happened before event Mirth. This follows by transitivity of the before relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 117 (frustrated) | 2.17858064339042617e-01 | 2.23587231005709292e-01 | 6.21996376898337755e-01 | B | 3.98409145892628436e-01 |
| 116 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 7 | `irth` | 2.81122047233582695e+00 | 1 |
| 2 | 8 | `Ġhappened` | 2.73131135432965166e+00 | 1 |
| 3 | 6 | `ĠM` | 2.72727869567332570e+00 | 1 |
| 4 | 22 | `Ġbefore` | 2.68747359674505359e+00 | 0 |
| 5 | 17 | `Ġby` | 2.64231052212904771e+00 | 0 |
| 6 | 1 | `iven` | 2.62448293847565894e+00 | 0 |
| 7 | 18 | `Ġtrans` | 2.62371180569155094e+00 | 0 |
| 8 | 10 | `Ġevent` | 2.53586401822133656e+00 | 1 |
| 9 | 21 | `Ġthe` | 2.53535866651960484e+00 | 0 |
| 10 | 19 | `itivity` | 2.46663652897218721e+00 | 0 |

- Overlap top-10 with defect-transition labels: `4/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample117_vs_116/token_table_117.csv`
- token_table consistent: `runs/cfa_case_study/sample117_vs_116/token_table_116.csv`
- pair overlay CSV: `runs/cfa_case_study/sample117_vs_116/pair_overlay_117_vs_116.csv`
- plot frustrated: `runs/cfa_case_study/sample117_vs_116/plot_case_117.png`
- plot pair: `runs/cfa_case_study/sample117_vs_116/plot_pair_compare_117_116.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
