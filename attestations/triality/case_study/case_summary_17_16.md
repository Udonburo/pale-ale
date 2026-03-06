# CFA Case Study: sample 17 vs 16

## Pair
- Frustrated sample_id: `17`
- Consistent contrast_sample_id: `16`
- World type: `reachability`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) There is a directed edge from Frost to Mirth.
2) There is a directed edge from Mirth to Solis.
Question: Which path relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, a directed path exists from Solis to Frost. This follows by transitivity of path reachability.
```
- Defect spans: `[{"start": 19, "end": 61, "label": "frustration_defect_v1", "text": "a directed path exists from Solis to Frost"}]`

### Consistent Answer
```text
Given these facts, a directed path exists from Frost to Solis. This follows by transitivity of path reachability.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 17 (frustrated) | 2.76480614973262040e-01 | 2.47719419766381038e-01 | 3.31209873057699167e-01 | A | 5.47292580844371268e-02 |
| 16 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 23 | `ability` | 2.77849269201910953e+00 | 0 |
| 2 | 21 | `Ġpath` | 2.76643801646828980e+00 | 0 |
| 3 | 9 | `Ġfrom` | 2.74462356312261768e+00 | 1 |
| 4 | 17 | `Ġby` | 2.62137002213012238e+00 | 0 |
| 5 | 18 | `Ġtrans` | 2.57468266755969299e+00 | 0 |
| 6 | 6 | `Ġdirected` | 2.55103370050832767e+00 | 1 |
| 7 | 19 | `itivity` | 2.46825752137028287e+00 | 0 |
| 8 | 10 | `ĠSol` | 2.38208240231029045e+00 | 1 |
| 9 | 1 | `iven` | 2.33159695394928956e+00 | 0 |
| 10 | 3 | `Ġfacts` | 2.32083082321973144e+00 | 0 |

- Overlap top-10 with defect-transition labels: `3/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample17_vs_16/token_table_17.csv`
- token_table consistent: `runs/cfa_case_study/sample17_vs_16/token_table_16.csv`
- pair overlay CSV: `runs/cfa_case_study/sample17_vs_16/pair_overlay_17_vs_16.csv`
- plot frustrated: `runs/cfa_case_study/sample17_vs_16/plot_case_17.png`
- plot pair: `runs/cfa_case_study/sample17_vs_16/plot_pair_compare_17_16.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
