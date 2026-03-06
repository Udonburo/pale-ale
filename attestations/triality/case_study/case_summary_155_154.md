# CFA Case Study: sample 155 vs 154

## Pair
- Frustrated sample_id: `155`
- Consistent contrast_sample_id: `154`
- World type: `reachability`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) There is a directed edge from Ember to Lumen.
2) There is a directed edge from Lumen to Riven.
Question: Which path relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, a directed path exists from Riven to Ember. This follows by transitivity of path reachability.
```
- Defect spans: `[{"start": 19, "end": 61, "label": "frustration_defect_v1", "text": "a directed path exists from Riven to Ember"}]`

### Consistent Answer
```text
Given these facts, a directed path exists from Ember to Riven. This follows by transitivity of path reachability.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 155 (frustrated) | 2.52811147186147167e-01 | 2.58136086433047696e-01 | 3.11736291084117212e-01 | B | 5.36002046510695163e-02 |
| 154 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 23 | `ability` | 2.77568011132180725e+00 | 0 |
| 2 | 21 | `Ġpath` | 2.75895093307361172e+00 | 0 |
| 3 | 17 | `Ġby` | 2.64968801572222024e+00 | 0 |
| 4 | 18 | `Ġtrans` | 2.63209802069828136e+00 | 0 |
| 5 | 6 | `Ġdirected` | 2.57011998343693637e+00 | 1 |
| 6 | 19 | `itivity` | 2.49429225846518232e+00 | 0 |
| 7 | 10 | `ĠR` | 2.44021947072302492e+00 | 1 |
| 8 | 3 | `Ġfacts` | 2.42128633486929079e+00 | 0 |
| 9 | 15 | `ĠThis` | 2.35276883566273209e+00 | 0 |
| 10 | 2 | `Ġthese` | 2.31820647636589250e+00 | 0 |

- Overlap top-10 with defect-transition labels: `2/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample155_vs_154/token_table_155.csv`
- token_table consistent: `runs/cfa_case_study/sample155_vs_154/token_table_154.csv`
- pair overlay CSV: `runs/cfa_case_study/sample155_vs_154/pair_overlay_155_vs_154.csv`
- plot frustrated: `runs/cfa_case_study/sample155_vs_154/plot_case_155.png`
- plot pair: `runs/cfa_case_study/sample155_vs_154/plot_pair_compare_155_154.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
