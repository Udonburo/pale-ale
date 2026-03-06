# CFA Case Study: sample 127 vs 126

## Pair
- Frustrated sample_id: `127`
- Consistent contrast_sample_id: `126`
- World type: `genealogy`

## CFA Snippets
### Frustrated Prompt
```text
Closed world facts:
1) Opalx is the parent of Vivid.
2) Vivid is the parent of Beryl.
Question: Which ancestor relation follows?
Answer in one short paragraph.
```
### Frustrated Answer
```text
Given these facts, Beryl is an ancestor of Opalx. This follows by transitivity of the parent relation.
```
- Defect spans: `[{"start": 19, "end": 48, "label": "frustration_defect_v1", "text": "Beryl is an ancestor of Opalx"}]`

### Consistent Answer
```text
Given these facts, Opalx is an ancestor of Beryl. This follows by transitivity of the parent relation.
```

## AUPRC Summary
| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |
|---|---:|---:|---:|---|---:|
| 127 (frustrated) | 2.18387916431394713e-01 | 2.12655545731918749e-01 | 7.10747531335766514e-01 | A | 4.92359614904371801e-01 |
| 126 (consistent) |  |  |  | None |  |

## Top 10 Tokens by E (Frustrated)
| Rank | Step | Token | E score | Transition label |
|---:|---:|---|---:|---:|
| 1 | 11 | `ĠOp` | 2.74772990342024270e+00 | 1 |
| 2 | 6 | `eryl` | 2.70035955190384769e+00 | 1 |
| 3 | 18 | `Ġtrans` | 2.66911004316379996e+00 | 0 |
| 4 | 10 | `Ġof` | 2.56266027065306634e+00 | 1 |
| 5 | 15 | `ĠThis` | 2.55050858460421592e+00 | 0 |
| 6 | 13 | `x` | 2.54564192870334161e+00 | 1 |
| 7 | 12 | `al` | 2.50438283481700896e+00 | 1 |
| 8 | 19 | `itivity` | 2.47259968663874208e+00 | 0 |
| 9 | 17 | `Ġby` | 2.45340671650870545e+00 | 0 |
| 10 | 9 | `Ġancestor` | 2.44474075475248354e+00 | 1 |

- Overlap top-10 with defect-transition labels: `6/10`

## Output Files
- token_table frustrated: `runs/cfa_case_study/sample127_vs_126/token_table_127.csv`
- token_table consistent: `runs/cfa_case_study/sample127_vs_126/token_table_126.csv`
- pair overlay CSV: `runs/cfa_case_study/sample127_vs_126/pair_overlay_127_vs_126.csv`
- plot frustrated: `runs/cfa_case_study/sample127_vs_126/plot_case_127.png`
- plot pair: `runs/cfa_case_study/sample127_vs_126/plot_pair_compare_127_126.png`

## Interpretation Notes (fill in)
- E spikes coincide with defect span transitions:
- Baseline behavior near defect span:
- Consistent vs frustrated divergence pattern:
