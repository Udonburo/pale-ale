# Tables regenerated from saved numerical inputs

No new observations. Interval endpoints are retained from the original analyses;
aggregate saved means do not suffice to rerun their paired-round bootstraps.

## Table 1 — direct binary generation

| N | Library ms | Direct ms | Reduction % | Ratio interval |
|---:|---:|---:|---:|---|
| 512 | 34.57 | 9.60 | 72.2 | [0.2693, 0.2855] |
| 4,096 | 97.56 | 52.83 | 45.8 | [0.5256, 0.5589] |

## Table 2 — order maintenance

| N | Wall ratio | Ratio interval | Reduction % | CPU ratio |
|---:|---:|---|---:|---:|
| 512 | 0.6279 | [0.6200, 0.6369] | 37.2 | 0.6821 |
| 4,096 | 0.3248 | [0.3211, 0.3285] | 67.5 | 0.3171 |

## Table 3 — selected sampling configurations

| Payoff | Method | N × k | Estimated MSE | 95% upper | Warm ms |
|---|---|---:|---:|---:|---:|
| One-sided | Direct binary array | 512 × 3 | 0.00058336 | 0.00061554 | 26.51 |
| One-sided | Order-maintained array | 2048 × 1 | 0.00032209 | 0.00032684 | 11.08 |
| One-sided | CRN | 2048 × 7 | 0.0006207 | 0.00071107 | 12.19 |
| One-sided | Antithetic | 4096 × 4 | 0.00050082 | 0.00056925 | 11.97 |
| Quadratic | Direct binary array | 2048 × 1 | 0.0024182 | 0.0027658 | 26.39 |
| Quadratic | Order-maintained array | 2048 × 1 | 0.0022989 | 0.0026453 | 10.94 |
| Quadratic | CRN | 4096 × 13 | 0.0064098 | 0.0075468 | 30.91 |
| Quadratic | Antithetic | 2048 × 55 | 0.0060657 | 0.0070546 | 70.00 |

## Table 4 — acquisition-inclusive accounting

| Payoff | Method | First evaluation s | At R=1000, ms/evaluation |
|---|---|---:|---:|
| One-sided | Direct binary array | 49.498 | 75.98 |
| One-sided | Order-maintained array | 53.725 | 64.79 |
| One-sided | CRN | 4.945 | 17.12 |
| One-sided | Antithetic | 4.975 | 16.93 |
| Quadratic | Direct binary array | 56.737 | 83.10 |
| Quadratic | Order-maintained array | 60.990 | 71.92 |
| Quadratic | CRN | 5.138 | 36.01 |
| Quadratic | Antithetic | 5.130 | 75.06 |

Cost convention: recorded mask-initialization cost is included for the baseline implementations.
This is not an intrinsic lower bound for aligned-innovation CRN.
First-use and reuse figures are component sums, not new timed end-to-end experiments.
