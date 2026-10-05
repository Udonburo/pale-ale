**Table C1. Fixed-workload repetitions, microseconds per step.** R63 and T are the preselected repair and tandem snapshots with $m=20,w=52$. Preparation is measured separately in nine batches of 200 calls; the remaining columns are medians of nine setup-inclusive sequences. The workload and tape are identical at every repetition, with step caches cleared. Python dispatch and host variation remain present.

| Workload / method | Preparation once | $T=1$ | $T=4$ | $T=64$ | $T=256$ |
| --- | --- | --- | --- | --- | --- |
| R63 / Dual reuse | 51.1 | 232.8 | 179.8 | 159.0 | 138.1 |
| R63 / Rebuild | 18.7 | 224.2 | 204.2 | 187.0 | 165.3 |
| R63 / Direct basis | 0.2 | 110.5 | 101.1 | 84.8 | 80.5 |
| R63 / Primal reuse | 9.5 | 127.3 | 111.8 | 89.2 | 82.2 |
| T / Dual reuse | 53.6 | 955.2 | 792.8 | 807.7 | 851.7 |
| T / Rebuild | 16.9 | 926.3 | 908.4 | 960.3 | 1012.7 |
| T / Direct basis | 0.4 | 456.9 | 448.8 | 468.7 | 483.1 |
| T / Primal reuse | 9.7 | 473.5 | 456.9 | 465.2 | 479.8 |
