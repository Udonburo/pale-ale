**Table 5. Staged workload diagnostics, microseconds.** R63 uses $m=20$ and tandem uses $m=20$; both have $w=52$. Construction includes the relevant cache initialization. Query consumes prepared constraints or bases; aggregation includes compaction and readouts. The demanded constraint set is identified in an untimed replay. Staged execution materializes queries and flows and includes Python dispatch, so its marginal phase medians are not a decomposition of the separately timed fused step.

| Workload / method | Source | Construction | Query | Aggregation | Staged total | Fused total |
| --- | --- | --- | --- | --- | --- | --- |
| R63 / Dual reuse | 5.5 | 23.5 | 91.3 | 9.4 | 133.6 | 166.5 |
| R63 / Rebuild | 5.5 | 28.6 | 88.5 | 9.2 | 132.8 | 190.3 |
| R63 / Direct basis | 5.4 | 13.3 | 88.3 | 9.0 | 116.7 | 103.1 |
| R63 / Primal reuse | 5.5 | 18.4 | 89.2 | 9.2 | 124.1 | 109.2 |
| T / Dual reuse | 7.6 | 32.3 | 507.5 | 143.5 | 686.7 | 988.7 |
| T / Rebuild | 8.7 | 38.0 | 517.9 | 153.2 | 716.7 | 1181.8 |
| T / Direct basis | 7.5 | 20.0 | 464.4 | 139.6 | 635.9 | 575.7 |
| T / Primal reuse | 8.0 | 23.6 | 472.5 | 147.0 | 655.0 | 598.6 |
