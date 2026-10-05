# Six exact executors: confirmation and workload diagnostics

The checked, executed source is in `confirmation/source/`. `legacy/` contains
unaltered copies of the earlier measured kernels; `executors.py` adds direct
interval basis counting, a transported nested primal frame, common model tables
and readouts. The [plan](PLAN.md) records the 17 cells, seeds, technical repeats
and separate diagnostic questions before confirmation.

The current wrapper corrects the even-K repair indicator and rejects tandem
widths below four. The measured source is preserved verbatim; see
`model-wrapper.diff.txt`. The existing timings belong to the measured source.
`check_model_domains.py` verifies identical model tables for all 17 published
cells and identical saved outputs for all six current methods on one seed per
cell (102 method trajectories, 13,800 steps).

## Findings

- Eight paired seeds, five technical repeats and six methods per cell: 4,080
  timings. All outputs agree at all 552,000 timed method steps. None are discarded.
- Rebuild/reuse paired cell medians: 1.055--1.201. Direct basis is 1.535--1.937
  times faster than reuse in every cell median.
- Fixed primal-frame transport is correct and competitive, but yields no
  consistent further advantage over direct basis on these cells.
- Five fixed workloads have exact expected total demands 55.184, 112.718,
  118.069, 62.871 and 143.096; their 4,096-shift means are 55.237, 112.640,
  118.129, 62.872 and 143.151. These are conditional diagnostics, not timing seeds.
- Saved-source replay: 816 trajectories, 110,400 method steps. Accounting:
  18,400 paired steps and 32,906 small rank partitions, all passing.

## Reproduce from the paper root

```text
python review_checks/primal_dual/confirmation/source/check.py
python review_checks/primal_dual/check_primal_counter.py
python review_checks/primal_dual/check_model_domains.py
python review_checks/primal_dual/check_point_sorting_bridge.py
python review_checks/primal_dual/replay.py
python review_checks/primal_dual/verify_workloads.py
python review_checks/primal_dual/analyze.py
```

`replay.py --quick` checks one small repair and tandem pair; full replay is the
default. It checks executed-source identity and all eight matrices before running.
`analyze.py` reads raw observations and recreates Tables 2--5/C1 and Figures
3--4/A1 without repeating timings. `analyze.py --timing-only` updates Table 2 and
timing findings alone: each method/direct ratio is calculated within seed before
taking the median, alongside the matched rebuild/reuse comparison.
`check_primal_counter.py` checks manuscript Algorithm 3 against explicit coset
enumeration and the saved compiled kernel, including non-reduced echelon bases,
rank deficiency and word/count limits. `workload.py` regenerates the fixed batches
and shifts; `phases.py` repeats the separate staged latency diagnostics.

`verify_workloads.py` reconstructs incoming histograms, query arrays and the
retained shifts, then checks every integer observation without a tolerance.
`workloads/expectations-exact.json` stores numerator/denominator pairs derived
from the prefix sets and source-row weights. The original floating summary
remains intact and is checked with explicit Python 3.11 addition order;
Python's later floating `sum` implementation is not the equality criterion.
`analyze.py` uses the exact fractions for theoretical table entries and curves,
while reading the unchanged original observations. The initial fraction file
can be constructed with `verify_workloads.py --write-exact` only when that file
does not exist; ordinary reproduction checks it without overwriting it.

The independent sorting bridge starts at the original direction columns. Its
192 full sorts compare 12,240 rank-assigned words, with both binary and Gray
index orders and zero/nonzero sorting shifts. It also checks 1,530 actual SciPy
unscrambled points, all eight retained matrices and 2,064 sampled assignments
at the saved generator sizes. This is an author regression, not the reviewer's
unavailable test implementation.

To obtain new host timings, use a **new output directory**:

```text
python review_checks/primal_dual/confirmation/source/benchmark.py --out fresh-timing-run
```

The benchmark resumes only with matching source and design. Do not replace the
retained host observations with a new run.

## Inputs and records

`confirmation/data/metadata.json` specifies cells, seed rule, versions and source.
`matrices.json` saves C and both direction-column sets. `timings.jsonl` contains
one trajectory time per method/round/seed. `trajectories/*.npz` saves canonical
histograms, readouts, both dual work profiles and per-width demands.

Rank/word vectors are MSB-first. U,V are unit lower triangular. Philox4x32-10
uses counter (index,domain,step,0), a 64-bit seed key, domains 1/2 for U/V and
3 for the shift at index zero. The first two 32-bit outputs form the packed
word; the required low bits are retained. The ideal uniform-bit theorem and
this reproducible pseudorandom source are distinct statements.

Tandem orders states by total length then queue-2 length. Its exact cutoffs
include threshold atoms. All states reachable within 200 steps are present,
plus an unused guard level. Dense tables, compaction and full output histories
are charged in every method.

`workloads/phases.json` preserves every staged timing sample. An untimed replay
identifies demanded constraints before construction; query/flow arrays are
materialized and Python dispatch remains. These stages are **not an additive
decomposition of the fused production trajectory**. The demand-discovery pass
is not a free implementable optimization. Fused totals are measured separately.
Fixed-workload repetitions keep the tape and queries unchanged but clear caches
and regenerate the source. Cache/host effects remain beyond preparation cost.
