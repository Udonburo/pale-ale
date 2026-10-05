# Primal/dual follow-up, 3 October 2026

This successor answers the current referee questions. Earlier measurements and
distributed packages keep their identities. Development may repair code; the
confirmation starts only after enumeration checks and development timings.
No confirmation cell, seed, or slow observation will be removed on performance
grounds. Any scientific change after that start requires a separate follow-up.

## Questions and methods

Compare six compiled, exact executors using the same inverse-LMS tape, complete
state order, transition thresholds, dense histogram output, and integer readouts:
coefficient reuse; matched incremental reconstruction; compact rank streaming;
full-map basis with endpoint prefixes; full-map basis with direct aligned
interval blocks; and transformed fixed suffix bases with those same direct
blocks. The last method shares each distinct fixed echelon vector across suffix
widths and solves for all of its transformed columns together. It retains the
fixed factorization C=B T, with U still included in the realized affine offsets.
This is a reference implementation constructed here, not a claimed verbatim
implementation of a published dynamic executor.

## Confirmation design (fixed before confirmation)

Repair: K=63, w in {30,52}, m in {8,12,16,20}; K=255, w=52,
m in {12,16,20}; horizon 100. Tandem: w in {30,52}, m in
{12,16,20}; horizon 200. All start at state zero. Total: 17 cells.
Each cell has eight independent paired seeds 103000000+100*cell+rep,
rep=0,...,7. Five timing rounds per seed use a deterministic shuffled method
order. Timing includes C extraction, model tables, fixed preparation, source,
updates, compaction, full step histograms, and integer readouts. It excludes
JIT warmup, verification, compression, and disk writes. Repeated rounds are
technical repeats, not additional independent randomizations. Report the median
of the five times for each seed, then paired ratios and all eight ratios.
The host is not isolated; preserve every timing observation.

The fixed matrix uses the existing exact SciPy Sobol adapter: coordinates 1,2,
bits=max(m,w), first m direction columns, C=C_y C_s^{-1}, MSB-first rank/word
vectors. It is common across seeds, models, horizons and methods at given m,w.
The shared Philox counter is (row index, domain, step, 0), with domains 1 for U,
2 for V, 3 for c (index zero). Seeds are 64-bit keys. Unit lower triangular
rows and shifts have the existing eager source law.

Repair uses the existing binary64-computed cutoffs and rates. Tandem orders
(q1,q2) by (q1+q2,q2), with service 1 for u<=7/16, service 2 for
7/16<u<=3/4, and arrival otherwise. Internal integer cutoffs are
7*2^(w-4)+1 and 12*2^(w-4)+1. The table contains all states reachable
through the horizon, plus one unused guard level. No live state is truncated.
Readouts are per-step total queue length and 1{q2>4}; repair retains its
per-step failed-machine count and upper-half indicator. Cumulative integer
readouts and every step histogram must agree across all methods.

Development uses seeds starting 102000000, separate from confirmation. Small
enumeration checks cover arbitrary, zero and deficient C, endpoint atoms and
both state models. Check transformed suffix images, pivots and forward offsets
independently against scalar GF(2) arithmetic. Snapshot the executing source
and save its identity when confirmation begins, so archived data have an exact
implementation. Save one canonical compressed trajectory per paired seed;
check every method and every timing repeat against it outside the timed region.

## Workload diagnostics

Use preselected incoming histograms at repair (K=63,m=12/20,w=52,t=50),
repair (K=255,m=20,w=52,t=50), and tandem (m=12/20,w=52,t=100),
all from development seed 102000000. Fix C,U,V, the incoming histogram and
thresholds. Independently resample c 4096 times. Compare exact prefix-diversity
predictions, sampled mean J, and Psi by width. Preserve the extracted workloads.
Also give a declared synthetic same-Q example with different prefix diversity.

Measure setup/source, constraint or basis construction, query, and aggregation/
readout on fixed workloads. Where construction is moved ahead of queries to
isolate it, label this a staged diagnostic and compare its output and total time
with the fused kernel; do not present its phase times as an exact decomposition
of the original fused trajectory. Repeat the identical workload 1,4,16,64,256
times with per-step caches cleared to measure preparation amortization, rather
than interpreting a changing-horizon trajectory as a fixed workload.

## Reporting

Retain losses. Separate the benefit of avoiding rank visits, the effect of direct
interval decomposition, and the effect of retaining primal or dual structure.
Use the findings to decide the manuscript's contribution statement. Do not
claim universal superiority, infer timings from XOR counts, or call the shared
transformation bound a bound on total query time. Literature comparisons and
the reproduction package are part of the same revision.
