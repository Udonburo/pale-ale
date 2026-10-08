# Exact State-Ranked Array-RQMC: Representations and Shared Work

**Aoi Kawasaki — author preprint, 6 October 2026.**
Submitted to *Monte Carlo Methods and Applications* (MCMA); not yet peer reviewed.
Submission status updated 8 October 2026.

[Paper DOI](https://doi.org/10.5281/zenodo.23166655) ·
[Main paper, 20 pages](zenodo/implicit-array-rqmc.pdf) ·
[Supplement, 6 pages](zenodo/retained-studies.pdf) ·
[Source and reproduction](../../papers/exact-state-ranked-array-rqmc/README.md) ·
[Public companion v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/exact-state-ranked-array-rqmc-v1.0.0)

## The computational problem

Array-RQMC sorts a population and assigns inputs by rank at each update.
Repeated states occupy different ranks and may receive different inputs.
An exact compressed executor must retain that assignment while recovering
the next population and its order.

This paper closes the update with exact rank-interval counts. Primal image
bases and dual constraints provide two counting representations; aggregation
and cumulative sums recover the next state histogram and rank intervals.
When complete states repeat and transitions admit short finite-word interval
partitions, the executor can avoid visiting every particle while preserving
the realized histogram path and supported readouts.

That preservation supports finite-population error assessment: a tractable
one-chain dynamic program can supply a reference mean, while population
execution samples the coupled estimator's error around that mean.

## Contributions and evidence

| Contribution | Where to inspect it |
| --- | --- |
| Primal and dual exact execution, including reranking | Main paper Sections 2–3; primal counter in Appendix D. |
| Exact conditional expected shared demand from query-prefix diversity, under the stated shift condition | Theorem 1, Proposition 2, and the rank-boundary analysis in Section 4. |
| A comparison that separates enumeration, interval decomposition, and retained structure | Six executors, 17 repair/tandem conditions, Table 2 and the seed-level ratios in Figure 1. |
| A concrete reason to preserve the coupled population | Supplement Section S4: finite-population estimator errors and the finite-word reference mean. |

In all 17 measured cells, median within-seed timing ratios favor **direct
basis** over the other non-enumerating methods. At $2^{20}$ particles, its
paired speed ratios against compact rank streaming are about **41–57× for
repair** and **10× for tandem**. Dual coefficient retention improves on matched
reconstruction by factors of **1.055–1.201**. Compact streaming wins in the
smallest-population conditions.

The comparison uses eight independent seeds per cell and five technical timing
repeats per seed, on one non-isolated host. Each method returns every step's
histogram and readout. Ratios are formed within seed before taking their median;
they are not ratios of independently aggregated medians. The work analysis
characterizes shared transformation demand; the measured execution ranking
also reflects query processing and aggregation.

[Read Table 2](../../papers/exact-state-ranked-array-rqmc/tables/primal_dual_times.md) ·
[Inspect the study design](../../papers/exact-state-ranked-array-rqmc/review_checks/primal_dual/README.md) ·
[View the paired observations](../../papers/exact-state-ranked-array-rqmc/figures/primal_dual_pairs.png)

## Reproduction download

Download [array-rqmc-repro-20261006.zip](https://github.com/Udonburo/pale-ale/releases/download/exact-state-ranked-array-rqmc-v1.0.0/array-rqmc-repro-20261006.zip)
from the fixed release. After extraction, follow its README. The
[distribution verification](verification.json) records the exact archive hash
and checks performed on a separate extracted copy.

The public companion contains all 4,080 original timings, the measured code,
current corrected executors, saved population paths, fixed-workload inputs,
the supplement's error-example inputs and the complete PDF build sources.
Measured source remains separate from the corrected wrapper. Reproduction
checks saved results; it does not repeat elapsed-time measurements.

## Versions and provenance

The published preprint's Data and code availability paragraph names the pre-publication
review archive `array-rqmc-review-20261005-estimator.zip`. The ZIP linked here
is its public counterpart. `PUBLIC_RELEASE.json` documents every retained
file's identity and the packaging differences. Private review correspondence
and unrelated historical packages are omitted. The original review archives
remain unchanged. The paper DOI record links this fixed public release.
The journal submission uses separately prepared PDFs with current DOI and
companion links; submission preparation did not replace the DOI-bound PDFs
or the v1.0.0 reproduction inputs.

## Cite

```bibtex
@misc{kawasaki2026exact,
  author = {Kawasaki, Aoi},
  title = {{Exact State-Ranked Array-RQMC: Representations and Shared Work}},
  year = {2026},
  doi = {10.5281/zenodo.23166655},
  url = {https://doi.org/10.5281/zenodo.23166655},
  note = {Author preprint, version 20261005-estimator}
}
```

For code or saved observations, also identify companion tag
`exact-state-ranked-array-rqmc-v1.0.0`. The
[citation metadata](../../papers/exact-state-ranked-array-rqmc/CITATION.cff)
records both identities.

## Published PDFs and licensing

The two files under `zenodo/` are exact copies of the PDFs published as
version `20261005-estimator`, DOI **10.5281/zenodo.23166655**.
`zenodo/CHECKSUMS-SHA256.txt` checks those two files. The inventory and README
are repository documentation, not additional files deposited with the DOI.
The all-versions DOI is **10.5281/zenodo.23166654**.

Publication materials and author-created saved data are CC BY 4.0. Code and
the Typst template are MPL-2.0. Dependencies retain their own licenses; see
the [per-file scope](../../papers/exact-state-ranked-array-rqmc/LICENSES.txt).
No third-party runtime libraries or fonts are distributed in the companion.

The earlier [binary-projection preprint](../binary-array-rqmc/README.md), its
DOI and its archived source remain a separate publication.
