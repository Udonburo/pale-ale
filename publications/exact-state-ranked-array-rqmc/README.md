# Exact State-Ranked Array-RQMC: Representations and Shared Work

**Aoi Kawasaki — author preprint, 6 October 2026.** Not peer reviewed.

[Paper DOI](https://doi.org/10.5281/zenodo.23166655) ·
[Main paper, 20 pages](zenodo/implicit-array-rqmc.pdf) ·
[Supplement, 6 pages](zenodo/retained-studies.pdf) ·
[Source and reproduction](../../papers/exact-state-ranked-array-rqmc/README.md) ·
[Public companion v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/exact-state-ranked-array-rqmc-v1.0.0)

The paper gives primal image-basis and dual constraint implementations of the
same state-ranked Array-RQMC empirical process. Exact counts propagate the
realized histogram path, reranking and supported readouts without visiting
every particle when complete states repeat and transitions admit short input
interval partitions. Query-prefix diversity determines the dual counter's
shared transformation demand.

Six compiled executors share the repair and tandem inputs. In all 17 measured
cells, median within-seed timing ratios favor direct basis over the other
non-enumerating methods. Dual coefficient retention improves on matched
reconstruction. The paper distinguishes those mechanisms, their costs and
their limits; it does not claim that dual reuse is the fastest executor.

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

The paper's Data and code availability paragraph names the pre-publication
review archive `array-rqmc-review-20261005-estimator.zip`. The ZIP linked here
is its public counterpart. `PUBLIC_RELEASE.json` documents every retained
file's identity and the packaging differences. Private review correspondence
and unrelated historical packages are omitted. The original review archives
remain unchanged. The paper DOI record links this fixed public release.

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
