# pale-ale

**Reproducible studies of learned systems and stochastic computation.**

Mathematical results, bounded experiments, and executable companions: from
structural observation and replay in language models to exact, lower-cost
simulation. Each study states its own assumptions, evidence, and limits.
Negative results remain part of the record.

[Publications](publications/README.md) ·
[Reproduce a result](#reproduce-a-result) ·
[Technical reports](#technical-reports) ·
[Amber](#amber)

## Latest preprint

### Exact Binary Projections for Array-RQMC

*Joint Laws and Pathwise-Preserving Execution* — Aoi Kawasaki, September 2026.

Which parts of a randomized point set does a simulator actually need?
This paper characterizes the joint binary law consumed from a specified
scrambled Sobol construction, generates it directly, and maintains state order
without changing finite-seed paths. An equal-covariance counterexample shows
why preserving covariance alone is insufficient.

In eight fixed benchmark cases, order maintenance reduces warm execution time
by **37.2% at 512 paths** and **67.5% at 4,096 paths**, relative to direct
generation with full state sorting. Total cost advantages depend on workload
and reuse. Preprint; not peer reviewed.

[Read the PDF](publications/binary-array-rqmc/zenodo/binary-array-rqmc.pdf) ·
[DOI](https://doi.org/10.5281/zenodo.22728405) ·
[Code and reproduction](papers/binary-array-rqmc/repro/README.md) ·
[Release v1.0.0](https://github.com/Udonburo/pale-ale/releases/tag/binary-array-rqmc-v1.0.0)

## Earlier papers and notes

| Study | Focus |
| --- | --- |
| [Sensitivity without reproducibility](publications/sensitivity-without-reproducibility/README.md) | Positive-control sensitivity versus fresh re-estimation of a representation instrument. |
| [Local mapping without iterative closure](publications/local-mapping-without-iterative-closure/README.md) | Input-output demonstrations and bounded Graph-XOR capability in Qwen3. |
| [Compression-interleaved parenthesization defects](publications/compression-interleaved-parenthesization-defects/README.md) | A predeclared null test across 24 replay-artifact-graph endpoints. |
| [Observer-relative closure signatures](publications/observer-relative-closure-signatures/README.md) | A bounded audit of existing language-model replay artifacts. |
| [Transport-first defect telemetry](publications/transport-first-defect-telemetry/README.md) | A mathematical formulation of transport and closure inconsistency. |
| [Structural replay under FP32](publications/structural-replay-fp32/README.md) | Dense-transformer replay evidence under a fixed precision and execution regime. |

The [publication catalog](publications/README.md) collects dates, DOIs, release
downloads, and source locations. Cite the specific study, not the repository
as a single empirical claim. See [citation metadata](CITATION.cff).

## Technical reports

These closed studies are repository reports, separate from the DOI publications.

- **[CRD: controlled calibration and a negative acquisition test](analysis/crd/TECHNICAL_REPORT.md).**
  Calibration succeeded in a constructed system; no primary seed met local
  action acquisition in the fixed 242-seed successor study.
  [Aggregate data and verification](analysis/crd/README.md).
- **[Gate12C-2: synthetic development negative](analysis/gate12c2_v2_balanced_prototype/TECHNICAL_REPORT.md).**
  The candidate failed its quantitative stability criterion and the bounded
  repair was insufficient. No real held-out evaluation was opened.
  [Closure record and retained implementation](docs/reference/gate12c2_control_plane_sunset.md).

## Reproduce a result

There is no single experiment behind this repository. Start with the companion
for the result you want to check.

- **Array-RQMC:** [19 tests, three kernels, and saved-data regeneration](papers/binary-array-rqmc/repro/README.md).
  The [archived capsule](publications/binary-array-rqmc/zenodo/reproducibility-capsule.zip)
  is self-contained; it does not include every original raw experiment.
- **FP32 structural replay:** [reproduction guide](docs/reproduce_gate12a.md)
  and [evidence atlas](docs/gate12a_evidence_atlas.md).
- **Other publications:** use the study's [publication page](publications/README.md)
  for its exact package, dependencies, and verification scope.

To check the catalog and tracked publication checksums from the repository root:

```sh
python publications/validate_catalog.py
```

Published deposits retain their original bytes. Working code and documentation
may evolve; versioned releases and their checksums identify the archived result.

## Amber

[**Open Amber**](https://amber-oversight.vercel.app/) — a browser-based companion
for reviewing evidence-linked agent traces. Imported traces are processed
locally; reviewers, not the application, make the disposition.

Amber is a technical prototype, not benchmark evidence or an automated judge
of correctness or safety.

## Navigate the repository

| Location | Contents |
| --- | --- |
| [publications/](publications/README.md) | Publication records and exact archive packages. |
| [papers/](papers/) | Manuscript sources and paper-specific computational companions. |
| [analysis/](analysis/) · [docs/](docs/) | Technical reports, verification notes, and reproduction guides. |
| [tools/](tools/) · [src/](src/) · [crates/](crates/) | Research utilities and Python/Rust implementations. |
| [apps/](apps/) | Interactive prototypes. |
| [workstream/](workstream/README.md) · [ABOUT/](ABOUT/README.md) | Research history and project orientation. |

Historical Gate names identify study checkpoints; their conventions are
explained in [Workstreams and Gates](ABOUT/WORKSTREAM_AND_GATES.md).
Claims remain study-specific: the work does not establish a universal model
quality or safety score, or a complete mechanistic account of language models.

## License

Repository software is [MPL-2.0](LICENSE). Publication materials, data, and
third-party artifacts follow their accompanying terms. The Array-RQMC release
uses **CC BY 4.0 for publication materials and saved data, and MPL-2.0 for code**;
see its [per-file license scope](publications/binary-array-rqmc/zenodo/LICENSES.txt).
