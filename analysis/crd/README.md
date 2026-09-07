# CRD technical report and evidence notes

[Read the technical report](TECHNICAL_REPORT.md).

This is a repository technical report about two completed controlled studies.
It has not been deposited on Zenodo or submitted as an arXiv preprint. It is
not an additional paper entry in the publication catalog. No new experiment
was run to produce it.

## Included material

- `TECHNICAL_REPORT.md`: reader-facing definitions, results, interpretation,
  limitations, and closure.
- `report_data.json`: selected aggregate numbers from the retained scientific
  results and postmortem; no seed tickets, blind keys, private cloud URLs, or
  machine-specific absolute paths.
- `figures/training_losses.png`: unchanged copy of the existing CRD-02
  postmortem figure. Its identity is recorded in the aggregate snapshot.
- `export_report_data.py`: standard-library-only, read-only extraction and
  checking of the report snapshot. It imports no production experiment code.

Run the limited self-contained check from the repository root:

```text
python -B analysis/crd/export_report_data.py --check
```

This checks headline snapshot consistency and the included figure's hash.
It cannot authenticate absent original data or reproduce the experiment.

With the retained local evidence available, use:

```text
python -B analysis/crd/export_report_data.py --source-root workstream/local --check
```

An equivalent retained root can be supplied. This verifies the fixed identities
of both result files, the CRD-02 receipt and protocol, checks the receipt-bound
primary metric export, recomputes A/F/V/B and descriptive C from its 242 records,
checks diagnostic pairing, and compares the extracted snapshot. Omitting
`--check` prints the extracted JSON to stdout; the script never writes files.
It is a report check, not a replacement official reducer or a full chain audit.

## Source identity and availability

Exact source paths relative to the retained root, byte counts, and SHA-256
values are in `report_data.json` under `source_files`. Principal identities:

| Source | SHA-256 |
|---|---|
| CRD-01 scientific result | `0ab0a113d300afe3470ca59c5016dac96116f1889c8896b67b53b74d861376b3` |
| CRD-02 complete result table | `b1972c84ea147c4f7653ce9de6cea46b9517980b7a68ffaa196a36ed50a7a8af` |
| CRD-02 final receipt | `8b5213596b2385c07289e9bafd87aaa99bbeb429f60d52d19d78481a2a8f41ef` |
| CRD-02 prospective protocol | `c07a4d68d2c8f52fc5f1513f731b96ceb55e650f1165a247606751c68fa470af` |

Original evidence remains under ignored local storage and is not made publicly
available by these source identifiers. A clean repository checkout cannot
reconstruct the official training/query chain or the 484 individual saved
training losses from this aggregate package. The figure's per-update percentile
series is not included as a separate raw table; full regeneration requires the
retained postmortem inputs and plotter identified by hash in the snapshot.
Hashes identify evidence, but do not substitute for access to it.

CRD-01's 2026-09-04 result review accepted constructed estimator/intervention
calibration and limited seed-coverage inference. Its private independent copy
was verified on 2026-09-05. CRD-02's completed result and saved-data postmortem
were accepted in the 2026-09-07 scientific close. The retained close note still
assigns CRD-02 off-site preservation as a separate task. This report makes no
claim of a new cloud verification or completed CRD-02 backup.

The completed CRD-02 run spans an original database and two continuation
layers. Its prior invalid/stopping records and value-preserving repair history
remain part of the evidence. Neither the final continuation alone nor this
report snapshot replaces those dependencies.

## Interpretation and report construction

The prespecified component flags and population decisions are distinguished
from post-hoc retention, alias, and minibatch-loss analysis. Seed tickets are
the independent-replicate operating model; sequences and cells are nested
measurement units. No extra population test or confidence interval was added
for the report. In particular, empirical percentile bands are not confidence
intervals, and CRD-01's endpoint-wise test is not CRD-02's single joint-event
test.

Scientific thresholds, original observations, and the inherited figure remain
unchanged. This report adds interpretation and a compact inspection surface,
not a new experiment or a new qualification decision.
