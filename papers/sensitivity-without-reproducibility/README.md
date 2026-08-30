# Sensitivity Without Reproducibility

Source for the technical report:

> *Sensitivity Without Reproducibility: A Measurement Boundary for an
> Operator-Valued Representation Instrument in Qwen Models*

## Release state

The scientific content is frozen. The tracked source binds the underlying
execution record to commit
`407d20fd4f074b9ef4524c82c8136874efaec476`.

The archival DOI is intentionally inserted only after this source branch is
merged and a Zenodo DOI is reserved. The final PDF and reproducibility capsule
are therefore release artifacts, not inputs to the scientific analysis.

## Files

- `main.tex` - complete manuscript and supplement
- `figure1.tex` - central sensitivity/reproducibility figure
- `figure2.tex` - exact square and evidential ladder
- `PRIMARY_SOURCE_CITATION_AUDIT.md` - claim-bounded source audit

## Build

From this directory:

```text
tectonic -X compile main.tex --outdir <temporary-output-directory>
```

The publication PDF is exported only after metadata, text, font, and
all-page visual checks.

## Scientific boundary

```text
visible-state causal use             POSITIVE on Qwen3.5-27B / Qwen3.6-27B
bounded operator instrument          POSITIVE on one Qwen3.6-27B substrate
fresh distribution requalification  NOT QUALIFIED (5/20 blocks)
broken-square sensitivity            59/60 layer-blocks
Stage E                              UNOPENED
representation-function coupling     UNTESTED
```

Preparing or building this manuscript performs no model execution, activation
extraction, or new scientific analysis.
