# Gate12A Frozen FP32 Dense-Transformer Report Bundle

Status: release-prep draft
Scope: public report bundle for the frozen FP32 Gate12A dense-transformer line; not a general software release and not a final theory claim

## What This Bundle Is

This bundle is the intended Zenodo-facing release surface for the current Gate12A report line in `pale-ale`.

It is meant to freeze one narrow empirical surface:

- the current Gate12A report PDF
- the paper source used to build that PDF
- the manifest and checksum material needed to inspect the release
- the implementation snapshot used as the software-side reference point
- the artifact bundles, or stable links to separately archived artifact bundles

The bundle is intentionally narrower than the repo as a whole.
It is a public-facing report surface for the frozen FP32 dense-transformer line, not a universal claim about all model classes and not a final statement of the broader structural research program.

## Intended Payload

The intended payload should include:

1. Report
   - `paper.pdf`

2. Paper source
   - `main.tex`
   - bibliography file(s)
   - figure source files

3. Release-side documentation
   - `README_RELEASE.md`
   - `PROVENANCE.md`
   - `ZENODO_METADATA_DRAFT.md`
   - `MANUAL_UPLOAD_RUNBOOK.md`

4. Integrity material
   - release file list
   - checksum file such as `SHA256SUMS.txt`

5. Software-side reference material
   - a frozen implementation snapshot

6. Evidence-side material
   - artifact bundle(s), if included directly
   - or stable links / record identifiers for separately archived artifact bundle(s)

## What This Bundle Does And Does Not Claim

This bundle is intended to support a narrow report claim:

- under a frozen FP32 dense-transformer regime, the Gate12A structural replay signature persists across the closed 3B/4B full-family-set lines
- phenotype remains family-conditioned and model-specific
- the clearest limited convergence appears at archive high tension
- transcript-only extensions and sidecar / exclusion results are recorded separately rather than collapsed into the same claim surface

This bundle does not claim:

- a universal law for all LLMs
- unchanged extension beyond the current frozen dense-transformer regime
- a final mechanistic explanation of hallucination or contradiction
- that the broader structural program is fully frozen in this release

## Upload Mode

The default upload mode for this report should remain a manual Zenodo upload.

Reason:

- the key `runs/` evidence is intentionally not represented as a plain tracked Git snapshot
- a GitHub-tag-only Zenodo sync would not, by itself, preserve the intended evidence-side release surface

So for this release:

- GitHub tag/release = software-side reference point
- Zenodo upload = report-facing bundle surface

If the artifact bundle is too large for the report record:

- publish the report bundle as one record
- publish the artifact bundle as a separate record
- cross-link them explicitly

A separate Zenodo record is preferred over a GitHub-only link when the evidence bundle is central to the release.

## Reader Orientation

A reader landing on this bundle should be able to answer, quickly:

- what the report claims
- what files produced the paper
- which software-side snapshot matches the report
- where the evidence lives
- how the commit and artifact lineage is bound

The detailed binding for that last point belongs in `PROVENANCE.md`.
