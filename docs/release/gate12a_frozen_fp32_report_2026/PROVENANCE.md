# Gate12A Frozen FP32 Dense-Transformer Report: Provenance

Status: release-prep draft
Scope: commit and artifact lineage for the frozen FP32 Gate12A report bundle

## Purpose

This file records the main software-side and artifact-side bindings that the report bundle is expected to preserve.

Its purpose is to keep four things distinct:

- the commit used as the software-side reference implementation
- the commit under which the main numerical summaries were generated
- any upstream artifact-generation exception that must be disclosed explicitly
- the report-facing bundle state assembled for Zenodo

## Main Commits Already Fixed By The Report Line

The current Gate12A report text distinguishes these commits:

1. Mainline artifact-summary commit
   - `084eb7878d8cb016243950e1cf4b4bd7379daaba`
   - role: artifact-frozen outputs for the mainline cross-model summaries reported in the manuscript

2. Archived reference implementation / sidecar-boundary commit
   - `8a14c94b999d823c1734b2582730c3ad4ea98d03`
   - role: archived release-time reference implementation preserving the same replay criteria together with the sidecar-boundary logic

3. Upstream Gate8 exception already disclosed in the manuscript
   - `58d06742f23a0bc7ba25c6ecde790e2e03b4324e`
   - role: one upstream Gate8 candidate-execution artifact for the `Qwen/Qwen2.5-3B-Instruct` transcript bundle was recovered from this earlier generation run rather than from the mainline artifact-summary commit

This exception is not a hidden drift. It is part of the intended public provenance surface and should remain explicitly documented in the release bundle.

## Report-Side Binding

The final Zenodo-facing report bundle should also record:

- the GitHub repository URL
- the release tag used for the software-side freeze
- the GitHub release URL corresponding to that tag
- the exact tagged commit hash
- the Zenodo DOI and record URL once minted

Those release-specific values should be filled in after the final tag exists.

## Expected Provenance Statement

The release bundle should preserve the following logic:

- the report-level numerical summaries are bound to the archived mainline artifact outputs generated under `084eb7878d8cb016243950e1cf4b4bd7379daaba`
- the software-side reference implementation preserved in the release is bound to `8a14c94b999d823c1734b2582730c3ad4ea98d03`
- one upstream Gate8 transcript candidate-execution artifact in the closed mainline is bound to `58d06742f23a0bc7ba25c6ecde790e2e03b4324e`
- the Zenodo bundle adds paper-facing release packaging on top of those already-disclosed bindings rather than replacing them

## Evidence-Side Binding

The bundle should also make clear which of these two evidence modes is being used:

1. Direct artifact inclusion
   - the run artifacts are included directly in the Zenodo upload

2. Separate artifact record
   - the report bundle points to a separate Zenodo artifact record
   - that separate record carries the evidence bundle and its own integrity files

If the second mode is used, record:

- the artifact record DOI
- the artifact record URL
- a short note explaining why the split record exists

## Integrity Files To Ship With The Bundle

The release bundle should ship with:

- a release file list
- `SHA256SUMS.txt`
- a release-binding file or equivalent release note containing the final tag and commit

If the artifact bundle is separate, it should ship its own checksum surface rather than relying only on the report bundle checksums.
