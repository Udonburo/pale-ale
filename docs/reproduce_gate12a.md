# Reproduce Gate12A

This guide keeps three layers separate: release identification, release-bundle
verification, and empirical regeneration. The April 2026 Gate12A public claim
surface is the frozen FP32 dense-transformer report release, not a promise that
the full empirical pipeline can be rebuilt from the selected-manifest bundle
alone.

## 1. Check DOI / Release / Included Bundle Names

Use the current citable-release table in [`../README.md`](../README.md) first.
The Gate12A frozen technical report DOI is:

- `10.5281/zenodo.19483162`

The local release bundle is
[`../publications/structural-replay-fp32/zenodo/`](../publications/structural-replay-fp32/zenodo/).
Its release-side files are:

- `paper.pdf`
- `paper-source.zip`
- `selected-manifests.zip`
- `README.txt`
- `github-release-body.md`
- `zenodo-description.md`
- `CHECKSUMS-SHA256.txt`

The mathematical telemetry note is a separate record:

- `10.5281/zenodo.19569052`

The earlier replication checkpoint / prior release surface is:

- `10.5281/zenodo.19340221`

## 2. Verify Selected Manifests / Checksums / Commit Bindings

The checksum layer verifies the release bundle files. It does not, by itself,
re-run the empirical pipeline.

On Windows PowerShell, compute the file hashes and compare them with
[`../publications/structural-replay-fp32/zenodo/CHECKSUMS-SHA256.txt`](../publications/structural-replay-fp32/zenodo/CHECKSUMS-SHA256.txt):

```powershell
Get-FileHash publications\structural-replay-fp32\zenodo\paper.pdf -Algorithm SHA256
Get-FileHash publications\structural-replay-fp32\zenodo\paper-source.zip -Algorithm SHA256
Get-FileHash publications\structural-replay-fp32\zenodo\selected-manifests.zip -Algorithm SHA256
Get-FileHash publications\structural-replay-fp32\zenodo\README.txt -Algorithm SHA256
Get-FileHash publications\structural-replay-fp32\zenodo\github-release-body.md -Algorithm SHA256
Get-FileHash publications\structural-replay-fp32\zenodo\zenodo-description.md -Algorithm SHA256
```

Then inspect
[`../publications/structural-replay-fp32/zenodo/README.txt`](../publications/structural-replay-fp32/zenodo/README.txt)
for the frozen commit bindings:

- downstream Gate12A replay summaries and structural quartet verdicts:
  `084eb7878d8cb016243950e1cf4b4bd7379daaba`
- one upstream Gate8 candidate-execution artifact exception:
  `58d06742f23a0bc7ba25c6ecde790e2e03b4324e`

The `selected-manifests.zip` file is included for provenance and
boundary-condition inspection. It is not a substitute for the full candidate
execution artifacts.

## 3. Re-Run Only When Full Artifacts Are Available

Only attempt a Gate12A empirical re-run when the required full
`candidate_execution`-equivalent artifacts are available locally or from a
separately archived artifact source. Do not treat the selected-manifest bundle
as enough for full regeneration.

When the full artifacts are present, keep the frozen-protocol boundary explicit:

- use the same precision/observation/replay surface described by the release
- keep generated outputs in a new local output root
- do not mix sidecar or protocol-expanding runs into the frozen Gate12A paper
  release surface

A local replay command should point at the full candidate-execution directory,
not at `selected-manifests.zip`:

```powershell
python tools/run_gate12a_family_replay.py --gate8-execution-dir <full-candidate-execution-dir> --out-root <new-output-root> --balanced-per-band 6 --top-k 3
```

If the full artifacts are not available, stop at DOI, checksum, selected
manifest, and commit-binding verification.
