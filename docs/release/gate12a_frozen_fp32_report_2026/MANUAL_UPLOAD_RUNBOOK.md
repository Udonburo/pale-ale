# Gate12A Frozen FP32 Dense-Transformer Report: Manual Upload Runbook

Status: release-prep draft
Scope: operational runbook for assembling and uploading the report-facing Zenodo bundle

## Why Manual Upload Is Still The Right Default

For this release, manual Zenodo upload remains the correct default.

Reason:

- the report bundle is a mixed release surface, not just a software tag
- it needs paper files, release-side documentation, integrity files, and evidence-side material or stable evidence links
- the key evidence is not fully represented by a plain GitHub tag alone

So the intended split remains:

- GitHub tag/release = software-side anchor
- Zenodo manual upload = report-facing release bundle

## Preconditions

Before assembling the report bundle:

1. the intended software-side commit must already be on `origin/main`
2. the docs-facing repo surface should already be pushed
3. the paper PDF and source must be finalized for the current release
4. the provenance language in `PROVENANCE.md` must match the manuscript
5. the artifact-inclusion mode must be chosen
   - direct artifact inclusion
   - separate artifact record with cross-link

## Step 1: Finalize Release Binding

Before tagging:

- choose the release tag name
- finalize `README_RELEASE.md`
- finalize `PROVENANCE.md`
- finalize `ZENODO_METADATA_DRAFT.md`

Do not hard-code the final tagged commit hash until the tag exists.

## Step 2: Tag The Release Commit

After the release-prep commit exists on `origin/main`, create and push the final tag.

Suggested pattern:

```powershell
git tag <final-tag-name>
git push origin <final-tag-name>
```

The final tag name should then be copied into:

- `PROVENANCE.md`
- `ZENODO_METADATA_DRAFT.md`

## Step 3: Assemble The Bundle

Assemble a release directory containing:

- `paper.pdf`
- paper source files
- `README_RELEASE.md`
- `PROVENANCE.md`
- `ZENODO_METADATA_DRAFT.md`
- file list
- `SHA256SUMS.txt`
- implementation snapshot
- artifact bundle or artifact-link note

If artifacts are not included directly, add a short note such as:

- `ARTIFACT_RECORD_LINKS.md`

with the exact DOI / URL once available.

## Step 4: Generate Checksums

After the directory is assembled, generate:

- `SHA256SUMS.txt`

The checksum surface should cover every shipped file in the report bundle.

## Step 5: Create The Zenodo Record

In Zenodo:

1. create a new upload
2. upload the assembled report bundle manually
3. use upload type `dataset` unless there is a later reason to switch
4. copy title and description from `ZENODO_METADATA_DRAFT.md`
5. add related identifiers for the GitHub repository and release tag

If the artifact bundle is a second record, create that record either first or immediately after and cross-link the two records before publish.

## Step 6: Final Cross-Check Before Publish

Confirm:

- the title matches the actual report scope
- the description does not overclaim beyond the frozen regime
- the provenance note preserves the `084eb...`, `8a14c...`, and `58d067...` distinctions
- checksums exist
- local-only files are excluded
- any separate artifact record is linked explicitly
