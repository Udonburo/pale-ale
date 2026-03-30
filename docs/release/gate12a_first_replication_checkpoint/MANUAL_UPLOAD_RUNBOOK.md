# Gate12A First Replication Checkpoint: Manual Upload Runbook

Status: release-prep draft
Scope: operational runbook for manual Zenodo upload of the first Gate12A replication checkpoint bundle

## Why Manual Upload Is Required

For this checkpoint, manual Zenodo upload is required.

Reason:

- the key `runs/` artifact directories are ignored by git in this repository
- a GitHub release alone would not carry the local `manifest/status/csv/jsonl` evidence that defines the checkpoint

So the correct split is:

- GitHub tag/release = software-side reference point
- Zenodo upload = manual dataset bundle that includes the evidence files from `runs/`

## Preconditions

Before assembling the bundle:

1. the target commit must already be on `origin/main`
2. the release tag must be chosen from that `origin/main` commit
3. the files listed in `BUNDLE_FILE_LIST.txt` must all exist locally
4. local-only drafting files must stay excluded

## Step 1: Finalize Release Binding

Use the fixed tag name:

- `gate12a-first-replication-checkpoint-2026-03-31`

Update the release-prep docs so that they carry:

- `CHECKPOINT_README.md`
- `ZENODO_METADATA_DRAFT.md`

with:

- final release tag name
- deterministic tag URL
- deterministic release URL

Do not try to hard-code the final tagged commit hash into a tracked file.
Instead, let the bundle assembly step generate `RELEASE_BINDING.json` from the clean tagged checkout.

## Step 2: Tag The Final Commit

After the metadata finalize commit exists on `origin/main`, create the tag:

```powershell
git tag gate12a-first-replication-checkpoint-2026-03-31
git push origin gate12a-first-replication-checkpoint-2026-03-31
```

Confirm the worktree is clean before assembling the bundle.

## Step 3: Assemble The Bundle

From the repository root, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File docs\release\gate12a_first_replication_checkpoint\assemble_bundle.ps1 -RepoRoot . -OutputRoot dist\zenodo -BundleName gate12a_first_replication_checkpoint_bundle -TagName gate12a-first-replication-checkpoint-2026-03-31
```

Expected outputs:

- `dist\zenodo\gate12a_first_replication_checkpoint_bundle\`
- `dist\zenodo\gate12a_first_replication_checkpoint_bundle.zip`

The assembled directory should include:

- `CHECKPOINT_README.md`
- `BUNDLE_FILE_LIST.txt`
- `ZENODO_METADATA_DRAFT.md`
- `RELEASE_BINDING.json`
- `SHA256SUMS.txt`
- the tracked memo/spec files
- the transcript-side run artifacts
- the briefing-side run artifacts

## Step 4: Sanity Check The Bundle

Confirm:

- `SHA256SUMS.txt` exists
- `RELEASE_BINDING.json` exists
- file count looks plausible
- transcript-side and briefing-side run files are both present
- no local-only `tense` drafts are included

## Step 5: Create The Zenodo Record

In Zenodo:

1. choose `+ New upload`
2. upload the bundle zip manually
3. set upload type to `dataset`
4. copy the title and abstract from `ZENODO_METADATA_DRAFT.md`
5. fill creators, keywords, and related identifiers

## Step 6: Final Cross-Check Before Publish

Before publishing, verify:

- title still says `first replication checkpoint`
- scope still says `transcript_v1 + briefing_v1`
- the description does not claim universal law or threshold law
- the related identifiers point to the final GitHub tag/release on `origin/main`

## Intentionally Excluded

Do not include:

- local-only `tense` drafts
- enriched local tagging copies
- future-family replication runs
- Rust refactor planning notes
