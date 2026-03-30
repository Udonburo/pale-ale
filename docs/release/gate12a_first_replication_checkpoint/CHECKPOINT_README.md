# Gate12A First Replication Checkpoint Bundle

Status: release-prep draft
Scope: supplemental bundle for a Zenodo replication checkpoint, not a general software release and not a final theory claim

## Upload Mode

This checkpoint should be uploaded to Zenodo as a manual dataset deposit.

Reason:

- the key `runs/` evidence is intentionally ignored by git in this repository
- a GitHub-tag-only Zenodo sync would omit the local run artifacts that carry the checkpoint evidence

So for this checkpoint:

- GitHub release/tag can still be used as the software-side reference point
- the Zenodo record itself should be created with `+ New upload` and a manually assembled bundle zip

## What This Bundle Is

This bundle fixes the first cross-family replication checkpoint for the Gate12A anchor-rich closure tension observable surface in `pale-ale`.

The checkpoint is intentionally narrow.
It preserves a reproducible observable surface and its first cross-family comparison over:

- `transcript_v1`
- `briefing_v1`

It does not claim:

- a universal law
- a family-independent threshold
- a global good/bad classifier
- a final theory of LLM closure quality
- Gate12B threshold doctrine

## Exact Release Binding

Fill these fields after the release tag is created from `origin/main`.

- Release tag: `<fill_after_origin_main_tag>`
- Release commit: `<fill_after_origin_main_commit>`
- Repository: `https://github.com/Udonburo/pale-ale`

## Included Material

This bundle includes four groups of files.

1. Tracked Gate12A memo line
   - `workstream/199_GATE12A_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`
   - `workstream/200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md`
   - `workstream/201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`
   - `workstream/202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md`

2. Gate12A spec context
   - `workstream/196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
   - `workstream/197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
   - `workstream/198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`

3. Transcript-side committed run artifacts
   - discrete connection surface
   - calibration surface
   - text-surface audit
   - first-pass phenotype read

4. Briefing-side committed run artifacts
   - discrete connection surface
   - calibration surface
   - text-surface audit
   - first-pass phenotype read

The exact file list is recorded in:

- `BUNDLE_FILE_LIST.txt`

## Key Scoped Facts Fixed By This Bundle

Under the current Gate12A observable surface:

- `zero_overlap_count = 0` in both families
- all currently defined triangles are anchor-rich in both families
- `triangles_with_all_anchor_count = 0` in both families
- directional subregime ordering remains aligned in both families
  - `residual_chord` flatter than `trusted_tree`
  - `anchor_qualified` flatter than `plain`
- `high residual != immediate failure` remains true on first-pass reading
- extreme-band phenotype mix shifts by family even when structural observable facts stay aligned

## What Is Intentionally Excluded

This bundle intentionally excludes:

- local-only `tense` drafting notes
- local-only enriched tagging copies
- unfinished release tags
- future-family replication surfaces
- Rust refactor planning material

## Integrity Files

This bundle should be shipped with:

- `BUNDLE_FILE_LIST.txt`
- `SHA256SUMS.txt`

Generate `SHA256SUMS.txt` after the bundle directory is assembled, using:

- `make_sha256sums.ps1`

## Reader Orientation

Read in this order:

1. `workstream/202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md`
2. `workstream/200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md`
3. `workstream/201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`
4. `workstream/199_GATE12A_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`
5. the paired transcript-side and briefing-side run artifacts listed in `BUNDLE_FILE_LIST.txt`
