# Gate12A First Replication Checkpoint: Zenodo Metadata Draft

Status: used for published Zenodo dataset `10.5281/zenodo.19340221`

## Recommended Title

Primary:

```text
Gate12A Anchor-Rich Closure Tension Observable Surface: Transcript_v1 and Briefing_v1 First Replication Checkpoint
```

Shorter:

```text
Gate12A Anchor-Rich Closure Tension: Transcript_v1 and Briefing_v1 First Replication Checkpoint
```

Conservative:

```text
Scoped Replication Checkpoint for the Gate12A Anchor-Rich Closure Tension Observable Surface
```

## Recommended Upload Type

- `dataset`

Rationale:

- this deposit is best treated as a reproducible empirical checkpoint bundle
- the software repo remains separately available through Git and GitHub
- the key `runs/` evidence is git-ignored locally, so this checkpoint should use manual Zenodo upload rather than GitHub auto-sync alone

## Published Record

- DOI: `10.5281/zenodo.19340221`
- Record URL: `https://zenodo.org/records/19340221`
- Title used for publication: `Gate12A anchor-rich closure tension observable surface: transcript_v1 + briefing_v1 first replication checkpoint`
- Authors used for publication: `Aoi Kawasaki (Independent Researcher)`

## Fixed Release Tag

Use this tag for the first replication checkpoint:

```text
gate12a-first-replication-checkpoint-2026-03-31
```

## Abstract

```text
This deposit records the first replication checkpoint for the Gate12A anchor-rich closure tension observable surface in pale-ale.

The checkpoint is intentionally scoped. It does not claim a universal law, a threshold doctrine, or a global correctness classifier. Instead, it fixes a reproducible observable surface and the first cross-family comparison over two rendering families: transcript_v1 and briefing_v1.

Under the current Gate12A observable surface, the tracked memo line recorded here establishes the following scoped empirical facts. In both families, zero-overlap transport collapse did not appear (`zero_overlap_count = 0`), all currently defined triangles were anchor-rich, and the directional subregime ordering remained stable (`residual_chord` flatter than `trusted_tree`, `anchor_qualified` flatter than `plain`). At the same time, first-pass phenotype mixtures shifted by family, and high residual did not behave as an immediate failure signal. This supports a narrow non-monotone reading of closure tension at the observable-surface level while explicitly withholding any family-independent threshold claim.

The deposit therefore fixes:
1. the exact tracked memo set for Gate12A empirical scope through the first cross-family comparison,
2. the committed-code run artifacts needed to inspect the current observable surface on transcript_v1 and briefing_v1,
3. the first-pass reading surfaces used for the current extreme-band comparison.

This is a replication checkpoint, not a final theory of LLM closure quality. Its purpose is to preserve a timestamped empirical surface that can be replicated, challenged, or extended by later families and later Gate12 lines.
```

## Description Notes

Keep the public description aligned with these limits:

- this is a `first replication checkpoint`
- this is about the `current Gate12A observable surface`
- this is about `anchor-rich closure tension`
- this is limited to `transcript_v1` and `briefing_v1`

Do not claim:

- universal law
- threshold law
- final theory
- family-independent good/bad classification

## Creators

Used for the published record:

- `Aoi Kawasaki (Independent Researcher)`

Suggested normalized fields for later reuse:

Suggested fields:

- family name
- given name
- affiliation
- ORCID if available

## Keywords

Recommended starting set:

- `LLM auditing`
- `symbolic trajectories`
- `closure tension`
- `holonomy`
- `discrete connection`
- `anchor-rich triangles`
- `non-monotone observable`
- `replication checkpoint`
- `transcript_v1`
- `briefing_v1`

## Suggested Communities

Use only if they match the actual upload context.

Recommended internal shortlist:

- LLM evaluation
- AI auditing
- reproducible research
- computational geometry of reasoning

## Related Identifiers

Used for the published record:

- GitHub repository URL: `https://github.com/Udonburo/pale-ale`
- GitHub release URL: `https://github.com/Udonburo/pale-ale/releases/tag/gate12a-first-replication-checkpoint-2026-03-31`
- Git tag URL: `https://github.com/Udonburo/pale-ale/tree/gate12a-first-replication-checkpoint-2026-03-31`

Optional later identifiers:

Recommended identifiers:

1. GitHub repository URL
   - relation note: source repository for code and tracked memos

2. GitHub release URL for the checkpoint tag
   - relation note: software/release companion to this bundle

3. Git tag URL for the exact release commit
   - relation note: exact code state paired with this deposit

4. Optional later paper/preprint URL
   - only if a paper exists

## Bundle Notes

Attach:

- `CHECKPOINT_README.md`
- `BUNDLE_FILE_LIST.txt`
- `SHA256SUMS.txt`
- all files listed in `BUNDLE_FILE_LIST.txt`

Do not attach:

- local-only `tense` drafts
- incomplete future-family runs
- Rust refactor planning notes
- a GitHub tag snapshot without the supplemental run-artifact bundle

## Release Binding Note

The exact tagged commit hash should be taken from:

- the final git tag target on `origin/main`
- the generated `RELEASE_BINDING.json` inside the assembled bundle
