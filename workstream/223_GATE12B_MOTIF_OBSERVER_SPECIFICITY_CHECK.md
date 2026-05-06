# Gate12B Motif Observer Specificity Check

Status: motif observer specificity memo draft
Role: bounded Gate12B dense-transformer family specificity check over existing Gate12A artifacts, not a checkpoint revision, not a release claim, not an invariant-law promotion, and not a Gate12A schema change
Date: 2026-05-06

This memo proceeds from:

- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`
- `217_GATE12B_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_OPENING_MEMO.md`
- `218_GATE12B_FIRST_OBSERVER_RELATIVE_COARSE_GRAINED_CLOSURE_SMOKE_MEMO.md`
- `219_GATE12B_QWEN_2_5_0_5B_THREE_FAMILY_OBSERVER_RELATIVE_COMPARISON_MEMO.md`
- `220_GATE12B_DENSE_TRANSFORMER_FAMILY_EFFECT_EXPANSION_MEMO.md`
- `221_GATE12B_ARCHIVE_STRICT_SUPPORT_SENSITIVITY_MEMO.md`
- `222_GATE12B_ARCHIVE_OBSERVER_SCOPE_EXPANSION_SENSITIVITY_MEMO.md`

## 0. Scope

This memo checks whether `cycle_motif_expansion_v1` is merely a broad
relation-signature extractor or whether it preserves the archive-family
specificity observed in `220` through `222`.

It compares:

- `transcript_v1`
- `briefing_v1`
- `archive_v1`

across:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

It uses:

- `observer_mode_set = cycle_motif_expansion_v1`
- `min_observer_support = 3`
- `min_scale_support = 3`
- `top_k = 1 / 3 / 5`

This is a CPU-only read-only secondary audit over existing Gate12A artifacts.

It does not:

- change the default `core_v1` observer set
- change Gate12A source artifacts
- add another observer design
- mix in sidecar or non-mainline architecture rows
- turn candidate rows into failure labels

## 1. Added Summary Tool

This pass adds a small read-only Gate12B run summarizer:

- `tools/summarize_gate12b_runs.py`

It reads one or more Gate12B run directories and emits:

- `gate12b_run_summary.csv`
- `gate12b_run_summary.json`
- `manifest.json`

The summary includes:

- manifest parameters
- candidate total / high / flat counts
- dominant relation signature by band
- observer support distribution
- scale support distribution
- gauge unstable count
- builder hash match against the current Gate12B runner
- `checksums.json` recomputation status

For this memo, the summary artifact is:

- `runs/gate12b_summary_motif_specificity_dense_transformer/`

The summary covered `36` Gate12B runs:

- `4` models
- `3` families
- `3` `top_k` settings

All `36` summary rows had:

- `checksum_status = ok`
- `builder_script_sha256_matches_current = true`

## 2. Family-Level Result

The key result is that `cycle_motif_expansion_v1` does not force all families
into the archive direction.

Across the `12` archive settings, the signature pair was stable:

- high side: `residual_chord=3`
- flat side: `residual_chord=1|trusted_tree=2`

Across transcript and briefing, the dominant direction was usually the reverse:

- high side: `residual_chord=1|trusted_tree=2`
- flat side: `residual_chord=3`

Family-level signature-pair counts:

| family | high=`residual_chord=3`, flat=`residual_chord=1\|trusted_tree=2` | high=`residual_chord=1\|trusted_tree=2`, flat=`residual_chord=3` | flat > high | flat = high | high > flat |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transcript_v1` | 3 / 12 | 9 / 12 | 3 / 12 | 1 / 12 | 8 / 12 |
| `briefing_v1` | 3 / 12 | 9 / 12 | 3 / 12 | 1 / 12 | 8 / 12 |
| `archive_v1` | 12 / 12 | 0 / 12 | 11 / 12 | 1 / 12 | 0 / 12 |

This preserves the `220` reading:

- archive remains the stable family-conditioned surface
- transcript and briefing remain model-conditioned

It also sharpens the `222` reading:

- motif observer expansion recovers archive strict observer support
- motif observer expansion does not erase family specificity

## 3. top_k = 3 Detail

The middle sensitivity setting is representative.

| family | model | total | high | flat | high signature | flat signature | observer support | scale support |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `transcript_v1` | `Qwen/Qwen2.5-0.5B` | 9 | 6 | 3 | `residual_chord=1\|trusted_tree=2` / 6 | `residual_chord=3` / 3 | `3:6\|4:3` | `3:9` |
| `transcript_v1` | `Qwen/Qwen2.5-3B-Instruct` | 7 | 4 | 3 | `residual_chord=1\|trusted_tree=2` / 4 | `residual_chord=3` / 3 | `3:4\|4:3` | `3:7` |
| `transcript_v1` | `meta-llama/Llama-3.2-3B-Instruct` | 7 | 4 | 3 | `residual_chord=1\|trusted_tree=2` / 4 | `residual_chord=3` / 3 | `3:4\|4:3` | `3:7` |
| `transcript_v1` | `Qwen/Qwen3-4B` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6\|4:3` | `3:9` |
| `briefing_v1` | `Qwen/Qwen2.5-0.5B` | 9 | 6 | 3 | `residual_chord=1\|trusted_tree=2` / 6 | `residual_chord=3` / 3 | `3:6\|4:3` | `3:9` |
| `briefing_v1` | `Qwen/Qwen2.5-3B-Instruct` | 9 | 6 | 3 | `residual_chord=1\|trusted_tree=2` / 6 | `residual_chord=3` / 3 | `3:6\|4:3` | `3:9` |
| `briefing_v1` | `meta-llama/Llama-3.2-3B-Instruct` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6\|4:3` | `3:9` |
| `briefing_v1` | `Qwen/Qwen3-4B` | 6 | 3 | 3 | `residual_chord=1\|trusted_tree=2` / 3 | `residual_chord=3` / 3 | `3:3\|4:3` | `3:6` |
| `archive_v1` | `Qwen/Qwen2.5-0.5B` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6\|4:3` | `3:9` |
| `archive_v1` | `Qwen/Qwen2.5-3B-Instruct` | 6 | 3 | 3 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 3 | `3:3\|4:3` | `3:6` |
| `archive_v1` | `meta-llama/Llama-3.2-3B-Instruct` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6\|4:3` | `3:9` |
| `archive_v1` | `Qwen/Qwen3-4B` | 9 | 3 | 6 | `residual_chord=3` / 3 | `residual_chord=1\|trusted_tree=2` / 6 | `3:6\|4:3` | `3:9` |

## 4. Gauge Boundary

Across the `36` motif specificity runs:

- `33` runs had `gauge_unstable_check_count = 0`
- `3` runs had `gauge_unstable_check_count = 16`

The three nonzero rows were all:

- `Qwen/Qwen3-4B`
- `briefing_v1`
- `top_k = 1 / 3 / 5`

All nonzero checks came from:

- `triangle:000358`
- band transition: `tense -> flat`
- max residual delta: `1.6653345369377348e-16`

`triangle:000358` is not an invariant candidate and is not a
gauge-stable candidate in the affected runs.

This is therefore recorded as the same threshold-boundary caveat seen in the
earlier Qwen3-4B briefing pass, not as a candidate-level instability.

## 5. Reading Boundary

This memo earns the bounded sentence:

- the motif observer expansion is not simply overfitting every family to the
  archive signature direction; archive is stable in one direction across all
  current dense-transformer motif strict settings, while transcript and
  briefing are mostly reversed and remain model-conditioned

It does not earn:

- that archive behavior is universal
- that transcript or briefing behavior is noise
- that `cycle_motif_expansion_v1` should become the default observer mode set
- that candidate dominance is a final metric
- that more observers should be added before this sensitivity line is reviewed

## 6. Short Sentence

The motif observer specificity check strengthens the archive-family read:
under `cycle_motif_expansion_v1`, archive keeps
high=`residual_chord=3` and flat=`residual_chord=1|trusted_tree=2` in `12/12`
strict motif settings, while transcript and briefing mostly show the reverse
direction and remain model-conditioned.
