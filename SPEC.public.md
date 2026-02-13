# Pale Ale - Public Specification

**Version: v1.0.1**

## Purpose

Pale Ale is an **auditor**. It does not generate content.
It evaluates inputs (`query`, `context`, `answer`) under fixed measurement and policy definitions,
then returns an **audit conclusion** (`status`) and supporting **evidence**.

Goals:

- **Authority:** For the same inputs and the same definitions, the tool produces effectively identical audit conclusions regardless of who runs it or where.
- **Reproducibility:** Output JSON is independently verifiable as an audit receipt.
- **Zero-BS UX:** `pale-ale eval ...` completes end-to-end without external scripts or preprocessing.

> Determinism targets invariance of audit-relevant outcomes (`status` and decision-relevant `evidence`),
> not bit-level identity of embedding vectors.

## Measurement Approach

`pale-ale` computes structural distance by:

1. Splitting embedding vectors into 8-dimensional blocks
2. Snapping each block to roots of the E8 lattice (240 roots, soft-snap with configurable top-k and beta)
3. Computing rotor/bivector-based distance metrics across block pairs

Three component scores are produced:

| Score | Meaning |
|---|---|
| `score_sem` | Semantic similarity (cosine-derived, normalized) |
| `score_struct` | Structural distance (E8 rotor decomposition) |
| `score_ratio` | Ratio of structural to semantic signal. Higher = more structural drift. |

## Status Levels

Audit conclusions use three status levels:

| Status | Meaning |
|---|---|
| `LUCID` | Structural and semantic scores are within normal range |
| `HAZY` | Elevated structural drift detected |
| `DELIRIUM` | High structural drift; warrants manual review |
| `UNKNOWN` | Evaluation could not complete (error) |

Status is determined by `max_score_ratio` across sentence pairs, compared against configurable thresholds (`th_ratio_hazy`, `th_ratio_delirium`).

## CLI Commands

Common options shown; run `pale-ale <command> --help` for the full list.

```
pale-ale eval <query> <context> <answer> [--json] [--offline]
pale-ale batch <input> [--out <path>] [--format jsonl|tsv] [--strict] [--dry-run] [--offline] [--json]
pale-ale report <input_ndjson> [--summary] [--top N] [--filter ...] [--find <substr>] [--json] [--tui]
pale-ale calibrate <input_ndjson> [--hazy-q <f64>] [--delirium-q <f64>] [--min-rows <N>] [--out <path>] [--json]
pale-ale model status|download|verify|path [--offline]
pale-ale model print-hashes [--json]
pale-ale model clear-cache [--yes] [--json]
pale-ale doctor [--offline]
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success (any status: LUCID, HAZY, DELIRIUM) |
| `1` | CLI usage error |
| `2` | Runtime error (dependency, environment, or processing failure) |

### `--json` Behavior

When `--json` is specified, output is always valid JSON, including on failure.
Failure output includes `status: "UNKNOWN"` and structured error information.

### `--offline` Behavior

- `eval --offline`: if model is not cached, outputs `UNKNOWN` with error code `MODEL_MISSING_OFFLINE` (exit 2).
- `model download --offline`: fails immediately (exit 2).
- `doctor --offline`: skips network checks.

## Output JSON Shape (eval)

```json
{
  "status": "LUCID | HAZY | DELIRIUM | UNKNOWN",
  "error": null,
  "data": {
    "scores": { "max_score_ratio": 0.0, "..." : "..." },
    "evidence": [ { "ctx_sentence_index": 0, "ans_sentence_index": 0, "score_ratio": 0.0, "..." : "..." } ]
  },
  "audit_trace": {
    "model": { "model_id": "...", "revision": "..." },
    "hashes": { "measurement_hash": "...", "policy_hash": "...", "inputs_hash": "..." },
    "config_source": "default | file | env",
    "warnings": [],
    "attestation_level": "ATTESTED | ATTESTED_WITH_POLICY | UNATTESTED"
  }
}
```

## Batch Output

`batch` writes one NDJSON row per input to `--out <path>` (default: `./pale-ale.batch.ndjson`).
Each row includes `row_index`, `id`, `status`, `error`, `data`, and `audit_trace`.
Output order matches input order.

## Constraints

- Vector length must be a multiple of 8
- Inputs must be finite `f64`
- All parameters (`alpha`, thresholds) must be finite
- Model identity is pinned: `sentence-transformers/all-MiniLM-L6-v2` at a fixed revision with BLAKE3-verified file hashes

## Known Limitations

- **VSCode-based terminals (Windows):** The TUI (`--tui`) may show flickering or residual frames on exit due to ConPTY limitations. Use Windows Terminal or a standalone terminal emulator for best results.
- **Behavior on macOS/Linux integrated terminals** has not been fully verified yet.
- **Sentence splitting** uses a simple boundary-character approach. Abbreviations like `U.S.` or `e.g.` may cause unexpected splits; determinism is prioritized over linguistic accuracy.

## Glossary

| Term | Definition |
|---|---|
| **Measurement** | The fixed pipeline from embedding generation through score computation |
| **Policy** | Rules that map measurements to status and extract evidence |
| **ATTESTED** | Result uses canonical measurement and canonical policy |
| **ATTESTED_WITH_POLICY** | Canonical measurement, custom policy |
| **UNATTESTED** | Non-canonical measurement (reference only) |
| **Evidence** | Sentence-pair fragments that explain an audit conclusion |
