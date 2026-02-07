# Pale-Ale Classic (E8 Edition) - Rust CLI Specification

**Spec v2026-02-05** (Target: Rust CLI `v1.0.1`)

<a id="sec-0"></a>

## 0. Purpose and Non-Negotiables

Pale-Ale Classic is an **auditor**. It does not generate content. It evaluates inputs (`query`, `context`, `answer`, etc.) under fixed measurement and policy definitions, then returns an **audit conclusion** and supporting **evidence**.

The following non-negotiables define this specification:

1. **Authority**
   For the same inputs and the same attested definitions, the tool SHOULD produce effectively identical audit conclusions (`status` and decision-relevant `evidence`) regardless of who runs it or where.
2. **Reproducibility**
   Output JSON MUST be independently verifiable as an audit receipt, with sufficient metadata to validate provenance and comparability.
3. **Zero-BS UX**
   `pale-ale eval ...` MUST complete end-to-end without requiring external Python scripts or manual preprocessing.

> Determinism in this specification does not mean bit-level identity of embedding vectors.
> The primary target is invariance of audit-relevant outcomes (`status` and decision-relevant `evidence`). See [Section 8](#sec-8).

<a id="sec-0-1"></a>

### 0.1 Scope Freeze (Classic Boundary)

- **E8 logic is frozen as Classic**: `240 roots / 8D blocks / Spin3(E8) measurement`.
- **Deferred to next-generation projects**: high-dimensional variants (for example, 432D), Leech-lattice expansion, dynamic phase systems, Hilbert-space extensions, probabilistic monad/sheaf composition, Hamiltonian/Lagrangian dynamics, and similar work.

Classic prioritizes a robust static-lattice auditor, not exploratory architecture expansion.

---

<a id="sec-1"></a>

## 1. Repository and Packaging Strategy

<a id="sec-1-1"></a>

### 1.1 Rust Workspace (Single Repo, Single Binary)

The Rust CLI MUST be implemented inside the `pale-ale-core` repository as a workspace (CLI included).

Rationale:

- supply-chain control,
- unified testing and versioning,
- shortest path to reproducible binary distribution.

Recommended structure:

```text
crates/
  core/        # E8 spin3 measurement engine (minimal change)
  embed/       # embedding generation + model management + attestation materials
  diagnose/    # status/evidence decisions and rule system (spec center)
  cli/         # clap UI + IO + JSON reporting (minimal logic)
```

<a id="sec-1-2"></a>

### 1.2 Python Positioning (Oracle / Maintenance)

The Python implementation (`pale-ale`) is **frozen (maintenance only)** and limited to:

- a regression **oracle** to detect major degradations during Rust migration,
- a historical/prototyping sandbox retained for reference.

**Canon is Rust.** The Rust implementation is authoritative.

---

<a id="sec-2"></a>

## 2. Fixed Responsibility Separation

This separation is mandatory and MUST NOT be blurred:

- `core`: computes distances/components **only** (no decisions)
- `embed`: generates embeddings **only** (no decisions)
- `diagnose`: owns **all** decision-making and accountability (`status` + `evidence`)
- `cli`: IO + formatting (including JSON) **only** (no business logic)

Violating this separation breaks audit accountability and debuggability.

---

<a id="sec-3"></a>

## 3. Attestation Doctrine

Attestation protects **Authority**: what can be treated as canonical measurement and canonical policy.

<a id="sec-3-1"></a>

### 3.1 Attestation Levels

`ATTESTED_WITH_CONFIG` is removed. The only valid levels are:

- **ATTESTED**
  - uses built-in embed with the official pinned model identity,
  - `measurement_hash` matches canonical,
  - `policy_hash` matches canonical,
  - eligible as a submit-worthy audit report.
- **ATTESTED_WITH_POLICY**
  - uses built-in embed with the official pinned model identity,
  - `measurement_hash` matches canonical,
  - `policy_hash` differs (thresholds, extraction rules, etc.),
  - interpreted as "measured with the official ruler, judged with custom policy".
- **UNATTESTED**
  - `measurement_hash` differs, or
  - external vectors / unofficial model / modified build / non-canonical embed path, etc.,
  - reference-only result.

<a id="sec-3-2"></a>

### 3.2 Absolute Rules

- If any measurement-critical element changes (even one bit in canonicalized config), the result MUST be labeled `UNATTESTED`.
- Results with different `measurement_hash` MUST NOT be compared as the same audit conclusion.

---

<a id="sec-4"></a>

## 4. Config Classification

All keys and parameters are classified into a fixed list for `v1.0.1`. Future revisions MUST update this list by explicit diff.

<a id="sec-4-1"></a>

### 4.1 Measurement-Critical (Change => `UNATTESTED`)

Any change in this section implies `UNATTESTED`.

#### Embedding / Tokenization

- model identity: `model_id`, `revision`, and required file hashes (minimum: `model.safetensors`, `tokenizer.json`)
- tokenizer behavior: normalizer / pre-tokenizer / truncation / padding / `max_len` / special tokens
- dtype and execution path: dtype (`fp16`, `fp32`, `int8`, etc.) and execution path (CPU-only is recommended for `v1.0.1`)
- pooling definition: **Masked Mean Pooling** (see [Section 7.2](#sec-7-2))
- whether L2 normalization is applied
- any rounding/quantization (if used, the method and precision MUST be fully fixed)

#### Text -> Embedding Preprocessing

- sentence splitting algorithm, maximum sentence count, whitespace/newline handling
- sentence ordering and overflow join rules (for example, max 64 sentences)
- character normalization policy (fixed explicitly, or explicitly "none")

#### Sentence Split v1 (Measurement-Critical)

1. Input is UTF-8. Only newline normalization is applied (`CRLF/CR -> LF`). No Unicode normalization is applied.
2. `LF` is a hard boundary. Split by `LF`, then process each line independently.
3. Scan each line left-to-right. Sentence boundaries are characters in `{'.','!','?','。','！','？','．','…'}`.
   Consecutive boundary characters (for example, `...`, `…`, `?!`) are treated as one boundary group.
   If trailing closers follow, they stay in the same sentence: `) ] } 」 』 】 》 〉 " '`.
4. Boundary characters are retained at sentence end.
5. Trim leading/trailing ASCII whitespace (`U+0020`, `U+0009`) for each sentence; drop empty sentences.
6. Preserve sentence order. If sentence count exceeds `MAX_SENTENCES (=64)`, keep the first 63 and merge the remainder into the final sentence joined by a single space (`" "`).
7. No special handling for abbreviations (for example, `U.S.`, `e.g.`); determinism is prioritized.

#### MeasurementConfig Fields (Sentence Split v1)

The following fields are fixed in `MeasurementConfig` for `v1.0.1` (all measurement-critical):

- `sentence_split_version`: `"v1"`
- `sentence_split_max_sentences`: `64`
- `sentence_split_normalize_newlines`: `true`
- `sentence_split_per_line`: `true`
- `sentence_split_boundary_chars`: `[".","!","?","。","！","？","．","…"]`
- `sentence_split_closing_chars`: `[")","]","}","」","』","】","》","〉","\"","'"]`
- `sentence_split_keep_boundary`: `true`
- `sentence_split_trim_ascii_ws`: `true`
- `sentence_split_overflow_strategy`: `"merge_tail"`
- `sentence_split_overflow_joiner`: `" "`
- `sentence_split_unicode_normalize`: `false`

#### Spin/E8 Measurement

- E8 snapshot parameters: `k`, `beta`, roots version (`240` fixed)
- structural distance aggregation: weights for `d_intra / d_inter / d_hct` (for example, `0.5/0.3/0.2`)
- semantic distance definition (for example, `0.5 * (1 - cos)` plus EPS behavior)
- score formulas: `score_sem_raw`, `score_sem_used`, `score_ratio`, and any required intermediate definitions

<a id="sec-4-2"></a>

### 4.2 Policy-Only (Change => `ATTESTED_WITH_POLICY`)

Any change here implies `ATTESTED_WITH_POLICY`, assuming measurement remains attested.

- status thresholds (`TH_*`)
- evidence extraction rules: thresholds, caps, ordering, extraction strategy
- exception/rescue rules (for example, FORMAT_OR_ORDER rescue)
- report formatting: display rounding and verbose/dev fields

#### PolicyConfig Fields (v1.0.1)

The following `PolicyConfig` keys are fixed for `v1.0.1` (all policy-only). Key names, semantics, and valid ranges are fixed in this version.
Defaults MAY evolve operationally, but any change MUST update `policy_hash` and MUST be declared by `policy_defaults_rev`.

Core:

- `policy_version`: `"v1"`
- `policy_profile`: `"classic"`
- `policy_defaults_rev`: for example, `"classic-1.0.1"`

Status (`LUCID < HAZY < DELIRIUM`):

- `status_levels`: `["LUCID","HAZY","DELIRIUM"]`
- `status_ratio_lucid_min`: `<f32>` in `[0,1]`
- `status_ratio_hazy_min`: `<f32>` in `[0,1]` (below this is `DELIRIUM`)
- `status_sem_raw_min`: `<f32>` in `[0,1]`
- `status_struct_min`: `<f32>` in `[0,1]`
- `th_ratio_hazy`: `<f32>` (default `1.5`)
- `th_ratio_delirium`: `<f32>` (default `2.2`)

Verdict mapping (v1):

- Let `r_max = max(score_ratio)` across selected measurement pairs.
- if `r_max >= th_ratio_delirium` => `DELIRIUM`
- else if `r_max >= th_ratio_hazy` => `HAZY`
- else => `LUCID`

Evidence (extraction, caps, ordering):

- `evidence_max_items`: `<u32>`
- `evidence_max_per_ctx_sentence`: `<u32>`
- `evidence_max_per_ans_sentence`: `<u32>`
- `max_evidence`: `<usize>` (default `6`)
- `max_evidence_per_answer`: `<usize>` (default `3`)
- `evidence_min_score_ratio`: `<f32>` in `[0,1]`
- `evidence_min_score_sem_raw`: `<f32>` in `[0,1]`
- `evidence_min_score_struct`: `<f32>` in `[0,1]`
- `evidence_sort_key`: `"score_ratio" | "score_sem_raw" | "score_struct"`
- `evidence_sort_desc`: `true | false`
- `evidence_context_window_sentences_before`: `<u32>`
- `evidence_context_window_sentences_after`: `<u32>`

Rules (rescue/escalation):

- `rule_format_or_order_enabled`: `true | false`
- `rule_format_or_order_min_ratio`: `<f32>` in `[0,1]`
- `rule_format_or_order_min_struct`: `<f32>` in `[0,1]`
- `rule_invalid_block_rate_high_enabled`: `true | false`
- `rule_invalid_block_rate_high_threshold`: `<f32>` in `[0,1]`
- `rule_invalid_block_rate_high_min_status`: `"HAZY" | "DELIRIUM"`

Report formatting:

- `report_round_digits`: `<u32>` (display rounding precision)
- `report_include_rule_trace`: `true | false`
- `report_include_evidence_text`: `true | false`
- `report_include_raw_scores`: `true | false`

---

<a id="sec-5"></a>

## 5. Hashing and Canonicalization

<a id="sec-5-1"></a>

### 5.1 Primary Hash Algorithm

Primary hash is **BLAKE3**. The following hashes MUST use BLAKE3 hex:

- `measurement_hash`
- `policy_hash`
- `inputs_hash`
- model file hashes (manifest and report)

BLAKE3 hex MUST be 64 lowercase hex characters (32 bytes).

If future constraints require SHA-256, it MAY be added as secondary fields. Primary MUST remain BLAKE3 to preserve comparability.

<a id="sec-5-2"></a>

### 5.2 Canonicalization (JCS / RFC 8785)

`measurement_hash` and `policy_hash` MUST be computed from:

- UTF-8 bytes of JCS (RFC 8785) canonicalized JSON.

Goal: identical meaning -> identical bytes -> identical hash.

Implementations MUST use a JCS-compliant serializer. Manual implementations MUST match JCS semantics, including number formatting.

<a id="sec-5-3"></a>

### 5.3 Hash Separation

- `measurement_hash = BLAKE3(JCS(MeasurementConfig))`
- `policy_hash      = BLAKE3(JCS(PolicyConfig))`

Optional build trace hashes (for example, `cargo_lock_hash`) MAY be included in `audit_trace.build`.

<a id="sec-5-4"></a>

### 5.4 Comparability Rules

- If `measurement_hash` differs: reports are not comparable.
- If `measurement_hash` matches and `policy_hash` differs:
  - `scores` are comparable.
  - `status` and `evidence` are not comparable (policy differs).

`comparability` SHOULD be `null` for standalone reports (no baseline provided).

---

<a id="sec-6"></a>

## 6. Config Sources and Environment Variables

<a id="sec-6-1"></a>

### 6.1 Env Var Namespacing

Environment variables MUST be split:

- `PA_MEASURE_*` for measurement-critical settings (changes force `UNATTESTED`)
- `PA_POLICY_*` for policy-only settings (changes yield `ATTESTED_WITH_POLICY` if measurement remains attested)

Operational convenience variables MAY be excluded from hash classification, while model identity verification remains mandatory. Example:

- `PA_CACHE_DIR` (general cache location)

Model location overrides are measurement-critical in `v1.0.1`:

- `PA_MEASURE_MODEL_DIR` (primary)
- `PALE_ALE_MODEL_DIR` (legacy alias, deprecated)

<a id="sec-6-2"></a>

### 6.2 Config Source Recording

Output JSON MUST include in `audit_trace`:

- `measurement_hash`, `policy_hash`
- `config_source`: `default | file | env`
- `attestation_level`

---

<a id="sec-7"></a>

## 7. Embedding Pipeline

<a id="sec-7-1"></a>

### 7.1 Fixed Model Identity

Official model identity MUST be pinned by:

- `model_id`
- `revision`
- a required-file manifest including hashes and sizes

For Classic `v1.0.1`, the canonical model identity is:

- `model_id = "sentence-transformers/all-MiniLM-L6-v2"`
- `revision = "e4ce9877abf3edfe10b0d82785e83bdcb973e22e"`

Minimum required files:

- `model.safetensors`
- `tokenizer.json`
- `config.json` (if required by the model architecture)

Canonical required-file BLAKE3 hashes:

- `model.safetensors = 8087e9bf97c265f8435ed268733ecf3791825ad24850fd5d84d89e32ee3a589a`
- `tokenizer.json = 82483bb4f0bdb81779f295ecc5a93285d2156834e994a2169f9800e4c8f250c1`
- `config.json = 02ba870d29dc00b373fe71bd273baca30586d6577def4130456e756c7b286890`

Manifest hashes MUST be 64 lowercase hex chars. Invalid manifest hashes MUST fail verification/download with an explicit error.

After download, file hashes MUST be verified. Any mismatch MUST return an error.
Download provenance sidecar metadata (`*.meta.json`) SHOULD be written atomically via same-directory temp file + rename.
Missing or invalid sidecar metadata MUST NOT break verification; provenance fields may be `null`.

<a id="sec-7-2"></a>

### 7.2 Masked Mean Pooling (Normative)

A naive `mean(dim=1)` is forbidden (padding contamination). The embedding pipeline MUST:

1. Expand `attention_mask` to `[B, S, 1]` and multiply into `hidden_state` (zero invalid tokens).
2. Sum over the sequence dimension.
3. Compute valid token count from mask sum and clamp with `min=1e-9`.
4. Divide `sum / count` to produce pooled vectors.
5. Apply L2 normalization.

If stabilization steps are added (for example, cast to `f32`, per-element `round(1e-6)`), the exact choice and procedure MUST be treated as measurement-critical.

---

<a id="sec-8"></a>

## 8. Determinism Policy

- Target: same inputs -> invariant `status` and decision-relevant `evidence` (in principle).
- Non-target: bit-level identity of embedding vectors.

Stabilization is expected in `embed` (fixed tokenizer, strict pooling, L2, fixed dtype, optional rounding). E8 snapping is expected to absorb small numeric drift as a final filter.

---

<a id="sec-9"></a>

## 9. Core Handling for None / Invalid Blocks

If `normalize8` returns `None` (for example, low norm), this MUST NOT be hidden. It MUST be recorded as measurement missingness.

Normative behavior:

- `core` SHOULD remain operational and SHOULD NOT crash on these cases.
- `spin3_components` MUST include:
  - `total_blocks`
  - `valid_blocks`
  - `invalid_blocks`
  - `invalid_block_rate`
- `diagnose` MUST be able to escalate or tag by missingness rate (policy-only).
  - Example: tag `INVALID_BLOCK_RATE_HIGH` and enforce minimum `HAZY`.
- `audit_trace.invalid_block_rate` MUST be present.

---

<a id="sec-10"></a>

## 10. CLI Contract

UX is part of the specification.

<a id="sec-10-1"></a>

### 10.1 Commands

Minimum command set:

- `pale-ale eval <query> <context> <answer> [--json] [--offline] [--config <path>]`
- `pale-ale inspect ... [--pair i:j] [--dev]`
- `pale-ale model status|download|verify|path [--offline]`
- `pale-ale model print-hashes [--json]` (dev helper, read-only)
- `pale-ale model clear-cache [--yes] [--json]` (dev helper, scoped to pinned model/revision)
- `pale-ale doctor [--offline]`

A standalone `embed` command is not required for `v1.0.1`. If added for debugging, outputs MUST be labeled `UNATTESTED`.

<a id="sec-10-2"></a>

### 10.2 Exit Codes and `--json` Behavior

Fixed exit codes:

- `0`: success (`LUCID`, `HAZY`, `DELIRIUM`, etc.)
- `1`: CLI usage error
- `2`: dependency/environment error (missing model, permission issues, offline download attempts, corruption, hash mismatch, etc.)
- `3`: internal error (captured panic, invariant violation, etc.)

`--json` contract:

- When `--json` is specified, the tool MUST output JSON even on failure.
- Failure output MUST include:
  - `status: "UNKNOWN"`
  - `audit_trace.error: { code, message }`
- Exit code MUST still follow the fixed classes above.

<a id="sec-10-3"></a>

### 10.3 `--offline` Contract

- `eval --offline`:
  - if model is not cached, MUST output JSON with `UNKNOWN` and error code `MODEL_MISSING_OFFLINE`,
  - exit code MUST be `2`.
- `model download --offline`: MUST fail immediately with error code `OFFLINE_FORBIDS_DOWNLOAD` (exit `2`).
- `model status|verify --offline`: MUST work if network is not required (success `0` / failure `2`).
- `doctor --offline`: MUST skip network checks and fail only for fatal local issues (exit `2`).

---

<a id="sec-11"></a>

## 11. Output JSON (Audit Report)

<a id="sec-11-1"></a>

### 11.1 Top-Level Shape

Root object MUST contain exactly these four top-level blocks:

- `status`
- `scores`
- `evidence`
- `audit_trace`

Additional top-level keys SHOULD NOT be added in `v1.0.1`.

<a id="sec-11-2"></a>

### 11.2 `audit_trace` Required Fields

`audit_trace` MUST include:

- `tool`: `{ name, version, features?, target_triple? }`
- `model`: `{ model_id, revision, files: [{ path, blake3, size_bytes }] }`
- `hashes`: `{ measurement_hash, policy_hash, inputs_hash }` (BLAKE3 hex)
- `config_source`: `default | file | env`
- `attestation_level`: `ATTESTED | ATTESTED_WITH_POLICY | UNATTESTED`
- `invalid_block_rate`
- `comparability`: `null` for standalone reports, or a struct for compare mode
- `error` (when present): `{ code, message }`
  - model download/verify diagnostics MAY include provenance in `error.details`:
    `final_url`, `etag`, `content_length`

Recommended build metadata:

- `build`: `{ git_commit?, git_dirty?, rustc_version?, profile?, cargo_lock_hash? }`

`inputs_hash` definition (normative):

- `inputs_hash = BLAKE3(JCS({ query: <string>, context: <string>, answer: <string> }))`

No trimming or normalization is applied unless explicitly defined as measurement-critical preprocessing.

`model verify --json` detail contract (CLI envelope):

- success: `data.details` MUST contain per-file entries with states (`MATCH`)
- failure due file-level issues: `data.details` MUST still be present and include per-file states (`MATCH`, `MISSING`, `MISMATCH`)
- detail entries SHOULD include provenance fields when available:
  `final_url`, `etag`, `content_length`, `size_bytes`
- `model print-hashes --json` SHOULD include `data.warning` clarifying hashes are computed from local cache bytes.

<a id="sec-11-3"></a>

### 11.3 Evidence Minimum Fields

Each `evidence` element MUST include at minimum:

- `ctx_sentence_index`, `ans_sentence_index`
- excerpt text fields (for example, `text`, `match_context`, etc.)
- `score_struct`, `score_sem`, `score_ratio`
- `tags: string[]`
- `rule_trace: string[]`

Evidence ordering and cap rules (deterministic v1):

- Group by `ans_sentence_index`.
- Inside each answer group, sort by:
  higher `score_ratio`, then lower `ctx_sentence_index`, then lower `ans_sentence_index`.
- Take `max_evidence_per_answer` per answer group.
- Merge candidates globally and sort by:
  higher `score_ratio`, then lower `ans_sentence_index`, then lower `ctx_sentence_index`.
- Take `max_evidence` total.

<a id="sec-11-4"></a>

### 11.4 Comparing Reports

Comparison interfaces MAY ship in `v1.0.2` or later. Intended interface:

- `pale-ale eval ... --baseline <report.json>`
- or `pale-ale compare <a.json> <b.json>`

Only when baseline/compare mode is used should `comparability` be non-null.

---

<a id="sec-12"></a>

## 12. Tests and CI

Required for `v1.0.1`:

1. **Golden E2E**: fixed input set -> key JSON fields (`status`, decision-critical `evidence`, and essential `audit_trace`) are stable.
2. **Python Oracle**: not bit-equality; detects major regressions (drift thresholds are acceptable).
3. **Unicode / Fuzz**: zero-width characters, combining marks, emoji, and long text MUST NOT crash the tool.
4. **Core Invariants**: symmetry, boundedness, and shuffle sensitivity (preserve existing foundations).

CI MUST:

- build and test on Windows / macOS / Linux,
- test `--offline` behavior with missing model: output `UNKNOWN` and exit `2`.

---

<a id="sec-13"></a>

## 13. Release Plan

- **v1.0.1 (Initial Rust CLI)**
  - `model download/verify/path`
  - `eval` producing a full JSON audit report (`audit_trace` included)
  - `embed` (Candle) with pooling/L2/dtype fixed
  - `diagnose` ported and treated as canonical
  - offline contract fixed
- **v1.0.2 (Patch)**
  - tighten tokenizer/splitting/pooling deltas
  - formalize compare/baseline and `comparability`
- **v1.1.x**
  - offline model pack (zip)
  - signed policy bundles (if needed)

---

<a id="sec-14"></a>

## 14. Implementation Order

Recommended PR sequence:

1. `MeasurementConfig` / `PolicyConfig` + JCS + BLAKE3 + `audit_trace` skeleton
2. `attestation_level()` + `--json` / exit codes / offline contract
3. `model download/verify/path` (built-in manifest; verification required)
4. `embed` (Candle inference + Masked Mean Pooling + L2 + fixed dtype)
5. `eval` E2E wiring (it may return `UNKNOWN` initially; structure MUST be correct)
6. Port `diagnose` (status + evidence), then finalize Golden E2E

---

<a id="glossary"></a>

## Glossary

- **Auditor / Audit Tool**: a tool that returns an audit conclusion and grounds; it does not generate content. See [Section 0](#sec-0), [Section 11](#sec-11).
- **Authority**: same inputs -> effectively same conclusion under the same attested measurement and policy. See [Section 0](#sec-0), [Section 3](#sec-3), [Section 8](#sec-8).
- **Reproducibility**: output report is independently verifiable as an audit receipt. See [Section 0](#sec-0), [Section 11.2](#sec-11-2).
- **Zero-BS UX**: `eval` completes without external preprocessing or Python. See [Section 0](#sec-0), [Section 10](#sec-10).
- **Measurement**: the ruler from embedding generation through score computation. See [Section 4.1](#sec-4-1), [Section 7](#sec-7).
- **Policy**: rules that map measurements to `status` and extract `evidence`. See [Section 4.2](#sec-4-2), [Section 11.3](#sec-11-3).
- **Measurement-Critical**: any change forces `UNATTESTED`. See [Section 4.1](#sec-4-1), [Section 3.2](#sec-3-2).
- **Policy-Only**: any change yields `ATTESTED_WITH_POLICY` (if measurement remains attested). See [Section 4.2](#sec-4-2), [Section 3.1](#sec-3-1).
- **Attestation**: classification of whether measurement and policy match canonical definitions. See [Section 3](#sec-3).
- **ATTESTED / ATTESTED_WITH_POLICY / UNATTESTED**: the only allowed attestation levels. See [Section 3.1](#sec-3-1).
- **Canonicalization (JCS / RFC 8785)**: canonical JSON serialization used for hashing config objects. See [Section 5.2](#sec-5-2).
- **measurement_hash / policy_hash**: hash identifiers for measurement and policy equivalence. See [Section 5.3](#sec-5-3), [Section 11.2](#sec-11-2).
- **BLAKE3**: primary hashing algorithm for this spec. See [Section 5.1](#sec-5-1).
- **Comparability**: whether reports can be compared; requires matching `measurement_hash`. See [Section 5.4](#sec-5-4), [Section 11.4](#sec-11-4).
- **Evidence**: ground fragments that explain an audit conclusion. See [Section 11.3](#sec-11-3).
- **Status**: the audit conclusion label (`LUCID`, `HAZY`, `DELIRIUM`, or `UNKNOWN` on failure). See [Section 11.1](#sec-11-1).
- **Masked Mean Pooling**: padding-safe mean pooling defined in [Section 7.2](#sec-7-2).
- **Invalid Block / invalid_block_rate**: recorded measurement missingness rate. See [Section 9](#sec-9).
- **Offline Mode (`--offline`)**: network-prohibited execution contract. See [Section 10.3](#sec-10-3).
