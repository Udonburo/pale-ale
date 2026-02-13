# pale-ale

Geometric auditing engine for LLM embeddings using E8 lattice decomposition.

`pale-ale` provides deterministic, structure-aware distance metrics that complement cosine similarity.
The workspace also includes the `pale-ale` CLI for end-to-end audit workflows (`eval`, `batch`, `report`, `calibrate`).

## Why This Exists

Cosine similarity is angle-only. It can miss block-wise structural drift that still matters for auditability.

`pale-ale` measures that drift by:

- Splitting vectors into 8D blocks
- Snapping blocks to the 240 roots of the E8 lattice
- Comparing rotor/bivector behavior across blocks

This yields a deterministic structural signal intended to complement, not replace, semantic similarity.

E8 is used because in 8 dimensions it offers an exceptionally symmetric and dense lattice structure (240 minimal roots; Weyl group order 696,729,600), giving a stable anchor set for block-wise structural comparison.

## Features

- Deterministic E8-based structural distance
- Zero-alloc hot path for k=1..3 and stack-only dynamic path
- Zero-copy NumPy integration for contiguous arrays (via PyO3)
- NaN/Inf rejection for all inputs and parameters
- BLAKE3-verified model integrity with offline mode support

## Quick Start

The binary is `pale-ale` (built from the `pale-ale-cli` crate in this workspace).

```bash
# Development path (fastest iteration; no install needed)
cargo run -p pale-ale-cli -- doctor
cargo run -p pale-ale-cli -- model status
cargo run -p pale-ale-cli -- eval "query" "context" "answer"
```

For a release binary:

```bash
cargo build -p pale-ale-cli --release
target/release/pale-ale doctor
target/release/pale-ale eval "query" "context" "answer"
```

On Windows, use `target\\release\\pale-ale.exe`.

If you also want the TUI report viewer (`report --tui`), build with:

```bash
cargo build -p pale-ale-cli --features cli-tui --release
```

Basic commands:

```bash
# Environment/model health
pale-ale doctor
pale-ale model status

# Single evaluation
pale-ale eval "query" "context" "answer"

# Batch run -> NDJSON report
pale-ale batch input.ndjson --out report_out.ndjson

# Report summary / filtering
pale-ale report report_out.ndjson --summary --top 20
pale-ale report report_out.ndjson --filter status=HAZY --find abc123

# Threshold calibration from batch output
pale-ale calibrate report_out.ndjson --json
```

Calibration produces a copy-pastable policy snippet:

```yaml
policy:
  th_ratio_hazy: 1.5
  th_ratio_delirium: 2.2
calibration:
  method: quantile_nearest_rank
  hazy_q: 0.9
  delirium_q: 0.98
  rows_total: 100
  rows_used: 92
  rows_err: 8
  min_ratio: 1.0
  max_ratio: 4.7
```

Paste the `policy` block into your policy config and re-run `eval`/`batch` with that policy.

## Audit Binding

When running with `--json`, audit-relevant fields are bound in the output envelope:

- `audit_trace.hashes.inputs_hash`: binds raw eval input tuple (`query`, `context`, `answer`)
- `audit_trace.hashes.measurement_hash`: binds measurement definition
- `audit_trace.hashes.policy_hash`: binds verdict policy
- `audit_trace.model.files[].blake3`: binds model artifacts to pinned hashes

This makes results replayable and reviewable as audit receipts.

## Batch NDJSON (Minimal)

Input row (JSONL):

```json
{"id":"1","query":"...","context":"...","answer":"..."}
```

Output row (NDJSON):

```json
{"row_index":0,"id":"1","inputs_hash":"...","status":"LUCID","error":null,"data":{...},"audit_trace":{...}}
```

## Report TUI

```bash
pale-ale report report_out.ndjson --tui
```

> **Note:** `--tui` requires building with `--features cli-tui`.
> **Note (Windows):** The TUI expects full alternate-screen support.
> VSCode-based editors (VSCode, Cursor, Windsurf, Trae, etc.) may show flickering or residual frames on exit due to ConPTY limitations.
> For best results, run `--tui` in **Windows Terminal** or another standalone terminal emulator.
> Behavior on macOS/Linux integrated terminals has not been fully verified yet.
> If `--tui` is unstable in your terminal, use `--summary` (always supported).
> `--json` with `--tui` requires `--summary`.

## Rust Library Usage

Rust library crate naming stays `pale-ale-core` for crates.io namespace safety.

```toml
[dependencies]
pale-ale-core = "1"
```

```rust
use pale_ale_core::{spin3_components, spin3_struct_distance};

fn main() {
    let u = vec![0.1_f64; 8];
    let v = vec![0.2_f64; 8];

    let d_struct = spin3_struct_distance(&u, &v).unwrap();
    let comp = spin3_components(&u, &v).unwrap();

    println!("d_struct = {:.6}", d_struct);
    println!("d_intra  = {:.6}", comp.d_intra);
    println!("d_hct    = {:.6}", comp.d_hct);
}
```

| Function | Description |
| --- | --- |
| `spin3_struct_distance(u, v)` | Structural-only distance (`0` = identity, `1` = maximally different) |
| `spin3_distance(u, v, alpha)` | Semantic + structural blend. `alpha` controls mixing weight. |
| `spin3_components(u, v)` | Detailed breakdown (intra, inter, hct, anchors) |

## Python Bindings

```bash
pip install pale-ale
```

For best performance with NumPy, pass contiguous `float64` arrays.
Non-contiguous arrays are safely copied internally.

## Model Cache and Integrity

- Model files are cached under the OS cache directory; override with `PA_MEASURE_MODEL_DIR`.
- Offline mode (`--offline`) forbids downloads.
- `pale-ale model print-hashes --json` prints cached BLAKE3 hashes for audit.
- `pale-ale model clear-cache --yes` removes only the pinned model revision cache.

## Constraints

- Vector length must be a multiple of 8
- Inputs must be finite `f64` (`float64` in Python)
- `alpha` must be finite

For the full CLI contract, output schema, and measurement details, see [SPEC.public.md](SPEC.public.md).

## MSRV

Rust 1.65+

## Roadmap

The project is preparing a technical whitepaper on geometric auditing with E8 lattices.

## License

Licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE) for details.
