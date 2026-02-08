# pale-ale-core

Geometric auditing engine for LLM embeddings using E8 lattices and geometric algebra. This crate provides deterministic, structure-aware distance metrics that complement cosine similarity.

## Overview

`pale-ale-core` computes a structural distance between embedding vectors by:

- Decomposing vectors into 8D blocks
- Snapping blocks to the 240 roots of the E8 lattice
- Comparing rotors/bivectors to measure flow and consistency

The result is a stable, deterministic signal designed for auditing and analysis.

Unlike cosine similarity (angle-only), this metric detects **block-wise structural drift and flow/topology breaks** that can be invisible in high-dimensional angle metrics.

## Features

- Deterministic E8-based structural distance
- Zero-copy Numpy integration for contiguous arrays
- Zero-alloc hot path for k=1..3 and stack-only dyn path
- NaN/Inf rejection for inputs and parameters
- Optional `python-inspect` feature for detailed diagnostics (the `inspect` feature is kept as a deprecated alias, documented here)
- Python bindings are gated behind the `python` feature (default is pure Rust); `numpy-support` and `python-inspect` imply `python`.
- Model files are cached under the OS cache directory; override with `PA_MEASURE_MODEL_DIR` (legacy: `PALE_ALE_MODEL_DIR`). Offline mode forbids downloads.
- The canonical model spec is pinned to `sentence-transformers/all-MiniLM-L6-v2@e4ce9877abf3edfe10b0d82785e83bdcb973e22e` with embedded BLAKE3 file hashes.
- `pale-ale model print-hashes --json` prints current cached BLAKE3 values and a copy/paste Rust constants block for audited canonical updates.
- `pale-ale model clear-cache --yes` removes only the pinned model/revision cache directory.

## CLI Report Viewer

`pale-ale report` inspects batch NDJSON output (`pale-ale.batch.ndjson`) without loading a model.

Summary examples:

```bash
pale-ale report pale-ale.batch.ndjson
pale-ale report pale-ale.batch.ndjson --summary --top 20 --filter status=HAZY
pale-ale report pale-ale.batch.ndjson --filter has_warning --find abc123 --json
```

Optional TUI build:

```bash
cargo build -p pale-ale-cli --features cli-tui
```

Then run:

```bash
pale-ale report pale-ale.batch.ndjson --tui
```

## CLI Calibration

`pale-ale calibrate` reads batch NDJSON rows and suggests policy ratio thresholds from historical `max_score_ratio` values.

```bash
pale-ale calibrate pale-ale.batch.ndjson --json
```

Human output is a copy-pastable snippet:

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

## Installation

### Python

Use the wrapper package:

```bash
pip install pale-ale
```

`pale-ale` is the higher-level wrapper/CLI; `pale-ale-core` provides the Rust engine and Python bindings.

Core-only (no CLI):

```bash
pip install pale-ale-core
```

### Rust

```toml
[dependencies]
pale-ale-core = "1.0.0"
```

## Usage (Python)

Directly using the core bindings. Note that `spin3_struct_distance` is the primary metric for structural auditing.

```python
import numpy as np
import pale_ale_core

# Must be float64, length % 8 == 0
# Use ascontiguousarray to ensure zero-copy passing to Rust
u = np.ascontiguousarray(np.random.rand(1536), dtype=np.float64)
v = np.ascontiguousarray(u + 0.1, dtype=np.float64)

# 1. Structural-only distance (0.0 .. 1.0) -> Recommended for auditing
d_struct = pale_ale_core.spin3_struct_distance(u, v)

# 2. Mixed distance (Semantic + Structural)
# alpha is a linear mixing weight: (1-alpha)*semantic + alpha*structural
d_mix = pale_ale_core.spin3_distance(u, v, alpha=0.15)

# 3. Detailed Breakdown
components = pale_ale_core.spin3_components(u, v)

print(f"Structural Dist: {d_struct:.6f}")
print(f"Intra-Block:     {components['intra']:.6f}")
print(f"Topology (HCT):  {components['hct']:.6f}")
```

### Python API Reference

| Function | Description |
| --- | --- |
| `spin3_struct_distance(u, v)` | Pure structural distance. `0` = identity, `1` = maximally different under this metric. |
| `spin3_distance(u, v, alpha)` | Mixed distance. Blends a cosine-like semantic distance (normalized dot) with structural distance. |
| `spin3_components(u, v)` | Returns a dictionary of detailed metrics (intra, inter, hct, anchors). |

## Usage (Rust)

```rust
use pale_ale_core::{spin3_components, spin3_struct_distance};

fn main() {
    let u: Vec<f64> = vec![0.1; 8];
    let v: Vec<f64> = vec![0.2; 8];

    let d = spin3_struct_distance(&u, &v).unwrap();
    let components = spin3_components(&u, &v).unwrap();

    println!("d_struct = {:.6}", d);
    println!("d_intra  = {:.6}", components.d_intra);
}
```

## Constraints

- Vector length must be a multiple of 8
- Inputs must be `float64` / `f64`
- Inputs must be finite (NaN/Inf are rejected)
- `alpha` must be finite

## MSRV

- Rust 1.65+

## Note on Numpy Contiguity

When using the Python bindings, contiguous Numpy arrays are borrowed zero-copy. Non-contiguous arrays (e.g., slices like `arr[::2]`) will fall back to an owned copy internally to ensure safety.

## Roadmap & Licensing Note

The project is preparing a Technical Whitepaper detailing the geometric properties of E8 lattices for AI auditing.

The current release is licensed under MPL-2.0 to preserve research integrity during the initial phase. The project is considering a future dual license (MIT / Apache-2.0) after the whitepaper publication to encourage wider adoption. No decision has been made yet.

## License

Licensed under the Mozilla Public License 2.0. See `LICENSE` for details.
