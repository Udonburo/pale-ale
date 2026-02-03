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
- Optional `inspect` feature for detailed diagnostics

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
