# pale-ale-core

[![PyO3](https://img.shields.io/badge/backend-PyO3-blue.svg)](https://pyo3.rs)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> *Geometry-aware structural distance for embedding evaluation.*

**Geometry-aware distance metrics for embedding pairs (Rust core for pale-ale).**

`pale-ale-core` provides a structural distance signal that is intentionally **not** the same thing as cosine similarity.
The goal is to distinguish:
* **Semantic proximity:** Topic-level similarity (what Cosine does).
* **Structural integrity:** Order, flow, and logical consistency (what this crate adds).

This crate is the high-performance backend for the Python package **[pale-ale](https://github.com/Udonburo/pale-ale)**.

---

## ⚡ The Problem: Cosine Similarity is Structure-Blind

Standard cosine similarity measures the angle between two vectors:

$$\cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

For modern LLM embeddings, cosine often remains high even when the **structure** is broken. `pale-ale-core` adds a second axis: **structure-sensitive distance** computed over fixed-size blocks.

| Pair | Cosine Similarity | Structural Distance | Typical outcome in pale-ale |
| :--- | :--- | :--- | :--- |
| "AI is great." vs "AI is great!" | High | Low | ✅ Robust (Surface change) |
| "AI is great." vs "Great is AI." | High | **Higher** | 🚨 Order/Logic Distortion |
| "The sky is blue." vs "The sky is green." | High | Varies | Semantic contradiction |

> **⚠️ Note:** These examples are illustrative. Exact values depend on the embedding model and preprocessing. This crate produces geometric signals, not truth judgments. Diagnosis labels are produced by [pale-ale](https://github.com/Udonburo/pale-ale) (Python), not by this crate.

---

## 📊 Return Value

`spin3_distance` returns a **distance score in `[0.0, 1.0]`**:

| Value | Meaning |
|-------|---------|
| `0.0` | Identical (no distance) |
| `1.0` | Maximally different |

The `alpha` parameter controls the semantic/structural mix:
* `alpha = 0.0` → Pure semantic distance (≈ `0.5 * (1 - cosine)`)
* `alpha = 1.0` → Pure structural distance (geometry-based)
* `alpha = None` → Default: `0.15`

---

## 🚀 Key Concepts (Implementation)

### 1. Block Decomposition (8D)
Embedding vectors are sliced into 8-dimensional blocks. This allows us to analyze local geometric properties rather than collapsing everything into a single scalar.

### 2. E8 Root System as a Codebook
We use the **240 roots of the E8 root system** as an optimal, high-symmetry reference codebook in 8D for blockwise "snapping":
* **`snap_soft`:** A Boltzmann-weighted mixture (top-k, beta=12.0) to find the nearest structural "anchor" in 8D space.
* **Why E8?** The E8 lattice provides an extremely dense sphere packing in 8 dimensions, making it an ideal "geometric ruler" for quantization-like stabilization.
* *Clarification: We are not claiming embeddings naturally live on an E8 lattice. We use E8 roots as a reference basis.*

### 3. Structure-Sensitive Metrics
The structural score (`d_struct`) is a composite of:
* **Intra-block:** Deformation within a block (continuous vs. snapped).
* **Inter-block:** Change of bivector "flow" (rotors) between adjacent blocks.
* **HCT (Holonomy-Curvature-Transport):** Higher-order consistency checks across the sequence.

---

## 📦 Installation

### Python

**Recommended for end users (CLI):** Install with CLI and full features:
```bash
pip install "pale-ale[spin]"
```

**Core only:** Low-level distance API without CLI:
```bash
pip install pale-ale-core
```

> This installs only the `pale_ale_core` module (no `pale-ale` CLI).

**Python usage (core only):**
```python
import pale_ale_core

u = [0.1] * 8
v = [0.2] * 8
d = pale_ale_core.spin3_distance(u, v, None)  # None -> default alpha=0.15
print(f"Distance: {d:.6f}")
```

### Rust

This crate is not yet published to crates.io. For now, use git dependency:

```toml
[dependencies]
pale-ale-core = { git = "https://github.com/Udonburo/pale-ale-core" }
```

---

## 🛠️ Usage (Rust)

```rust
use pale_ale_core::spin3_distance;

fn main() {
    // Input vectors must be a multiple of 8 dimensions (e.g. 768, 1024, 1536)
    let u: Vec<f64> = vec![/* ... */];
    let v: Vec<f64> = vec![/* ... */];

    // alpha controls the mix:
    // 0.0 = semantic-only (Cosine-like)
    // 1.0 = structural-only (Geometry-based)
    match spin3_distance(&u, &v, Some(0.5)) {
        Ok(d) => println!("Geometric Distance: {:.6}", d),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

---

## 🛡️ Developer: Audit Mode (feature: `inspect`)

> **Note:** This is a source-build feature, not available in prebuilt PyPI wheels.

To see *why* a score is low, enable the `inspect` feature when building from source:

```bash
maturin develop --release --features inspect
```

This enables `spin3_inspect(...)` in Python, returning a dictionary of sub-components (`intra`, `inter`, `hct`, `semantic`, etc.). Useful for debugging and research.

> Rust developers can access the same internals directly via the source code.

---

## ⚠️ Constraints

* **Dimensions:** Input vectors must have length divisible by 8. If your model uses non-standard dimensions, please pad with zeros.
* **Compute Cost:** E8 snapping evaluates 240 roots per block. This is computationally heavier than simple cosine similarity, by design.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

The Python wrapper [pale-ale](https://github.com/Udonburo/pale-ale) is also Apache-2.0 licensed.
