# pale-ale

**Post-hoc geometric auditing of LLM outputs via Gate1 rotor diagnostics and Gate2 Cl⁺(8) telemetry.**

pale-ale measures geometric failure signals from model outputs after inference. Gate1 computes local rotor diagnostics in `SimpleRotor29`; Gate2 upgrades to the closed even subalgebra `Cl⁺(8)` (128 dimensions) to measure composition-dependent observables — holonomy, closure error, and higher-grade energy leakage. It does not modify the model; it measures what the model has already produced.

## How It Works

LLM outputs are projected into 8-dimensional blocks. For each adjacent pair of vectors, pale-ale constructs a **rotor** — the Clifford algebra element that rotates one vector into another. These rotors are then **composed** along the answer trajectory.

In flat space, composing rotors around a closed loop yields the identity. When it doesn't, the residual is **holonomy** — a direct, model-agnostic measurement of geometric path-dependence in the output.

```
ans_0 → ans_1 → ans_2 → ... → ans_N
  R_01     R_12     R_23
         ↓ compose ↓
    R_total  vs  R_direct(ans_0 → ans_N)
         ↓ compare ↓
       closure error (H1-B)
```

## Gate Architecture

pale-ale uses a staged **gate pipeline** — each gate measures a different geometric property, with increasing algebraic complexity.

### Gate 1 — Rotor Diagnostics (v4.0.0)

Operates in `SimpleRotor29` (scalar + 28 bivector components). Measures:

- **AUC** over rotor-derived structural distances
- **Linking sanity** between answer units and context
- **Collapse / entropy gates** for degenerate outputs
- **Run validity** with threshold-based pass/fail

```bash
pale-ale gate1 run --input gate1_input.json --out ./gate1_out \
  --dataset-revision-id rev1 --dataset-hash-blake3 abc123 \
  --spec-hash-raw-blake3 def456 --spec-hash-blake3 ghi789 \
  --unitization-id sentence_split_v1 \
  --rotor-encoder-id encoder@rev \
  --rotor-encoder-preproc-id preproc_v1 \
  --vec8-postproc-id postproc_v1 \
  --evaluation-mode supervised_v1
```

### Gate 2 — Holonomy Telemetry (v4.1.0)

Upgrades representation to `Even128` — the full 128-dimensional even subalgebra Cl⁺(8). Measures composition-dependent observables that are invisible in Gate 1:

| Metric | What it measures |
|---|---|
| **H1-B** (Closure Error) | Discrepancy between composed adjacent rotors and the direct endpoint rotor |
| **H2** (Triangle Holonomy) | Loop closure failure for 3-step triangles — non-zero values indicate path-dependent geometry |
| **H3** (Higher-Grade Energy) | Energy leaking into grades 4, 6, 8 after composition — measures departure from pure spin group |

```bash
pale-ale gate2 run --input gate2_input.json --out ./gate2_out \
  --dataset-revision-id rev1 --dataset-hash-blake3 abc123 \
  --spec-hash-raw-blake3 def456 --spec-hash-blake3 ghi789 \
  --unitization-id sentence_split_v1 \
  --rotor-encoder-id encoder@rev \
  --rotor-encoder-preproc-id preproc_v1 \
  --vec8-postproc-id postproc_v1 \
  --evaluation-mode-id supervised_v1
```

**Gate 2 is telemetry-only in v4.1.0** — it measures and records, but does not invalidate runs. Thresholding is reserved for future versions.

**Output artifacts** (`manifest.json`, `summary.csv`, `samples.csv`) are deterministic: UTF-8/LF, `{:.17e}` float formatting, no NaN/Inf, stable key/column/row ordering.

## Mathematical Foundation

| Component | Detail |
|---|---|
| **Algebra** | Even subalgebra Cl⁺(8) — 128 basis blades across grades {0, 2, 4, 6, 8} |
| **Blade sign** | Swap-count with popcount (`swapcount_popcount_v1`) |
| **Composition** | Strict left-fold over time-reversed sequence, normalize once at end |
| **Distance** | Projective chordal: `d = √(2(1 − min(1, |⟨R₁, R₂⟩|)))` |
| **Determinism** | Fixed accumulation order, `total_cmp` sorting, no intermediate normalization |

The composition order ensures correct temporal application: older rotors act first in the sandwich product `R x ~R`.

Full specifications: [SPEC.phase4.md](SPEC.phase4.md) (Gate 1) · [SPEC.phase4.gate2.md](SPEC.phase4.gate2.md) (Gate 2)

## Workspace Structure

```
crates/
  rotor/       ← leaf math: SimpleRotor29, Even128, Cl⁺(8) algebra (no deps)
  diagnose/    ← metrics, orchestrator, artifact writer, manifest validator
  cli/         ← thin CLI shell (gate1 run, gate2 run, eval, batch, ...)
  embed/       ← model loading and embedding
  modelspec/   ← model specification and verification
```

## Additional CLI Commands

Beyond gate runs, pale-ale includes tools for interactive evaluation and batch processing:

```bash
# Single evaluation
pale-ale eval "query" "context" "answer"

# Batch run → NDJSON report
pale-ale batch input.ndjson --out report_out.ndjson

# Report summary / filtering
pale-ale report report_out.ndjson --summary --top 20

# Threshold calibration from batch output
pale-ale calibrate report_out.ndjson --json

# Environment / model health
pale-ale doctor
pale-ale model status
```

## Audit Binding

When running with `--json`, audit fields are bound in the output envelope:

- `audit_trace.hashes.inputs_hash` — binds raw eval input
- `audit_trace.hashes.measurement_hash` — binds measurement definition
- `audit_trace.hashes.policy_hash` — binds verdict policy
- `audit_trace.model.files[].blake3` — binds model artifacts to pinned hashes

## Rust Library

```toml
[dependencies]
pale-ale-core = "1"
```

```rust
use pale_ale_core::{spin3_components, spin3_struct_distance};

let u = vec![0.1_f64; 8];
let v = vec![0.2_f64; 8];

let d = spin3_struct_distance(&u, &v).unwrap();
let c = spin3_components(&u, &v).unwrap();
println!("d_struct={:.6} d_intra={:.6} d_hct={:.6}", d, c.d_intra, c.d_hct);
```

## Distribution

Current canonical distribution is this monorepo (GitHub tags/releases).

- CLI: build from source (`cargo build -p pale-ale-cli --release`)
- Rust crates in this workspace: use Cargo path dependencies or published crate versions as applicable
- Legacy PyPI packages are pre-monorepo artifacts and retained for reference only (not the current release channel)

## Constraints

- Vector length must be a multiple of 8
- All inputs must be finite `f64`
- Determinism is a hard requirement: identical inputs always produce identical outputs

## Building

```bash
cargo build -p pale-ale-cli --release
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

MSRV: Rust 1.65+

## Roadmap

- **v4.1.1+**: Threshold calibration for Gate 2 observables
- **v4.2+**: Gate 3 — topological features (persistent homology)
- **v4.3+**: Gate 4–5 — conformal fusion, integrated audit pipeline

## License

Licensed under the [Mozilla Public License 2.0](LICENSE).
