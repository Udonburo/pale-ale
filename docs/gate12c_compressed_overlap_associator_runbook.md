# Gate12C-1 Compressed-Overlap Associator Runbook

Status: implementation-only runbook

This runner implements the Gate12C-1 equal-rank alpha measurement contract from:

- `workstream/234_GATE12C_EQUAL_RANK_ALPHA_IMPLEMENTATION_CONTRACT.md`

It is a deterministic artifact reader over Gate12A outputs. It does not run model inference, regenerate Gate12A artifacts, consume Gate12B overlays, add rectangular rank-mismatch support, emit physical-claim terminology, or define a scientific null-excess threshold.

Real empirical execution requires a separate predeclared plan.

## Command

```text
python tools/run_gate12c_compressed_overlap_associator.py \
  --gate12a-dir PATH_TO_GATE12A_ARTIFACT_DIR \
  --out-dir PATH_TO_GATE12C1_OUTPUT_DIR \
  --orientation-null-seed SEED \
  --orientation-null-requested-draw-count N \
  --orientation-null-max-attempt-count M
```

Required arguments:

```text
--gate12a-dir
--out-dir
--orientation-null-seed
--orientation-null-requested-draw-count
--orientation-null-max-attempt-count
```

Validation:

```text
orientation_null_requested_draw_count > 0
orientation_null_max_attempt_count >= orientation_null_requested_draw_count
```

The runner intentionally does not choose a scientific null draw count.

## Tolerances

The implementation exposes these CLI tolerance arguments and records them in `manifest.json`:

| Argument | Default |
| --- | ---: |
| `--tau-overlap-sv-min` | `1e-8` |
| `--tau-overlap-singular-value-abs-error` | `1e-8` |
| `--tau-transport-reconstruction-fro` | `1e-8` |
| `--tau-ordinary-associator-fro` | `1e-10` |
| `--tau-no-compression-associator-fro` | `1e-10` |
| `--tau-split-rel` | `1e-3` |
| `--tau-gauge-operator-covariance-fro` | `1e-8` |
| `--tau-gauge-scalar-delta-abs` | `1e-10` |
| `--epsilon` | `1e-12` |

These are engineering tolerances. They are not empirical discovery thresholds.

## Input Surface

The runner reads the seven Gate12A artifacts required by the contract:

```text
manifest.json
node_local_object_registry.jsonl
node_local_object_arrays.npz
transport_relation_registry.jsonl
transport_operator_arrays.npz
explicit_triangle_cycle_registry.jsonl
triangle_holonomy_registry.jsonl
```

The runner reuses the Gate12C-0 loading and reconstruction discipline:

```text
C_(target<-source) = U_target.T @ U_source
```

It validates:

- stored overlap singular spectra
- stored Gate12A polar or partial-isometry transport
- active ranks
- source/output aliasing and nested output paths
- source artifact immutability across required files

Traversal order is reconstructed from `node_id_path` and edge source/target ids. Lexical `edge_id_path` order is not used as traversal order.

## Measurement

For each eligible root:

```text
v0 -> v1 -> v2 -> v0
M0 = C_(v1<-v0)
M1 = C_(v2<-v1)
M2 = C_(v0<-v2)
```

For each:

```text
q in {1, ..., r - 1}
```

the runner computes:

```text
L_q = Q_q(M2 @ M1) @ M0
R_q = M2 @ Q_q(M1 @ M0)
A_q = L_q - R_q
```

`Q_q` is the compact-SVD best rank-`q` approximation. It is evaluated only for stable observed cuts:

```text
gap_q(M) = (sigma_q - sigma_(q+1)) / max(sigma_1, epsilon)
gap_q > tau_split_rel
```

Near-degenerate observed rows are retained with status fields, but they are not active measurements and are never aggregation eligible.

## Gauge Check

The gauge algorithm id is:

```text
deterministic_node_signed_permutation_gauge_v1
```

For each node id and rank, the implementation builds a signed permutation from:

```text
canonical_json_utf8([
  "gate12c1_gauge_signed_permutation_v1",
  node_id,
  rank
])
```

with:

```text
ensure_ascii = false
separators = (",", ":")
UTF-8 encoding
```

The edge overlaps transform as:

```text
C_(target<-source)' = G_target.T @ C_(target<-source) @ G_source
```

For comparable stable rows, the operator covariance check is:

```text
A_q_transformed ~= G_root.T @ A_q @ G_root
```

The scalar check compares Frobenius norms. Comparable failures are contract failures.

## Orientation Null

The orientation-null mode is:

```text
cycle_shared_spectrum_preserving_operator_null_v1
```

For each cycle and attempt, the runner creates one randomized triangle:

```text
C_e^(b) = L_e^(b) Sigma_e R_e^(b).T
```

The same randomized triangle is reused for all three roots and every `q` in that cycle attempt.

The null is spectrum-preserving and edge-wise coherence-destroying. It is not a gauge transform and is not required to be globally node-basis-realizable.

### Seed Encoding

Each orientation matrix seed is:

```text
SHA256(canonical_json_utf8([
  "gate12c1_orientation_null_v1",
  orientation_null_seed,
  cycle_id,
  edge_id,
  draw_index,
  left_or_right_orientation_label
]))
```

Encoding is fixed as Python JSON with:

```text
ensure_ascii = false
separators = (",", ":")
UTF-8 encoding
```

`draw_index` is a JSON integer. All other values are JSON strings. The implementation does not use naive string concatenation.

### Orthogonal Generator

The orthogonal generator id is:

```text
sha256_counter_box_muller_qr_sign_normalized_v1
```

Algorithm:

1. Compute the SHA-256 digest of the canonical seed bytes.
2. Fill a byte stream with `SHA256(seed_digest || uint64_be(counter))`.
3. Convert each 8-byte big-endian unsigned integer `x` to:

```text
u = (x + 0.5) / 2^64
```

4. Convert uniform pairs to standard normals with Box-Muller:

```text
z0 = sqrt(-2 log u1) * cos(2 pi u2)
z1 = sqrt(-2 log u1) * sin(2 pi u2)
```

5. Fill the `r x r` matrix row-major.
6. Run QR decomposition.
7. Multiply each Q column by `sign(diag(R))`, replacing zero signs with `+1`.

The tests include golden-vector coverage for canonical seed bytes, counter-stream bytes, and the pre-QR normal matrix.

## Cycle-Level Attempt Loop

The attempt loop is cycle-level:

1. Create one null triangle for the cycle attempt.
2. Evaluate that triangle for every root/q row.
3. Append a scalar to a row only when that row's null cuts are stable.
4. Increment that row's invalid-cut count otherwise.
5. Continue until every row has the requested valid count or the cycle reaches the max attempt count.

Rows may complete at different attempt indices. Completed rows do not receive further null values while unfinished rows continue.

## Null Summaries

For each row:

```text
p_upper = (1 + count(a_b >= a_obs)) / (1 + B_valid)
MAD = median(abs(a_b - median(a_b)))
std = sqrt(mean((a_b - mean(a_b))^2))
orientation_null_scale_degenerate = MAD <= epsilon
```

If scale-degenerate:

```text
orientation_null_robust_z = null
orientation_null_excess_status = scale_degenerate
```

Otherwise, complete observed rows use:

```text
orientation_null_robust_z =
  (a_obs - median(a_b)) / (1.4826 * MAD(a_b) + epsilon)
orientation_null_excess_status = descriptive_only
```

Unavailable numeric values are JSON `null`, never NaN or infinity.

## Outputs

The output artifact family is:

```text
manifest.json
triangle_associator_registry.jsonl
triangle_associator_arrays.npz
cycle_associator_summary.jsonl
compression_sweep_summary.csv
gauge_stability_summary.json
spectral_orientation_null_summary.jsonl
gate12c_status.json
gate12c_read.md
checksums.json
```

The NPZ contains:

```text
compressed_overlap_left_operator
compressed_overlap_right_operator
compressed_overlap_associator_operator
```

Registry row order is deterministic:

```text
cycle_id
root_rotation_index
compression_rank_q
```

`operator_array_index` matches the registry row index exactly.

## Process Status

Exit `0` means the implementation completed validly. It does not mean null excess was observed.

Valid data outcomes include:

- no descriptive null excess
- zero aggregation-eligible rows
- insufficient valid null draws recorded correctly
- scale-degenerate null summaries

Exit `1` is reserved for contract or implementation failures such as missing artifacts, reconstruction mismatch, ordinary null failure, no-compression null failure, gauge failure, source mutation, nondeterministic output, or invalid JSON numeric output.

`gate12c_status.json` separates process status, measurement counts, control counts, aggregation eligibility counts, and orientation-null descriptive statuses.
