# Gate12A Discrete Connection Audit

Status: spec-only draft
Role: first Gate12A runner spec for discrete connection audit over fixed upstream artifact families, not connection Laplacian energy, not discrete action, not spin-network dynamics, and not generic graph search
Date: 2026-03-29

Gate12A proceeds from:

- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`
- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`

The current in-worktree Gate12A empirical memo draft is now recorded in:

- `199_GATE12A_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md`

## 0. Scope

Gate12A is the first runner-spec draft under the Gate12 constitution and implementation contract.

Gate12A does:

- fix the exact upstream artifact families accepted by the first runner
- fix the first CLI surface and output directory shape
- fix the deterministic edge pass that emits transport rows
- fix the bounded triangle-only join pass for first holonomy reads

Gate12A does not:

- construct or expose a rich graph object as public output
- search arbitrary cycles
- define energy, action, or variational judgment
- widen the line beyond flat node, edge, and triangle artifacts

## 1. Upstream Input Family

The first runner consumes exactly two upstream artifact families.

### 1.1 Node Local Object Family

This family must provide:

- one `manifest.json`
- one `node_local_object_registry.jsonl`
- one `node_local_object_arrays.npz`

The runner reads from this family only:

- explicit node ids
- auxiliary basis factors
- projector ranks

### 1.2 Explicit Relation-Seed Family

This family must provide:

- one `manifest.json`
- one `explicit_relation_seed_registry.jsonl`

The relation-seed manifest must contain at least:

- `run_id`
- `schema_version`
- `relation_seed_mode = "explicit_edge_seed_v1"`

Required row keys:

- `edge_id`
- `source_node_id`
- `target_node_id`
- `relation_kind`
- `anchor_qualified`
- `anchor_relation_id`

Allowed `relation_kind` values:

- `trusted_tree`
- `residual_chord`

No other upstream source is in scope.

The runner must not:

- mine repo docs directly
- infer hidden edges from prose
- synthesize relation rows from a monolithic graph class

## 2. Runner Surface

The first runner should exist as:

- `tools/run_gate12a_discrete_connection_audit.py`

Required CLI arguments:

- `--node-artifact-dir`
- `--relation-seed-dir`
- `--out-dir`

Optional arguments may tune thresholds, but may not change:

- the artifact family
- the transport law
- the triangle-only cycle scope

## 3. Output Directory Layout

The runner must emit exactly the Gate12A artifact family defined in:

- `197_GATE12A_DISCRETE_CONNECTION_IMPLEMENTATION_CONTRACT.md`

That means:

- `manifest.json`
- `node_local_object_registry.jsonl`
- `node_local_object_arrays.npz`
- `transport_relation_registry.jsonl`
- `transport_operator_arrays.npz`
- `explicit_triangle_cycle_registry.jsonl`
- `triangle_holonomy_registry.jsonl`
- `triangle_holonomy_arrays.npz`
- `gate12a_discrete_connection_status.json`
- `gate12a_discrete_connection_policy_compare.csv`
- `gate12a_discrete_connection_read.md`
- `checksums.json`

## 4. First Edge Pass

The runner must process relation-seed rows in one deterministic pass.

For each explicit relation row it must:

- load the source and target auxiliary basis factors
- compute `M_(j<-i) := U_j^T U_i`
- compute the compact SVD
- determine `k`
- emit the transport case
- emit the square-padded `transport_matrix_local`
- emit the basis-invariant compatibility row

The edge pass must not:

- collapse multiple explicit relations into one aggregate edge
- smooth incompatibility away into a scalar-only summary
- skip zero-overlap rows silently

If `k = 0`, the row must still be emitted as:

- `transport_case = undefined_zero_overlap`
- `transport_level_compatibility_status = undefined`

and its `transport_matrix_local` row must be zero-filled.

## 5. First Triangle Join Pass

The runner must perform one bounded triangle-only join pass after the edge pass.

The admissible cycle family is:

- explicit cycles of length `3` only

The join rule is:

- take two explicit transport rows that share a middle node
- require one explicit closing leg already present in the relation registry
- require at least one `residual_chord` among the three legs
- canonicalize by declared `base_node_id` plus ordered edge-id path

For v1, `ordered edge-id path` means lexicographic order on edge-id strings after the declared `base_node_id` is fixed.

The runner must not:

- run general DFS or BFS cycle search
- enumerate arbitrary simple cycles
- extend beyond path length `3`

## 6. First Holonomy Pass

For each admissible explicit triangle cycle, the runner must:

- read the three square-padded source-to-target local maps
- require `transport_case = equal_rank_orthogonal` on all three legs before any `defined` holonomy read
- require one common node rank `k > 0` across the full triangle
- extract the active `k x k` block from each leg
- multiply those active blocks in the declared cycle order
- compare the product to `I_k`

If the three legs do not share one common node rank `k > 0`, the cycle must be emitted as:

- `holonomy_status = rank_chain_undefined`

If any leg is not `equal_rank_orthogonal`, the cycle must be emitted as:

- `holonomy_status = equal_rank_required`

No fallback holonomy is allowed.

## 7. Determinism Rules

The runner must be deterministic under fixed:

- upstream node-local-object artifacts
- upstream relation-seed rows
- thresholds
- code revision

Cycle ordering must be deterministic.

Registry row ordering must be deterministic.

## 8. Status Surface

The runner must emit:

- `graph_object_policy_status = flat_artifact_only`
- `transport_operator_surface_status = defined`
- `basis_invariant_compatibility_status = defined`
- `triangle_holonomy_scope_status = explicit_triangle_only`

The runner may emit `triangle_holonomy_status` only as:

- `defined_or_equal_rank_required_or_rank_chain_undefined_only`

## 9. Read Surface

The read markdown must state plainly:

- which upstream artifact families were accepted
- that the public implementation surface stayed flat-artifact only
- that the transport law was overlap/SVD based
- that cycle search was triangle-only
- how many zero-overlap edges were emitted
- how many defined holonomy rows were emitted

## 10. Forbidden Shortcuts

The runner spec forbids:

- NetworkX-style or framework-style graph construction as the public surface
- hidden edge synthesis
- heuristic diff transport in place of the declared operator law
- generic loop search
- direct Laplacian or action claims from this runner

## 11. Short Sentence

The Gate12A sentence is:

- `Gate12A takes one node-object family plus one explicit relation-seed family, emits flat edge operators by one deterministic pass, and measures holonomy on explicit triangles only.`

The shortest acceptable memory hook is:

- `stream rows, emit operators, join triangles, stop.`
