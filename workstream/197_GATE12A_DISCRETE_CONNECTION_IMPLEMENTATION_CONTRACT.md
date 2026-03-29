# Gate12A Discrete Connection Implementation Contract

Status: spec-only draft
Role: Gate12 implementation contract for flat discrete-connection artifacts, not connection Laplacian energy, not discrete action, not spin-network state dynamics, not sheaf or cohomology doctrine, and not a rich graph-object framework
Date: 2026-03-29

Gate12A proceeds from:

- `196_GATE12_DISCRETE_CONNECTION_CONSTITUTION.md`
- `195_GATE11_ADMISSIBILITY_MEMBRANE_FREEZE.md`
- `26_GATE9_GRAPH_GAUGE_CONSTITUTION.md`
- `08_GATE6A_ARTIFACT_SCHEMA.md`

The current in-worktree Gate12A first runner spec draft is now recorded in:

- `198_GATE12A_DISCRETE_CONNECTION_AUDIT.md`

## 0. Scope

Gate12A is the first implementation-contract draft under the Gate12 constitution.

Gate12A does:

- define the flat public artifact family for node-local objects, explicit transport relations, and explicit triangle-cycle holonomy
- fix the first cold transport operator law in explicit linear-algebra terms
- fix the first public basis-invariant compatibility judgment
- fix the first bounded loop scope as explicit triangle cycles only

Gate12A does not:

- introduce a mandatory rich graph object as the public implementation surface
- define connection Laplacian, energy, or action
- generalize to arbitrary cycle search
- promote bundle, Yang-Mills, Berry, sheaf, cohomology, or spin-network language into the contract surface

## 1. Public Implementation Stance

Gate12A is artifact-first.

The public implementation surface is not:

- one monolithic in-memory graph object
- a framework-specific graph class
- an implicit adjacency structure hidden behind helper code

The public implementation surface is:

- flat row-indexed JSONL registries
- deterministic array artifacts for local operators
- explicit run manifests and checksums

Internal helper indices or adjacency caches may exist during execution, but they are not public doctrine and may not replace the flat artifact family as the contract surface.

## 2. Run Directory Layout

Any Gate12A implementation must emit a run directory of the form:

- `runs/gate12a_discrete_connection_<run_id>/`

Required files:

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

## 3. manifest.json

Required fields:

- `run_id`
- `schema_version`
- `method_id = "gate12a_discrete_connection_v1"`
- `code_git_commit`
- `builder_script_sha256`
- `graph_object_policy = "flat_artifact_only_v1"`
- `local_object_mode = "projector_factor_public_basis_aux_v1"`
- `transport_operator_mode = "polar_overlap_v1"`
- `rank_mismatch_mode = "svd_partial_isometry_v1"`
- `cycle_mode = "explicit_triangle_only_v1"`
- `compatibility_mode = "singular_spectrum_basis_invariant_v1"`
- `holonomy_mode = "triangle_equal_rank_orthogonal_fro_residual_v1"`
- `tau_overlap_sv_min`
- `tau_transport_gap_fro`
- `tau_holonomy_residual_fro`
- `input_manifest_refs`

`input_manifest_refs` must point to exactly:

- one upstream node-local-object artifact family
- one upstream explicit relation-seed artifact family

The implementation surface may not mine repo docs or repo-wide historical text directly.

## 4. node_local_object_registry.jsonl

One JSON object per explicit node.

Required keys:

- `node_id`
- `node_label`
- `basis_array_index`
- `projector_rank`
- `local_object_status`

Gate12A does not require a public node subtype taxonomy in v1.

Allowed `local_object_status` values:

- `defined`
- `undefined_rank_zero`
- `undefined_aux_basis_missing`

## 5. node_local_object_arrays.npz

This file carries the auxiliary representation of the public local object.

Required arrays:

### A. `basis_factor`

Shape:

- `[N, d_model, r_max]`

Policy:

- columns are orthonormal where active
- inactive trailing columns are zero-padded
- the public projector is implicit as `P_i = U_i U_i^T`
- dense full `P_i in R^(d x d)` is not a required v1 artifact

### B. `rank_active`

Shape:

- `[N]`

## 6. transport_relation_registry.jsonl

One JSON object per explicit transport-bearing relation.

Required keys:

- `edge_id`
- `source_node_id`
- `target_node_id`
- `relation_kind`
- `anchor_qualified`
- `anchor_relation_id`
- `source_rank`
- `target_rank`
- `overlap_rank`
- `transport_case`
- `operator_array_index`
- `compatibility_gap_fro`
- `transport_level_compatibility_status`

Allowed `relation_kind` values:

- `trusted_tree`
- `residual_chord`

Allowed `transport_case` values:

- `equal_rank_orthogonal`
- `rank_mismatch_partial_isometry`
- `undefined_zero_overlap`

`anchor_qualified = true` means only that an explicit anchor relation qualifies the edge judgment.

It does not create a third edge ontology and does not replace the underlying `trusted_tree` or `residual_chord` relation kind.

## 7. First Transport Operator Law

Let node `i` carry auxiliary orthonormal basis factor `U_i in R^(d x r_i)` and node `j` carry `U_j in R^(d x r_j)`.

Define the overlap operator:

- `M_(j<-i) := U_j^T U_i`

Let the compact singular value decomposition be:

- `M_(j<-i) = A Sigma B^T`

where active singular values satisfy:

- `sigma_l > tau_overlap_sv_min`

Let `k` be the count of active singular values.

The first transport law is:

- if `r_i = r_j = k`, define `T_(j<-i) := A B^T`
- if `k > 0` but the ranks do not match, define the active transport operator as the partial isometry `T_(j<-i)^act := A_k B_k^T`
- if `k = 0`, the transport operator is undefined and must be emitted as `undefined_zero_overlap`

For `undefined_zero_overlap`, the corresponding `transport_matrix_local` row must be zero-filled.

The public stored transport operator is the source-to-target local map embedded into square padding:

- rows are indexed by target-local coordinates
- columns are indexed by source-local coordinates
- the active rectangular block is written into the top-left `r_j x r_i` corner
- the remaining rows and columns are zero-padded to `r_max x r_max`

So `transport_matrix_local` is a square storage surface for a rectangular source/target map, not a claim that every active transport operator is intrinsically square.

Gate12A does not allow:

- an arbitrary learned transport map
- a diff-style heuristic in place of an explicit linear operator
- a transport definition that is not recoverable from the emitted auxiliary local-object factors

## 8. First Public Compatibility Invariant

The first public compatibility judgment is basis-invariant.

It is derived from the active singular spectrum of `M_(j<-i)`, not from raw basis coordinates alone.

Required public scalar:

- `compatibility_gap_fro := ||I_k - Sigma_k||_F`

Required public judgment:

- `transport_level_compatibility_status = compatible` if `compatibility_gap_fro <= tau_transport_gap_fro`
- `transport_level_compatibility_status = incompatible` if `compatibility_gap_fro > tau_transport_gap_fro`
- `transport_level_compatibility_status = undefined` if `k = 0`

In v1, this basis-invariant compatibility judgment is the public invariant target.

## 9. transport_operator_arrays.npz

Required arrays:

### A. `transport_matrix_local`

Shape:

- `[E, r_max, r_max]`

Policy:

- each row stores the zero-padded source-to-target local transport map with target rows and source columns
- dense full `d x d` transport matrices are not required in v1

### B. `overlap_singular_values`

Shape:

- `[E, r_max]`

### C. `active_rank`

Shape:

- `[E]`

## 10. Explicit Triangle Cycle Scope

Gate12A does not perform generic cycle enumeration.

The only allowed cycle family in v1 is:

- explicit triangle cycles of length `3`

That means:

- no arbitrary simple-cycle search
- no path-length growth beyond `3`
- no implicit closure

An admissible triangle cycle must contain:

- three explicit named transport relations
- an explicit return leg already materialized in the relation registry
- at least one `residual_chord` relation

## 11. explicit_triangle_cycle_registry.jsonl

One JSON object per admissible explicit triangle cycle.

Required keys:

- `cycle_id`
- `base_node_id`
- `edge_id_path`
- `node_id_path`
- `cycle_length`
- `cycle_status`

Required fixed value:

- `cycle_length = 3`

Allowed `cycle_status` values:

- `admissible_explicit_triangle`

## 12. Triangle Holonomy Law

Let an admissible triangle cycle be:

- `C = (n0 -> n1 -> n2 -> n0)`

with associated active local transport operators:

- `T_(1<-0)`
- `T_(2<-1)`
- `T_(0<-2)`

Holonomy is defined only when all three transport legs satisfy both:

- `transport_case = equal_rank_orthogonal`
- the three legs share one common node rank `k > 0`

For holonomy composition, each leg contributes its active `k x k` local block from the square-padded storage surface described in Section 7.

No rank-mismatch partial-isometry leg may participate in a `defined` holonomy row.

In that case define:

- `H_C := T_(0<-2)^(act,k) T_(2<-1)^(act,k) T_(1<-0)^(act,k)`

and the first public holonomy residual:

- `holonomy_residual_fro := ||H_C - I_k||_F`

Here `T_(j<-i)^(act,k)` means the active `k x k` block of the stored square-padded transport matrix for an `equal_rank_orthogonal` leg.

If the three legs do not share one common node rank `k > 0`, the cycle must be emitted as:

- `holonomy_status = rank_chain_undefined`

If any leg is not `equal_rank_orthogonal`, the cycle must be emitted as:

- `holonomy_status = equal_rank_required`

If any required closing leg is absent, the cycle must not be emitted as holonomy-bearing at all.

## 13. triangle_holonomy_registry.jsonl

One JSON object per emitted triangle holonomy read.

Required keys:

- `cycle_id`
- `base_node_id`
- `holonomy_rank`
- `holonomy_residual_fro`
- `holonomy_status`

Allowed `holonomy_status` values:

- `defined`
- `equal_rank_required`
- `rank_chain_undefined`

## 14. triangle_holonomy_arrays.npz

Required arrays:

### A. `holonomy_matrix_local`

Shape:

- `[C, r_max, r_max]`

Policy:

- each row stores the zero-padded local holonomy operator for `defined` cycles
- undefined cycles are zero-filled and marked only through the registry

## 15. gate12a_discrete_connection_status.json

Required status keys:

- `graph_object_policy_status`
- `transport_operator_surface_status`
- `basis_invariant_compatibility_status`
- `triangle_holonomy_scope_status`
- `triangle_holonomy_status`

Required status values in a successful v1 emission:

- `graph_object_policy_status = flat_artifact_only`
- `transport_operator_surface_status = defined`
- `basis_invariant_compatibility_status = defined`
- `triangle_holonomy_scope_status = explicit_triangle_only`
- `triangle_holonomy_status = defined_or_equal_rank_required_or_rank_chain_undefined_only`

## 16. gate12a_discrete_connection_policy_compare.csv

This file must contain one deterministic summary row that echoes the status surface and the run-level counts for:

- nodes
- transport relations
- explicit triangle cycles
- defined triangle holonomy rows

## 17. gate12a_discrete_connection_read.md

This file must state in plain language:

- that the public implementation surface is flat artifacts rather than a rich graph object
- what transport operator law was used
- what basis-invariant compatibility judgment was used
- that only explicit triangle cycles were measured
- whether any defined holonomy rows were emitted

## 18. Forbidden Shortcuts

Gate12A forbids:

- NetworkX-style or framework-style graph classes as the public output surface
- silent contraction of anchor qualification into a new edge type
- replacing the linear transport operator with a heuristic diff score
- arbitrary cycle mining
- longer-loop search in v1
- energy, action, or variational claims derived from this contract alone

## 19. Short Sentence

The Gate12A sentence is:

- `Gate12A emits flat node, edge, and triangle artifacts; transport is an explicit overlap-derived operator, compatibility is basis-invariant, and holonomy is measured on explicit triangles only.`

The harshest acceptable memory hook is:

- `no rich graph object, no fake transport heuristic, no generic loop search.`
