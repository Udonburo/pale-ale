#!/usr/bin/env python3
"""Development-only synthetic laboratory for Gate12C-2.

This module is deliberately separate from the frozen Gate12C-1 runner.  It
implements the smallest auditable vertical slice needed to develop the S0
true-null regime and the N1 role-constrained frame-reassignment null:

* deterministic, typed, jointly realizable synthetic graph generation;
* FP64 compression-composition residual diagnostics;
* explicit undefined/degenerate statuses;
* an independent joint-realizability checker;
* role-constrained frame reassignment with incident-edge reconstruction; and
* a development report schema that cannot be mistaken for locked calibration.

It does not authorize a locked synthetic test or a real held-out execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GRAPH_SCHEMA_VERSION = "gate12c2_synthetic_graph_v0.1"
DIAGNOSTIC_SCHEMA_VERSION = "gate12c2_residual_diagnostics_v0.1"
DEVELOPMENT_REPORT_SCHEMA_VERSION = "gate12c2_development_report_v0.1"
GENERATOR_ID = "gate12c2_s0_joint_frames_pcg64_v0.1"
N1_ID = "gate12c2_n1_role_constrained_frame_reassignment_v0.1"
SEED_DERIVATION_ID = "sha256_canonical_json_to_uint64_v1"
REFERENCE_DTYPE = "float64"

DEFAULT_NUMERIC_ATOL = 1.0e-10
DEFAULT_DEGENERACY_ATOL = 1.0e-12
DEFAULT_RELATIVE_GAP_MIN = 1.0e-8


class Gate12C2DevelopmentError(ValueError):
    """Raised when a development graph or operation violates its contract."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _derived_seed(master_seed: str, *parts: object) -> int:
    payload = [SEED_DERIVATION_ID, str(master_seed), *parts]
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _seed_receipt(master_seed: str) -> str:
    return hashlib.sha256(
        _canonical_json_bytes([SEED_DERIVATION_ID, str(master_seed)])
    ).hexdigest()


def _rng(master_seed: str, *parts: object) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_derived_seed(master_seed, *parts)))


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < array.shape[1]:
        raise Gate12C2DevelopmentError(
            "frame candidates must be two-dimensional with rows >= columns"
        )
    q_matrix, r_matrix = np.linalg.qr(array, mode="reduced")
    diagonal = np.diag(r_matrix)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return np.asarray(q_matrix * signs, dtype=np.float64)


@dataclass(frozen=True)
class NodeFrame:
    node_id: str
    role: str
    family: str
    frame: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        array = np.asarray(self.frame, dtype=np.float64)
        if array.ndim != 2:
            raise Gate12C2DevelopmentError("node frames must be matrices")
        object.__setattr__(self, "frame", array.copy())

    @property
    def ambient_dim(self) -> int:
        return int(self.frame.shape[0])

    @property
    def local_rank(self) -> int:
        return int(self.frame.shape[1])

    @property
    def stratum(self) -> tuple[str, str, int, int]:
        return (self.role, self.family, self.ambient_dim, self.local_rank)


@dataclass(frozen=True)
class EdgeOverlap:
    edge_id: str
    source_node_id: str
    target_node_id: str
    matrix: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        array = np.asarray(self.matrix, dtype=np.float64)
        if array.ndim != 2:
            raise Gate12C2DevelopmentError("edge overlaps must be matrices")
        object.__setattr__(self, "matrix", array.copy())


@dataclass(frozen=True)
class SyntheticGraph:
    replicate_id: str
    regime: str
    nodes: tuple[NodeFrame, ...]
    edges: tuple[EdgeOverlap, ...]
    cycle_node_ids: tuple[str, str, str]
    generator_id: str
    seed_receipt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GRAPH_SCHEMA_VERSION

    def node_map(self) -> dict[str, NodeFrame]:
        result = {node.node_id: node for node in self.nodes}
        if len(result) != len(self.nodes):
            raise Gate12C2DevelopmentError(
                f"duplicate node ID in replicate {self.replicate_id}"
            )
        return result

    def edge_map(self) -> dict[tuple[str, str], EdgeOverlap]:
        result = {
            (edge.source_node_id, edge.target_node_id): edge for edge in self.edges
        }
        if len(result) != len(self.edges):
            raise Gate12C2DevelopmentError(
                f"duplicate directed edge in replicate {self.replicate_id}"
            )
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "replicate_id": self.replicate_id,
            "regime": self.regime,
            "generator_id": self.generator_id,
            "seed_receipt": self.seed_receipt,
            "reference_dtype": REFERENCE_DTYPE,
            "cycle_node_ids": list(self.cycle_node_ids),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "role": node.role,
                    "family": node.family,
                    "ambient_dim": node.ambient_dim,
                    "local_rank": node.local_rank,
                    "frame_sha256": hashlib.sha256(
                        np.ascontiguousarray(node.frame).tobytes()
                    ).hexdigest(),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "matrix_sha256": hashlib.sha256(
                        np.ascontiguousarray(edge.matrix).tobytes()
                    ).hexdigest(),
                }
                for edge in self.edges
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResidualDiagnostics:
    q: int
    eligibility_status: str
    numerical_status: str
    defect: float | None
    tail_left: float | None
    tail_right: float | None
    propagated_left: float | None
    propagated_right: float | None
    alignment: float | None
    propagation_left: float | None
    propagation_right: float | None
    alignment_status: str
    propagation_left_status: str
    propagation_right_status: str
    matrix_identity_error: float | None
    squared_identity_error: float | None
    relative_gap_left: float | None
    relative_gap_right: float | None
    schema_version: str = DIAGNOSTIC_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "q": self.q,
            "eligibility_status": self.eligibility_status,
            "numerical_status": self.numerical_status,
            "a_q": self.defect,
            "u_q": self.tail_left,
            "v_q": self.tail_right,
            "x_q": self.propagated_left,
            "y_q": self.propagated_right,
            "c_q": self.alignment,
            "p_L_q": self.propagation_left,
            "p_R_q": self.propagation_right,
            "alignment_status": self.alignment_status,
            "p_L_status": self.propagation_left_status,
            "p_R_status": self.propagation_right_status,
            "matrix_identity_error": self.matrix_identity_error,
            "squared_identity_error": self.squared_identity_error,
            "relative_gap_left": self.relative_gap_left,
            "relative_gap_right": self.relative_gap_right,
            "reference_dtype": REFERENCE_DTYPE,
        }


def _edge_from_frames(
    source: NodeFrame,
    target: NodeFrame,
    *,
    edge_id: str,
) -> EdgeOverlap:
    if source.ambient_dim != target.ambient_dim:
        raise Gate12C2DevelopmentError(
            f"ambient dimension mismatch for {source.node_id}->{target.node_id}"
        )
    matrix = np.asarray(target.frame.T @ source.frame, dtype=np.float64)
    return EdgeOverlap(
        edge_id=edge_id,
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        matrix=matrix,
    )


def graph_from_nodes(
    *,
    replicate_id: str,
    regime: str,
    nodes: Sequence[NodeFrame],
    cycle_node_ids: tuple[str, str, str],
    generator_id: str,
    seed_receipt: str,
    metadata: Mapping[str, Any] | None = None,
) -> SyntheticGraph:
    node_map = {node.node_id: node for node in nodes}
    if len(node_map) != len(nodes):
        raise Gate12C2DevelopmentError("node IDs must be unique")
    if any(node_id not in node_map for node_id in cycle_node_ids):
        raise Gate12C2DevelopmentError("cycle references an unknown node")

    n0, n1, n2 = (node_map[node_id] for node_id in cycle_node_ids)
    edges = (
        _edge_from_frames(n0, n1, edge_id=f"{replicate_id}:e0"),
        _edge_from_frames(n1, n2, edge_id=f"{replicate_id}:e1"),
        _edge_from_frames(n2, n0, edge_id=f"{replicate_id}:e2"),
    )
    return SyntheticGraph(
        replicate_id=replicate_id,
        regime=regime,
        nodes=tuple(nodes),
        edges=edges,
        cycle_node_ids=cycle_node_ids,
        generator_id=generator_id,
        seed_receipt=seed_receipt,
        metadata=dict(metadata or {}),
    )


def generate_s0_cohort(
    *,
    replicate_count: int,
    master_seed: str,
    ambient_dim: int = 8,
    local_rank: int = 3,
    frame_noise: float = 0.35,
    family: str = "synthetic_primary",
) -> tuple[SyntheticGraph, ...]:
    """Generate exchangeable graph-consistent S0 replicates.

    Each role has a fixed population center, while every replicate receives an
    independent role-conditioned orthonormal frame.  N1 reassignment is
    therefore exchangeable within the declared role/family/rank strata.
    """

    if replicate_count < 2:
        raise Gate12C2DevelopmentError("S0 requires at least two replicates")
    if local_rank < 2 or ambient_dim < local_rank:
        raise Gate12C2DevelopmentError(
            "S0 requires ambient_dim >= local_rank >= 2"
        )
    if not math.isfinite(frame_noise) or frame_noise <= 0.0:
        raise Gate12C2DevelopmentError("frame_noise must be finite and positive")

    roles = ("input", "bridge", "output")
    role_centers: dict[str, np.ndarray] = {}
    for role in roles:
        candidate = _rng(master_seed, "role_center", role).normal(
            size=(ambient_dim, local_rank)
        )
        role_centers[role] = _orthonormalize(candidate)

    cohort: list[SyntheticGraph] = []
    receipt = _seed_receipt(master_seed)
    for replicate_index in range(replicate_count):
        replicate_id = f"s0-{replicate_index:06d}"
        nodes: list[NodeFrame] = []
        for node_index, role in enumerate(roles):
            noise = _rng(
                master_seed, "replicate", replicate_index, "role", role
            ).normal(size=(ambient_dim, local_rank))
            frame = _orthonormalize(role_centers[role] + frame_noise * noise)
            nodes.append(
                NodeFrame(
                    node_id=f"{replicate_id}:n{node_index}",
                    role=role,
                    family=family,
                    frame=frame,
                )
            )
        cohort.append(
            graph_from_nodes(
                replicate_id=replicate_id,
                regime="S0_true_null",
                nodes=nodes,
                cycle_node_ids=tuple(node.node_id for node in nodes),
                generator_id=GENERATOR_ID,
                seed_receipt=receipt,
                metadata={
                    "replicate_index": replicate_index,
                    "ambient_dim": ambient_dim,
                    "local_rank": local_rank,
                    "frame_noise": float(frame_noise),
                    "seed_derivation_id": SEED_DERIVATION_ID,
                    "bit_generator": "numpy.PCG64",
                },
            )
        )
    return tuple(cohort)


def check_joint_realizability(
    graph: SyntheticGraph,
    *,
    atol: float = DEFAULT_NUMERIC_ATOL,
) -> dict[str, Any]:
    node_map = graph.node_map()
    failures: list[dict[str, Any]] = []
    maximum_error = 0.0
    for edge in graph.edges:
        source = node_map.get(edge.source_node_id)
        target = node_map.get(edge.target_node_id)
        if source is None or target is None:
            failures.append(
                {
                    "edge_id": edge.edge_id,
                    "reason": "unknown_endpoint",
                }
            )
            continue
        expected = np.asarray(target.frame.T @ source.frame, dtype=np.float64)
        if expected.shape != edge.matrix.shape:
            failures.append(
                {
                    "edge_id": edge.edge_id,
                    "reason": "shape_mismatch",
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(edge.matrix.shape),
                }
            )
            continue
        error = float(np.linalg.norm(edge.matrix - expected, ord="fro"))
        maximum_error = max(maximum_error, error)
        if not math.isfinite(error) or error > atol:
            failures.append(
                {
                    "edge_id": edge.edge_id,
                    "reason": "overlap_mismatch",
                    "frobenius_error": error,
                }
            )
    return {
        "status": "pass" if not failures else "fail",
        "edge_count": len(graph.edges),
        "maximum_frobenius_error": maximum_error,
        "atol": float(atol),
        "failures": failures,
    }


def n1_role_constrained_reassignment(
    graphs: Sequence[SyntheticGraph],
    *,
    reassignment_seed: str,
) -> tuple[SyntheticGraph, ...]:
    """Reassign frames within frozen strata and reconstruct every edge.

    Donors are assigned by a deterministic cyclic derangement within each
    role/family/ambient-dimension/rank stratum.  No observed edge matrix is
    copied or independently oriented.
    """

    if len(graphs) < 2:
        raise Gate12C2DevelopmentError("N1 requires at least two graphs")
    graph_ids = [graph.replicate_id for graph in graphs]
    if len(set(graph_ids)) != len(graph_ids):
        raise Gate12C2DevelopmentError("replicate IDs must be unique")

    members: dict[
        tuple[str, str, int, int], list[tuple[str, NodeFrame]]
    ] = defaultdict(list)
    for graph in graphs:
        for node in graph.nodes:
            members[node.stratum].append((graph.replicate_id, node))

    donor_for: dict[tuple[str, str], tuple[str, NodeFrame]] = {}
    for stratum, entries in sorted(members.items()):
        if len(entries) < 2:
            raise Gate12C2DevelopmentError(
                f"N1 stratum has fewer than two members: {stratum!r}"
            )
        ordered = sorted(
            entries,
            key=lambda item: hashlib.sha256(
                _canonical_json_bytes(
                    [
                        N1_ID,
                        reassignment_seed,
                        list(stratum),
                        item[0],
                        item[1].node_id,
                    ]
                )
            ).hexdigest(),
        )
        for index, target in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            donor_for[(target[0], target[1].node_id)] = donor

    result: list[SyntheticGraph] = []
    receipt = _seed_receipt(reassignment_seed)
    for graph in graphs:
        reassigned_nodes: list[NodeFrame] = []
        donor_manifest: dict[str, str] = {}
        for node in graph.nodes:
            donor_graph_id, donor = donor_for[(graph.replicate_id, node.node_id)]
            if donor_graph_id == graph.replicate_id and donor.node_id == node.node_id:
                raise Gate12C2DevelopmentError("N1 produced a fixed-point donor")
            if donor.stratum != node.stratum:
                raise Gate12C2DevelopmentError("N1 crossed a frozen stratum")
            reassigned_nodes.append(
                NodeFrame(
                    node_id=node.node_id,
                    role=node.role,
                    family=node.family,
                    frame=donor.frame,
                )
            )
            donor_manifest[node.node_id] = f"{donor_graph_id}/{donor.node_id}"
        result.append(
            graph_from_nodes(
                replicate_id=graph.replicate_id,
                regime=graph.regime,
                nodes=reassigned_nodes,
                cycle_node_ids=graph.cycle_node_ids,
                generator_id=N1_ID,
                seed_receipt=receipt,
                metadata={
                    "source_generator_id": graph.generator_id,
                    "source_seed_receipt": graph.seed_receipt,
                    "reassignment_seed_derivation_id": SEED_DERIVATION_ID,
                    "donor_node_ids": donor_manifest,
                    "null_construction": "N1_role_constrained_frame_reassignment",
                },
            )
        )
    return tuple(result)


def _truncated_reconstruction(
    matrix: np.ndarray,
    q: int,
    *,
    relative_gap_min: float,
) -> tuple[np.ndarray | None, str, float | None]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise Gate12C2DevelopmentError("compression input must be a matrix")
    maximum_rank = min(array.shape)
    if q <= 0 or q > maximum_rank:
        return None, "invalid_q", None

    left, singular_values, right_t = np.linalg.svd(array, full_matrices=False)
    if not np.all(np.isfinite(singular_values)):
        return None, "nonfinite_spectrum", None
    if q == maximum_rank:
        reconstruction = (left[:, :q] * singular_values[:q]) @ right_t[:q, :]
        return np.asarray(reconstruction, dtype=np.float64), "full_rank_control", None

    scale = max(float(singular_values[0]), np.finfo(np.float64).tiny)
    relative_gap = float((singular_values[q - 1] - singular_values[q]) / scale)
    if relative_gap <= relative_gap_min:
        return None, "unstable_spectral_cut", relative_gap
    reconstruction = (left[:, :q] * singular_values[:q]) @ right_t[:q, :]
    return np.asarray(reconstruction, dtype=np.float64), "eligible", relative_gap


def residual_diagnostics(
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    q: int,
    relative_gap_min: float = DEFAULT_RELATIVE_GAP_MIN,
    numerical_atol: float = DEFAULT_NUMERIC_ATOL,
    degeneracy_atol: float = DEFAULT_DEGENERACY_ATOL,
) -> ResidualDiagnostics:
    """Compute the exact Gate12C residual decomposition in FP64."""

    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in (m0, m1, m2)]
    if any(matrix.ndim != 2 for matrix in matrices):
        raise Gate12C2DevelopmentError("m0, m1, and m2 must be matrices")
    if not (matrices[1].shape[1] == matrices[0].shape[0]):
        raise Gate12C2DevelopmentError("m1 @ m0 is not conformable")
    if not (matrices[2].shape[1] == matrices[1].shape[0]):
        raise Gate12C2DevelopmentError("m2 @ m1 is not conformable")

    m0_array, m1_array, m2_array = matrices
    product_left = np.asarray(m2_array @ m1_array, dtype=np.float64)
    product_right = np.asarray(m1_array @ m0_array, dtype=np.float64)
    q_left, left_status, left_gap = _truncated_reconstruction(
        product_left, q, relative_gap_min=relative_gap_min
    )
    q_right, right_status, right_gap = _truncated_reconstruction(
        product_right, q, relative_gap_min=relative_gap_min
    )
    if q_left is None or q_right is None:
        status = (
            left_status if left_status != "eligible" else right_status
        )
        return ResidualDiagnostics(
            q=q,
            eligibility_status=status,
            numerical_status="not_evaluated",
            defect=None,
            tail_left=None,
            tail_right=None,
            propagated_left=None,
            propagated_right=None,
            alignment=None,
            propagation_left=None,
            propagation_right=None,
            alignment_status="not_evaluated",
            propagation_left_status="not_evaluated",
            propagation_right_status="not_evaluated",
            matrix_identity_error=None,
            squared_identity_error=None,
            relative_gap_left=left_gap,
            relative_gap_right=right_gap,
        )

    residual_left = np.asarray(product_left - q_left, dtype=np.float64)
    residual_right = np.asarray(product_right - q_right, dtype=np.float64)
    propagated_left_matrix = np.asarray(residual_left @ m0_array, dtype=np.float64)
    propagated_right_matrix = np.asarray(m2_array @ residual_right, dtype=np.float64)
    left_parenthesization = np.asarray(q_left @ m0_array, dtype=np.float64)
    right_parenthesization = np.asarray(m2_array @ q_right, dtype=np.float64)
    defect_matrix = np.asarray(
        left_parenthesization - right_parenthesization, dtype=np.float64
    )
    decomposition_matrix = np.asarray(
        propagated_right_matrix - propagated_left_matrix, dtype=np.float64
    )

    defect = float(np.linalg.norm(defect_matrix, ord="fro"))
    tail_left = float(np.linalg.norm(residual_left, ord="fro"))
    tail_right = float(np.linalg.norm(residual_right, ord="fro"))
    propagated_left = float(np.linalg.norm(propagated_left_matrix, ord="fro"))
    propagated_right = float(np.linalg.norm(propagated_right_matrix, ord="fro"))
    inner_product = float(
        np.real(np.vdot(propagated_left_matrix, propagated_right_matrix))
    )

    if propagated_left <= degeneracy_atol or propagated_right <= degeneracy_atol:
        alignment = None
        zero_sides = []
        if propagated_left <= degeneracy_atol:
            zero_sides.append("x")
        if propagated_right <= degeneracy_atol:
            zero_sides.append("y")
        alignment_status = "undefined_degenerate_" + "_".join(zero_sides)
    else:
        alignment = float(
            inner_product / (propagated_left * propagated_right)
        )
        alignment = max(-1.0, min(1.0, alignment))
        alignment_status = "defined"

    if tail_left <= degeneracy_atol:
        propagation_left = None
        propagation_left_status = "undefined_degenerate_u"
    else:
        propagation_left = float(propagated_left / tail_left)
        propagation_left_status = "defined"

    if tail_right <= degeneracy_atol:
        propagation_right = None
        propagation_right_status = "undefined_degenerate_v"
    else:
        propagation_right = float(propagated_right / tail_right)
        propagation_right_status = "defined"

    matrix_identity_error = float(
        np.linalg.norm(defect_matrix - decomposition_matrix, ord="fro")
    )
    squared_rhs = (
        propagated_left * propagated_left
        + propagated_right * propagated_right
        - 2.0 * inner_product
    )
    squared_identity_error = float(abs(defect * defect - squared_rhs))
    finite_values = (
        defect,
        tail_left,
        tail_right,
        propagated_left,
        propagated_right,
        matrix_identity_error,
        squared_identity_error,
    )
    numerical_status = (
        "pass"
        if all(math.isfinite(value) for value in finite_values)
        and matrix_identity_error <= numerical_atol
        and squared_identity_error <= numerical_atol
        else "fail"
    )
    eligibility_status = (
        "full_rank_control"
        if left_status == right_status == "full_rank_control"
        else "eligible"
    )
    return ResidualDiagnostics(
        q=q,
        eligibility_status=eligibility_status,
        numerical_status=numerical_status,
        defect=defect,
        tail_left=tail_left,
        tail_right=tail_right,
        propagated_left=propagated_left,
        propagated_right=propagated_right,
        alignment=alignment,
        propagation_left=propagation_left,
        propagation_right=propagation_right,
        alignment_status=alignment_status,
        propagation_left_status=propagation_left_status,
        propagation_right_status=propagation_right_status,
        matrix_identity_error=matrix_identity_error,
        squared_identity_error=squared_identity_error,
        relative_gap_left=left_gap,
        relative_gap_right=right_gap,
    )


def cycle_matrices(graph: SyntheticGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n0, n1, n2 = graph.cycle_node_ids
    edges = graph.edge_map()
    required = ((n0, n1), (n1, n2), (n2, n0))
    missing = [pair for pair in required if pair not in edges]
    if missing:
        raise Gate12C2DevelopmentError(
            f"cycle is missing directed edges: {missing!r}"
        )
    return tuple(np.asarray(edges[pair].matrix, dtype=np.float64) for pair in required)


def graph_residual_diagnostics(
    graph: SyntheticGraph,
    *,
    q: int,
    relative_gap_min: float = DEFAULT_RELATIVE_GAP_MIN,
    numerical_atol: float = DEFAULT_NUMERIC_ATOL,
    degeneracy_atol: float = DEFAULT_DEGENERACY_ATOL,
) -> ResidualDiagnostics:
    m0, m1, m2 = cycle_matrices(graph)
    return residual_diagnostics(
        m0,
        m1,
        m2,
        q=q,
        relative_gap_min=relative_gap_min,
        numerical_atol=numerical_atol,
        degeneracy_atol=degeneracy_atol,
    )


def edge_spectrum_marginal_discrepancy(
    observed: Sequence[SyntheticGraph],
    comparison: Sequence[SyntheticGraph],
) -> dict[str, Any]:
    def flattened_singular_values(graphs: Sequence[SyntheticGraph]) -> np.ndarray:
        values: list[float] = []
        for graph in graphs:
            for edge in graph.edges:
                values.extend(
                    float(value)
                    for value in np.linalg.svd(edge.matrix, compute_uv=False)
                )
        return np.sort(np.asarray(values, dtype=np.float64))

    observed_values = flattened_singular_values(observed)
    comparison_values = flattened_singular_values(comparison)
    if observed_values.shape != comparison_values.shape:
        raise Gate12C2DevelopmentError(
            "marginal spectrum comparison requires equal edge-spectrum counts"
        )
    absolute = np.abs(observed_values - comparison_values)
    return {
        "summary_id": "sorted_edge_singular_value_discrepancy_v1",
        "value_count": int(observed_values.size),
        "mean_absolute_sorted_difference": float(np.mean(absolute)),
        "maximum_absolute_sorted_difference": float(np.max(absolute)),
        "observed_mean": float(np.mean(observed_values)),
        "comparison_mean": float(np.mean(comparison_values)),
        "mean_shift": float(
            np.mean(comparison_values) - np.mean(observed_values)
        ),
    }


def development_s0_n1_report(
    observed: Sequence[SyntheticGraph],
    comparison: Sequence[SyntheticGraph],
    *,
    q: int,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Build an auditable development summary without estimating type-I error."""

    observed_by_id = {graph.replicate_id: graph for graph in observed}
    comparison_by_id = {graph.replicate_id: graph for graph in comparison}
    if set(observed_by_id) != set(comparison_by_id):
        raise Gate12C2DevelopmentError(
            "observed and comparison cohorts must share replicate IDs"
        )

    rows: list[dict[str, Any]] = []
    numerical_failures = 0
    ineligible = 0
    degenerate_alignment = 0
    for replicate_id in sorted(observed_by_id):
        observed_diagnostic = graph_residual_diagnostics(
            observed_by_id[replicate_id], q=q
        )
        comparison_diagnostic = graph_residual_diagnostics(
            comparison_by_id[replicate_id], q=q
        )
        if (
            observed_diagnostic.defect is None
            or comparison_diagnostic.defect is None
        ):
            ineligible += 1
            log_ratio = None
        else:
            log_ratio = float(
                math.log(observed_diagnostic.defect + epsilon)
                - math.log(comparison_diagnostic.defect + epsilon)
            )
        numerical_failures += int(
            observed_diagnostic.numerical_status == "fail"
            or comparison_diagnostic.numerical_status == "fail"
        )
        degenerate_alignment += int(
            observed_diagnostic.alignment is None
            or comparison_diagnostic.alignment is None
        )
        rows.append(
            {
                "replicate_id": replicate_id,
                "observed": observed_diagnostic.as_dict(),
                "comparison": comparison_diagnostic.as_dict(),
                "log_observed_to_comparison_defect": log_ratio,
            }
        )

    informative_log_ratios = [
        float(row["log_observed_to_comparison_defect"])
        for row in rows
        if row["log_observed_to_comparison_defect"] is not None
    ]
    observed_realizability = [
        check_joint_realizability(graph)["status"] == "pass" for graph in observed
    ]
    comparison_realizability = [
        check_joint_realizability(graph)["status"] == "pass"
        for graph in comparison
    ]

    return {
        "schema_version": DEVELOPMENT_REPORT_SCHEMA_VERSION,
        "epistemic_status": "development_only",
        "regime": "S0_true_null",
        "null_candidate_id": N1_ID,
        "independent_unit": "synthetic_graph_replicate",
        "reference_dtype": REFERENCE_DTYPE,
        "q": int(q),
        "replicate_count": len(rows),
        "informative_count": len(informative_log_ratios),
        "ineligible_count": ineligible,
        "numerical_failure_count": numerical_failures,
        "degenerate_alignment_pair_count": degenerate_alignment,
        "joint_realizability": {
            "observed_pass_count": sum(observed_realizability),
            "comparison_pass_count": sum(comparison_realizability),
            "observed_total": len(observed_realizability),
            "comparison_total": len(comparison_realizability),
        },
        "edge_spectrum_marginal_discrepancy": (
            edge_spectrum_marginal_discrepancy(observed, comparison)
        ),
        "directional_summary": {
            "negative_log_ratio_count": sum(
                value < 0.0 for value in informative_log_ratios
            ),
            "positive_log_ratio_count": sum(
                value > 0.0 for value in informative_log_ratios
            ),
            "zero_log_ratio_count": sum(
                value == 0.0 for value in informative_log_ratios
            ),
            "median_log_ratio": (
                float(np.median(informative_log_ratios))
                if informative_log_ratios
                else None
            ),
        },
        "type_i_calibration": {
            "status": "not_estimated_without_frozen_decision_rule",
            "false_positive_rate": None,
            "acceptance_threshold": None,
            "reason": (
                "A development pairing is not a locked outer calibration "
                "experiment and does not define a confirmatory decision rule."
            ),
        },
        "rows": rows,
    }


def manifests(graphs: Iterable[SyntheticGraph]) -> list[dict[str, Any]]:
    return [graph.manifest() for graph in graphs]
