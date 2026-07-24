#!/usr/bin/env python3
"""Development-only synthetic laboratory for Gate12C-2.

This module is deliberately separate from the frozen Gate12C-1 runner.  It
implements auditable development infrastructure for S0/S1/S2 and the N1
role-constrained frame-reassignment null:

* deterministic, typed, jointly realizable synthetic graph generation;
* FP64 compression-composition residual diagnostics;
* explicit undefined/degenerate statuses;
* direct and independently assembled block-Gram realizability checks;
* role-constrained frame reassignment with incident-edge reconstruction; and
* complete-pipeline outer calibration and nested inner-draw diagnostics.

It does not authorize a locked synthetic test or a real held-out execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GRAPH_SCHEMA_VERSION = "gate12c2_synthetic_graph_v0.1"
DIAGNOSTIC_SCHEMA_VERSION = "gate12c2_residual_diagnostics_v0.2"
DEVELOPMENT_REPORT_SCHEMA_VERSION = "gate12c2_development_report_v0.2"
PIPELINE_DECISION_SCHEMA_VERSION = "gate12c2_pipeline_decision_v0.1"
OUTER_CALIBRATION_SCHEMA_VERSION = "gate12c2_outer_calibration_v0.1"
MECHANISM_CONTROL_SCHEMA_VERSION = "gate12c2_mechanism_control_v0.1"
INNER_DRAW_STABILITY_SCHEMA_VERSION = "gate12c2_inner_draw_stability_v0.1"
GENERATOR_ID = "gate12c2_s0_joint_frames_pcg64_v0.1"
S1_SHARED_COUPLING_GENERATOR_ID = (
    "gate12c2_s1_shared_node_coupling_reverse_v0.1"
)
N1_ID = "gate12c2_n1_role_constrained_frame_reassignment_v0.1"
S2_UNCONSTRAINED_ORIENTATION_ID = (
    "gate12c2_s2_independent_edge_orientation_stress_v0.1"
)
SEED_DERIVATION_ID = "sha256_canonical_json_to_uint64_v1"
REFERENCE_DTYPE = "float64"
N1_ESCALATION_POLICY = (
    "N1 is the sole primary candidate. N2 may be opened only after a "
    "documented failure of at least one predeclared required N1 gate; "
    "locked-candidate winner selection is prohibited. N3 requires a new "
    "contract."
)

DEFAULT_NUMERIC_ATOL = 1.0e-10
DEFAULT_DEGENERACY_ATOL = 1.0e-12
DEFAULT_RELATIVE_GAP_MIN = 1.0e-8
DEFAULT_HOLM_ALPHA = 0.05
DEFAULT_PRIMARY_ZERO_TOLERANCE = 1.0e-12
PIPELINE_ENDPOINT_COUNT = 24


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
    product_singular_values_left: tuple[float, ...]
    product_singular_values_right: tuple[float, ...]
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
            "product_singular_values_left": list(
                self.product_singular_values_left
            ),
            "product_singular_values_right": list(
                self.product_singular_values_right
            ),
            "reference_dtype": REFERENCE_DTYPE,
        }


@dataclass(frozen=True)
class EndpointDecisionInput:
    """One endpoint entering the complete 24-endpoint decision hierarchy."""

    case_id: str
    case_order: int
    model: str
    family: str
    q: int
    coverage_complete: bool
    informative: bool
    median_log_ratio: float | None
    raw_p: float


@dataclass(frozen=True)
class ResidualMechanismControl:
    """Algebraic diagnostic control, not an end-to-end graph S1 case."""

    mechanism: str
    level: float
    q: int
    m0: np.ndarray = field(repr=False)
    m1: np.ndarray = field(repr=False)
    m2: np.ndarray = field(repr=False)
    declared_control: str
    diagnostics: ResidualDiagnostics
    schema_version: str = MECHANISM_CONTROL_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epistemic_status": "development_algebraic_control_only",
            "mechanism": self.mechanism,
            "level": float(self.level),
            "q": int(self.q),
            "declared_control": self.declared_control,
            "joint_graph_realizability_claimed": False,
            "end_to_end_s1_satisfied": False,
            "matrix_sha256": {
                name: hashlib.sha256(
                    np.ascontiguousarray(matrix).tobytes()
                ).hexdigest()
                for name, matrix in (
                    ("m0", self.m0),
                    ("m1", self.m1),
                    ("m2", self.m2),
                )
            },
            "diagnostics": self.diagnostics.as_dict(),
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


def generate_s1_shared_node_coupling_cohort(
    *,
    replicate_count: int,
    master_seed: str,
    effect_strength: float,
    ambient_dim: int = 7,
    local_rank: int = 3,
    observed_mismatch_floor: float = 0.01,
    family: str = "synthetic_s1_alignment",
) -> tuple[SyntheticGraph, ...]:
    """Generate a graph-realizable graded reverse-direction S1 cohort.

    Within each observed graph, bridge and output share one latent frame plus a
    fixed small output offset. This creates a nonzero observed defect while
    retaining a known graph-level coupling. Across replicates the shared frame
    moves along a deterministic Stiefel tangent direction. N1 independently
    reassigns bridge and output frames, destroying that coupling. The effect
    strength controls the cross-replicate frame differences.
    """

    if replicate_count < 4:
        raise Gate12C2DevelopmentError(
            "S1 shared-coupling cohort requires at least four replicates"
        )
    if local_rank < 2 or ambient_dim <= local_rank:
        raise Gate12C2DevelopmentError(
            "S1 requires ambient_dim > local_rank >= 2"
        )
    if not math.isfinite(effect_strength) or not 0.0 < effect_strength <= 0.5:
        raise Gate12C2DevelopmentError(
            "S1 effect_strength must lie in the development range (0, 0.5]"
        )
    if (
        not math.isfinite(observed_mismatch_floor)
        or not 0.0 <= observed_mismatch_floor <= 0.05
    ):
        raise Gate12C2DevelopmentError(
            "observed_mismatch_floor must lie in [0, 0.05]"
        )

    input_frame = _orthonormalize(
        _rng(master_seed, "S1", "input_frame").normal(
            size=(ambient_dim, local_rank)
        )
    )
    shared_center = _orthonormalize(
        _rng(master_seed, "S1", "shared_center").normal(
            size=(ambient_dim, local_rank)
        )
    )
    raw_direction = _rng(master_seed, "S1", "tangent_direction").normal(
        size=(ambient_dim, local_rank)
    )
    tangent_direction = np.asarray(
        raw_direction
        - shared_center @ (shared_center.T @ raw_direction),
        dtype=np.float64,
    )
    direction_norm = float(np.linalg.norm(tangent_direction, ord="fro"))
    if direction_norm <= DEFAULT_DEGENERACY_ATOL:
        raise Gate12C2DevelopmentError(
            "S1 tangent direction is numerically degenerate"
        )
    tangent_direction = np.asarray(
        tangent_direction / direction_norm * math.sqrt(local_rank),
        dtype=np.float64,
    )
    raw_output_offset = _rng(master_seed, "S1", "output_offset").normal(
        size=(ambient_dim, local_rank)
    )
    output_offset = np.asarray(
        raw_output_offset
        - shared_center @ (shared_center.T @ raw_output_offset),
        dtype=np.float64,
    )
    output_offset_norm = float(np.linalg.norm(output_offset, ord="fro"))
    if output_offset_norm <= DEFAULT_DEGENERACY_ATOL:
        raise Gate12C2DevelopmentError(
            "S1 output offset is numerically degenerate"
        )
    output_offset = np.asarray(
        output_offset / output_offset_norm * math.sqrt(local_rank),
        dtype=np.float64,
    )

    receipt = _seed_receipt(master_seed)
    latent_positions = np.linspace(-1.0, 1.0, replicate_count)
    graphs: list[SyntheticGraph] = []
    for replicate_index, latent_position in enumerate(latent_positions):
        replicate_id = f"s1-{replicate_index:06d}"
        shared_frame = _orthonormalize(
            shared_center
            + effect_strength * float(latent_position) * tangent_direction
        )
        output_frame = _orthonormalize(
            shared_frame + observed_mismatch_floor * output_offset
        )
        nodes = (
            NodeFrame(
                node_id=f"{replicate_id}:n0",
                role="input",
                family=family,
                frame=input_frame,
            ),
            NodeFrame(
                node_id=f"{replicate_id}:n1",
                role="bridge",
                family=family,
                frame=shared_frame,
            ),
            NodeFrame(
                node_id=f"{replicate_id}:n2",
                role="output",
                family=family,
                frame=output_frame,
            ),
        )
        graphs.append(
            graph_from_nodes(
                replicate_id=replicate_id,
                regime="S1_known_reverse_shared_node_coupling",
                nodes=nodes,
                cycle_node_ids=tuple(node.node_id for node in nodes),
                generator_id=S1_SHARED_COUPLING_GENERATOR_ID,
                seed_receipt=receipt,
                metadata={
                    "replicate_index": replicate_index,
                    "latent_position": float(latent_position),
                    "effect_strength": float(effect_strength),
                    "mechanism": (
                        "shared_node_residual_cancellation_with_fixed_offset"
                    ),
                    "known_direction": "observed_defect_smaller_than_N1",
                    "bridge_output_frame_identity": (
                        observed_mismatch_floor == 0.0
                    ),
                    "observed_mismatch_floor": float(
                        observed_mismatch_floor
                    ),
                    "ambient_dim": ambient_dim,
                    "local_rank": local_rank,
                    "seed_derivation_id": SEED_DERIVATION_ID,
                },
            )
        )
    return tuple(graphs)


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
        "checker_id": "direct_edge_reconstruction_v1",
        "status": "pass" if not failures else "fail",
        "edge_count": len(graph.edges),
        "maximum_frobenius_error": maximum_error,
        "atol": float(atol),
        "failures": failures,
    }


def check_block_gram_realizability(
    graph: SyntheticGraph,
    *,
    atol: float = DEFAULT_NUMERIC_ATOL,
) -> dict[str, Any]:
    """Check realizability through an independently assembled block Gram.

    Unlike :func:`check_joint_realizability`, this path never calls the edge
    constructor and does not reconstruct one edge at a time.  It concatenates
    all node frames, checks the resulting global Gram matrix, and compares its
    off-diagonal blocks with the stored directed overlaps.
    """

    nodes = tuple(sorted(graph.nodes, key=lambda node: node.node_id))
    failures: list[dict[str, Any]] = []
    if not nodes:
        return {
            "checker_id": "global_block_gram_v1",
            "status": "fail",
            "failures": [{"reason": "empty_node_set"}],
            "atol": float(atol),
        }

    ambient_dims = {node.ambient_dim for node in nodes}
    if len(ambient_dims) != 1:
        return {
            "checker_id": "global_block_gram_v1",
            "status": "fail",
            "failures": [
                {
                    "reason": "mixed_ambient_dimensions",
                    "ambient_dimensions": sorted(ambient_dims),
                }
            ],
            "atol": float(atol),
        }

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    frame_blocks: list[np.ndarray] = []
    maximum_diagonal_error = 0.0
    for node in nodes:
        start = cursor
        cursor += node.local_rank
        offsets[node.node_id] = (start, cursor)
        frame_blocks.append(np.asarray(node.frame, dtype=np.float64))
        diagonal_error = float(
            np.linalg.norm(
                node.frame.T @ node.frame - np.eye(node.local_rank),
                ord="fro",
            )
        )
        maximum_diagonal_error = max(maximum_diagonal_error, diagonal_error)
        if not math.isfinite(diagonal_error) or diagonal_error > atol:
            failures.append(
                {
                    "node_id": node.node_id,
                    "reason": "nonorthonormal_frame",
                    "frobenius_error": diagonal_error,
                }
            )

    concatenated = np.concatenate(frame_blocks, axis=1)
    block_gram = np.asarray(concatenated.T @ concatenated, dtype=np.float64)
    symmetry_error = float(
        np.linalg.norm(block_gram - block_gram.T, ord="fro")
    )
    eigenvalues = np.linalg.eigvalsh(
        0.5 * np.asarray(block_gram + block_gram.T, dtype=np.float64)
    )
    minimum_eigenvalue = float(np.min(eigenvalues))
    numerical_rank = int(
        np.linalg.matrix_rank(block_gram, tol=max(atol, np.finfo(float).eps))
    )
    ambient_dim = next(iter(ambient_dims))
    if not math.isfinite(symmetry_error) or symmetry_error > atol:
        failures.append(
            {
                "reason": "block_gram_not_symmetric",
                "frobenius_error": symmetry_error,
            }
        )
    if not math.isfinite(minimum_eigenvalue) or minimum_eigenvalue < -atol:
        failures.append(
            {
                "reason": "block_gram_not_psd",
                "minimum_eigenvalue": minimum_eigenvalue,
            }
        )
    if numerical_rank > ambient_dim:
        failures.append(
            {
                "reason": "block_gram_rank_exceeds_ambient_dimension",
                "numerical_rank": numerical_rank,
                "ambient_dim": ambient_dim,
            }
        )

    maximum_edge_block_error = 0.0
    for edge in graph.edges:
        if (
            edge.source_node_id not in offsets
            or edge.target_node_id not in offsets
        ):
            failures.append(
                {
                    "edge_id": edge.edge_id,
                    "reason": "unknown_endpoint",
                }
            )
            continue
        source_start, source_stop = offsets[edge.source_node_id]
        target_start, target_stop = offsets[edge.target_node_id]
        expected = block_gram[
            target_start:target_stop,
            source_start:source_stop,
        ]
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
        maximum_edge_block_error = max(maximum_edge_block_error, error)
        if not math.isfinite(error) or error > atol:
            failures.append(
                {
                    "edge_id": edge.edge_id,
                    "reason": "block_overlap_mismatch",
                    "frobenius_error": error,
                }
            )

    return {
        "checker_id": "global_block_gram_v1",
        "status": "pass" if not failures else "fail",
        "node_count": len(nodes),
        "edge_count": len(graph.edges),
        "block_gram_dimension": int(block_gram.shape[0]),
        "ambient_dim": int(ambient_dim),
        "numerical_rank": numerical_rank,
        "minimum_eigenvalue": minimum_eigenvalue,
        "symmetry_error": symmetry_error,
        "maximum_diagonal_frobenius_error": maximum_diagonal_error,
        "maximum_edge_block_frobenius_error": maximum_edge_block_error,
        "atol": float(atol),
        "failures": failures,
    }


def n1_reassignment_audit(
    source: Sequence[SyntheticGraph],
    reassigned: Sequence[SyntheticGraph],
    *,
    atol: float = DEFAULT_NUMERIC_ATOL,
) -> dict[str, Any]:
    """Audit N1 assignment strength separately from realizability."""

    source_by_id = {graph.replicate_id: graph for graph in source}
    reassigned_by_id = {graph.replicate_id: graph for graph in reassigned}
    if len(source_by_id) != len(source) or len(reassigned_by_id) != len(reassigned):
        raise Gate12C2DevelopmentError("N1 audit requires unique replicate IDs")
    if set(source_by_id) != set(reassigned_by_id):
        raise Gate12C2DevelopmentError(
            "N1 audit requires identical source and reassigned replicate IDs"
        )

    source_nodes: dict[str, tuple[str, NodeFrame]] = {}
    stratum_members: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
    for graph in source:
        for node in graph.nodes:
            reference = f"{graph.replicate_id}/{node.node_id}"
            if reference in source_nodes:
                raise Gate12C2DevelopmentError(
                    f"duplicate source-node reference: {reference}"
                )
            source_nodes[reference] = (graph.replicate_id, node)
            stratum_members[node.stratum].append(reference)

    failures: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    donor_use: Counter[str] = Counter()
    fixed_point_count = 0
    same_graph_assignment_count = 0
    crossed_stratum_count = 0
    frame_mismatch_count = 0
    for graph in reassigned:
        donor_manifest = graph.metadata.get("donor_node_ids")
        if not isinstance(donor_manifest, Mapping):
            failures.append(
                {
                    "replicate_id": graph.replicate_id,
                    "reason": "missing_donor_manifest",
                }
            )
            continue
        source_node_map = source_by_id[graph.replicate_id].node_map()
        for node in graph.nodes:
            donor_reference = donor_manifest.get(node.node_id)
            if not isinstance(donor_reference, str):
                failures.append(
                    {
                        "target_node_id": node.node_id,
                        "reason": "missing_donor_reference",
                    }
                )
                continue
            donor_entry = source_nodes.get(donor_reference)
            if donor_entry is None:
                failures.append(
                    {
                        "target_node_id": node.node_id,
                        "reason": "unknown_donor_reference",
                        "donor_reference": donor_reference,
                    }
                )
                continue
            donor_graph_id, donor = donor_entry
            source_node = source_node_map.get(node.node_id)
            target_reference = f"{graph.replicate_id}/{node.node_id}"
            is_fixed_point = donor_reference == target_reference
            is_same_graph = donor_graph_id == graph.replicate_id
            crossed_stratum = donor.stratum != node.stratum
            frame_error = float(np.linalg.norm(node.frame - donor.frame, ord="fro"))
            frame_mismatch = not math.isfinite(frame_error) or frame_error > atol
            fixed_point_count += int(is_fixed_point)
            same_graph_assignment_count += int(is_same_graph)
            crossed_stratum_count += int(crossed_stratum)
            frame_mismatch_count += int(frame_mismatch)
            donor_use[donor_reference] += 1
            assignments.append(
                {
                    "target_reference": target_reference,
                    "donor_reference": donor_reference,
                    "target_stratum": list(node.stratum),
                    "fixed_point": is_fixed_point,
                    "same_graph_assignment": is_same_graph,
                    "crossed_stratum": crossed_stratum,
                    "donor_frame_frobenius_error": frame_error,
                    "source_frame_changed": (
                        None
                        if source_node is None
                        else not np.allclose(
                            node.frame,
                            source_node.frame,
                            atol=atol,
                            rtol=0.0,
                        )
                    ),
                }
            )

    unused_donors = sorted(set(source_nodes) - set(donor_use))
    reused_donors = {
        donor: count for donor, count in sorted(donor_use.items()) if count > 1
    }
    derangement_ineligible = {
        repr(stratum): len(members)
        for stratum, members in sorted(stratum_members.items())
        if len(members) < 2
    }
    stratum_sizes = [len(members) for members in stratum_members.values()]
    status = (
        "pass"
        if not failures
        and fixed_point_count == 0
        and same_graph_assignment_count == 0
        and crossed_stratum_count == 0
        and frame_mismatch_count == 0
        and not unused_donors
        and not reused_donors
        and not derangement_ineligible
        else "fail"
    )
    return {
        "audit_id": "n1_role_constrained_derangement_audit_v1",
        "status": status,
        "assignment_count": len(assignments),
        "stratum_count": len(stratum_members),
        "stratum_size_min": min(stratum_sizes) if stratum_sizes else None,
        "stratum_size_median": (
            float(np.median(stratum_sizes)) if stratum_sizes else None
        ),
        "stratum_size_max": max(stratum_sizes) if stratum_sizes else None,
        "stratum_sizes": {
            repr(stratum): len(members)
            for stratum, members in sorted(stratum_members.items())
        },
        "fixed_point_count": fixed_point_count,
        "fixed_point_rate": (
            fixed_point_count / len(assignments) if assignments else None
        ),
        "same_graph_assignment_count": same_graph_assignment_count,
        "crossed_stratum_count": crossed_stratum_count,
        "frame_mismatch_count": frame_mismatch_count,
        "unique_donor_count": len(donor_use),
        "unused_donor_references": unused_donors,
        "reused_donor_counts": reused_donors,
        "derangement_ineligible_strata": derangement_ineligible,
        "failures": failures,
        "assignments": assignments,
        "atol": float(atol),
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


def s2_graph_unconstrained_orientation_draw(
    graphs: Sequence[SyntheticGraph],
    *,
    orientation_seed: str,
    draw_index: int = 0,
) -> tuple[SyntheticGraph, ...]:
    """Create the deliberately invalid comparison used by the S2 stress test.

    Every edge singular spectrum is preserved, but left and right singular
    frames are sampled independently per edge.  This is not a candidate null;
    it is the graph-coupling destruction that S2 must diagnose.
    """

    if draw_index < 0:
        raise Gate12C2DevelopmentError("draw_index must be nonnegative")
    receipt = _seed_receipt(orientation_seed)
    result: list[SyntheticGraph] = []
    for graph in graphs:
        edges: list[EdgeOverlap] = []
        for edge in graph.edges:
            rows, columns = edge.matrix.shape
            singular_values = np.linalg.svd(
                edge.matrix,
                compute_uv=False,
            )
            inner = len(singular_values)
            left = _orthonormalize(
                _rng(
                    orientation_seed,
                    "S2",
                    draw_index,
                    graph.replicate_id,
                    edge.edge_id,
                    "left",
                ).normal(size=(rows, inner))
            )
            right = _orthonormalize(
                _rng(
                    orientation_seed,
                    "S2",
                    draw_index,
                    graph.replicate_id,
                    edge.edge_id,
                    "right",
                ).normal(size=(columns, inner))
            )
            matrix = np.asarray(
                (left * singular_values) @ right.T,
                dtype=np.float64,
            )
            edges.append(
                EdgeOverlap(
                    edge_id=edge.edge_id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    matrix=matrix,
                )
            )
        result.append(
            SyntheticGraph(
                replicate_id=graph.replicate_id,
                regime="S2_graph_unconstrained_orientation_stress",
                nodes=graph.nodes,
                edges=tuple(edges),
                cycle_node_ids=graph.cycle_node_ids,
                generator_id=S2_UNCONSTRAINED_ORIENTATION_ID,
                seed_receipt=receipt,
                metadata={
                    "source_generator_id": graph.generator_id,
                    "source_seed_receipt": graph.seed_receipt,
                    "orientation_draw_index": int(draw_index),
                    "orientation_seed_derivation_id": SEED_DERIVATION_ID,
                    "edge_singular_spectra_preserved": True,
                    "shared_node_realizability_deliberately_destroyed": True,
                    "candidate_null": False,
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
    product_singular_values_left = tuple(
        float(value)
        for value in np.linalg.svd(product_left, compute_uv=False)
    )
    product_singular_values_right = tuple(
        float(value)
        for value in np.linalg.svd(product_right, compute_uv=False)
    )
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
            product_singular_values_left=product_singular_values_left,
            product_singular_values_right=product_singular_values_right,
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
        product_singular_values_left=product_singular_values_left,
        product_singular_values_right=product_singular_values_right,
    )


def development_residual_mechanism_controls(
    *,
    tail_levels: Sequence[float] = (0.10, 0.20, 0.35, 0.50),
    propagation_levels: Sequence[float] = (0.30, 0.45, 0.65, 0.90),
    alignment_angles: Sequence[float] = (0.10, 0.30, 0.60, 1.00),
    fixed_tail: float = 0.20,
    fixed_propagated_magnitude: float = 0.20,
) -> dict[str, tuple[ResidualMechanismControl, ...]]:
    """Construct separate algebraic controls for tail, propagation, alignment.

    These controls verify that the residual diagnostics resolve the three
    mechanisms.  They are intentionally not mislabeled as the graph-realizable
    S1 regime required before locked calibration.
    """

    controls: dict[str, list[ResidualMechanismControl]] = {
        "tail": [],
        "propagation": [],
        "alignment": [],
    }
    identity = np.eye(2, dtype=np.float64)

    for level in tail_levels:
        numeric = float(level)
        if not 0.0 < numeric < 1.0:
            raise Gate12C2DevelopmentError(
                "tail control levels must lie strictly between 0 and 1"
            )
        m0 = np.diag([numeric, 1.0])
        m2 = np.diag([1.0, numeric])
        diagnostic = residual_diagnostics(m0, identity, m2, q=1)
        controls["tail"].append(
            ResidualMechanismControl(
                mechanism="tail",
                level=numeric,
                q=1,
                m0=m0,
                m1=identity,
                m2=m2,
                declared_control=(
                    "p_L_q=p_R_q=1 and c_q=0; u_q=v_q=level"
                ),
                diagnostics=diagnostic,
            )
        )

    if not 0.0 < fixed_tail < 1.0:
        raise Gate12C2DevelopmentError(
            "fixed_tail must lie strictly between 0 and 1"
        )
    for level in propagation_levels:
        numeric = float(level)
        if not fixed_tail < numeric <= 1.0:
            raise Gate12C2DevelopmentError(
                "propagation levels must exceed fixed_tail and be at most 1"
            )
        m0 = np.diag([fixed_tail, numeric])
        m2 = np.diag([numeric, fixed_tail])
        diagnostic = residual_diagnostics(m0, identity, m2, q=1)
        controls["propagation"].append(
            ResidualMechanismControl(
                mechanism="propagation",
                level=numeric,
                q=1,
                m0=m0,
                m1=identity,
                m2=m2,
                declared_control=(
                    f"u_q=v_q={fixed_tail}; c_q=0; "
                    "p_L_q=p_R_q=level"
                ),
                diagnostics=diagnostic,
            )
        )

    if not 0.0 < fixed_propagated_magnitude < 1.0:
        raise Gate12C2DevelopmentError(
            "fixed_propagated_magnitude must lie in (0, 1)"
        )
    leading = 1.0
    for angle in alignment_angles:
        numeric = float(angle)
        if not 0.0 < numeric < math.pi / 2.0:
            raise Gate12C2DevelopmentError(
                "alignment angles must lie strictly between 0 and pi/2"
            )
        sine_squared = math.sin(numeric) ** 2
        cosine_squared = math.cos(numeric) ** 2
        if cosine_squared <= np.finfo(np.float64).eps:
            residual_squared = (
                fixed_propagated_magnitude**2
                / (leading * leading * sine_squared)
            )
        else:
            linear = leading * leading * sine_squared
            discriminant = (
                linear * linear
                + 4.0
                * cosine_squared
                * fixed_propagated_magnitude**2
            )
            residual_squared = (
                -linear + math.sqrt(discriminant)
            ) / (2.0 * cosine_squared)
        residual_singular_value = math.sqrt(residual_squared)
        if not 0.0 < residual_singular_value < leading:
            raise Gate12C2DevelopmentError(
                "alignment control produced an invalid spectral ordering"
            )
        rotation = np.asarray(
            [
                [math.cos(numeric), -math.sin(numeric)],
                [math.sin(numeric), math.cos(numeric)],
            ],
            dtype=np.float64,
        )
        m2 = np.diag([leading, residual_singular_value])
        m0 = np.asarray(
            rotation
            @ np.diag([leading, residual_singular_value])
            @ rotation.T,
            dtype=np.float64,
        )
        diagnostic = residual_diagnostics(m0, identity, m2, q=1)
        controls["alignment"].append(
            ResidualMechanismControl(
                mechanism="alignment",
                level=numeric,
                q=1,
                m0=m0,
                m1=identity,
                m2=m2,
                declared_control=(
                    "x_q and y_q fixed at "
                    f"{fixed_propagated_magnitude}; angle changes c_q"
                ),
                diagnostics=diagnostic,
            )
        )

    return {
        mechanism: tuple(rows)
        for mechanism, rows in controls.items()
    }


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


def development_s2_null_inflation_report(
    observed: Sequence[SyntheticGraph],
    unconstrained: Sequence[SyntheticGraph],
    *,
    q: int,
) -> dict[str, Any]:
    """Report S2 component movement without treating the stressor as a null."""

    observed_by_id = {graph.replicate_id: graph for graph in observed}
    comparison_by_id = {
        graph.replicate_id: graph for graph in unconstrained
    }
    if (
        len(observed_by_id) != len(observed)
        or len(comparison_by_id) != len(unconstrained)
        or set(observed_by_id) != set(comparison_by_id)
    ):
        raise Gate12C2DevelopmentError(
            "S2 report requires unique, matching replicate IDs"
        )

    component_fields = (
        "a_q",
        "u_q",
        "v_q",
        "x_q",
        "y_q",
        "c_q",
        "p_L_q",
        "p_R_q",
    )
    rows: list[dict[str, Any]] = []
    defect_inflation_count = 0
    valid_pair_count = 0
    for replicate_id in sorted(observed_by_id):
        observed_diagnostic = graph_residual_diagnostics(
            observed_by_id[replicate_id],
            q=q,
        ).as_dict()
        comparison_diagnostic = graph_residual_diagnostics(
            comparison_by_id[replicate_id],
            q=q,
        ).as_dict()
        differences: dict[str, float | None] = {}
        for field_name in component_fields:
            observed_value = observed_diagnostic[field_name]
            comparison_value = comparison_diagnostic[field_name]
            differences[field_name] = (
                None
                if observed_value is None or comparison_value is None
                else float(comparison_value - observed_value)
            )
        if (
            observed_diagnostic["a_q"] is not None
            and comparison_diagnostic["a_q"] is not None
        ):
            valid_pair_count += 1
            defect_inflation_count += int(
                comparison_diagnostic["a_q"] > observed_diagnostic["a_q"]
            )
        rows.append(
            {
                "replicate_id": replicate_id,
                "observed": observed_diagnostic,
                "graph_unconstrained_comparison": comparison_diagnostic,
                "comparison_minus_observed": differences,
            }
        )

    direct_realizability_fail_count = sum(
        check_joint_realizability(graph)["status"] == "fail"
        for graph in unconstrained
    )
    block_gram_fail_count = sum(
        check_block_gram_realizability(graph)["status"] == "fail"
        for graph in unconstrained
    )
    component_differences: dict[str, list[float]] = {
        field_name: [] for field_name in component_fields
    }
    for row in rows:
        for field_name, value in row["comparison_minus_observed"].items():
            if value is not None:
                component_differences[field_name].append(float(value))
    return {
        "schema_version": DEVELOPMENT_REPORT_SCHEMA_VERSION,
        "epistemic_status": "development_s2_stress_only",
        "regime": "S2_null_inflation",
        "observed_process_modified": False,
        "comparison_id": S2_UNCONSTRAINED_ORIENTATION_ID,
        "comparison_is_candidate_null": False,
        "comparison_graph_constraint": "deliberately_violated",
        "q": int(q),
        "replicate_count": len(rows),
        "valid_pair_count": valid_pair_count,
        "defect_inflation_count": defect_inflation_count,
        "defect_inflation_rate": (
            defect_inflation_count / valid_pair_count
            if valid_pair_count
            else None
        ),
        "realizability_failure_count": {
            "direct_checker": direct_realizability_fail_count,
            "block_gram_checker": block_gram_fail_count,
        },
        "edge_spectrum_marginal_discrepancy": (
            edge_spectrum_marginal_discrepancy(
                observed,
                unconstrained,
            )
        ),
        "component_difference_medians": {
            field_name: (
                float(np.median(values)) if values else None
            )
            for field_name, values in component_differences.items()
        },
        "identification_boundary": (
            "A negative observed/comparison ratio is insufficient. S2 is "
            "identified only through separate observed and comparison "
            "movements in a, tails, propagation, and alignment."
        ),
        "rows": rows,
    }


def development_s1_known_reverse_report(
    observed: Sequence[SyntheticGraph],
    comparison: Sequence[SyntheticGraph],
    *,
    q: int,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Summarize a graph-realizable S1 minimal example without claiming power."""

    observed_by_id = {graph.replicate_id: graph for graph in observed}
    comparison_by_id = {graph.replicate_id: graph for graph in comparison}
    if (
        len(observed_by_id) != len(observed)
        or len(comparison_by_id) != len(comparison)
        or set(observed_by_id) != set(comparison_by_id)
    ):
        raise Gate12C2DevelopmentError(
            "S1 report requires unique, matching replicate IDs"
        )
    effect_strengths = {
        float(graph.metadata["effect_strength"]) for graph in observed
    }
    if len(effect_strengths) != 1:
        raise Gate12C2DevelopmentError(
            "one S1 report must contain one effect strength"
        )
    mismatch_floors = {
        float(graph.metadata["observed_mismatch_floor"])
        for graph in observed
    }
    if len(mismatch_floors) != 1:
        raise Gate12C2DevelopmentError(
            "one S1 report must contain one observed mismatch floor"
        )

    component_fields = (
        "a_q",
        "u_q",
        "v_q",
        "x_q",
        "y_q",
        "c_q",
        "p_L_q",
        "p_R_q",
    )
    rows: list[dict[str, Any]] = []
    log_ratios: list[float] = []
    smaller_count = 0
    component_observed: dict[str, list[float]] = {
        field_name: [] for field_name in component_fields
    }
    component_comparison: dict[str, list[float]] = {
        field_name: [] for field_name in component_fields
    }
    for replicate_id in sorted(observed_by_id):
        observed_diagnostic = graph_residual_diagnostics(
            observed_by_id[replicate_id],
            q=q,
        ).as_dict()
        comparison_diagnostic = graph_residual_diagnostics(
            comparison_by_id[replicate_id],
            q=q,
        ).as_dict()
        observed_defect = observed_diagnostic["a_q"]
        comparison_defect = comparison_diagnostic["a_q"]
        log_ratio = None
        if observed_defect is not None and comparison_defect is not None:
            log_ratio = float(
                math.log(observed_defect + epsilon)
                - math.log(comparison_defect + epsilon)
            )
            log_ratios.append(log_ratio)
            smaller_count += int(observed_defect < comparison_defect)
        for field_name in component_fields:
            observed_value = observed_diagnostic[field_name]
            comparison_value = comparison_diagnostic[field_name]
            if observed_value is not None:
                component_observed[field_name].append(float(observed_value))
            if comparison_value is not None:
                component_comparison[field_name].append(
                    float(comparison_value)
                )
        rows.append(
            {
                "replicate_id": replicate_id,
                "observed": observed_diagnostic,
                "N1_comparison": comparison_diagnostic,
                "log_observed_to_N1_defect": log_ratio,
            }
        )

    return {
        "schema_version": DEVELOPMENT_REPORT_SCHEMA_VERSION,
        "epistemic_status": "development_s1_minimal_example_only",
        "regime": "S1_known_reverse_shared_node_coupling",
        "generator_id": S1_SHARED_COUPLING_GENERATOR_ID,
        "null_candidate_id": N1_ID,
        "effect_strength": next(iter(effect_strengths)),
        "observed_mismatch_floor": next(iter(mismatch_floors)),
        "effect_mechanism": (
            "shared_node_residual_cancellation_with_fixed_offset"
        ),
        "known_direction": "observed_defect_smaller_than_N1",
        "q": int(q),
        "replicate_count": len(rows),
        "informative_pair_count": len(log_ratios),
        "observed_smaller_count": smaller_count,
        "observed_smaller_rate": (
            smaller_count / len(log_ratios) if log_ratios else None
        ),
        "median_log_observed_to_N1_defect": (
            float(np.median(log_ratios)) if log_ratios else None
        ),
        "component_medians": {
            field_name: {
                "observed": (
                    float(np.median(component_observed[field_name]))
                    if component_observed[field_name]
                    else None
                ),
                "N1": (
                    float(np.median(component_comparison[field_name]))
                    if component_comparison[field_name]
                    else None
                ),
            }
            for field_name in component_fields
        },
        "joint_realizability": {
            "observed_direct_pass_count": sum(
                check_joint_realizability(graph)["status"] == "pass"
                for graph in observed
            ),
            "N1_direct_pass_count": sum(
                check_joint_realizability(graph)["status"] == "pass"
                for graph in comparison
            ),
            "observed_block_gram_pass_count": sum(
                check_block_gram_realizability(graph)["status"] == "pass"
                for graph in observed
            ),
            "N1_block_gram_pass_count": sum(
                check_block_gram_realizability(graph)["status"] == "pass"
                for graph in comparison
            ),
        },
        "n1_assignment_audit": n1_reassignment_audit(
            observed,
            comparison,
        ),
        "power": {
            "status": "not_estimated_without_outer_experiments",
            "estimate": None,
        },
        "rows": rows,
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
    observed_block_gram = [
        check_block_gram_realizability(graph)["status"] == "pass"
        for graph in observed
    ]
    comparison_block_gram = [
        check_block_gram_realizability(graph)["status"] == "pass"
        for graph in comparison
    ]
    observed_status_counts = Counter(
        str(row["observed"]["eligibility_status"]) for row in rows
    )
    comparison_status_counts = Counter(
        str(row["comparison"]["eligibility_status"]) for row in rows
    )

    return {
        "schema_version": DEVELOPMENT_REPORT_SCHEMA_VERSION,
        "epistemic_status": "development_only",
        "regime": "S0_true_null",
        "null_candidate_id": N1_ID,
        "candidate_selection_policy": N1_ESCALATION_POLICY,
        "independent_unit": "synthetic_graph_replicate",
        "reference_dtype": REFERENCE_DTYPE,
        "q": int(q),
        "replicate_count": len(rows),
        "informative_count": len(informative_log_ratios),
        "ineligible_count": ineligible,
        "numerical_failure_count": numerical_failures,
        "degenerate_alignment_pair_count": degenerate_alignment,
        "joint_realizability": {
            "direct_checker_id": "direct_edge_reconstruction_v1",
            "independent_checker_id": "global_block_gram_v1",
            "observed_pass_count": sum(observed_realizability),
            "comparison_pass_count": sum(comparison_realizability),
            "observed_total": len(observed_realizability),
            "comparison_total": len(comparison_realizability),
            "observed_block_gram_pass_count": sum(observed_block_gram),
            "comparison_block_gram_pass_count": sum(comparison_block_gram),
        },
        "n1_assignment_audit": n1_reassignment_audit(
            observed,
            comparison,
        ),
        "edge_spectrum_marginal_discrepancy": (
            edge_spectrum_marginal_discrepancy(observed, comparison)
        ),
        "diagnostic_only_not_matching_constraints": {
            "observed_eligibility_status_counts": dict(
                sorted(observed_status_counts.items())
            ),
            "comparison_eligibility_status_counts": dict(
                sorted(comparison_status_counts.items())
            ),
            "valid_pair_rate": (
                len(informative_log_ratios) / len(rows) if rows else None
            ),
            "retry_count": 0,
            "retry_policy": "no_retry_in_current_generator",
            "split_gap_fields": [
                "relative_gap_left",
                "relative_gap_right",
            ],
            "product_spectrum_fields": [
                "product_singular_values_left",
                "product_singular_values_right",
            ],
        },
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


def _holm_adjusted_p_values(
    endpoints: Sequence[EndpointDecisionInput],
) -> dict[tuple[str, int], tuple[float, int]]:
    if len(endpoints) != PIPELINE_ENDPOINT_COUNT:
        raise Gate12C2DevelopmentError(
            "the Gate12C pipeline decision requires exactly 24 endpoints"
        )
    ordered = sorted(
        endpoints,
        key=lambda endpoint: (
            float(endpoint.raw_p),
            int(endpoint.case_order),
            int(endpoint.q),
        ),
    )
    adjusted: dict[tuple[str, int], tuple[float, int]] = {}
    running = 0.0
    for position, endpoint in enumerate(ordered, start=1):
        raw_p = float(endpoint.raw_p)
        if not math.isfinite(raw_p) or not 0.0 <= raw_p <= 1.0:
            raise Gate12C2DevelopmentError(
                f"invalid raw p-value for {endpoint.case_id}/q={endpoint.q}"
            )
        multiplier = PIPELINE_ENDPOINT_COUNT - position + 1
        running = max(running, min(1.0, raw_p * multiplier))
        adjusted[(endpoint.case_id, endpoint.q)] = (float(running), position)
    return adjusted


def _sign_with_tolerance(
    value: float | None,
    *,
    zero_tolerance: float,
) -> int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    numeric = float(value)
    if numeric > zero_tolerance:
        return 1
    if numeric < -zero_tolerance:
        return -1
    return 0


def complete_pipeline_decision(
    endpoints: Sequence[EndpointDecisionInput],
    *,
    holm_alpha: float = DEFAULT_HOLM_ALPHA,
    zero_tolerance: float = DEFAULT_PRIMARY_ZERO_TOLERANCE,
) -> dict[str, Any]:
    """Run one complete Gate12C-style 24-endpoint decision hierarchy.

    This function defines the unit that an outer S0 calibration must repeat.
    It intentionally refuses single-triangle or single-endpoint substitutes.
    """

    if not 0.0 < holm_alpha < 1.0:
        raise Gate12C2DevelopmentError("holm_alpha must lie in (0, 1)")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise Gate12C2DevelopmentError(
            "zero_tolerance must be finite and nonnegative"
        )
    keys = [(endpoint.case_id, endpoint.q) for endpoint in endpoints]
    if len(set(keys)) != len(keys):
        raise Gate12C2DevelopmentError("duplicate case/q endpoint")
    case_orders = {
        endpoint.case_id: endpoint.case_order for endpoint in endpoints
    }
    if len(case_orders) != 12 or len(set(case_orders.values())) != 12:
        raise Gate12C2DevelopmentError(
            "pipeline decision requires 12 uniquely ordered cases"
        )
    by_case: dict[str, list[EndpointDecisionInput]] = defaultdict(list)
    for endpoint in endpoints:
        if endpoint.q not in {1, 2}:
            raise Gate12C2DevelopmentError("each endpoint q must be 1 or 2")
        by_case[endpoint.case_id].append(endpoint)
    if any(
        {endpoint.q for endpoint in rows} != {1, 2} or len(rows) != 2
        for rows in by_case.values()
    ):
        raise Gate12C2DevelopmentError(
            "each of the 12 cases must contain exactly q=1 and q=2"
        )
    for case_id, rows in by_case.items():
        metadata = {
            (row.case_order, row.model, row.family) for row in rows
        }
        if len(metadata) != 1:
            raise Gate12C2DevelopmentError(
                f"case metadata differs across q endpoints: {case_id}"
            )

    adjusted = _holm_adjusted_p_values(endpoints)
    endpoint_rows: list[dict[str, Any]] = []
    rows_by_case: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for endpoint in sorted(
        endpoints, key=lambda row: (row.case_order, row.q)
    ):
        median = endpoint.median_log_ratio
        if median is not None and not math.isfinite(float(median)):
            raise Gate12C2DevelopmentError(
                f"non-finite median for {endpoint.case_id}/q={endpoint.q}"
            )
        adjusted_p, sort_position = adjusted[(endpoint.case_id, endpoint.q)]
        q_support = bool(
            endpoint.coverage_complete
            and endpoint.informative
            and median is not None
            and float(median) > zero_tolerance
            and adjusted_p < holm_alpha
        )
        row = {
            "endpoint_id": f"{endpoint.case_id}:q{endpoint.q}",
            "case_id": endpoint.case_id,
            "case_order": int(endpoint.case_order),
            "model": endpoint.model,
            "family": endpoint.family,
            "q": int(endpoint.q),
            "coverage_complete": bool(endpoint.coverage_complete),
            "informative": bool(endpoint.informative),
            "median_log_ratio": (
                None if median is None else float(median)
            ),
            "raw_p": float(endpoint.raw_p),
            "holm_adjusted_p": adjusted_p,
            "holm_sort_position": sort_position,
            "q_support": q_support,
            "run_support": False,
            "q_discordant_run": None,
        }
        endpoint_rows.append(row)
        rows_by_case[endpoint.case_id][endpoint.q] = row

    run_rows: list[dict[str, Any]] = []
    for case_id, by_q in sorted(
        rows_by_case.items(), key=lambda item: item[1][1]["case_order"]
    ):
        q1 = by_q[1]
        q2 = by_q[2]
        run_support = bool(q1["q_support"] and q2["q_support"])
        sign_q1 = _sign_with_tolerance(
            q1["median_log_ratio"],
            zero_tolerance=zero_tolerance,
        )
        sign_q2 = _sign_with_tolerance(
            q2["median_log_ratio"],
            zero_tolerance=zero_tolerance,
        )
        q_discordant = (
            None
            if sign_q1 is None or sign_q2 is None
            else bool(
                bool(q1["q_support"]) != bool(q2["q_support"])
                or sign_q1 != sign_q2
            )
        )
        q1["run_support"] = run_support
        q2["run_support"] = run_support
        q1["q_discordant_run"] = q_discordant
        q2["q_discordant_run"] = q_discordant
        run_rows.append(
            {
                "case_id": case_id,
                "case_order": q1["case_order"],
                "model": q1["model"],
                "family": q1["family"],
                "run_support": run_support,
                "q_discordant_run": q_discordant,
            }
        )

    coverage_limited = any(
        not row["coverage_complete"] or not row["informative"]
        for row in endpoint_rows
    )
    q_discordant_count = sum(
        row["q_discordant_run"] is True for row in run_rows
    )
    supporting_runs = [row for row in run_rows if row["run_support"]]
    support_count = len(supporting_runs)
    family_counts = Counter(row["family"] for row in supporting_runs)
    model_counts = Counter(row["model"] for row in supporting_runs)
    all_families = {row["family"] for row in run_rows}
    all_models = {row["model"] for row in run_rows}
    breadth_pass = bool(
        all(family_counts[family] >= 3 for family in all_families)
        and all(model_counts[model] >= 2 for model in all_models)
    )
    if coverage_limited:
        grid_outcome = "coverage_limited"
    elif q_discordant_count >= 6:
        grid_outcome = "mixed_q"
    elif support_count == 12:
        grid_outcome = "strong_broad"
    elif support_count >= 10 and breadth_pass:
        grid_outcome = "broad_replicated"
    elif support_count == 0:
        grid_outcome = "no_directional_support"
    else:
        grid_outcome = "partial_or_structured"

    directional_grid_positive = grid_outcome in {
        "strong_broad",
        "broad_replicated",
        "partial_or_structured",
    }
    return {
        "schema_version": PIPELINE_DECISION_SCHEMA_VERSION,
        "epistemic_status": "development_calibration_unit",
        "independent_unit": "complete_24_endpoint_outer_experiment",
        "holm_alpha": float(holm_alpha),
        "zero_tolerance": float(zero_tolerance),
        "endpoint_count": len(endpoint_rows),
        "q_support_count": sum(row["q_support"] for row in endpoint_rows),
        "any_endpoint_support": any(
            row["q_support"] for row in endpoint_rows
        ),
        "supporting_run_count": support_count,
        "any_run_support": support_count > 0,
        "q_discordant_run_count": q_discordant_count,
        "grid_outcome": grid_outcome,
        "directional_grid_positive": directional_grid_positive,
        "endpoint_rows": endpoint_rows,
        "run_rows": run_rows,
    }


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float | int]:
    if total <= 0 or successes < 0 or successes > total:
        raise Gate12C2DevelopmentError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "successes": int(successes),
        "total": int(total),
        "estimate": float(proportion),
        "wilson_95_lower": float(max(0.0, center - half_width)),
        "wilson_95_upper": float(min(1.0, center + half_width)),
    }


def summarize_outer_calibration(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize repeated complete-pipeline outer experiments.

    Endpoint FPR, family-wise FPR, run-level FPR, and grid-level positive rate
    remain separate estimands.  No acceptance threshold is supplied here.
    """

    if not decisions:
        raise Gate12C2DevelopmentError(
            "outer calibration requires at least one complete experiment"
        )
    expected_endpoint_ids: tuple[str, ...] | None = None
    endpoint_successes: Counter[str] = Counter()
    any_endpoint_count = 0
    any_run_count = 0
    grid_positive_count = 0
    outcome_counts: Counter[str] = Counter()
    for index, decision in enumerate(decisions):
        if (
            decision.get("schema_version")
            != PIPELINE_DECISION_SCHEMA_VERSION
            or decision.get("independent_unit")
            != "complete_24_endpoint_outer_experiment"
            or int(decision.get("endpoint_count", -1))
            != PIPELINE_ENDPOINT_COUNT
        ):
            raise Gate12C2DevelopmentError(
                f"outer experiment {index} is not a complete pipeline decision"
            )
        rows = decision.get("endpoint_rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise Gate12C2DevelopmentError(
                f"outer experiment {index} has invalid endpoint rows"
            )
        endpoint_ids = tuple(str(row["endpoint_id"]) for row in rows)
        if expected_endpoint_ids is None:
            expected_endpoint_ids = endpoint_ids
        elif endpoint_ids != expected_endpoint_ids:
            raise Gate12C2DevelopmentError(
                "outer experiments must use the same ordered endpoint family"
            )
        for row in rows:
            endpoint_successes[str(row["endpoint_id"])] += int(
                bool(row["q_support"])
            )
        any_endpoint_count += int(bool(decision["any_endpoint_support"]))
        any_run_count += int(bool(decision["any_run_support"]))
        grid_positive_count += int(bool(decision["directional_grid_positive"]))
        outcome_counts[str(decision["grid_outcome"])] += 1

    total = len(decisions)
    assert expected_endpoint_ids is not None
    return {
        "schema_version": OUTER_CALIBRATION_SCHEMA_VERSION,
        "epistemic_status": "development_only_no_acceptance_threshold",
        "outer_independent_unit": "complete_24_endpoint_outer_experiment",
        "outer_experiment_count": total,
        "type_i_estimands": {
            "endpoint_fpr": (
                "P(q_support for a named case/q endpoint under S0)"
            ),
            "family_wise_fpr": (
                "P(at least one of 24 endpoints has q_support under S0)"
            ),
            "run_level_fpr": (
                "P(at least one case has both q endpoints supported under S0)"
            ),
            "grid_level_positive_rate": (
                "P(final grid outcome is strong_broad, broad_replicated, "
                "or partial_or_structured under S0)"
            ),
        },
        "endpoint_fpr": {
            endpoint_id: _wilson_interval(
                endpoint_successes[endpoint_id],
                total,
            )
            for endpoint_id in expected_endpoint_ids
        },
        "family_wise_fpr": _wilson_interval(any_endpoint_count, total),
        "run_level_fpr": _wilson_interval(any_run_count, total),
        "grid_level_positive_rate": _wilson_interval(
            grid_positive_count,
            total,
        ),
        "grid_outcome_counts": dict(sorted(outcome_counts.items())),
        "acceptance_rule": {
            "status": "not_frozen",
            "warning": (
                "Sample counts alone do not define acceptance. FPR, power, "
                "null-inflation, nuisance-match, and instability thresholds "
                "must be frozen before locked execution."
            ),
        },
    }


def nested_inner_draw_stability(
    ordered_null_draws: Sequence[float],
    *,
    observed_value: float,
    prefix_counts: Sequence[int] = (255, 511, 1023),
    decision_alpha: float = 0.05,
    runtime_seconds_by_prefix: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Compare nested prefixes without using calibration performance.

    The same deterministic stream is used throughout, so 255 is a prefix of
    511, which is a prefix of 1023.  Draw count may later be selected from
    stability, Monte Carlo precision, and runtime only—not from whichever
    count produces the most favorable FPR or power.
    """

    values = np.asarray(ordered_null_draws, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise Gate12C2DevelopmentError(
            "ordered_null_draws must be a nonempty finite one-dimensional stream"
        )
    observed = float(observed_value)
    if not math.isfinite(observed):
        raise Gate12C2DevelopmentError("observed_value must be finite")
    counts = tuple(int(count) for count in prefix_counts)
    if (
        not counts
        or any(count <= 0 for count in counts)
        or tuple(sorted(set(counts))) != counts
        or counts[-1] > values.size
    ):
        raise Gate12C2DevelopmentError(
            "prefix_counts must be unique, increasing, positive, and available"
        )
    if not 0.0 < decision_alpha < 1.0:
        raise Gate12C2DevelopmentError("decision_alpha must lie in (0, 1)")
    runtimes = {
        int(count): float(seconds)
        for count, seconds in (runtime_seconds_by_prefix or {}).items()
    }
    if any(
        count not in counts
        or not math.isfinite(seconds)
        or seconds < 0.0
        for count, seconds in runtimes.items()
    ):
        raise Gate12C2DevelopmentError(
            "runtime entries must be finite, nonnegative prefix values"
        )

    rows: list[dict[str, Any]] = []
    for count in counts:
        prefix = values[:count]
        lower_p = float((1 + np.sum(prefix <= observed)) / (count + 1))
        upper_p = float((1 + np.sum(prefix >= observed)) / (count + 1))
        rows.append(
            {
                "draw_count": count,
                "null_mean": float(np.mean(prefix)),
                "null_median": float(np.median(prefix)),
                "null_standard_deviation": float(np.std(prefix, ddof=1))
                if count > 1
                else 0.0,
                "null_quantile_05": float(np.quantile(prefix, 0.05)),
                "null_quantile_95": float(np.quantile(prefix, 0.95)),
                "lower_tail_monte_carlo_p": lower_p,
                "upper_tail_monte_carlo_p": upper_p,
                "lower_tail_decision": lower_p < decision_alpha,
                "upper_tail_decision": upper_p < decision_alpha,
                "minimum_attainable_monte_carlo_p": 1.0 / (count + 1),
                "runtime_seconds": runtimes.get(count),
            }
        )

    reference = rows[-1]
    median_scale = max(
        abs(float(reference["null_median"])),
        np.finfo(np.float64).tiny,
    )
    for row in rows:
        row["absolute_median_difference_vs_largest_prefix"] = abs(
            float(row["null_median"]) - float(reference["null_median"])
        )
        row["relative_median_difference_vs_largest_prefix"] = (
            row["absolute_median_difference_vs_largest_prefix"]
            / median_scale
        )
        row["lower_tail_decision_disagrees_with_largest_prefix"] = bool(
            row["lower_tail_decision"] != reference["lower_tail_decision"]
        )
        row["upper_tail_decision_disagrees_with_largest_prefix"] = bool(
            row["upper_tail_decision"] != reference["upper_tail_decision"]
        )

    return {
        "schema_version": INNER_DRAW_STABILITY_SCHEMA_VERSION,
        "epistemic_status": "development_draw_count_stability_only",
        "stream_order_sha256": hashlib.sha256(
            np.ascontiguousarray(values).tobytes()
        ).hexdigest(),
        "ordered_stream_length": int(values.size),
        "prefix_counts": list(counts),
        "nested_prefix_contract": all(
            np.array_equal(values[:left], values[:right][:left])
            for left, right in zip(counts, counts[1:])
        ),
        "observed_value": observed,
        "decision_alpha": float(decision_alpha),
        "selection_basis_allowed": [
            "decision_stability",
            "Monte_Carlo_precision",
            "runtime",
        ],
        "selection_basis_prohibited": [
            "best_observed_FPR",
            "best_observed_power",
            "most_favorable_direction",
        ],
        "selected_draw_count": None,
        "rows": rows,
    }


def manifests(graphs: Iterable[SyntheticGraph]) -> list[dict[str, Any]]:
    return [graph.manifest() for graph in graphs]
