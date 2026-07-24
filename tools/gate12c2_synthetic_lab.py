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
PIPELINE_DECISION_SCHEMA_VERSION = "gate12c2_pipeline_decision_v0.2"
OUTER_CALIBRATION_SCHEMA_VERSION = "gate12c2_outer_calibration_v0.2"
MECHANISM_CONTROL_SCHEMA_VERSION = "gate12c2_mechanism_control_v0.1"
INNER_DRAW_STABILITY_SCHEMA_VERSION = "gate12c2_inner_draw_stability_v0.2"
SEED_NAMESPACE_SCHEMA_VERSION = "gate12c2_seed_namespace_v0.2"
ACCEPTED_DRAW_STREAM_SCHEMA_VERSION = "gate12c2_accepted_draw_stream_v0.1"
OUTER_EXPERIMENT_SCHEMA_VERSION = "gate12c2_outer_experiment_v0.2"
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
C2_CONTRACT_VERSION = "0.2"
OBJECT_REFERENCE_DIAGNOSTIC_KERNEL = "object_fp64_reference_v0.1"
BATCHED_DIAGNOSTIC_KERNEL = "batched_fp64_equivalent_v0.1"
ALLOWED_DIAGNOSTIC_KERNELS = frozenset(
    {
        OBJECT_REFERENCE_DIAGNOSTIC_KERNEL,
        BATCHED_DIAGNOSTIC_KERNEL,
    }
)
PRIMARY_ALTERNATIVE = "observed_smaller_than_null"
ALLOWED_ALTERNATIVES = frozenset(
    {
        PRIMARY_ALTERNATIVE,
        "observed_larger_than_null",
    }
)
CALIBRATION_PROMOTION_OUTCOMES = frozenset(
    {
        "strong_broad",
        "broad_replicated",
    }
)
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
REFERENCE_BLOCK_COUNT_BY_FAMILY = {
    "family-0": 128,
    "family-1": 200,
    "family-2": 128,
}
ONE_SIDED_95_Z = 1.6448536269514722
S0_MAX_POINT_FPR = 0.05
S0_MAX_ONE_SIDED_95_UPPER = 0.07
S1_MIN_POINT_POWER = 0.80
S1_MIN_ONE_SIDED_95_LOWER = 0.75
S2_MIN_POINT_IDENTIFICATION = 0.80
S2_MIN_ONE_SIDED_95_LOWER = 0.75
S2_MIN_LOG_NULL_INFLATION = 0.05


class Gate12C2DevelopmentError(ValueError):
    """Raised when a development graph or operation violates its contract."""


@dataclass(frozen=True)
class OuterSeedNamespace:
    """Typed, scheduling-invariant seed key for one attempted draw."""

    surface_id: str
    null_candidate_id: str
    regime_id: str
    effect_strength: float | None
    outer_experiment_index: int
    case_or_endpoint_id: str
    cycle_or_root_id: str
    draw_attempt_index: int
    contract_version: str = C2_CONTRACT_VERSION
    schema_version: str = SEED_NAMESPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.surface_id not in {"development", "locked_synthetic"}:
            raise Gate12C2DevelopmentError(
                "surface_id must be development or locked_synthetic"
            )
        for name in (
            "null_candidate_id",
            "regime_id",
            "case_or_endpoint_id",
            "cycle_or_root_id",
            "contract_version",
        ):
            if not str(getattr(self, name)).strip():
                raise Gate12C2DevelopmentError(
                    f"seed namespace field {name} must be nonempty"
                )
        if self.effect_strength is not None and not math.isfinite(
            float(self.effect_strength)
        ):
            raise Gate12C2DevelopmentError(
                "effect_strength must be finite when present"
            )
        if self.outer_experiment_index < 0 or self.draw_attempt_index < 0:
            raise Gate12C2DevelopmentError(
                "outer and draw attempt indices must be nonnegative"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "surface_id": self.surface_id,
            "null_candidate_id": self.null_candidate_id,
            "regime_id": self.regime_id,
            "effect_strength": (
                None
                if self.effect_strength is None
                else float(self.effect_strength)
            ),
            "outer_experiment_index": int(self.outer_experiment_index),
            "case_or_endpoint_id": self.case_or_endpoint_id,
            "cycle_or_root_id": self.cycle_or_root_id,
            "draw_attempt_index": int(self.draw_attempt_index),
        }


@dataclass(frozen=True)
class NullDrawAttempt:
    """One auditable attempted null draw before valid-prefix extraction."""

    attempt_index: int
    accepted: bool
    value: float | None
    rejection_reason: str | None
    accepted_draw_index: int | None
    seed_namespace_sha256: str

    def __post_init__(self) -> None:
        if self.attempt_index < 0:
            raise Gate12C2DevelopmentError(
                "draw attempt index must be nonnegative"
            )
        if self.accepted:
            if (
                self.value is None
                or not math.isfinite(float(self.value))
                or self.accepted_draw_index is None
                or self.accepted_draw_index < 0
                or self.rejection_reason is not None
            ):
                raise Gate12C2DevelopmentError(
                    "accepted draw requires finite value, accepted index, "
                    "and no rejection reason"
                )
        elif (
            self.value is not None
            or self.accepted_draw_index is not None
            or not str(self.rejection_reason or "").strip()
        ):
            raise Gate12C2DevelopmentError(
                "rejected draw requires a reason and no value or accepted index"
            )
        if len(self.seed_namespace_sha256) != 64:
            raise Gate12C2DevelopmentError(
                "seed namespace receipt must be a SHA-256 hex digest"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": int(self.attempt_index),
            "accepted": bool(self.accepted),
            "value": None if self.value is None else float(self.value),
            "rejection_reason": self.rejection_reason,
            "accepted_draw_index": self.accepted_draw_index,
            "seed_namespace_sha256": self.seed_namespace_sha256,
        }


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


def typed_seed_receipt(
    master_seed: str,
    namespace: OuterSeedNamespace,
) -> dict[str, Any]:
    """Return the exact canonical namespace and deterministic seed receipt."""

    namespace_payload = namespace.as_dict()
    namespace_bytes = _canonical_json_bytes(namespace_payload)
    seed_payload = [
        SEED_DERIVATION_ID,
        str(master_seed),
        namespace_payload,
    ]
    digest = hashlib.sha256(_canonical_json_bytes(seed_payload)).digest()
    return {
        "schema_version": SEED_NAMESPACE_SCHEMA_VERSION,
        "seed_derivation_id": SEED_DERIVATION_ID,
        "namespace": namespace_payload,
        "namespace_sha256": hashlib.sha256(namespace_bytes).hexdigest(),
        "seed_receipt_sha256": hashlib.sha256(
            _canonical_json_bytes(seed_payload)
        ).hexdigest(),
        "seed_uint64": int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        ),
    }


def typed_seed_token(
    master_seed: str,
    namespace: OuterSeedNamespace,
) -> str:
    """Return a non-secret deterministic token suitable for nested generators."""

    return str(typed_seed_receipt(master_seed, namespace)["seed_receipt_sha256"])


def c2_freeze_candidate_specification() -> dict[str, Any]:
    """Return the contract-v0.2 decision schema without opening locked data."""

    return {
        "schema_version": "gate12c2_freeze_candidate_v0.2",
        "epistemic_status": "development_freeze_candidate",
        "contract_version": C2_CONTRACT_VERSION,
        "alternative": PRIMARY_ALTERNATIVE,
        "directional_effect": "-median_log_observed_to_null_defect",
        "directional_p": "lower_tail_one_sided_sign_p",
        "multiplicity": "Holm_over_24_case_q_endpoints",
        "promotion_outcomes": sorted(CALIBRATION_PROMOTION_OUTCOMES),
        "partial_or_structured_is_promotional": False,
        "candidate_policy": N1_ESCALATION_POLICY,
        "S0_gates": {
            "family_wise_fpr": {
                "maximum_point_estimate": S0_MAX_POINT_FPR,
                "maximum_one_sided_95_upper": (
                    S0_MAX_ONE_SIDED_95_UPPER
                ),
            },
            "claim_promotion_false_rate": {
                "maximum_point_estimate": S0_MAX_POINT_FPR,
                "maximum_one_sided_95_upper": (
                    S0_MAX_ONE_SIDED_95_UPPER
                ),
            },
        },
        "S1_gate": {
            "minimum_primary_effect_power": S1_MIN_POINT_POWER,
            "minimum_one_sided_95_lower": S1_MIN_ONE_SIDED_95_LOWER,
        },
        "S2_gate": {
            "minimum_identification_rate": S2_MIN_POINT_IDENTIFICATION,
            "minimum_one_sided_95_lower": (
                S2_MIN_ONE_SIDED_95_LOWER
            ),
            "minimum_log_stressor_to_N1_null_defect": (
                S2_MIN_LOG_NULL_INFLATION
            ),
            "component_rule": (
                "x_increased_or_y_increased_or_c_decreased"
            ),
        },
        "inner_draw_selection": {
            "prefix_counts": [255, 511, 1023],
            "prefix_basis": "accepted_valid_draw_index",
            "minimum_endpoint_decision_agreement": 0.99,
            "maximum_absolute_endpoint_median_log_ratio_shift": 0.05,
            "maximum_absolute_S0_family_wise_fpr_shift": 0.01,
            "prohibited_selection_bases": [
                "best_observed_FPR",
                "best_observed_power",
                "most_favorable_direction",
            ],
        },
        "reference_block_hierarchy": {
            "source": "Gate12C-1 rendering-family block counts",
            "block_count_by_family": dict(
                sorted(REFERENCE_BLOCK_COUNT_BY_FAMILY.items())
            ),
            "use": (
                "development calibration reference; a later real-held-out "
                "contract must independently freeze its own schedule"
            ),
            "locked_schedule_frozen": False,
        },
        "seed_namespace_fields": [
            "contract_version",
            "surface_id",
            "null_candidate_id",
            "regime_id",
            "effect_strength",
            "outer_experiment_index",
            "case_or_endpoint_id",
            "cycle_or_root_id",
            "draw_attempt_index",
        ],
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }


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
class BatchedResidualDiagnostics:
    """FP64 residual diagnostics for a stack of conformable matrix triples.

    Optional scalar values use NaN internally and are converted back to
    ``None`` by :meth:`row`.  The object-level ``ResidualDiagnostics`` path
    remains the reference implementation; this container exists only to make
    equivalent development calibration computationally tractable.
    """

    q: int
    eligibility_status: tuple[str, ...]
    numerical_status: tuple[str, ...]
    defect: np.ndarray = field(repr=False)
    tail_left: np.ndarray = field(repr=False)
    tail_right: np.ndarray = field(repr=False)
    propagated_left: np.ndarray = field(repr=False)
    propagated_right: np.ndarray = field(repr=False)
    alignment: np.ndarray = field(repr=False)
    propagation_left: np.ndarray = field(repr=False)
    propagation_right: np.ndarray = field(repr=False)
    alignment_status: tuple[str, ...]
    propagation_left_status: tuple[str, ...]
    propagation_right_status: tuple[str, ...]
    matrix_identity_error: np.ndarray = field(repr=False)
    squared_identity_error: np.ndarray = field(repr=False)
    relative_gap_left: np.ndarray = field(repr=False)
    relative_gap_right: np.ndarray = field(repr=False)
    product_singular_values_left: np.ndarray = field(repr=False)
    product_singular_values_right: np.ndarray = field(repr=False)
    schema_version: str = DIAGNOSTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        numeric_vector_names = (
            "defect",
            "tail_left",
            "tail_right",
            "propagated_left",
            "propagated_right",
            "alignment",
            "propagation_left",
            "propagation_right",
            "matrix_identity_error",
            "squared_identity_error",
            "relative_gap_left",
            "relative_gap_right",
        )
        vectors: dict[str, np.ndarray] = {}
        for name in numeric_vector_names:
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 1:
                raise Gate12C2DevelopmentError(
                    f"batched diagnostic field {name} must be one-dimensional"
                )
            vectors[name] = array.copy()
            object.__setattr__(self, name, vectors[name])
        batch_size = len(vectors["defect"])
        if any(len(array) != batch_size for array in vectors.values()):
            raise Gate12C2DevelopmentError(
                "batched diagnostic vectors must have one shared batch size"
            )
        for name in (
            "eligibility_status",
            "numerical_status",
            "alignment_status",
            "propagation_left_status",
            "propagation_right_status",
        ):
            statuses = tuple(str(value) for value in getattr(self, name))
            if len(statuses) != batch_size:
                raise Gate12C2DevelopmentError(
                    f"batched diagnostic status field {name} has wrong length"
                )
            object.__setattr__(self, name, statuses)
        for name in (
            "product_singular_values_left",
            "product_singular_values_right",
        ):
            spectra = np.asarray(getattr(self, name), dtype=np.float64)
            if spectra.ndim != 2 or spectra.shape[0] != batch_size:
                raise Gate12C2DevelopmentError(
                    f"batched diagnostic spectrum field {name} has wrong shape"
                )
            object.__setattr__(self, name, spectra.copy())

    def __len__(self) -> int:
        return int(self.defect.shape[0])

    @staticmethod
    def _optional(value: float) -> float | None:
        numeric = float(value)
        return None if math.isnan(numeric) else numeric

    def row(self, index: int) -> ResidualDiagnostics:
        """Materialize one scalar row in the exact reference result type."""

        if index < 0 or index >= len(self):
            raise IndexError(index)
        return ResidualDiagnostics(
            q=int(self.q),
            eligibility_status=self.eligibility_status[index],
            numerical_status=self.numerical_status[index],
            defect=self._optional(self.defect[index]),
            tail_left=self._optional(self.tail_left[index]),
            tail_right=self._optional(self.tail_right[index]),
            propagated_left=self._optional(self.propagated_left[index]),
            propagated_right=self._optional(self.propagated_right[index]),
            alignment=self._optional(self.alignment[index]),
            propagation_left=self._optional(self.propagation_left[index]),
            propagation_right=self._optional(self.propagation_right[index]),
            alignment_status=self.alignment_status[index],
            propagation_left_status=self.propagation_left_status[index],
            propagation_right_status=self.propagation_right_status[index],
            matrix_identity_error=self._optional(
                self.matrix_identity_error[index]
            ),
            squared_identity_error=self._optional(
                self.squared_identity_error[index]
            ),
            relative_gap_left=self._optional(self.relative_gap_left[index]),
            relative_gap_right=self._optional(self.relative_gap_right[index]),
            product_singular_values_left=tuple(
                float(value)
                for value in self.product_singular_values_left[index]
            ),
            product_singular_values_right=tuple(
                float(value)
                for value in self.product_singular_values_right[index]
            ),
        )


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
    directional_raw_p: float
    alternative: str = PRIMARY_ALTERNATIVE


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


@dataclass(frozen=True)
class N1ArrayReassignment:
    """Array-native representation of one exact N1 reassignment draw."""

    replicate_ids: tuple[str, ...]
    donor_indices: np.ndarray = field(repr=False)
    reassigned_frames: np.ndarray = field(repr=False)
    m0: np.ndarray = field(repr=False)
    m1: np.ndarray = field(repr=False)
    m2: np.ndarray = field(repr=False)
    audit: Mapping[str, Any]
    kernel_id: str = "n1_array_reassignment_equivalent_v0.1"

    def __post_init__(self) -> None:
        donor_indices = np.asarray(self.donor_indices, dtype=np.int64)
        frames = np.asarray(self.reassigned_frames, dtype=np.float64)
        matrices = [
            np.asarray(getattr(self, name), dtype=np.float64)
            for name in ("m0", "m1", "m2")
        ]
        batch_size = len(self.replicate_ids)
        if donor_indices.shape != (batch_size, 3):
            raise Gate12C2DevelopmentError(
                "N1 array donor indices must have shape (batch, 3)"
            )
        if frames.ndim != 4 or frames.shape[:2] != (batch_size, 3):
            raise Gate12C2DevelopmentError(
                "N1 array frames must have shape (batch, 3, ambient, rank)"
            )
        if any(
            matrix.ndim != 3 or matrix.shape[0] != batch_size
            for matrix in matrices
        ):
            raise Gate12C2DevelopmentError(
                "N1 array cycle matrices must share the reassignment batch"
            )
        object.__setattr__(self, "donor_indices", donor_indices.copy())
        object.__setattr__(self, "reassigned_frames", frames.copy())
        for name, matrix in zip(("m0", "m1", "m2"), matrices, strict=True):
            object.__setattr__(self, name, matrix.copy())


def n1_role_constrained_array_reassignment(
    graphs: Sequence[SyntheticGraph],
    *,
    reassignment_seed: str,
) -> N1ArrayReassignment:
    """Reproduce N1 exactly without materializing node and graph objects.

    This kernel is intentionally limited to the three-node, one-cycle cohort
    used by the C-2 outer laboratory.  It uses the same canonical hash sort and
    cyclic donor assignment as :func:`n1_role_constrained_reassignment`.
    """

    if len(graphs) < 2:
        raise Gate12C2DevelopmentError("N1 requires at least two graphs")
    replicate_ids = tuple(graph.replicate_id for graph in graphs)
    if len(set(replicate_ids)) != len(replicate_ids):
        raise Gate12C2DevelopmentError("replicate IDs must be unique")

    cycle_nodes: list[tuple[NodeFrame, NodeFrame, NodeFrame]] = []
    for graph in graphs:
        if len(graph.nodes) != 3 or len(graph.cycle_node_ids) != 3:
            raise Gate12C2DevelopmentError(
                "N1 array kernel requires exactly three nodes in one cycle"
            )
        node_map = graph.node_map()
        ordered = tuple(node_map[node_id] for node_id in graph.cycle_node_ids)
        if {node.node_id for node in ordered} != {
            node.node_id for node in graph.nodes
        }:
            raise Gate12C2DevelopmentError(
                "N1 array kernel requires every node to lie on the cycle"
            )
        cycle_nodes.append(ordered)

    frame_shapes = {
        tuple(node.frame.shape)
        for ordered in cycle_nodes
        for node in ordered
    }
    if len(frame_shapes) != 1:
        raise Gate12C2DevelopmentError(
            "N1 array kernel requires one shared node-frame shape"
        )
    role_strata = tuple(cycle_nodes[0][role_index].stratum for role_index in range(3))
    if len(set(role_strata)) != 3:
        raise Gate12C2DevelopmentError(
            "N1 array kernel requires three distinct cycle-node strata"
        )
    for ordered in cycle_nodes:
        if tuple(node.stratum for node in ordered) != role_strata:
            raise Gate12C2DevelopmentError(
                "N1 array kernel requires one fixed stratum per cycle role"
            )

    batch_size = len(graphs)
    donor_indices = np.empty((batch_size, 3), dtype=np.int64)
    for role_index, stratum in enumerate(role_strata):
        target_indices = sorted(
            range(batch_size),
            key=lambda graph_index: hashlib.sha256(
                _canonical_json_bytes(
                    [
                        N1_ID,
                        reassignment_seed,
                        list(stratum),
                        replicate_ids[graph_index],
                        cycle_nodes[graph_index][role_index].node_id,
                    ]
                )
            ).hexdigest(),
        )
        for position, target_index in enumerate(target_indices):
            donor_indices[target_index, role_index] = target_indices[
                (position + 1) % batch_size
            ]

    source_frames = np.stack(
        [
            np.stack([node.frame for node in ordered], axis=0)
            for ordered in cycle_nodes
        ],
        axis=0,
    )
    role_indices = np.arange(3, dtype=np.int64)[None, :]
    reassigned_frames = source_frames[donor_indices, role_indices]
    m0 = np.asarray(
        np.swapaxes(reassigned_frames[:, 1], 1, 2)
        @ reassigned_frames[:, 0],
        dtype=np.float64,
    )
    m1 = np.asarray(
        np.swapaxes(reassigned_frames[:, 2], 1, 2)
        @ reassigned_frames[:, 1],
        dtype=np.float64,
    )
    m2 = np.asarray(
        np.swapaxes(reassigned_frames[:, 0], 1, 2)
        @ reassigned_frames[:, 2],
        dtype=np.float64,
    )

    fixed_point_count = int(
        np.sum(
            donor_indices
            == np.arange(batch_size, dtype=np.int64)[:, None]
        )
    )
    reused_donor_counts: dict[str, int] = {}
    unused_donor_count = 0
    for role_index in range(3):
        counts = np.bincount(
            donor_indices[:, role_index],
            minlength=batch_size,
        )
        unused_donor_count += int(np.sum(counts == 0))
        for donor_index in np.flatnonzero(counts > 1):
            reused_donor_counts[
                f"role-{role_index}/source-{int(donor_index)}"
            ] = int(counts[donor_index])
    audit_status = (
        "pass"
        if fixed_point_count == 0
        and unused_donor_count == 0
        and not reused_donor_counts
        else "fail"
    )
    audit = {
        "audit_id": "n1_array_derangement_audit_v0.1",
        "status": audit_status,
        "assignment_count": int(batch_size * 3),
        "stratum_count": 3,
        "stratum_size": int(batch_size),
        "fixed_point_count": fixed_point_count,
        "same_graph_assignment_count": fixed_point_count,
        "crossed_stratum_count": 0,
        "unused_donor_count": unused_donor_count,
        "reused_donor_counts": reused_donor_counts,
        "canonical_hash_sort_shared_with_object_generator": True,
        "edge_construction": "target_frame_transpose_times_source_frame",
    }
    return N1ArrayReassignment(
        replicate_ids=replicate_ids,
        donor_indices=donor_indices,
        reassigned_frames=reassigned_frames,
        m0=m0,
        m1=m1,
        m2=m2,
        audit=audit,
    )


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


def batched_residual_diagnostics(
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    q: int,
    relative_gap_min: float = DEFAULT_RELATIVE_GAP_MIN,
    numerical_atol: float = DEFAULT_NUMERIC_ATOL,
    degeneracy_atol: float = DEFAULT_DEGENERACY_ATOL,
) -> BatchedResidualDiagnostics:
    """Vectorized FP64 equivalent of :func:`residual_diagnostics`.

    The leading axis is the batch axis.  This development kernel deliberately
    preserves the scalar cut, degeneracy, and numerical-status rules.  It does
    not alter the estimand, null construction, or decision hierarchy.
    """

    arrays = [
        np.asarray(matrix, dtype=np.float64) for matrix in (m0, m1, m2)
    ]
    if any(array.ndim != 3 for array in arrays):
        raise Gate12C2DevelopmentError(
            "batched m0, m1, and m2 must have shape (batch, rows, columns)"
        )
    batch_sizes = {int(array.shape[0]) for array in arrays}
    if len(batch_sizes) != 1:
        raise Gate12C2DevelopmentError(
            "batched m0, m1, and m2 must share one batch size"
        )
    batch_size = next(iter(batch_sizes))
    if batch_size <= 0:
        raise Gate12C2DevelopmentError(
            "batched residual diagnostics require at least one matrix triple"
        )
    m0_array, m1_array, m2_array = arrays
    if m1_array.shape[2] != m0_array.shape[1]:
        raise Gate12C2DevelopmentError("batched m1 @ m0 is not conformable")
    if m2_array.shape[2] != m1_array.shape[1]:
        raise Gate12C2DevelopmentError("batched m2 @ m1 is not conformable")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise Gate12C2DevelopmentError(
            "batched development diagnostics require finite FP64 inputs"
        )

    product_left = np.asarray(m2_array @ m1_array, dtype=np.float64)
    product_right = np.asarray(m1_array @ m0_array, dtype=np.float64)
    left_u, left_s, left_vh = np.linalg.svd(
        product_left,
        full_matrices=False,
    )
    right_u, right_s, right_vh = np.linalg.svd(
        product_right,
        full_matrices=False,
    )
    if not np.all(np.isfinite(left_s)) or not np.all(np.isfinite(right_s)):
        raise Gate12C2DevelopmentError(
            "batched development SVD returned a nonfinite spectrum"
        )

    maximum_rank_left = min(product_left.shape[-2:])
    maximum_rank_right = min(product_right.shape[-2:])
    left_gap = np.full(batch_size, np.nan, dtype=np.float64)
    right_gap = np.full(batch_size, np.nan, dtype=np.float64)

    def cut_status(
        singular_values: np.ndarray,
        maximum_rank: int,
        gap_output: np.ndarray,
    ) -> tuple[str, ...]:
        if q <= 0 or q > maximum_rank:
            return tuple("invalid_q" for _ in range(batch_size))
        if q == maximum_rank:
            return tuple("full_rank_control" for _ in range(batch_size))
        scale = np.maximum(
            singular_values[:, 0],
            np.finfo(np.float64).tiny,
        )
        gap_output[:] = (
            singular_values[:, q - 1] - singular_values[:, q]
        ) / scale
        return tuple(
            "eligible" if value > relative_gap_min else "unstable_spectral_cut"
            for value in gap_output
        )

    left_status = cut_status(left_s, maximum_rank_left, left_gap)
    right_status = cut_status(right_s, maximum_rank_right, right_gap)
    left_reconstructable = np.asarray(
        [
            status in {"eligible", "full_rank_control"}
            for status in left_status
        ],
        dtype=bool,
    )
    right_reconstructable = np.asarray(
        [
            status in {"eligible", "full_rank_control"}
            for status in right_status
        ],
        dtype=bool,
    )
    active = left_reconstructable & right_reconstructable

    q_left = np.full_like(product_left, np.nan, dtype=np.float64)
    q_right = np.full_like(product_right, np.nan, dtype=np.float64)
    if q > 0 and np.any(left_reconstructable):
        mask = left_reconstructable
        q_left[mask] = (
            left_u[mask, :, :q] * left_s[mask, None, :q]
        ) @ left_vh[mask, :q, :]
    if q > 0 and np.any(right_reconstructable):
        mask = right_reconstructable
        q_right[mask] = (
            right_u[mask, :, :q] * right_s[mask, None, :q]
        ) @ right_vh[mask, :q, :]

    optional_names = (
        "defect",
        "tail_left",
        "tail_right",
        "propagated_left",
        "propagated_right",
        "alignment",
        "propagation_left",
        "propagation_right",
        "matrix_identity_error",
        "squared_identity_error",
    )
    values = {
        name: np.full(batch_size, np.nan, dtype=np.float64)
        for name in optional_names
    }
    eligibility_status: list[str] = []
    numerical_status = ["not_evaluated"] * batch_size
    alignment_status = ["not_evaluated"] * batch_size
    propagation_left_status = ["not_evaluated"] * batch_size
    propagation_right_status = ["not_evaluated"] * batch_size

    for index in range(batch_size):
        if not active[index]:
            eligibility_status.append(
                left_status[index]
                if left_status[index] != "eligible"
                else right_status[index]
            )
        elif (
            left_status[index] == right_status[index] == "full_rank_control"
        ):
            eligibility_status.append("full_rank_control")
        else:
            eligibility_status.append("eligible")

    if np.any(active):
        indices = np.flatnonzero(active)
        product_left_active = product_left[active]
        product_right_active = product_right[active]
        q_left_active = q_left[active]
        q_right_active = q_right[active]
        m0_active = m0_array[active]
        m2_active = m2_array[active]
        residual_left = np.asarray(
            product_left_active - q_left_active,
            dtype=np.float64,
        )
        residual_right = np.asarray(
            product_right_active - q_right_active,
            dtype=np.float64,
        )
        propagated_left_matrix = np.asarray(
            residual_left @ m0_active,
            dtype=np.float64,
        )
        propagated_right_matrix = np.asarray(
            m2_active @ residual_right,
            dtype=np.float64,
        )
        left_parenthesization = np.asarray(
            q_left_active @ m0_active,
            dtype=np.float64,
        )
        right_parenthesization = np.asarray(
            m2_active @ q_right_active,
            dtype=np.float64,
        )
        defect_matrix = np.asarray(
            left_parenthesization - right_parenthesization,
            dtype=np.float64,
        )
        decomposition_matrix = np.asarray(
            propagated_right_matrix - propagated_left_matrix,
            dtype=np.float64,
        )

        def frobenius_norm(stack: np.ndarray) -> np.ndarray:
            return np.sqrt(
                np.sum(stack * stack, axis=(-2, -1), dtype=np.float64)
            )

        defect = frobenius_norm(defect_matrix)
        tail_left = frobenius_norm(residual_left)
        tail_right = frobenius_norm(residual_right)
        propagated_left = frobenius_norm(propagated_left_matrix)
        propagated_right = frobenius_norm(propagated_right_matrix)
        inner_product = np.sum(
            propagated_left_matrix * propagated_right_matrix,
            axis=(-2, -1),
            dtype=np.float64,
        )
        matrix_identity_error = frobenius_norm(
            defect_matrix - decomposition_matrix
        )
        squared_rhs = (
            propagated_left * propagated_left
            + propagated_right * propagated_right
            - 2.0 * inner_product
        )
        squared_identity_error = np.abs(defect * defect - squared_rhs)
        for name, active_values in (
            ("defect", defect),
            ("tail_left", tail_left),
            ("tail_right", tail_right),
            ("propagated_left", propagated_left),
            ("propagated_right", propagated_right),
            ("matrix_identity_error", matrix_identity_error),
            ("squared_identity_error", squared_identity_error),
        ):
            values[name][active] = active_values

        alignment_defined = (
            (propagated_left > degeneracy_atol)
            & (propagated_right > degeneracy_atol)
        )
        alignment_values = np.full(len(indices), np.nan, dtype=np.float64)
        alignment_values[alignment_defined] = np.clip(
            inner_product[alignment_defined]
            / (
                propagated_left[alignment_defined]
                * propagated_right[alignment_defined]
            ),
            -1.0,
            1.0,
        )
        values["alignment"][active] = alignment_values

        propagation_left_defined = tail_left > degeneracy_atol
        propagation_right_defined = tail_right > degeneracy_atol
        propagation_left_values = np.full(
            len(indices),
            np.nan,
            dtype=np.float64,
        )
        propagation_right_values = np.full(
            len(indices),
            np.nan,
            dtype=np.float64,
        )
        propagation_left_values[propagation_left_defined] = (
            propagated_left[propagation_left_defined]
            / tail_left[propagation_left_defined]
        )
        propagation_right_values[propagation_right_defined] = (
            propagated_right[propagation_right_defined]
            / tail_right[propagation_right_defined]
        )
        values["propagation_left"][active] = propagation_left_values
        values["propagation_right"][active] = propagation_right_values

        finite = (
            np.isfinite(defect)
            & np.isfinite(tail_left)
            & np.isfinite(tail_right)
            & np.isfinite(propagated_left)
            & np.isfinite(propagated_right)
            & np.isfinite(matrix_identity_error)
            & np.isfinite(squared_identity_error)
        )
        numerical_pass = (
            finite
            & (matrix_identity_error <= numerical_atol)
            & (squared_identity_error <= numerical_atol)
        )
        for local_index, global_index in enumerate(indices):
            numerical_status[global_index] = (
                "pass" if numerical_pass[local_index] else "fail"
            )
            zero_sides: list[str] = []
            if propagated_left[local_index] <= degeneracy_atol:
                zero_sides.append("x")
            if propagated_right[local_index] <= degeneracy_atol:
                zero_sides.append("y")
            alignment_status[global_index] = (
                "defined"
                if not zero_sides
                else "undefined_degenerate_" + "_".join(zero_sides)
            )
            propagation_left_status[global_index] = (
                "defined"
                if propagation_left_defined[local_index]
                else "undefined_degenerate_u"
            )
            propagation_right_status[global_index] = (
                "defined"
                if propagation_right_defined[local_index]
                else "undefined_degenerate_v"
            )

    return BatchedResidualDiagnostics(
        q=q,
        eligibility_status=tuple(eligibility_status),
        numerical_status=tuple(numerical_status),
        defect=values["defect"],
        tail_left=values["tail_left"],
        tail_right=values["tail_right"],
        propagated_left=values["propagated_left"],
        propagated_right=values["propagated_right"],
        alignment=values["alignment"],
        propagation_left=values["propagation_left"],
        propagation_right=values["propagation_right"],
        alignment_status=tuple(alignment_status),
        propagation_left_status=tuple(propagation_left_status),
        propagation_right_status=tuple(propagation_right_status),
        matrix_identity_error=values["matrix_identity_error"],
        squared_identity_error=values["squared_identity_error"],
        relative_gap_left=left_gap,
        relative_gap_right=right_gap,
        product_singular_values_left=left_s,
        product_singular_values_right=right_s,
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


def batched_graph_residual_diagnostics(
    graphs: Sequence[SyntheticGraph],
    *,
    q: int,
    relative_gap_min: float = DEFAULT_RELATIVE_GAP_MIN,
    numerical_atol: float = DEFAULT_NUMERIC_ATOL,
    degeneracy_atol: float = DEFAULT_DEGENERACY_ATOL,
) -> BatchedResidualDiagnostics:
    """Evaluate one graph cohort through the vectorized FP64 kernel."""

    if not graphs:
        raise Gate12C2DevelopmentError(
            "batched graph diagnostics require at least one graph"
        )
    triples = [cycle_matrices(graph) for graph in graphs]
    shapes = {
        tuple(matrix.shape)
        for triple in triples
        for matrix in triple
    }
    if len(shapes) != 1:
        raise Gate12C2DevelopmentError(
            "batched graph diagnostics require one shared square edge shape"
        )
    m0 = np.stack([triple[0] for triple in triples], axis=0)
    m1 = np.stack([triple[1] for triple in triples], axis=0)
    m2 = np.stack([triple[2] for triple in triples], axis=0)
    return batched_residual_diagnostics(
        m0,
        m1,
        m2,
        q=q,
        relative_gap_min=relative_gap_min,
        numerical_atol=numerical_atol,
        degeneracy_atol=degeneracy_atol,
    )


def cohort_residual_diagnostics(
    graphs: Sequence[SyntheticGraph],
    *,
    q: int,
    diagnostic_kernel: str = OBJECT_REFERENCE_DIAGNOSTIC_KERNEL,
) -> tuple[ResidualDiagnostics, ...]:
    """Dispatch one cohort through a named, receipt-visible FP64 kernel."""

    if diagnostic_kernel not in ALLOWED_DIAGNOSTIC_KERNELS:
        raise Gate12C2DevelopmentError(
            f"unsupported diagnostic kernel: {diagnostic_kernel!r}"
        )
    if diagnostic_kernel == OBJECT_REFERENCE_DIAGNOSTIC_KERNEL:
        return tuple(
            graph_residual_diagnostics(graph, q=q) for graph in graphs
        )
    batch = batched_graph_residual_diagnostics(graphs, q=q)
    return tuple(batch.row(index) for index in range(len(batch)))


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


def _outer_case_grid() -> tuple[dict[str, Any], ...]:
    """Return the fixed 4-model by 3-family synthetic case surface."""

    return tuple(
        {
            "case_id": f"case-{case_order:02d}",
            "case_order": case_order,
            "model": f"model-{case_order % 4}",
            "family": f"family-{case_order // 4}",
        }
        for case_order in range(12)
    )


def reference_block_count_schedule() -> dict[str, int]:
    """Return the C-1-shaped 128/200/128 family schedule for development."""

    return {
        str(case["case_id"]): REFERENCE_BLOCK_COUNT_BY_FAMILY[
            str(case["family"])
        ]
        for case in _outer_case_grid()
    }


def _resolve_block_count_schedule(
    block_count: int | Mapping[str, int],
) -> dict[str, int]:
    cases = _outer_case_grid()
    case_ids = {str(case["case_id"]) for case in cases}
    if isinstance(block_count, Mapping):
        supplied = {str(key): int(value) for key, value in block_count.items()}
        if set(supplied) != case_ids:
            raise Gate12C2DevelopmentError(
                "case-specific block schedule must cover the fixed 12 cases exactly"
            )
        schedule = supplied
    else:
        schedule = {
            str(case["case_id"]): int(block_count) for case in cases
        }
    if any(value < 4 for value in schedule.values()):
        raise Gate12C2DevelopmentError(
            "outer experiments require at least four source blocks per case"
        )
    return schedule


def _block_count_receipt(
    schedule: Mapping[str, int],
) -> dict[str, Any]:
    values = set(int(value) for value in schedule.values())
    return {
        "mode": (
            "uniform" if len(values) == 1 else "case_specific"
        ),
        "uniform_block_count": (
            next(iter(values)) if len(values) == 1 else None
        ),
        "block_count_by_case": {
            key: int(schedule[key]) for key in sorted(schedule)
        },
    }


def _outer_observed_cohort(
    *,
    regime_id: str,
    master_seed: str,
    outer_experiment_index: int,
    case: Mapping[str, Any],
    block_count: int,
    effect_strength: float | None,
) -> tuple[tuple[SyntheticGraph, ...], dict[str, Any]]:
    namespace = OuterSeedNamespace(
        surface_id="development",
        null_candidate_id=N1_ID,
        regime_id=regime_id,
        effect_strength=effect_strength,
        outer_experiment_index=outer_experiment_index,
        case_or_endpoint_id=str(case["case_id"]),
        cycle_or_root_id="observed_block_cohort",
        draw_attempt_index=0,
    )
    receipt = typed_seed_receipt(master_seed, namespace)
    seed_token = str(receipt["seed_receipt_sha256"])
    family = (
        f"outer:{case['family']}:{case['case_id']}:{regime_id}"
    )
    if regime_id in {"S0_true_null", "S2_null_inflation"}:
        if effect_strength not in {None, 0.0}:
            raise Gate12C2DevelopmentError(
                "S0/S2 outer experiments require zero or absent effect strength"
            )
        cohort = generate_s0_cohort(
            replicate_count=block_count,
            master_seed=seed_token,
            family=family,
        )
    elif regime_id == "S1_known_reverse_shared_node_coupling":
        if effect_strength is None or effect_strength <= 0.0:
            raise Gate12C2DevelopmentError(
                "S1 outer experiments require a positive effect strength"
            )
        cohort = generate_s1_shared_node_coupling_cohort(
            replicate_count=block_count,
            master_seed=seed_token,
            effect_strength=float(effect_strength),
            family=family,
        )
    else:
        raise Gate12C2DevelopmentError(
            f"unsupported outer experiment regime: {regime_id!r}"
        )
    return cohort, receipt


def run_development_outer_experiment(
    *,
    regime_id: str,
    master_seed: str,
    outer_experiment_index: int,
    block_count: int | Mapping[str, int],
    inner_valid_draw_count: int,
    effect_strength: float | None = None,
    max_draw_attempts: int | None = None,
    epsilon: float = 1.0e-12,
    diagnostic_kernel: str = OBJECT_REFERENCE_DIAGNOSTIC_KERNEL,
) -> dict[str, Any]:
    """Run one graph-derived 12-case by 2-q development experiment.

    ``block_count`` may be one uniform integer or an exact 12-case mapping.
    q=1 and q=2 share the exact observed blocks and attempted N1 draws within
    each case. The resulting dependence is therefore carried into the complete
    endpoint-family decision rather than replaced by independent endpoint
    toys. Locked surface IDs are intentionally unavailable in this runner.
    """

    if outer_experiment_index < 0:
        raise Gate12C2DevelopmentError(
            "outer_experiment_index must be nonnegative"
        )
    block_schedule = _resolve_block_count_schedule(block_count)
    if inner_valid_draw_count <= 0:
        raise Gate12C2DevelopmentError(
            "inner_valid_draw_count must be positive"
        )
    attempt_limit = (
        max(inner_valid_draw_count * 4, inner_valid_draw_count + 8)
        if max_draw_attempts is None
        else int(max_draw_attempts)
    )
    if attempt_limit < inner_valid_draw_count:
        raise Gate12C2DevelopmentError(
            "max_draw_attempts cannot be below inner_valid_draw_count"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise Gate12C2DevelopmentError(
            "epsilon must be finite and positive"
        )
    if diagnostic_kernel not in ALLOWED_DIAGNOSTIC_KERNELS:
        raise Gate12C2DevelopmentError(
            f"unsupported diagnostic kernel: {diagnostic_kernel!r}"
        )

    endpoint_inputs: list[EndpointDecisionInput] = []
    endpoint_receipts: list[dict[str, Any]] = []
    case_receipts: list[dict[str, Any]] = []
    for case in _outer_case_grid():
        case_block_count = block_schedule[str(case["case_id"])]
        observed, observed_seed = _outer_observed_cohort(
            regime_id=regime_id,
            master_seed=master_seed,
            outer_experiment_index=outer_experiment_index,
            case=case,
            block_count=case_block_count,
            effect_strength=effect_strength,
        )
        observed_diagnostics = {
            (block_index, q): diagnostic
            for q in (1, 2)
            for block_index, diagnostic in enumerate(
                cohort_residual_diagnostics(
                    observed,
                    q=q,
                    diagnostic_kernel=diagnostic_kernel,
                )
            )
        }
        attempts_by_endpoint_block: dict[
            tuple[int, int], list[NullDrawAttempt]
        ] = {
            (q, block_index): []
            for q in (1, 2)
            for block_index in range(case_block_count)
        }
        accepted_counts: Counter[tuple[int, int]] = Counter()
        n1_audit_failure_count = 0
        for attempt_index in range(attempt_limit):
            if all(
                accepted_counts[(q, block_index)]
                >= inner_valid_draw_count
                for q in (1, 2)
                for block_index in range(case_block_count)
            ):
                break
            draw_namespace = OuterSeedNamespace(
                surface_id="development",
                null_candidate_id=N1_ID,
                regime_id=regime_id,
                effect_strength=effect_strength,
                outer_experiment_index=outer_experiment_index,
                case_or_endpoint_id=str(case["case_id"]),
                cycle_or_root_id="N1_reassigned_block_cohort",
                draw_attempt_index=attempt_index,
            )
            draw_seed = typed_seed_token(master_seed, draw_namespace)
            draw_seed_receipt = typed_seed_receipt(
                master_seed,
                draw_namespace,
            )
            comparison_batches: dict[
                int,
                BatchedResidualDiagnostics | tuple[ResidualDiagnostics, ...],
            ]
            if diagnostic_kernel == BATCHED_DIAGNOSTIC_KERNEL:
                array_comparison = n1_role_constrained_array_reassignment(
                    observed,
                    reassignment_seed=draw_seed,
                )
                audit_pass = array_comparison.audit["status"] == "pass"
                comparison_batches = {
                    q: batched_residual_diagnostics(
                        array_comparison.m0,
                        array_comparison.m1,
                        array_comparison.m2,
                        q=q,
                    )
                    for q in (1, 2)
                }
            else:
                comparison = n1_role_constrained_reassignment(
                    observed,
                    reassignment_seed=draw_seed,
                )
                audit = n1_reassignment_audit(observed, comparison)
                audit_pass = audit["status"] == "pass"
                comparison_batches = {
                    q: cohort_residual_diagnostics(
                        comparison,
                        q=q,
                        diagnostic_kernel=diagnostic_kernel,
                    )
                    for q in (1, 2)
                }
            n1_audit_failure_count += int(not audit_pass)
            for block_index in range(case_block_count):
                for q in (1, 2):
                    key = (q, block_index)
                    if accepted_counts[key] >= inner_valid_draw_count:
                        continue
                    observed_diagnostic = observed_diagnostics[
                        (block_index, q)
                    ]
                    batch_or_rows = comparison_batches[q]
                    if isinstance(
                        batch_or_rows,
                        BatchedResidualDiagnostics,
                    ):
                        batch_defect = float(
                            batch_or_rows.defect[block_index]
                        )
                        comparison_defect = (
                            None if math.isnan(batch_defect) else batch_defect
                        )
                        comparison_eligibility_status = (
                            batch_or_rows.eligibility_status[block_index]
                        )
                        comparison_numerical_status = (
                            batch_or_rows.numerical_status[block_index]
                        )
                    else:
                        comparison_diagnostic = batch_or_rows[block_index]
                        comparison_defect = comparison_diagnostic.defect
                        comparison_eligibility_status = (
                            comparison_diagnostic.eligibility_status
                        )
                        comparison_numerical_status = (
                            comparison_diagnostic.numerical_status
                        )
                    if not audit_pass:
                        accepted = False
                        value = None
                        reason = "n1_assignment_audit_failed"
                    elif observed_diagnostic.defect is None:
                        accepted = False
                        value = None
                        reason = (
                            "observed_" + observed_diagnostic.eligibility_status
                        )
                    elif comparison_defect is None:
                        accepted = False
                        value = None
                        reason = (
                            "null_" + comparison_eligibility_status
                        )
                    elif comparison_numerical_status != "pass":
                        accepted = False
                        value = None
                        reason = "null_numerical_failure"
                    else:
                        accepted = True
                        value = float(comparison_defect)
                        reason = None
                    accepted_index = (
                        accepted_counts[key] if accepted else None
                    )
                    attempts_by_endpoint_block[key].append(
                        NullDrawAttempt(
                            attempt_index=len(
                                attempts_by_endpoint_block[key]
                            ),
                            accepted=accepted,
                            value=value,
                            rejection_reason=reason,
                            accepted_draw_index=accepted_index,
                            seed_namespace_sha256=str(
                                draw_seed_receipt["namespace_sha256"]
                            ),
                        )
                    )
                    accepted_counts[key] += int(accepted)

        case_endpoint_receipts: list[dict[str, Any]] = []
        for q in (1, 2):
            block_scores: list[float] = []
            block_rows: list[dict[str, Any]] = []
            for block_index, graph in enumerate(observed):
                observed_diagnostic = observed_diagnostics[(block_index, q)]
                stream = accepted_valid_draw_stream(
                    attempts_by_endpoint_block[(q, block_index)],
                    required_valid_count=inner_valid_draw_count,
                )
                score = None
                null_median = None
                if (
                    observed_diagnostic.defect is not None
                    and stream["complete"]
                ):
                    null_median = float(
                        np.median(stream["accepted_values"])
                    )
                    score = float(
                        math.log(observed_diagnostic.defect + epsilon)
                        - math.log(null_median + epsilon)
                    )
                    block_scores.append(score)
                block_rows.append(
                    {
                        "source_block_id": graph.replicate_id,
                        "q": q,
                        "observed": observed_diagnostic.as_dict(),
                        "inner_stream_complete": bool(stream["complete"]),
                        "inner_attempt_count": int(
                            stream["attempt_count_supplied"]
                        ),
                        "inner_accepted_count": int(
                            stream["accepted_count_supplied"]
                        ),
                        "inner_rejection_reason_counts": (
                            stream["rejection_reason_counts"]
                        ),
                        "null_defect_median": null_median,
                        "block_log_observed_to_N1_defect": score,
                    }
                )
            sign_test = exact_directional_sign_p(
                block_scores,
                alternative=PRIMARY_ALTERNATIVE,
            )
            coverage_complete = len(block_scores) == case_block_count
            directional_raw_p = (
                float(sign_test["directional_raw_p"])
                if coverage_complete
                else 1.0
            )
            endpoint_inputs.append(
                EndpointDecisionInput(
                    case_id=str(case["case_id"]),
                    case_order=int(case["case_order"]),
                    model=str(case["model"]),
                    family=str(case["family"]),
                    q=q,
                    coverage_complete=coverage_complete,
                    informative=(
                        coverage_complete
                        and sign_test["test_status"] == "informative"
                    ),
                    median_log_ratio=(
                        float(np.median(block_scores))
                        if block_scores
                        else None
                    ),
                    directional_raw_p=directional_raw_p,
                    alternative=PRIMARY_ALTERNATIVE,
                )
            )
            case_endpoint_receipts.append(
                {
                    "endpoint_id": f"{case['case_id']}:q{q}",
                    "q": q,
                    "expected_block_count": case_block_count,
                    "represented_block_count": len(block_scores),
                    "coverage_complete": coverage_complete,
                    "sign_test": sign_test,
                    "block_rows": block_rows,
                }
            )
        endpoint_receipts.extend(case_endpoint_receipts)
        case_receipts.append(
            {
                **dict(case),
                "expected_block_count": case_block_count,
                "observed_seed_receipt": observed_seed,
                "observed_manifest_sha256": hashlib.sha256(
                    _canonical_json_bytes(manifests(observed))
                ).hexdigest(),
                "n1_audit_failure_count": n1_audit_failure_count,
                "endpoint_ids": [
                    row["endpoint_id"] for row in case_endpoint_receipts
                ],
            }
        )

    decision = complete_pipeline_decision(endpoint_inputs)
    return {
        "schema_version": OUTER_EXPERIMENT_SCHEMA_VERSION,
        "epistemic_status": "development_outer_experiment_only",
        "contract_version": C2_CONTRACT_VERSION,
        "surface_id": "development",
        "locked_execution_authorized": False,
        "regime_id": regime_id,
        "effect_strength": (
            None if effect_strength is None else float(effect_strength)
        ),
        "outer_experiment_index": int(outer_experiment_index),
        "block_count_schedule": _block_count_receipt(block_schedule),
        "inner_valid_draw_count": int(inner_valid_draw_count),
        "max_draw_attempts": int(attempt_limit),
        "diagnostic_kernel": diagnostic_kernel,
        "dependency_structure": (
            "q1_q2_share_observed_blocks_and_N1_draws_within_case"
        ),
        "alternative": PRIMARY_ALTERNATIVE,
        "case_receipts": case_receipts,
        "endpoint_receipts": endpoint_receipts,
        "pipeline_decision": decision,
    }


def run_development_outer_calibration(
    *,
    regime_id: str,
    master_seed: str,
    outer_experiment_count: int,
    block_count: int | Mapping[str, int],
    inner_valid_draw_count: int,
    effect_strength: float | None = None,
    max_draw_attempts: int | None = None,
    diagnostic_kernel: str = OBJECT_REFERENCE_DIAGNOSTIC_KERNEL,
) -> dict[str, Any]:
    """Repeat complete graph-derived outer experiments in development only."""

    if outer_experiment_count <= 0:
        raise Gate12C2DevelopmentError(
            "outer_experiment_count must be positive"
        )
    experiments = tuple(
        run_development_outer_experiment(
            regime_id=regime_id,
            master_seed=master_seed,
            outer_experiment_index=outer_index,
            block_count=block_count,
            inner_valid_draw_count=inner_valid_draw_count,
            effect_strength=effect_strength,
            max_draw_attempts=max_draw_attempts,
            diagnostic_kernel=diagnostic_kernel,
        )
        for outer_index in range(outer_experiment_count)
    )
    decisions = tuple(
        experiment["pipeline_decision"] for experiment in experiments
    )
    if regime_id == "S0_true_null":
        summary = summarize_outer_calibration(decisions)
    elif regime_id == "S1_known_reverse_shared_node_coupling":
        promotion_interval = _wilson_interval(
            sum(bool(decision["claim_promotion"]) for decision in decisions),
            len(decisions),
        )
        summary = {
            "schema_version": OUTER_CALIBRATION_SCHEMA_VERSION,
            "epistemic_status": "development_power_assessment_only",
            "regime_id": regime_id,
            "effect_strength": float(effect_strength),
            "outer_experiment_count": len(decisions),
            "claim_promotion_power": promotion_interval,
            "power_gate": {
                "minimum_point_estimate": S1_MIN_POINT_POWER,
                "minimum_one_sided_95_lower": (
                    S1_MIN_ONE_SIDED_95_LOWER
                ),
                "pass": bool(
                    promotion_interval["estimate"] >= S1_MIN_POINT_POWER
                    and promotion_interval["wilson_one_sided_95_lower"]
                    >= S1_MIN_ONE_SIDED_95_LOWER
                ),
                "locked_execution_authorized": False,
            },
            "grid_outcome_counts": dict(
                sorted(
                    Counter(
                        str(decision["grid_outcome"])
                        for decision in decisions
                    ).items()
                )
            ),
        }
    else:
        raise Gate12C2DevelopmentError(
            f"unsupported calibration regime: {regime_id!r}"
        )
    return {
        "schema_version": OUTER_CALIBRATION_SCHEMA_VERSION,
        "epistemic_status": "development_calibration_only",
        "regime_id": regime_id,
        "outer_experiment_count": len(experiments),
        "diagnostic_kernel": diagnostic_kernel,
        "summary": summary,
        "experiments": list(experiments),
    }


def _component_median(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    field_name: str,
) -> float | None:
    values = [
        float(row[arm][field_name])
        for row in rows
        if row[arm][field_name] is not None
    ]
    return float(np.median(values)) if values else None


def run_development_s2_identification_experiment(
    *,
    master_seed: str,
    outer_experiment_index: int,
    block_count: int | Mapping[str, int],
    inner_valid_draw_count: int,
    max_draw_attempts: int | None = None,
    minimum_log_null_inflation: float = S2_MIN_LOG_NULL_INFLATION,
    epsilon: float = 1.0e-12,
    diagnostic_kernel: str = OBJECT_REFERENCE_DIAGNOSTIC_KERNEL,
) -> dict[str, Any]:
    """Pair N1 and the graph-unconstrained stressor on identical observations."""

    if outer_experiment_index < 0:
        raise Gate12C2DevelopmentError(
            "outer_experiment_index must be nonnegative"
        )
    block_schedule = _resolve_block_count_schedule(block_count)
    if inner_valid_draw_count <= 0:
        raise Gate12C2DevelopmentError(
            "S2 requires at least four blocks and one valid inner draw"
        )
    if (
        not math.isfinite(minimum_log_null_inflation)
        or minimum_log_null_inflation < 0.0
    ):
        raise Gate12C2DevelopmentError(
            "minimum_log_null_inflation must be finite and nonnegative"
        )
    attempt_limit = (
        max(inner_valid_draw_count * 4, inner_valid_draw_count + 8)
        if max_draw_attempts is None
        else int(max_draw_attempts)
    )
    if attempt_limit < inner_valid_draw_count:
        raise Gate12C2DevelopmentError(
            "max_draw_attempts cannot be below inner_valid_draw_count"
        )
    if diagnostic_kernel not in ALLOWED_DIAGNOSTIC_KERNELS:
        raise Gate12C2DevelopmentError(
            f"unsupported diagnostic kernel: {diagnostic_kernel!r}"
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
    endpoint_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case in _outer_case_grid():
        case_block_count = block_schedule[str(case["case_id"])]
        observed, observed_seed = _outer_observed_cohort(
            regime_id="S2_null_inflation",
            master_seed=master_seed,
            outer_experiment_index=outer_experiment_index,
            case=case,
            block_count=case_block_count,
            effect_strength=None,
        )
        observed_manifest_sha256 = hashlib.sha256(
            _canonical_json_bytes(manifests(observed))
        ).hexdigest()
        observed_diagnostics = {
            q: cohort_residual_diagnostics(
                observed,
                q=q,
                diagnostic_kernel=diagnostic_kernel,
            )
            for q in (1, 2)
        }
        pair_rows: dict[tuple[int, int], list[dict[str, Any]]] = {
            (q, block_index): []
            for q in (1, 2)
            for block_index in range(case_block_count)
        }
        pair_attempts: dict[tuple[int, int], list[NullDrawAttempt]] = {
            key: [] for key in pair_rows
        }
        for attempt_index in range(attempt_limit):
            if all(
                len(pair_rows[key]) >= inner_valid_draw_count
                for key in pair_rows
            ):
                break
            n1_namespace = OuterSeedNamespace(
                surface_id="development",
                null_candidate_id=N1_ID,
                regime_id="S2_null_inflation",
                effect_strength=None,
                outer_experiment_index=outer_experiment_index,
                case_or_endpoint_id=str(case["case_id"]),
                cycle_or_root_id="N1_paired_null_cohort",
                draw_attempt_index=attempt_index,
            )
            stress_namespace = OuterSeedNamespace(
                surface_id="development",
                null_candidate_id=S2_UNCONSTRAINED_ORIENTATION_ID,
                regime_id="S2_null_inflation",
                effect_strength=None,
                outer_experiment_index=outer_experiment_index,
                case_or_endpoint_id=str(case["case_id"]),
                cycle_or_root_id="graph_unconstrained_stress_cohort",
                draw_attempt_index=attempt_index,
            )
            n1_receipt = typed_seed_receipt(master_seed, n1_namespace)
            stress_receipt = typed_seed_receipt(
                master_seed,
                stress_namespace,
            )
            pair_seed_sha256 = hashlib.sha256(
                _canonical_json_bytes(
                    [
                        n1_receipt["namespace_sha256"],
                        stress_receipt["namespace_sha256"],
                    ]
                )
            ).hexdigest()
            n1_graphs = n1_role_constrained_reassignment(
                observed,
                reassignment_seed=str(n1_receipt["seed_receipt_sha256"]),
            )
            stress_graphs = s2_graph_unconstrained_orientation_draw(
                observed,
                orientation_seed=str(
                    stress_receipt["seed_receipt_sha256"]
                ),
                draw_index=attempt_index,
            )
            n1_audit_pass = (
                n1_reassignment_audit(observed, n1_graphs)["status"] == "pass"
            )
            n1_diagnostics = {
                q: cohort_residual_diagnostics(
                    n1_graphs,
                    q=q,
                    diagnostic_kernel=diagnostic_kernel,
                )
                for q in (1, 2)
            }
            stress_diagnostics = {
                q: cohort_residual_diagnostics(
                    stress_graphs,
                    q=q,
                    diagnostic_kernel=diagnostic_kernel,
                )
                for q in (1, 2)
            }
            for block_index in range(case_block_count):
                n1_realizable = bool(
                    check_joint_realizability(n1_graphs[block_index])[
                        "status"
                    ]
                    == "pass"
                    and check_block_gram_realizability(
                        n1_graphs[block_index]
                    )["status"]
                    == "pass"
                )
                stress_nonrealizable = bool(
                    check_joint_realizability(stress_graphs[block_index])[
                        "status"
                    ]
                    == "fail"
                    and check_block_gram_realizability(
                        stress_graphs[block_index]
                    )["status"]
                    == "fail"
                )
                for q in (1, 2):
                    key = (q, block_index)
                    if len(pair_rows[key]) >= inner_valid_draw_count:
                        continue
                    observed_diagnostic = observed_diagnostics[q][block_index]
                    n1_diagnostic = n1_diagnostics[q][block_index]
                    stress_diagnostic = stress_diagnostics[q][block_index]
                    if not n1_audit_pass:
                        reason = "n1_assignment_audit_failed"
                    elif not n1_realizable:
                        reason = "n1_realizability_failed"
                    elif not stress_nonrealizable:
                        reason = "stressor_failed_to_break_realizability"
                    elif observed_diagnostic.defect is None:
                        reason = (
                            "observed_" + observed_diagnostic.eligibility_status
                        )
                    elif n1_diagnostic.defect is None:
                        reason = "n1_" + n1_diagnostic.eligibility_status
                    elif stress_diagnostic.defect is None:
                        reason = (
                            "stressor_" + stress_diagnostic.eligibility_status
                        )
                    else:
                        reason = None
                    accepted_index = (
                        len(pair_rows[key]) if reason is None else None
                    )
                    pair_attempts[key].append(
                        NullDrawAttempt(
                            attempt_index=len(pair_attempts[key]),
                            accepted=reason is None,
                            value=(
                                float(stress_diagnostic.defect)
                                if reason is None
                                and stress_diagnostic.defect is not None
                                else None
                            ),
                            rejection_reason=reason,
                            accepted_draw_index=accepted_index,
                            seed_namespace_sha256=pair_seed_sha256,
                        )
                    )
                    if reason is None:
                        pair_rows[key].append(
                            {
                                "attempt_index": attempt_index,
                                "observed": observed_diagnostic.as_dict(),
                                "N1": n1_diagnostic.as_dict(),
                                "graph_unconstrained_stressor": (
                                    stress_diagnostic.as_dict()
                                ),
                                "N1_realizable": n1_realizable,
                                "stressor_nonrealizable": stress_nonrealizable,
                                "observed_manifest_sha256": (
                                    observed_manifest_sha256
                                ),
                            }
                        )

        case_endpoint_rows: list[dict[str, Any]] = []
        for q in (1, 2):
            completed_blocks = 0
            per_block_component_medians: list[dict[str, Any]] = []
            rejection_counts: Counter[str] = Counter()
            for block_index in range(case_block_count):
                key = (q, block_index)
                stream = accepted_valid_draw_stream(
                    pair_attempts[key],
                    required_valid_count=inner_valid_draw_count,
                )
                rejection_counts.update(stream["rejection_reason_counts"])
                rows = pair_rows[key][:inner_valid_draw_count]
                if not stream["complete"] or len(rows) != inner_valid_draw_count:
                    continue
                completed_blocks += 1
                per_block_component_medians.append(
                    {
                        "source_block_id": observed[block_index].replicate_id,
                        "observed": {
                            field_name: _component_median(
                                rows,
                                "observed",
                                field_name,
                            )
                            for field_name in component_fields
                        },
                        "N1": {
                            field_name: _component_median(
                                rows,
                                "N1",
                                field_name,
                            )
                            for field_name in component_fields
                        },
                        "graph_unconstrained_stressor": {
                            field_name: _component_median(
                                rows,
                                "graph_unconstrained_stressor",
                                field_name,
                            )
                            for field_name in component_fields
                        },
                        "attempt_count": stream["attempt_count_supplied"],
                        "accepted_count": stream["accepted_count_supplied"],
                    }
                )

            def across_blocks(arm: str, field_name: str) -> float | None:
                values = [
                    float(row[arm][field_name])
                    for row in per_block_component_medians
                    if row[arm][field_name] is not None
                ]
                return float(np.median(values)) if values else None

            component_medians = {
                arm: {
                    field_name: across_blocks(arm, field_name)
                    for field_name in component_fields
                }
                for arm in (
                    "observed",
                    "N1",
                    "graph_unconstrained_stressor",
                )
            }
            n1_defect = component_medians["N1"]["a_q"]
            stress_defect = component_medians[
                "graph_unconstrained_stressor"
            ]["a_q"]
            log_null_inflation = (
                None
                if n1_defect is None or stress_defect is None
                else float(
                    math.log(stress_defect + epsilon)
                    - math.log(n1_defect + epsilon)
                )
            )
            n1_x = component_medians["N1"]["x_q"]
            n1_y = component_medians["N1"]["y_q"]
            n1_c = component_medians["N1"]["c_q"]
            stress_x = component_medians[
                "graph_unconstrained_stressor"
            ]["x_q"]
            stress_y = component_medians[
                "graph_unconstrained_stressor"
            ]["y_q"]
            stress_c = component_medians[
                "graph_unconstrained_stressor"
            ]["c_q"]
            channels = {
                "x_increased": bool(
                    n1_x is not None
                    and stress_x is not None
                    and stress_x > n1_x + DEFAULT_PRIMARY_ZERO_TOLERANCE
                ),
                "y_increased": bool(
                    n1_y is not None
                    and stress_y is not None
                    and stress_y > n1_y + DEFAULT_PRIMARY_ZERO_TOLERANCE
                ),
                "c_decreased": bool(
                    n1_c is not None
                    and stress_c is not None
                    and stress_c < n1_c - DEFAULT_PRIMARY_ZERO_TOLERANCE
                ),
            }
            coverage_complete = completed_blocks == case_block_count
            endpoint_identified = bool(
                coverage_complete
                and log_null_inflation is not None
                and log_null_inflation > minimum_log_null_inflation
                and any(channels.values())
            )
            row = {
                "endpoint_id": f"{case['case_id']}:q{q}",
                **dict(case),
                "q": q,
                "coverage_complete": coverage_complete,
                "completed_block_count": completed_blocks,
                "expected_block_count": case_block_count,
                "observed_process_modified": False,
                "observed_manifest_sha256": observed_manifest_sha256,
                "log_stressor_to_N1_null_defect": log_null_inflation,
                "minimum_log_null_inflation": (
                    float(minimum_log_null_inflation)
                ),
                "inflation_consistent_channels": channels,
                "endpoint_identified": endpoint_identified,
                "component_medians": component_medians,
                "rejection_reason_counts": dict(sorted(rejection_counts.items())),
                "block_rows": per_block_component_medians,
            }
            endpoint_rows.append(row)
            case_endpoint_rows.append(row)
        case_identified = bool(
            len(case_endpoint_rows) == 2
            and all(row["endpoint_identified"] for row in case_endpoint_rows)
        )
        case_rows.append(
            {
                **dict(case),
                "expected_block_count": case_block_count,
                "q1_identified": bool(
                    case_endpoint_rows[0]["endpoint_identified"]
                ),
                "q2_identified": bool(
                    case_endpoint_rows[1]["endpoint_identified"]
                ),
                "case_identified": case_identified,
                "observed_seed_receipt": observed_seed,
            }
        )

    identified_cases = [row for row in case_rows if row["case_identified"]]
    family_counts = Counter(row["family"] for row in identified_cases)
    model_counts = Counter(row["model"] for row in identified_cases)
    all_families = {row["family"] for row in case_rows}
    all_models = {row["model"] for row in case_rows}
    breadth_pass = bool(
        all(family_counts[family] >= 3 for family in all_families)
        and all(model_counts[model] >= 2 for model in all_models)
    )
    identified_case_count = len(identified_cases)
    identification_success = bool(
        identified_case_count == 12
        or (identified_case_count >= 10 and breadth_pass)
    )
    return {
        "schema_version": OUTER_EXPERIMENT_SCHEMA_VERSION,
        "epistemic_status": "development_s2_identification_only",
        "contract_version": C2_CONTRACT_VERSION,
        "surface_id": "development",
        "locked_execution_authorized": False,
        "regime_id": "S2_null_inflation",
        "outer_experiment_index": int(outer_experiment_index),
        "block_count_schedule": _block_count_receipt(block_schedule),
        "inner_valid_draw_count": int(inner_valid_draw_count),
        "diagnostic_kernel": diagnostic_kernel,
        "observed_process_modified": False,
        "paired_null_arms": [
            N1_ID,
            S2_UNCONSTRAINED_ORIENTATION_ID,
        ],
        "identified_case_count": identified_case_count,
        "breadth_pass": breadth_pass,
        "identification_success": identification_success,
        "endpoint_rows": endpoint_rows,
        "case_rows": case_rows,
    }


def run_development_s2_identification_calibration(
    *,
    master_seed: str,
    outer_experiment_count: int,
    block_count: int | Mapping[str, int],
    inner_valid_draw_count: int,
    max_draw_attempts: int | None = None,
    minimum_log_null_inflation: float = S2_MIN_LOG_NULL_INFLATION,
    diagnostic_kernel: str = OBJECT_REFERENCE_DIAGNOSTIC_KERNEL,
) -> dict[str, Any]:
    """Estimate the development S2 attribution rate over full outer units."""

    if outer_experiment_count <= 0:
        raise Gate12C2DevelopmentError(
            "outer_experiment_count must be positive"
        )
    experiments = tuple(
        run_development_s2_identification_experiment(
            master_seed=master_seed,
            outer_experiment_index=outer_index,
            block_count=block_count,
            inner_valid_draw_count=inner_valid_draw_count,
            max_draw_attempts=max_draw_attempts,
            minimum_log_null_inflation=minimum_log_null_inflation,
            diagnostic_kernel=diagnostic_kernel,
        )
        for outer_index in range(outer_experiment_count)
    )
    interval = _wilson_interval(
        sum(
            bool(experiment["identification_success"])
            for experiment in experiments
        ),
        len(experiments),
    )
    return {
        "schema_version": OUTER_CALIBRATION_SCHEMA_VERSION,
        "epistemic_status": "development_s2_calibration_only",
        "regime_id": "S2_null_inflation",
        "outer_experiment_count": len(experiments),
        "diagnostic_kernel": diagnostic_kernel,
        "identification_rate": interval,
        "identification_gate": {
            "minimum_point_estimate": S2_MIN_POINT_IDENTIFICATION,
            "minimum_one_sided_95_lower": (
                S2_MIN_ONE_SIDED_95_LOWER
            ),
            "minimum_log_null_inflation": float(
                minimum_log_null_inflation
            ),
            "required_component_channels": [
                "x_increased",
                "y_increased",
                "c_decreased",
            ],
            "channel_rule": "at_least_one",
            "pass": bool(
                interval["estimate"] >= S2_MIN_POINT_IDENTIFICATION
                and interval["wilson_one_sided_95_lower"]
                >= S2_MIN_ONE_SIDED_95_LOWER
            ),
            "locked_execution_authorized": False,
        },
        "experiments": list(experiments),
    }


def _directional_effect(
    median_log_ratio: float | None,
    *,
    alternative: str,
) -> float | None:
    if alternative not in ALLOWED_ALTERNATIVES:
        raise Gate12C2DevelopmentError(
            f"unsupported directional alternative: {alternative!r}"
        )
    if median_log_ratio is None:
        return None
    numeric = float(median_log_ratio)
    if not math.isfinite(numeric):
        raise Gate12C2DevelopmentError(
            "median log ratio must be finite when present"
        )
    return (
        -numeric
        if alternative == "observed_smaller_than_null"
        else numeric
    )


def exact_directional_sign_p(
    block_scores: Sequence[float],
    *,
    alternative: str = PRIMARY_ALTERNATIVE,
    zero_tolerance: float = DEFAULT_PRIMARY_ZERO_TOLERANCE,
) -> dict[str, Any]:
    """Exact one-sided sign value in the explicitly declared direction."""

    if alternative not in ALLOWED_ALTERNATIVES:
        raise Gate12C2DevelopmentError(
            f"unsupported directional alternative: {alternative!r}"
        )
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise Gate12C2DevelopmentError(
            "zero_tolerance must be finite and nonnegative"
        )
    signs = [
        _sign_with_tolerance(float(value), zero_tolerance=zero_tolerance)
        for value in block_scores
    ]
    if any(sign is None for sign in signs):
        raise Gate12C2DevelopmentError(
            "block scores must be finite for the sign test"
        )
    positive_count = sum(sign == 1 for sign in signs)
    negative_count = sum(sign == -1 for sign in signs)
    tie_count = sum(sign == 0 for sign in signs)
    directional_count = (
        negative_count
        if alternative == "observed_smaller_than_null"
        else positive_count
    )
    informative_count = positive_count + negative_count
    if informative_count == 0:
        raw_p = 1.0
        status = "non_informative"
    else:
        numerator = sum(
            math.comb(informative_count, count)
            for count in range(directional_count, informative_count + 1)
        )
        raw_p = float(numerator / (2**informative_count))
        status = "informative"
    return {
        "alternative": alternative,
        "test_status": status,
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "tie_count": int(tie_count),
        "directional_count": int(directional_count),
        "informative_count": int(informative_count),
        "directional_raw_p": raw_p,
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
            float(endpoint.directional_raw_p),
            int(endpoint.case_order),
            int(endpoint.q),
        ),
    )
    adjusted: dict[tuple[str, int], tuple[float, int]] = {}
    running = 0.0
    for position, endpoint in enumerate(ordered, start=1):
        raw_p = float(endpoint.directional_raw_p)
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
    """Run one complete reverse-direction 24-endpoint decision hierarchy.

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
    alternatives = {endpoint.alternative for endpoint in endpoints}
    if alternatives != {PRIMARY_ALTERNATIVE}:
        raise Gate12C2DevelopmentError(
            "Gate12C-2 calibration requires the explicit "
            f"{PRIMARY_ALTERNATIVE!r} alternative"
        )
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
        directional_effect = _directional_effect(
            median,
            alternative=endpoint.alternative,
        )
        q_directional_support = bool(
            endpoint.coverage_complete
            and endpoint.informative
            and directional_effect is not None
            and directional_effect > zero_tolerance
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
            "alternative": endpoint.alternative,
            "median_log_ratio": (
                None if median is None else float(median)
            ),
            "directional_effect": directional_effect,
            "directional_raw_p": float(endpoint.directional_raw_p),
            "holm_adjusted_directional_p": adjusted_p,
            "holm_sort_position": sort_position,
            "q_directional_support": q_directional_support,
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
        run_support = bool(
            q1["q_directional_support"]
            and q2["q_directional_support"]
        )
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
                bool(q1["q_directional_support"])
                != bool(q2["q_directional_support"])
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

    claim_promotion = grid_outcome in CALIBRATION_PROMOTION_OUTCOMES
    return {
        "schema_version": PIPELINE_DECISION_SCHEMA_VERSION,
        "epistemic_status": "development_calibration_unit",
        "outer_monte_carlo_unit": "complete_24_endpoint_outer_experiment",
        "alternative": PRIMARY_ALTERNATIVE,
        "holm_alpha": float(holm_alpha),
        "zero_tolerance": float(zero_tolerance),
        "endpoint_count": len(endpoint_rows),
        "q_directional_support_count": sum(
            row["q_directional_support"] for row in endpoint_rows
        ),
        "any_endpoint_support": any(
            row["q_directional_support"] for row in endpoint_rows
        ),
        "supporting_run_count": support_count,
        "any_run_support": support_count > 0,
        "q_discordant_run_count": q_discordant_count,
        "grid_outcome": grid_outcome,
        "claim_promotion": claim_promotion,
        "promotion_outcomes": sorted(CALIBRATION_PROMOTION_OUTCOMES),
        "partial_or_structured_is_promotional": False,
        "endpoint_rows": endpoint_rows,
        "run_rows": run_rows,
    }


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = ONE_SIDED_95_Z,
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
        "confidence_level": 0.95,
        "interval_sidedness": "one_sided_bounds",
        "wilson_one_sided_95_lower": float(
            max(0.0, center - half_width)
        ),
        "wilson_one_sided_95_upper": float(
            min(1.0, center + half_width)
        ),
    }


def _assess_s0_calibration_gates(
    *,
    family_wise: Mapping[str, Any],
    claim_promotion: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply contract-v0.2 S0 gates to development estimates."""

    def passes(interval: Mapping[str, Any]) -> bool:
        return bool(
            float(interval["estimate"]) <= S0_MAX_POINT_FPR
            and float(interval["wilson_one_sided_95_upper"])
            <= S0_MAX_ONE_SIDED_95_UPPER
        )

    family_wise_pass = passes(family_wise)
    promotion_pass = passes(claim_promotion)
    return {
        "status": "development_assessment_under_contract_v0.2",
        "contract_version": C2_CONTRACT_VERSION,
        "family_wise_safety": {
            "maximum_point_estimate": S0_MAX_POINT_FPR,
            "maximum_one_sided_95_upper": (
                S0_MAX_ONE_SIDED_95_UPPER
            ),
            "pass": family_wise_pass,
        },
        "claim_promotion_safety": {
            "promotion_outcomes": sorted(CALIBRATION_PROMOTION_OUTCOMES),
            "partial_or_structured_included": False,
            "maximum_point_estimate": S0_MAX_POINT_FPR,
            "maximum_one_sided_95_upper": (
                S0_MAX_ONE_SIDED_95_UPPER
            ),
            "pass": promotion_pass,
        },
        "overall_pass": bool(family_wise_pass and promotion_pass),
        "locked_execution_authorized": False,
    }


def summarize_outer_calibration(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize repeated complete-pipeline outer experiments.

    Endpoint FPR, family-wise FPR, run-level FPR, and claim-promotion rate
    remain separate estimands. Contract-v0.2 safety gates are applied without
    upgrading the development-only epistemic status.
    """

    if not decisions:
        raise Gate12C2DevelopmentError(
            "outer calibration requires at least one complete experiment"
        )
    expected_endpoint_ids: tuple[str, ...] | None = None
    endpoint_successes: Counter[str] = Counter()
    any_endpoint_count = 0
    any_run_count = 0
    claim_promotion_count = 0
    outcome_counts: Counter[str] = Counter()
    for index, decision in enumerate(decisions):
        if (
            decision.get("schema_version")
            != PIPELINE_DECISION_SCHEMA_VERSION
            or decision.get("outer_monte_carlo_unit")
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
                bool(row["q_directional_support"])
            )
        any_endpoint_count += int(bool(decision["any_endpoint_support"]))
        any_run_count += int(bool(decision["any_run_support"]))
        claim_promotion_count += int(bool(decision["claim_promotion"]))
        outcome_counts[str(decision["grid_outcome"])] += 1

    total = len(decisions)
    assert expected_endpoint_ids is not None
    return {
        "schema_version": OUTER_CALIBRATION_SCHEMA_VERSION,
        "epistemic_status": (
            "development_only_contract_v0.2_gates_applied"
        ),
        "outer_monte_carlo_unit": "complete_24_endpoint_outer_experiment",
        "alternative": PRIMARY_ALTERNATIVE,
        "outer_experiment_count": total,
        "type_i_estimands": {
            "endpoint_fpr": (
                "P(reverse-direction support for a named case/q endpoint "
                "under S0)"
            ),
            "family_wise_fpr": (
                "P(at least one of 24 endpoints supports the reverse "
                "direction under S0)"
            ),
            "run_level_fpr": (
                "P(at least one case has both q endpoints supported under S0)"
            ),
            "claim_promotion_false_rate": (
                "P(final grid outcome is strong_broad or broad_replicated "
                "under S0)"
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
        "claim_promotion_false_rate": _wilson_interval(
            claim_promotion_count,
            total,
        ),
        "grid_outcome_counts": dict(sorted(outcome_counts.items())),
        "calibration_gate_assessment": _assess_s0_calibration_gates(
            family_wise=_wilson_interval(any_endpoint_count, total),
            claim_promotion=_wilson_interval(
                claim_promotion_count,
                total,
            ),
        ),
    }


def accepted_valid_draw_stream(
    attempts: Sequence[NullDrawAttempt],
    *,
    required_valid_count: int,
) -> dict[str, Any]:
    """Extract an auditable valid-draw prefix from ordered attempts."""

    if required_valid_count <= 0:
        raise Gate12C2DevelopmentError(
            "required_valid_count must be positive"
        )
    ordered = tuple(attempts)
    expected_attempt_indices = tuple(range(len(ordered)))
    actual_attempt_indices = tuple(attempt.attempt_index for attempt in ordered)
    if actual_attempt_indices != expected_attempt_indices:
        raise Gate12C2DevelopmentError(
            "draw attempts must be contiguous and ordered from zero"
        )
    accepted = [attempt for attempt in ordered if attempt.accepted]
    expected_accepted_indices = list(range(len(accepted)))
    actual_accepted_indices = [
        int(attempt.accepted_draw_index)
        for attempt in accepted
        if attempt.accepted_draw_index is not None
    ]
    if actual_accepted_indices != expected_accepted_indices:
        raise Gate12C2DevelopmentError(
            "accepted draw indices must be contiguous in attempt order"
        )
    prefix = accepted[:required_valid_count]
    complete = len(prefix) == required_valid_count
    final_attempt_index = (
        prefix[-1].attempt_index if complete and prefix else None
    )
    return {
        "schema_version": ACCEPTED_DRAW_STREAM_SCHEMA_VERSION,
        "required_valid_count": int(required_valid_count),
        "attempt_count_supplied": len(ordered),
        "accepted_count_supplied": len(accepted),
        "complete": complete,
        "final_attempt_index": final_attempt_index,
        "accepted_values": [
            float(attempt.value)
            for attempt in prefix
            if attempt.value is not None
        ],
        "accepted_seed_namespace_sha256": [
            attempt.seed_namespace_sha256 for attempt in prefix
        ],
        "rejection_reason_counts": dict(
            sorted(
                Counter(
                    str(attempt.rejection_reason)
                    for attempt in ordered
                    if not attempt.accepted
                ).items()
            )
        ),
        "attempts": [attempt.as_dict() for attempt in ordered],
    }


def nested_inner_draw_stability_from_attempts(
    attempts: Sequence[NullDrawAttempt],
    *,
    observed_value: float,
    prefix_counts: Sequence[int] = (255, 511, 1023),
    decision_alpha: float = 0.05,
    runtime_seconds_by_prefix: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Evaluate nested prefixes after rejection, never by attempt prefix."""

    counts = tuple(int(count) for count in prefix_counts)
    if not counts:
        raise Gate12C2DevelopmentError("prefix_counts must be nonempty")
    stream = accepted_valid_draw_stream(
        attempts,
        required_valid_count=max(counts),
    )
    if not stream["complete"]:
        raise Gate12C2DevelopmentError(
            "insufficient accepted valid draws for the largest prefix"
        )
    report = nested_inner_draw_stability(
        stream["accepted_values"],
        observed_value=observed_value,
        prefix_counts=counts,
        decision_alpha=decision_alpha,
        runtime_seconds_by_prefix=runtime_seconds_by_prefix,
    )
    report["draw_stream_basis"] = "accepted_valid_draw_index"
    report["attempt_count_to_largest_prefix"] = (
        int(stream["final_attempt_index"]) + 1
    )
    report["rejection_reason_counts"] = stream["rejection_reason_counts"]
    report["accepted_seed_namespace_sha256"] = (
        stream["accepted_seed_namespace_sha256"]
    )
    return report


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
