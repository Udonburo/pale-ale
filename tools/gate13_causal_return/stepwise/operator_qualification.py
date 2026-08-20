"""Fresh split-half operator qualification for the stepwise substrate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from tools.gate13_causal_return.track_b.operator_core import (
    QUALIFIED,
    build_operator_packet,
    json_ready,
)
from tools.gate13_causal_return.track_b.split_half_floor import (
    align_frame,
    packet_disagreement,
)


LAYER_SET = (12, 24, 35)
TOKEN_POSITION = "LAST_PROMPT_TOKEN_BEFORE_FORCED_CHOICE"
ACTIVATION_REPRESENTATION = "BF16_RESIDUAL_STREAM_CAST_FLOAT32_FOR_STORAGE_FLOAT64_FOR_ESTIMATION"
FRAME_RANK = 4
FRAME_RELATIVE_SINGULAR_TOLERANCE = 1.0e-6
EDGE_RIDGE_RELATIVE = 1.0e-3
EDGE_RANK_TOLERANCE = 1.0e-8
EDGE_CONDITION_CEILING = 1.0e6
MINIMUM_NODE_SUPPORT_PER_HALF = 24
MINIMUM_QUALIFIED_LAYER_COUNT = 2
SPLIT_HALF_SINGULAR_FLOOR_MAX = 0.20
BROKEN_SENSITIVITY_ABSOLUTE_MARGIN = 0.05
BROKEN_SENSITIVITY_MULTIPLIER = 2.0

EXACT_PATH_P = ("phase0_state0", "phase0_state1", "phase1_state1")
EXACT_PATH_Q = ("phase0_state0", "phase1_state0", "phase1_state1")
BROKEN_PATH_Q = ("phase0_state0", "phase1_state0", "phase1_state1_broken")


class OperatorQualificationError(ValueError):
    """Fail-closed operator qualification error."""


def _matrix(value: Any, *, context: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise OperatorQualificationError(f"{context} must be a finite rank-2 matrix")
    return matrix


def _canonicalize_frame_sign(frame: np.ndarray) -> np.ndarray:
    result = np.asarray(frame, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            result[:, column] *= -1.0
    return result


def estimate_frame(activations: Any, *, rank: int = FRAME_RANK) -> dict[str, Any]:
    values = _matrix(activations, context="activations")
    if values.shape[0] < MINIMUM_NODE_SUPPORT_PER_HALF:
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "INSUFFICIENT_NODE_SUPPORT",
            "sample_count": int(values.shape[0]),
        }
    if rank <= 0 or rank > min(values.shape):
        raise OperatorQualificationError("invalid frame rank")
    mean = values.mean(axis=0)
    centered = values - mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    threshold = max(
        np.finfo(np.float64).eps,
        float(singular_values[0]) * FRAME_RELATIVE_SINGULAR_TOLERANCE,
    )
    numerical_rank = int(np.sum(singular_values > threshold))
    if numerical_rank < rank:
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "RANK_DEFICIENT_FRAME",
            "sample_count": int(values.shape[0]),
            "numerical_rank": numerical_rank,
            "singular_values": singular_values,
            "threshold": threshold,
        }
    frame = _canonicalize_frame_sign(vt[:rank].T)
    coordinates = centered @ frame
    return {
        "status": QUALIFIED,
        "sample_count": int(values.shape[0]),
        "mean": mean,
        "frame": frame,
        "coordinates": coordinates,
        "singular_values": singular_values,
        "numerical_rank": numerical_rank,
        "threshold": threshold,
    }


def align_half_frames(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    candidate_activations: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aligned: dict[str, dict[str, Any]] = {}
    if set(reference) != set(candidate) or set(candidate) != set(candidate_activations):
        raise OperatorQualificationError("node sets differ across halves")
    for node in sorted(reference):
        first = reference[node]
        second = candidate[node]
        if first["status"] != QUALIFIED or second["status"] != QUALIFIED:
            aligned[node] = {
                "status": "UNQUALIFIED",
                "rejection_reason": "FRAME_NOT_QUALIFIED",
            }
            continue
        alignment = align_frame(first["frame"], second["frame"])
        values = _matrix(candidate_activations[node], context=f"candidate activations {node}")
        frame = np.asarray(alignment["aligned"], dtype=np.float64)
        mean = values.mean(axis=0)
        aligned[node] = {
            **dict(second),
            "frame": frame,
            "mean": mean,
            "coordinates": (values - mean) @ frame,
            "gauge_alignment": alignment,
        }
    return aligned


def fit_edge_map(
    source_coordinates: Any,
    target_coordinates: Any,
    *,
    ridge_relative: float = EDGE_RIDGE_RELATIVE,
) -> dict[str, Any]:
    source = _matrix(source_coordinates, context="source coordinates")
    target = _matrix(target_coordinates, context="target coordinates")
    if source.shape != target.shape:
        raise OperatorQualificationError("paired edge coordinates must have equal shapes")
    rank = source.shape[1]
    gram = source.T @ source
    scale = max(float(np.trace(gram)) / max(rank, 1), np.finfo(np.float64).eps)
    ridge = float(ridge_relative) * scale
    edge = target.T @ source @ np.linalg.inv(gram + ridge * np.eye(rank))
    singular_values = np.linalg.svd(edge, compute_uv=False)
    numerical_rank = int(np.sum(singular_values > EDGE_RANK_TOLERANCE))
    condition = float(
        np.inf if singular_values[-1] <= 0 else singular_values[0] / singular_values[-1]
    )
    status = (
        QUALIFIED
        if numerical_rank == rank and condition <= EDGE_CONDITION_CEILING
        else "UNQUALIFIED"
    )
    return {
        "status": status,
        "matrix": edge,
        "singular_values": singular_values,
        "rank": numerical_rank,
        "condition": condition,
        "ridge": ridge,
        "rejection_reason": None
        if status == QUALIFIED
        else ("RANK_DEFICIENT" if numerical_rank != rank else "ILL_CONDITIONED"),
    }


def _edge(frames: Mapping[str, Mapping[str, Any]], source: str, target: str) -> dict[str, Any]:
    if frames[source]["status"] != QUALIFIED or frames[target]["status"] != QUALIFIED:
        return {"status": "UNQUALIFIED", "rejection_reason": "FRAME_NOT_QUALIFIED"}
    return fit_edge_map(frames[source]["coordinates"], frames[target]["coordinates"])


def _packet(
    edges: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    path_p: Sequence[str],
    path_q: Sequence[str],
    topology_id: str,
) -> dict[str, Any]:
    required = [
        *(zip(path_p[:-1], path_p[1:])),
        *(zip(path_q[:-1], path_q[1:])),
    ]
    failing = [pair for pair in required if edges[pair]["status"] != QUALIFIED]
    if failing:
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "EDGE_NOT_QUALIFIED",
            "failing_edges": [list(pair) for pair in failing],
        }
    packet = build_operator_packet(
        path_p_edges=[edges[pair]["matrix"] for pair in zip(path_p[:-1], path_p[1:])],
        path_q_edges=[edges[pair]["matrix"] for pair in zip(path_q[:-1], path_q[1:])],
        source_node=path_p[0],
        target_node=path_p[-1],
        path_p_id="->".join(path_p),
        path_q_id="->".join(path_q),
        topology_id=topology_id,
        rank_tolerance=EDGE_RANK_TOLERANCE,
        condition_ceiling=EDGE_CONDITION_CEILING,
    )
    packet["status"] = (
        QUALIFIED
        if packet["path_polar"]["status"] == QUALIFIED
        and packet["edge_polar"]["status"] == QUALIFIED
        else "UNQUALIFIED"
    )
    return packet


def build_half_packets(frames: Mapping[str, Mapping[str, Any]], *, half_id: str) -> dict[str, Any]:
    edge_pairs = {
        *zip(EXACT_PATH_P[:-1], EXACT_PATH_P[1:]),
        *zip(EXACT_PATH_Q[:-1], EXACT_PATH_Q[1:]),
        *zip(BROKEN_PATH_Q[:-1], BROKEN_PATH_Q[1:]),
    }
    edges = {pair: _edge(frames, pair[0], pair[1]) for pair in sorted(edge_pairs)}
    exact = _packet(
        edges,
        path_p=EXACT_PATH_P,
        path_q=EXACT_PATH_Q,
        topology_id="stepwise_exact_square_v1",
    )
    broken = _packet(
        edges,
        path_p=EXACT_PATH_P,
        path_q=BROKEN_PATH_Q,
        topology_id="stepwise_broken_square_positive_control_v1",
    )
    return {"half_id": half_id, "edges": edges, "exact_square": exact, "broken_square": broken}


def _normalized_delta(packet: Mapping[str, Any]) -> float:
    delta = np.asarray(packet["raw"]["Delta_pq"], dtype=np.float64)
    return float(np.linalg.norm(delta, ord="fro") / np.sqrt(delta.shape[0]))


def _singular_floor(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    values = []
    for key in ("singular_values_p", "singular_values_q"):
        left = np.asarray(first["raw"][key], dtype=np.float64)
        right = np.asarray(second["raw"][key], dtype=np.float64)
        values.append(float(np.linalg.norm(left - right) / np.sqrt(left.size)))
    return max(values)


def qualify_layer(
    half_1_activations: Mapping[str, Any],
    half_2_activations: Mapping[str, Any],
    *,
    layer: int,
) -> dict[str, Any]:
    first_frames = {node: estimate_frame(values) for node, values in half_1_activations.items()}
    second_native = {node: estimate_frame(values) for node, values in half_2_activations.items()}
    second_frames = align_half_frames(first_frames, second_native, half_2_activations)
    first = build_half_packets(first_frames, half_id="half_1")
    second = build_half_packets(second_frames, half_id="half_2_aligned")
    packets = [first["exact_square"], second["exact_square"], first["broken_square"], second["broken_square"]]
    if any(packet.get("status") != QUALIFIED for packet in packets):
        return {
            "layer": layer,
            "status": "FAIL",
            "reason": "INSUFFICIENT_SUPPORT_OR_OPERATOR_QUALIFICATION",
            "half_1_frames": first_frames,
            "half_2_frames_native": second_native,
            "half_2_frames_aligned": second_frames,
            "half_1": first,
            "half_2": second,
        }
    floor = _singular_floor(first["exact_square"], second["exact_square"])
    exact_effects = [_normalized_delta(first["exact_square"]), _normalized_delta(second["exact_square"])]
    broken_effects = [_normalized_delta(first["broken_square"]), _normalized_delta(second["broken_square"])]
    sensitivity_threshold = max(
        BROKEN_SENSITIVITY_MULTIPLIER * floor,
        floor + BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
        max(exact_effects) + BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
    )
    reproducible = floor <= SPLIT_HALF_SINGULAR_FLOOR_MAX
    sensitive = min(broken_effects) > sensitivity_threshold
    return {
        "layer": layer,
        "status": "PASS" if reproducible and sensitive else "FAIL",
        "split_half_singular_floor": floor,
        "split_half_floor_ceiling": SPLIT_HALF_SINGULAR_FLOOR_MAX,
        "exact_square_normalized_delta_by_half": exact_effects,
        "broken_square_normalized_delta_by_half": broken_effects,
        "broken_sensitivity_threshold": sensitivity_threshold,
        "reproducible": reproducible,
        "broken_square_sensitive": sensitive,
        "packet_disagreement": packet_disagreement(
            first["exact_square"], second["exact_square"]
        ),
        "half_1_frames": first_frames,
        "half_2_frames_native": second_native,
        "half_2_frames_aligned": second_frames,
        "half_1": first,
        "half_2": second,
    }


def qualify_track_b(
    activations: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(activations) != {"half_1", "half_2"}:
        raise OperatorQualificationError("exactly two independent halves are required")
    layer_results = []
    for layer in LAYER_SET:
        layer_results.append(
            qualify_layer(
                activations["half_1"][layer],
                activations["half_2"][layer],
                layer=layer,
            )
        )
    pass_count = sum(result["status"] == "PASS" for result in layer_results)
    status = "PASS" if pass_count >= MINIMUM_QUALIFIED_LAYER_COUNT else "FAIL"
    return json_ready(
        {
            "schema_version": "gate13_stepwise_fresh_operator_qualification_v1",
            "status": status,
            "primary_question": (
                "operator packet is independently re-estimable and sensitive to the "
                "broken-square control above its split-half floor"
            ),
            "layer_set": list(LAYER_SET),
            "token_position": TOKEN_POSITION,
            "activation_representation": ACTIVATION_REPRESENTATION,
            "frame_rank": FRAME_RANK,
            "minimum_node_support_per_half": MINIMUM_NODE_SUPPORT_PER_HALF,
            "minimum_qualified_layer_count": MINIMUM_QUALIFIED_LAYER_COUNT,
            "qualified_layer_count": pass_count,
            "layers": layer_results,
            "generic_nonzero_holonomy_as_pass_condition": False,
            "track_c_opened": False,
        }
    )


def operator_lock_payload() -> dict[str, Any]:
    return {
        "layer_set": list(LAYER_SET),
        "token_position": TOKEN_POSITION,
        "activation_representation": ACTIVATION_REPRESENTATION,
        "frame_estimator": "CENTERED_SVD_WITH_DETERMINISTIC_COLUMN_SIGN",
        "frame_rank": FRAME_RANK,
        "frame_relative_singular_tolerance": FRAME_RELATIVE_SINGULAR_TOLERANCE,
        "edge_estimator": "PAIRED_RIDGE_LINEAR_MAP_IN_NODE_LOCAL_COORDINATES",
        "edge_ridge_relative": EDGE_RIDGE_RELATIVE,
        "edge_rank_tolerance": EDGE_RANK_TOLERANCE,
        "edge_condition_ceiling": EDGE_CONDITION_CEILING,
        "minimum_node_support_per_half": MINIMUM_NODE_SUPPORT_PER_HALF,
        "exact_square_path_p": list(EXACT_PATH_P),
        "exact_square_path_q": list(EXACT_PATH_Q),
        "broken_square_path_q": list(BROKEN_PATH_Q),
        "split_half_singular_floor_max": SPLIT_HALF_SINGULAR_FLOOR_MAX,
        "broken_sensitivity_multiplier": BROKEN_SENSITIVITY_MULTIPLIER,
        "broken_sensitivity_absolute_margin": BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
        "minimum_qualified_layer_count": MINIMUM_QUALIFIED_LAYER_COUNT,
        "raw_packet_required": ["P_p", "P_q", "Delta_pq", "singular spectra"],
        "polar_packet_required": ["S_p", "S_q", "H_path where qualified", "H_edge where qualified"],
        "bootstrap_role": "SECONDARY_ONLY_NOT_A_SUBSTITUTE_FOR_INDEPENDENT_HALVES",
        "generic_nonzero_holonomy_as_novelty_or_pass_condition": False,
    }

