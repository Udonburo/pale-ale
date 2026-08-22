"""Fail-closed validator and model-free Review 2 design utilities.

This module is deliberately incapable of model or Modal execution.  Its only
optional empirical input is the already-collected Qwen3.6-27B fresh-B
activation directory.  It never reads a Track C outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "gate13_track_c_review2_validator_v1"
MODEL_REPOSITORY = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
FROZEN_TEMPLATE = "natural_rule_v1"
FROZEN_LAYERS = (21, 43, 62)
GRID_CLOUD_SIZES = (8, 12, 16, 24)
NODES = (
    "phase0_state0",
    "phase0_state1",
    "phase1_state0",
    "phase1_state1",
    "phase1_state1_broken",
)
SOURCE_NODE = "phase0_state0"
EXACT_PATH_P = ("phase0_state0", "phase0_state1", "phase1_state1")
EXACT_PATH_Q = ("phase0_state0", "phase1_state0", "phase1_state1")
BROKEN_PATH_Q = ("phase0_state0", "phase1_state0", "phase1_state1_broken")

FRAME_RANK = 4
FRAME_RELATIVE_SINGULAR_TOLERANCE = 1.0e-6
EDGE_RIDGE_RELATIVE = 1.0e-3
EDGE_RANK_TOLERANCE = 1.0e-8
EDGE_CONDITION_CEILING = 1.0e6
SPLIT_HALF_SPECTRAL_FLOOR_CEILING = 0.20
BROKEN_SENSITIVITY_MULTIPLIER = 2.0
BROKEN_SENSITIVITY_ABSOLUTE_MARGIN = 0.05
HALF_FEATURE_LOG_RATIO_CEILING = math.log(4.0)
NUMERICAL_EPSILON = 1.0e-12
SUBSAMPLE_SEED = 13_602_026
PERMUTATION_SEED = 13_602_027
SENSITIVITY_SEED = 13_602_028
DEPTH_LEVELS = (2, 4, 6, 8)
FRONTIER_BLOCK_COUNTS = (16, 20, 24, 28, 32)
FRONTIER_BEHAVIOR_EPISODES = (8, 16, 24, 32)
FUTURE_MODAL_BUDGET_USD = 65.0
EXISTING_B_OPERATOR_COST_USD = 1.02497632
EXISTING_B_OPERATOR_FORWARD_COUNT = 240
EXISTING_ACQUISITION_AND_PREFLIGHT_COST_USD = 0.05520247 + 0.06848531
COST_CONTINGENCY_MULTIPLIER = 1.25
PLANNING_REPRESENTATION_NOISE_SD_AT_N24 = 0.35
PLANNING_BEHAVIOR_NOISE_SD_AT_E24 = 0.50
BEHAVIOR_RMS_RELATIVE_SE_CEILING = 0.15

REQUIRED_FILES = (
    "PANEL_CLOSEOUT_BINDING.md",
    "TRACK_C_REVIEW2_PROTOCOL.md",
    "track_c_estimand_lock_candidate.json",
    "track_c_sensitivity_and_cost.json",
    "track_c_prior_art_collision_matrix.md",
    "track_c_review2_validator.py",
)
TERMINAL_STATES = {
    "REVIEW2_READY_FOR_HUMAN_AUTHORIZATION",
    "REVIEW2_NO_FEASIBLE_DESIGN",
    "REVIEW2_ESTIMAND_BLOCKER",
    "REVIEW2_PROVENANCE_BLOCKER",
}


class Review2ValidationError(ValueError):
    """Raised when a Review 2 invariant fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_matrix(value: Any, *, context: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise Review2ValidationError(f"{context} must be a finite rank-2 matrix")
    return matrix


def _canonicalize_frame_sign(frame: np.ndarray) -> np.ndarray:
    result = np.asarray(frame, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def estimate_frame(values: Any, *, minimum_support: int) -> dict[str, Any]:
    """Estimate the frozen rank-four centered frame at a diagnostic support.

    The historical B implementation required 24 rows.  Review 2 relaxes only
    that support guard for the explicitly diagnostic 8/12/16/24 downsampling
    grid; it does not revise the immutable B result.
    """

    activations = _finite_matrix(values, context="activations")
    sample_count, hidden_size = activations.shape
    if sample_count < minimum_support:
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "INSUFFICIENT_NODE_SUPPORT",
            "sample_count": sample_count,
        }
    if FRAME_RANK > min(sample_count - 1, hidden_size):
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "RANK_DEFICIENT_FRAME",
            "sample_count": sample_count,
            "numerical_rank": max(0, min(sample_count - 1, hidden_size)),
        }

    mean = activations.mean(axis=0)
    centered = activations - mean
    # The sample Gram route returns the same right singular subspace while
    # avoiding a large full decomposition of a 24 x 5120 matrix.
    gram = centered @ centered.T
    eigenvalues, left_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left_vectors = left_vectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(
        np.finfo(np.float64).eps,
        leading * FRAME_RELATIVE_SINGULAR_TOLERANCE,
    )
    numerical_rank = int(np.sum(singular_values > threshold))
    if numerical_rank < FRAME_RANK:
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "RANK_DEFICIENT_FRAME",
            "sample_count": sample_count,
            "numerical_rank": numerical_rank,
            "singular_values": singular_values,
            "threshold": threshold,
        }
    frame = centered.T @ left_vectors[:, :FRAME_RANK]
    frame /= singular_values[:FRAME_RANK][None, :]
    frame = _canonicalize_frame_sign(frame)
    return {
        "status": "QUALIFIED",
        "sample_count": sample_count,
        "mean": mean,
        "frame": frame,
        "coordinates": centered @ frame,
        "singular_values": singular_values,
        "numerical_rank": numerical_rank,
        "threshold": threshold,
    }


def fit_edge_map(source_coordinates: Any, target_coordinates: Any) -> dict[str, Any]:
    source = _finite_matrix(source_coordinates, context="source coordinates")
    target = _finite_matrix(target_coordinates, context="target coordinates")
    if source.shape != target.shape:
        raise Review2ValidationError("paired edge coordinates must have equal shapes")
    rank = source.shape[1]
    gram = source.T @ source
    scale = max(float(np.trace(gram)) / max(rank, 1), np.finfo(np.float64).eps)
    ridge = EDGE_RIDGE_RELATIVE * scale
    matrix = target.T @ source @ np.linalg.inv(gram + ridge * np.eye(rank))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    numerical_rank = int(np.sum(singular_values > EDGE_RANK_TOLERANCE))
    condition = float(
        np.inf
        if singular_values[-1] <= 0.0
        else singular_values[0] / singular_values[-1]
    )
    qualified = numerical_rank == rank and condition <= EDGE_CONDITION_CEILING
    return {
        "status": "QUALIFIED" if qualified else "UNQUALIFIED",
        "matrix": matrix,
        "singular_values": singular_values,
        "rank": numerical_rank,
        "condition": condition,
        "ridge": ridge,
    }


def _compose(edges: Sequence[np.ndarray]) -> np.ndarray:
    if not edges:
        return np.eye(FRAME_RANK, dtype=np.float64)
    product = np.eye(edges[0].shape[0], dtype=np.float64)
    for edge in edges:
        product = np.asarray(edge, dtype=np.float64) @ product
    return product


def _path_packet(
    edges: Mapping[tuple[str, str], Mapping[str, Any]],
    path: Sequence[str],
) -> dict[str, Any]:
    pairs = list(zip(path[:-1], path[1:]))
    failing = [pair for pair in pairs if edges[pair]["status"] != "QUALIFIED"]
    if failing:
        return {"status": "UNQUALIFIED", "failing_edges": [list(pair) for pair in failing]}
    matrix = _compose([np.asarray(edges[pair]["matrix"]) for pair in pairs])
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values > EDGE_RANK_TOLERANCE))
    condition = float(
        np.inf
        if singular_values[-1] <= 0.0
        else singular_values[0] / singular_values[-1]
    )
    qualified = rank == matrix.shape[0] and condition <= EDGE_CONDITION_CEILING
    return {
        "status": "QUALIFIED" if qualified else "UNQUALIFIED",
        "matrix": matrix,
        "singular_values": singular_values,
        "rank": rank,
        "condition": condition,
    }


def build_half_packets(frames: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    edge_pairs = {
        *zip(EXACT_PATH_P[:-1], EXACT_PATH_P[1:]),
        *zip(EXACT_PATH_Q[:-1], EXACT_PATH_Q[1:]),
        *zip(BROKEN_PATH_Q[:-1], BROKEN_PATH_Q[1:]),
    }
    if any(frames[node]["status"] != "QUALIFIED" for node in NODES):
        return {"status": "UNQUALIFIED", "rejection_reason": "FRAME_NOT_QUALIFIED"}
    edges = {
        pair: fit_edge_map(
            frames[pair[0]]["coordinates"],
            frames[pair[1]]["coordinates"],
        )
        for pair in sorted(edge_pairs)
    }
    path_p = _path_packet(edges, EXACT_PATH_P)
    path_q = _path_packet(edges, EXACT_PATH_Q)
    broken_q = _path_packet(edges, BROKEN_PATH_Q)
    packets = (path_p, path_q, broken_q)
    if any(packet["status"] != "QUALIFIED" for packet in packets):
        return {
            "status": "UNQUALIFIED",
            "rejection_reason": "EDGE_OR_PATH_NOT_QUALIFIED",
            "edges": edges,
            "path_p": path_p,
            "path_q": path_q,
            "broken_q": broken_q,
        }
    exact_delta = path_p["matrix"] - path_q["matrix"]
    broken_delta = path_p["matrix"] - broken_q["matrix"]
    conditions = [
        *(float(edge["condition"]) for edge in edges.values()),
        float(path_p["condition"]),
        float(path_q["condition"]),
        float(broken_q["condition"]),
    ]
    return {
        "status": "QUALIFIED",
        "edges": edges,
        "path_p": path_p,
        "path_q": path_q,
        "broken_q": broken_q,
        "exact_delta": exact_delta,
        "broken_delta": broken_delta,
        "maximum_condition": max(conditions),
    }


def _normalized_delta(delta: np.ndarray) -> float:
    return float(np.linalg.norm(delta, ord="fro") / math.sqrt(delta.shape[0]))


def _spectral_floor(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    disagreements = []
    for key in ("path_p", "path_q"):
        left = np.asarray(first[key]["singular_values"], dtype=np.float64)
        right = np.asarray(second[key]["singular_values"], dtype=np.float64)
        disagreements.append(float(np.linalg.norm(left - right) / math.sqrt(left.size)))
    return max(disagreements)


def source_weighted_crossfit_scalar(
    delta: Any,
    held_out_source_activations: Any,
    training_source_frame: Any,
) -> float:
    """Compute tr(Delta Sigma Delta^T) / tr(Sigma) in one source gauge."""

    operator = _finite_matrix(delta, context="Delta")
    held_out = _finite_matrix(
        held_out_source_activations,
        context="held-out source activations",
    )
    frame = _finite_matrix(training_source_frame, context="training source frame")
    if operator.shape != (frame.shape[1], frame.shape[1]):
        raise Review2ValidationError("Delta and source-frame dimensions differ")
    projected = held_out @ frame
    projected -= projected.mean(axis=0, keepdims=True)
    covariance = projected.T @ projected / max(projected.shape[0] - 1, 1)
    denominator = float(np.trace(covariance))
    if not np.isfinite(denominator) or denominator <= NUMERICAL_EPSILON:
        raise Review2ValidationError("held-out source covariance has zero trace")
    numerator = float(np.trace(operator @ covariance @ operator.T))
    value = numerator / denominator
    if not np.isfinite(value) or value < 0.0:
        raise Review2ValidationError("cross-fitted scalar is not finite and nonnegative")
    return value


def block_behavioral_summary(
    path_p_margins: Any,
    path_q_margins: Any,
    *,
    expected_episodes: int = 24,
) -> dict[str, float]:
    """Aggregate matched path margins to the one-row-per-block outcomes."""

    path_p = np.asarray(path_p_margins, dtype=np.float64)
    path_q = np.asarray(path_q_margins, dtype=np.float64)
    if (
        path_p.ndim != 1
        or path_q.ndim != 1
        or path_p.shape != path_q.shape
        or path_p.size != expected_episodes
        or not np.all(np.isfinite(path_p))
        or not np.all(np.isfinite(path_q))
    ):
        raise Review2ValidationError(
            "a block requires two finite matched margin vectors at frozen support"
        )
    return {
        "rms_equivalent_path_margin_discrepancy": float(
            np.sqrt(np.mean(np.square(path_p - path_q)))
        ),
        "mean_path_averaged_margin": float(np.mean(0.5 * (path_p + path_q))),
    }


def standardized_analysis_designs(
    rollout_depth: Any,
    mean_path_averaged_margin: Any,
    representation_feature: Any,
) -> dict[str, np.ndarray]:
    """Build the frozen nuisance and full block-level design matrices."""

    columns = []
    for name, values in (
        ("rollout depth", rollout_depth),
        ("mean path-averaged margin", mean_path_averaged_margin),
        ("representation feature", representation_feature),
    ):
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise Review2ValidationError(f"{name} must be a finite vector")
        columns.append(vector)
    if len({column.size for column in columns}) != 1:
        raise Review2ValidationError("analysis columns have unequal block counts")
    standardized = [_standardize(column) for column in columns]
    nuisance = np.column_stack(
        [np.ones(columns[0].size), standardized[0], standardized[1]]
    )
    full = np.column_stack([nuisance, standardized[2]])
    _design_matrix(nuisance, context="nuisance design")
    _design_matrix(full, context="full design")
    return {
        "nuisance": nuisance,
        "full": full,
        "standardized_representation_feature": standardized[2],
    }


def qualify_layer_crossfit(
    half_1: Mapping[str, Any],
    half_2: Mapping[str, Any],
    *,
    layer: int,
    minimum_support: int,
) -> dict[str, Any]:
    """Qualify a layer without ever multiplying across half-specific gauges."""

    first_frames = {
        node: estimate_frame(half_1[node], minimum_support=minimum_support)
        for node in NODES
    }
    second_frames = {
        node: estimate_frame(half_2[node], minimum_support=minimum_support)
        for node in NODES
    }
    first = build_half_packets(first_frames)
    second = build_half_packets(second_frames)
    if first["status"] != "QUALIFIED" or second["status"] != "QUALIFIED":
        return {
            "layer": layer,
            "status": "FAIL",
            "rank_and_conditioning": False,
            "reason": "FRAME_EDGE_OR_PATH_UNQUALIFIED",
        }

    first_scalar = source_weighted_crossfit_scalar(
        first["exact_delta"],
        half_2[SOURCE_NODE],
        first_frames[SOURCE_NODE]["frame"],
    )
    second_scalar = source_weighted_crossfit_scalar(
        second["exact_delta"],
        half_1[SOURCE_NODE],
        second_frames[SOURCE_NODE]["frame"],
    )
    feature_log_ratio = abs(
        math.log(
            max(first_scalar, NUMERICAL_EPSILON)
            / max(second_scalar, NUMERICAL_EPSILON)
        )
    )
    spectral_floor = _spectral_floor(first, second)
    exact_responses = [
        _normalized_delta(first["exact_delta"]),
        _normalized_delta(second["exact_delta"]),
    ]
    broken_responses = [
        _normalized_delta(first["broken_delta"]),
        _normalized_delta(second["broken_delta"]),
    ]
    broken_threshold = max(
        BROKEN_SENSITIVITY_MULTIPLIER * spectral_floor,
        spectral_floor + BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
        max(exact_responses) + BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
    )
    gates = {
        "rank_and_conditioning": max(
            float(first["maximum_condition"]),
            float(second["maximum_condition"]),
        )
        <= EDGE_CONDITION_CEILING,
        "exact_square_packet_reproducibility": (
            spectral_floor <= SPLIT_HALF_SPECTRAL_FLOOR_CEILING
        ),
        "broken_square_sensitivity": min(broken_responses) > broken_threshold,
    }
    return {
        "layer": layer,
        "status": "PASS" if all(gates.values()) else "FAIL",
        **gates,
        "layer_half_feature_log_ratio_diagnostic": feature_log_ratio,
        "crossfit_scalar_by_training_half": [first_scalar, second_scalar],
        "symmetric_layer_scalar": 0.5 * (first_scalar + second_scalar),
        "split_half_spectral_floor": spectral_floor,
        "exact_square_normalized_delta_by_half": exact_responses,
        "broken_square_normalized_delta_by_half": broken_responses,
        "broken_sensitivity_threshold": broken_threshold,
        "maximum_condition": max(
            float(first["maximum_condition"]),
            float(second["maximum_condition"]),
        ),
    }


_ACTIVATION_RE = re.compile(
    r"^b-(half_[12])-(?P<sample>\d{2})-(?P<node>phase[01]_state[01](?:_broken)?)\.npz$"
)


def load_existing_b_activations(source: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Load only existing B NPZ activations; response/outcome files are ignored."""

    source = source.resolve()
    activation_root = source / "activations"
    if not activation_root.is_dir():
        raise Review2ValidationError(f"missing activation directory: {activation_root}")
    rows: dict[str, dict[int, dict[str, list[tuple[int, np.ndarray]]]]] = {
        half: {layer: {node: [] for node in NODES} for layer in FROZEN_LAYERS}
        for half in ("half_1", "half_2")
    }
    observed_files = 0
    for path in sorted(activation_root.glob("half_*/*.npz")):
        match = _ACTIVATION_RE.match(path.name)
        if match is None:
            raise Review2ValidationError(f"unexpected B activation filename: {path.name}")
        half = match.group(1)
        node = match.group("node")
        sample = int(match.group("sample"))
        if node not in NODES:
            raise Review2ValidationError(f"unexpected B activation node: {node}")
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {f"layer_{layer}" for layer in FROZEN_LAYERS}:
                raise Review2ValidationError(f"activation layer set drift: {path}")
            for layer in FROZEN_LAYERS:
                vector = np.asarray(stored[f"layer_{layer}"], dtype=np.float64)
                if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                    raise Review2ValidationError(f"invalid activation vector: {path}:layer_{layer}")
                rows[half][layer][node].append((sample, vector))
        observed_files += 1
    if observed_files != 2 * 24 * len(NODES):
        raise Review2ValidationError(
            f"expected 240 existing B activation files, found {observed_files}"
        )

    result: dict[str, dict[int, dict[str, np.ndarray]]] = {
        half: {layer: {} for layer in FROZEN_LAYERS}
        for half in ("half_1", "half_2")
    }
    for half in result:
        for layer in FROZEN_LAYERS:
            for node in NODES:
                values = sorted(rows[half][layer][node], key=lambda item: item[0])
                indices = [index for index, _vector in values]
                if indices != list(range(24)):
                    raise Review2ValidationError(
                        f"B sample index drift: {half}:layer_{layer}:{node}"
                    )
                result[half][layer][node] = np.stack(
                    [vector for _index, vector in values], axis=0
                )
    return result


def _subsample_indices(size: int, replicate: int, half_index: int) -> np.ndarray:
    if size == 24:
        return np.arange(24, dtype=np.int64)
    seed = np.random.SeedSequence([SUBSAMPLE_SEED, size, replicate, half_index])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(24, size=size, replace=False))


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise Review2ValidationError("cannot summarize an empty metric")
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def run_fixed_grid_downsampling(
    source: Path,
    *,
    replicates: int = 24,
) -> dict[str, Any]:
    if replicates <= 0:
        raise Review2ValidationError("replicates must be positive")
    clouds = load_existing_b_activations(source)
    per_size_rows: dict[int, list[dict[str, Any]]] = {}
    full_feature: float | None = None
    for size in GRID_CLOUD_SIZES:
        rows = []
        replicate_count = 1 if size == 24 else replicates
        for replicate in range(replicate_count):
            indices = {
                "half_1": _subsample_indices(size, replicate, 1),
                "half_2": _subsample_indices(size, replicate, 2),
            }
            layers = []
            for layer in FROZEN_LAYERS:
                halves = {
                    half: {
                        node: clouds[half][layer][node][indices[half]]
                        for node in NODES
                    }
                    for half in ("half_1", "half_2")
                }
                layers.append(
                    qualify_layer_crossfit(
                        halves["half_1"],
                        halves["half_2"],
                        layer=layer,
                        minimum_support=size,
                    )
                )
            complete = all("crossfit_scalar_by_training_half" in row for row in layers)
            feature_by_training_half = (
                [
                    float(
                        np.mean(
                            [
                                row["crossfit_scalar_by_training_half"][half_index]
                                for row in layers
                            ]
                        )
                    )
                    for half_index in (0, 1)
                ]
                if complete
                else None
            )
            block_feature_log_ratio = (
                abs(
                    math.log(
                        max(feature_by_training_half[0], NUMERICAL_EPSILON)
                        / max(feature_by_training_half[1], NUMERICAL_EPSILON)
                    )
                )
                if feature_by_training_half is not None
                else None
            )
            split_half_feature_qualified = (
                block_feature_log_ratio is not None
                and block_feature_log_ratio <= HALF_FEATURE_LOG_RATIO_CEILING
            )
            valid = (
                all(layer_result["status"] == "PASS" for layer_result in layers)
                and split_half_feature_qualified
            )
            feature = (
                float(np.mean(feature_by_training_half))
                if feature_by_training_half is not None
                else None
            )
            rows.append(
                {
                    "replicate": replicate,
                    "indices": {key: value.tolist() for key, value in indices.items()},
                    "status": "PASS" if valid else "FAIL",
                    "feature": feature,
                    "feature_by_training_half": feature_by_training_half,
                    "split_half_feature_qualified": split_half_feature_qualified,
                    "block_feature_half_log_ratio": block_feature_log_ratio,
                    "layers": layers,
                }
            )
        per_size_rows[size] = rows
        if size == 24:
            full_feature = rows[0]["feature"]
    if full_feature is None or full_feature <= NUMERICAL_EPSILON:
        raise Review2ValidationError("full-support B feature is unavailable or zero")

    summaries = []
    for size in GRID_CLOUD_SIZES:
        rows = per_size_rows[size]
        features = [float(row["feature"]) for row in rows if row["feature"] is not None]
        layer_rows = [layer for row in rows for layer in row["layers"]]
        summaries.append(
            {
                "samples_per_node_per_half": size,
                "replicate_count": len(rows),
                "all_layer_gate_pass_count": sum(row["status"] == "PASS" for row in rows),
                "all_layer_gate_pass_rate": float(
                    np.mean([row["status"] == "PASS" for row in rows])
                ),
                "per_layer_gate_pass_rate": {
                    str(layer): float(
                        np.mean(
                            [
                                row["layers"][FROZEN_LAYERS.index(layer)]["status"] == "PASS"
                                for row in rows
                            ]
                        )
                    )
                    for layer in FROZEN_LAYERS
                },
                "primary_feature": _quantiles(features),
                "absolute_log_error_to_full_feature": _quantiles(
                    abs(math.log(max(value, NUMERICAL_EPSILON) / full_feature))
                    for value in features
                ),
                "block_feature_half_log_ratio": _quantiles(
                    float(row["block_feature_half_log_ratio"])
                    for row in rows
                    if row["block_feature_half_log_ratio"] is not None
                ),
                "layer_half_feature_log_ratio_diagnostic": _quantiles(
                    float(row["layer_half_feature_log_ratio_diagnostic"])
                    for row in layer_rows
                    if "layer_half_feature_log_ratio_diagnostic" in row
                ),
                "split_half_spectral_floor": _quantiles(
                    float(row["split_half_spectral_floor"])
                    for row in layer_rows
                    if "split_half_spectral_floor" in row
                ),
                "maximum_condition": _quantiles(
                    float(row["maximum_condition"])
                    for row in layer_rows
                    if "maximum_condition" in row
                ),
            }
        )
    subset_schedule = {
        str(size): [row["indices"] for row in per_size_rows[size]]
        for size in GRID_CLOUD_SIZES
    }
    return {
        "schema_version": "gate13_track_c_review2_b_downsampling_v1",
        "source_kind": "EXISTING_QWEN3_6_27B_FRESH_B_ACTIVATIONS_ONLY",
        "source_path": source.as_posix(),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "layers": list(FROZEN_LAYERS),
        "grid_samples_per_node_per_half": list(GRID_CLOUD_SIZES),
        "replicates_requested_below_full_support": replicates,
        "subsample_seed": SUBSAMPLE_SEED,
        "subsample_schedule_sha256": canonical_sha256(subset_schedule),
        "qualification_thresholds": {
            "frame_rank": FRAME_RANK,
            "edge_condition_ceiling": EDGE_CONDITION_CEILING,
            "split_half_spectral_floor_ceiling": SPLIT_HALF_SPECTRAL_FLOOR_CEILING,
            "half_feature_log_ratio_ceiling": HALF_FEATURE_LOG_RATIO_CEILING,
            "broken_sensitivity_multiplier": BROKEN_SENSITIVITY_MULTIPLIER,
            "broken_sensitivity_absolute_margin": BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
            "all_three_frozen_layers_required": True,
        },
        "full_support_primary_feature": full_feature,
        "grid": summaries,
        "track_c_outcome_read": False,
        "model_calls": 0,
    }


def _design_matrix(values: Any, *, context: str) -> np.ndarray:
    matrix = _finite_matrix(values, context=context)
    if matrix.shape[0] <= matrix.shape[1]:
        raise Review2ValidationError(f"{context} has no residual degrees of freedom")
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise Review2ValidationError(f"{context} is rank deficient")
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > EDGE_CONDITION_CEILING:
        raise Review2ValidationError(f"{context} is ill conditioned")
    return matrix


def _lobo_sse(outcome: Any, design: Any) -> float:
    y = np.asarray(outcome, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise Review2ValidationError("outcome must be a finite vector")
    x = _design_matrix(design, context="prediction design")
    if x.shape[0] != y.size:
        raise Review2ValidationError("outcome and design row counts differ")
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)
    if np.any(1.0 - leverage <= NUMERICAL_EPSILON):
        raise Review2ValidationError("leave-one-block-out leverage is one")
    held_out_residual = residual / (1.0 - leverage)
    return float(held_out_residual @ held_out_residual)


def relative_lobo_sse_reduction(
    outcome: Any,
    nuisance_design: Any,
    representation_feature: Any,
) -> dict[str, float]:
    """Freeze 1 - SSE_full/SSE_nuisance using leave-one-block-out errors."""

    nuisance = _design_matrix(nuisance_design, context="nuisance design")
    feature = np.asarray(representation_feature, dtype=np.float64)
    if feature.ndim != 1 or feature.size != nuisance.shape[0]:
        raise Review2ValidationError("representation feature has the wrong shape")
    if not np.all(np.isfinite(feature)):
        raise Review2ValidationError("representation feature must be finite")
    full = np.column_stack([nuisance, feature])
    nuisance_sse = _lobo_sse(outcome, nuisance)
    full_sse = _lobo_sse(outcome, full)
    if nuisance_sse <= NUMERICAL_EPSILON:
        raise Review2ValidationError("nuisance held-out SSE is zero")
    return {
        "sse_nuisance": nuisance_sse,
        "sse_full": full_sse,
        "relative_held_out_sse_reduction": 1.0 - full_sse / nuisance_sse,
    }


def nuisance_preserving_permutation_test(
    outcome: Any,
    nuisance_design: Any,
    representation_feature: Any,
    rollout_depth: Any,
    *,
    permutations: int = 99_999,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Freedman-Lane residual permutation within frozen depth strata."""

    if permutations <= 0:
        raise Review2ValidationError("permutations must be positive")
    y = np.asarray(outcome, dtype=np.float64)
    nuisance = _design_matrix(nuisance_design, context="nuisance design")
    depth = np.asarray(rollout_depth)
    if y.ndim != 1 or depth.ndim != 1 or y.size != nuisance.shape[0] or depth.size != y.size:
        raise Review2ValidationError("permutation inputs have incompatible shapes")
    observed = relative_lobo_sse_reduction(y, nuisance, representation_feature)
    beta, *_ = np.linalg.lstsq(nuisance, y, rcond=None)
    fitted = nuisance @ beta
    residual = y - fitted
    strata = [np.flatnonzero(depth == value) for value in sorted(set(depth.tolist()))]
    if any(indices.size < 2 for indices in strata):
        raise Review2ValidationError("each rollout-depth stratum needs at least two blocks")
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _index in range(permutations):
        permuted = residual.copy()
        for indices in strata:
            permuted[indices] = residual[rng.permutation(indices)]
        synthetic = fitted + permuted
        statistic = relative_lobo_sse_reduction(
            synthetic,
            nuisance,
            representation_feature,
        )["relative_held_out_sse_reduction"]
        if statistic >= observed["relative_held_out_sse_reduction"]:
            exceedances += 1
    return {
        **observed,
        "procedure": "FREEDMAN_LANE_NUISANCE_RESIDUAL_PERMUTATION_WITHIN_ROLLOUT_DEPTH",
        "permutations": permutations,
        "seed": seed,
        "exceedances": exceedances,
        "p_value_one_sided": (1.0 + exceedances) / (1.0 + permutations),
    }


def depth_schedule(block_count: int) -> list[int]:
    if block_count <= 0 or block_count % len(DEPTH_LEVELS) != 0:
        raise Review2ValidationError("block count must be positive and divisible by four")
    repeats = block_count // len(DEPTH_LEVELS)
    return [depth for depth in DEPTH_LEVELS for _index in range(repeats)]


def forward_and_cost_forecast(
    block_count: int,
    cloud_size: int,
    behavior_episodes: int,
) -> dict[str, Any]:
    depths = depth_schedule(block_count)
    map_forwards = block_count * 2 * len(NODES) * cloud_size
    # Each path uses `depth` self-fed transitions plus one common endpoint
    # forced-choice probe.  External register flips themselves are not calls.
    behavior_forwards = sum(
        2 * behavior_episodes * (depth + 1)
        for depth in depths
    )
    total = map_forwards + behavior_forwards
    empirical_rate = EXISTING_B_OPERATOR_COST_USD / EXISTING_B_OPERATOR_FORWARD_COUNT
    empirical_linear = (
        total * empirical_rate + EXISTING_ACQUISITION_AND_PREFLIGHT_COST_USD
    )
    planning_upper = empirical_linear * COST_CONTINGENCY_MULTIPLIER
    return {
        "map_activation_forwards": map_forwards,
        "behavior_forwards": behavior_forwards,
        "total_scientific_forwards": total,
        "depth_schedule": depths,
        "depth_counts": {
            str(depth): depths.count(depth)
            for depth in DEPTH_LEVELS
        },
        "empirical_qwen3_6_fresh_b_cost_per_forward_usd": empirical_rate,
        "fixed_acquisition_and_preflight_reference_usd": (
            EXISTING_ACQUISITION_AND_PREFLIGHT_COST_USD
        ),
        "empirical_linear_forecast_usd": empirical_linear,
        "contingency_multiplier": COST_CONTINGENCY_MULTIPLIER,
        "planning_upper_usd": planning_upper,
        "gpu_rate_basis_only_not_allocation": "HISTORICAL_QWEN3_6_27B_A100_80GB_FRESH_B",
    }


def _standardize(vector: np.ndarray) -> np.ndarray:
    centered = np.asarray(vector, dtype=np.float64) - float(np.mean(vector))
    scale = float(np.std(centered, ddof=1))
    if scale <= NUMERICAL_EPSILON:
        raise Review2ValidationError("simulation vector has zero variance")
    return centered / scale


def _simulate_statistic(
    *,
    block_count: int,
    cloud_size: int,
    behavior_episodes: int,
    latent_partial_correlation: float,
    rng: np.random.Generator,
) -> float:
    depths = np.asarray(depth_schedule(block_count), dtype=np.float64)
    depth_z = _standardize(depths)
    margin = _standardize(0.30 * depth_z + rng.normal(size=block_count))
    independent_feature = _standardize(rng.normal(size=block_count))
    latent_feature = _standardize(
        0.25 * margin + math.sqrt(1.0 - 0.25**2) * independent_feature
    )
    representation_noise_sd = (
        PLANNING_REPRESENTATION_NOISE_SD_AT_N24
        * math.sqrt(24.0 / cloud_size)
    )
    observed_feature = _standardize(
        latent_feature + rng.normal(scale=representation_noise_sd, size=block_count)
    )
    structural_noise = _standardize(rng.normal(size=block_count))
    rho = float(latent_partial_correlation)
    residual_outcome = rho * independent_feature + math.sqrt(max(0.0, 1.0 - rho**2)) * structural_noise
    behavior_noise_sd = (
        PLANNING_BEHAVIOR_NOISE_SD_AT_E24
        * math.sqrt(24.0 / behavior_episodes)
    )
    outcome = (
        0.40 * depth_z
        + 0.35 * margin
        + residual_outcome
        + rng.normal(scale=behavior_noise_sd, size=block_count)
    )
    nuisance = np.column_stack([np.ones(block_count), depth_z, margin])
    return relative_lobo_sse_reduction(
        outcome,
        nuisance,
        observed_feature,
    )["relative_held_out_sse_reduction"]


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, ddof=1, keepdims=True)
    if np.any(scale <= NUMERICAL_EPSILON):
        raise Review2ValidationError("simulation batch contains a zero-variance row")
    return centered / scale


def _batch_lobo_sse(outcome: np.ndarray, design: np.ndarray) -> np.ndarray:
    gram = np.einsum("sni,snj->sij", design, design)
    inverse = np.linalg.inv(gram)
    xty = np.einsum("sni,sn->si", design, outcome)
    beta = np.einsum("sij,sj->si", inverse, xty)
    fitted = np.einsum("sni,si->sn", design, beta)
    residual = outcome - fitted
    leverage = np.einsum("sni,sij,snj->sn", design, inverse, design)
    if np.any(1.0 - leverage <= NUMERICAL_EPSILON):
        raise Review2ValidationError("simulation LOBO leverage reached one")
    held_out = residual / (1.0 - leverage)
    return np.einsum("sn,sn->s", held_out, held_out)


def _simulate_statistics_batch(
    *,
    block_count: int,
    cloud_size: int,
    behavior_episodes: int,
    latent_partial_correlation: float,
    simulations: int,
    seed_components: Sequence[int],
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_components)))
    depths = np.asarray(depth_schedule(block_count), dtype=np.float64)
    depth_z = _standardize(depths)
    tiled_depth = np.broadcast_to(depth_z, (simulations, block_count))
    margin = _standardize_rows(
        0.30 * tiled_depth + rng.normal(size=(simulations, block_count))
    )
    independent_feature = _standardize_rows(
        rng.normal(size=(simulations, block_count))
    )
    latent_feature = _standardize_rows(
        0.25 * margin + math.sqrt(1.0 - 0.25**2) * independent_feature
    )
    representation_noise_sd = (
        PLANNING_REPRESENTATION_NOISE_SD_AT_N24
        * math.sqrt(24.0 / cloud_size)
    )
    observed_feature = _standardize_rows(
        latent_feature
        + rng.normal(
            scale=representation_noise_sd,
            size=(simulations, block_count),
        )
    )
    structural_noise = _standardize_rows(
        rng.normal(size=(simulations, block_count))
    )
    rho = float(latent_partial_correlation)
    residual_outcome = (
        rho * independent_feature
        + math.sqrt(max(0.0, 1.0 - rho**2)) * structural_noise
    )
    behavior_noise_sd = (
        PLANNING_BEHAVIOR_NOISE_SD_AT_E24
        * math.sqrt(24.0 / behavior_episodes)
    )
    outcome = (
        0.40 * tiled_depth
        + 0.35 * margin
        + residual_outcome
        + rng.normal(
            scale=behavior_noise_sd,
            size=(simulations, block_count),
        )
    )
    ones = np.ones_like(tiled_depth)
    nuisance = np.stack([ones, tiled_depth, margin], axis=2)
    full = np.concatenate([nuisance, observed_feature[:, :, None]], axis=2)
    nuisance_sse = _batch_lobo_sse(outcome, nuisance)
    full_sse = _batch_lobo_sse(outcome, full)
    return 1.0 - full_sse / nuisance_sse


def simulate_sensitivity_frontier(
    *,
    simulations: int = 1_000,
    effect_grid: Sequence[float] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
) -> dict[str, Any]:
    """Model-free LOBO sensitivity and empirical-cost frontier.

    This is a design sensitivity calculation, not a claim about the unknown
    Track C effect distribution.  No Track C data are accepted by the API.
    """

    if simulations < 100:
        raise Review2ValidationError("at least 100 simulations are required")
    effects = tuple(float(value) for value in effect_grid)
    if not effects or effects != tuple(sorted(effects)) or effects[0] <= 0.0 or effects[-1] >= 1.0:
        raise Review2ValidationError("effect grid must be strictly between zero and one")

    null_critical: dict[int, float] = {}
    for block_count in FRONTIER_BLOCK_COUNTS:
        null_values = _simulate_statistics_batch(
            block_count=block_count,
            cloud_size=24,
            behavior_episodes=24,
            latent_partial_correlation=0.0,
            simulations=simulations,
            seed_components=[SENSITIVITY_SEED, block_count, 0],
        )
        null_critical[block_count] = float(np.quantile(null_values, 0.95))

    rows = []
    for block_count in FRONTIER_BLOCK_COUNTS:
        for cloud_size in GRID_CLOUD_SIZES:
            for behavior_episodes in FRONTIER_BEHAVIOR_EPISODES:
                powers = []
                for effect_index, effect in enumerate(effects, 1):
                    statistics = _simulate_statistics_batch(
                        block_count=block_count,
                        cloud_size=cloud_size,
                        behavior_episodes=behavior_episodes,
                        latent_partial_correlation=effect,
                        simulations=simulations,
                        seed_components=[
                            SENSITIVITY_SEED,
                            block_count,
                            cloud_size,
                            behavior_episodes,
                            effect_index,
                        ],
                    )
                    exceedances = int(
                        np.sum(statistics >= null_critical[block_count])
                    )
                    powers.append(
                        {
                            "latent_partial_correlation": effect,
                            "estimated_power": exceedances / simulations,
                        }
                    )
                detectable = next(
                    (
                        row["latent_partial_correlation"]
                        for row in powers
                        if row["estimated_power"] >= 0.80
                    ),
                    None,
                )
                cost = forward_and_cost_forecast(
                    block_count,
                    cloud_size,
                    behavior_episodes,
                )
                rms_relative_se = 1.0 / math.sqrt(2.0 * behavior_episodes)
                rows.append(
                    {
                        "block_count": block_count,
                        "cloud_samples_per_node_per_half": cloud_size,
                        "behavior_episodes_per_block": behavior_episodes,
                        "null_95pct_lobo_statistic": null_critical[block_count],
                        "power_curve": powers,
                        "minimum_grid_latent_partial_correlation_at_80pct_power": detectable,
                        "behavior_rms_relative_se_reference": rms_relative_se,
                        "cloud_empirical_validity_eligible": cloud_size == 24,
                        "behavior_precision_eligible": (
                            rms_relative_se <= BEHAVIOR_RMS_RELATIVE_SE_CEILING
                        ),
                        "budget_eligible": (
                            cost["planning_upper_usd"] <= FUTURE_MODAL_BUDGET_USD
                        ),
                        "cost_forecast": cost,
                    }
                )

    eligible = [
        row
        for row in rows
        if row["cloud_empirical_validity_eligible"]
        and row["behavior_precision_eligible"]
        and row["budget_eligible"]
        and row["minimum_grid_latent_partial_correlation_at_80pct_power"] is not None
    ]
    if not eligible:
        selected = None
    else:
        selected = min(
            eligible,
            key=lambda row: (
                row["minimum_grid_latent_partial_correlation_at_80pct_power"],
                -row["block_count"],
                row["cost_forecast"]["planning_upper_usd"],
            ),
        )
    return {
        "schema_version": "gate13_track_c_review2_model_free_sensitivity_v1",
        "simulation_count_per_cell_and_effect": simulations,
        "simulation_seed": SENSITIVITY_SEED,
        "effect_grid_latent_partial_correlation": list(effects),
        "primary_statistic": "1-SSE_full_LOBO/SSE_nuisance_LOBO",
        "one_sided_alpha": 0.05,
        "target_power": 0.80,
        "planning_assumptions": {
            "standardized_representation_noise_sd_at_n24": (
                PLANNING_REPRESENTATION_NOISE_SD_AT_N24
            ),
            "representation_noise_scaling": "sqrt(24/cloud_size)",
            "standardized_behavior_noise_sd_at_e24": PLANNING_BEHAVIOR_NOISE_SD_AT_E24,
            "behavior_noise_scaling": "sqrt(24/behavior_episodes)",
            "depth_levels": list(DEPTH_LEVELS),
            "depth_balance": "EQUAL_COUNTS_PROSPECTIVELY_ASSIGNED",
            "interpretation": "MODEL_FREE_DESIGN_SENSITIVITY_NOT_AN_EFFECT_FORECAST",
        },
        "future_modal_budget": {
            "hard_ceiling_usd": FUTURE_MODAL_BUDGET_USD,
            "cost_basis": "HISTORICAL_QWEN3_6_27B_FRESH_B_PROVIDER_COST_WITH_25PCT_CONTINGENCY",
            "allocation_created": False,
        },
        "frontier": rows,
        "selected_cell": selected,
        "model_calls": 0,
        "track_c_outcome_read": False,
    }


def build_sensitivity_and_cost_payload(
    source: Path,
    *,
    replicates: int = 24,
    simulations: int = 1_000,
) -> dict[str, Any]:
    """Build the tracked, outcome-blind Review 2 planning payload."""

    source = source.resolve()
    downsampling = run_fixed_grid_downsampling(source, replicates=replicates)
    sensitivity = simulate_sensitivity_frontier(simulations=simulations)
    selected = sensitivity["selected_cell"]
    if selected is None:
        return {
            "schema_version": "gate13_track_c_review2_sensitivity_and_cost_v1",
            "terminal_recommendation": "REVIEW2_NO_FEASIBLE_DESIGN",
            "model_calls": 0,
            "track_c_outcome_read": False,
        }

    selected_key = (
        selected["block_count"],
        selected["cloud_samples_per_node_per_half"],
        selected["behavior_episodes_per_block"],
    )
    frozen_proposal = (20, 24, 24)
    if selected_key != frozen_proposal:
        raise Review2ValidationError(
            f"deterministic selection drifted from {frozen_proposal} to {selected_key}"
        )
    minimum_qualified = next(
        row
        for row in sensitivity["frontier"]
        if (
            row["block_count"],
            row["cloud_samples_per_node_per_half"],
            row["behavior_episodes_per_block"],
        )
        == (16, 24, 24)
    )

    relative_source = (
        "workstream/local/gate13_causal_return_outputs/checkpoint_panel/retrieved/"
        "qwen3_6_27b/executions/b786d648-8ea6-564b-a1cd-0f797c614a00/"
        "fresh_square_operator"
    )
    downsampling["source_path"] = relative_source
    artifact_manifest = _load_json(source / "artifact_manifest.json")
    frontier = []
    for row in sensitivity["frontier"]:
        cost = row["cost_forecast"]
        frontier.append(
            {
                "block_count": row["block_count"],
                "cloud_samples_per_node_per_half": row[
                    "cloud_samples_per_node_per_half"
                ],
                "behavior_episodes_per_block": row[
                    "behavior_episodes_per_block"
                ],
                "null_95pct_lobo_statistic": row["null_95pct_lobo_statistic"],
                "minimum_grid_latent_partial_correlation_at_80pct_power": row[
                    "minimum_grid_latent_partial_correlation_at_80pct_power"
                ],
                "behavior_rms_relative_se_reference": row[
                    "behavior_rms_relative_se_reference"
                ],
                "total_scientific_forwards": cost["total_scientific_forwards"],
                "planning_upper_usd": cost["planning_upper_usd"],
                "cloud_empirical_validity_eligible": row[
                    "cloud_empirical_validity_eligible"
                ],
                "behavior_precision_eligible": row[
                    "behavior_precision_eligible"
                ],
                "budget_eligible": row["budget_eligible"],
            }
        )

    return {
        "schema_version": "gate13_track_c_review2_sensitivity_and_cost_v1",
        "generated_date": "2026-08-23",
        "scope": "REVIEW2_ESTIMAND_DESIGN_ONLY",
        "source_provenance": {
            "source_kind": "EXISTING_QWEN3_6_27B_FRESH_B_ACTIVATIONS_ONLY",
            "source_path": relative_source,
            "artifact_manifest_sha256": sha256_file(source / "artifact_manifest.json"),
            "artifact_manifest_payload_sha256": artifact_manifest.get(
                "manifest_payload_sha256"
            ),
            "qualification_result_sha256": sha256_file(
                source / "qualification_result.json"
            ),
            "terminal_state_sha256": sha256_file(source / "terminal_state.json"),
            "response_or_logit_files_read": False,
        },
        "existing_b_fixed_grid_downsampling": downsampling,
        "model_free_sensitivity": {
            "simulation_count_per_cell_and_effect": sensitivity[
                "simulation_count_per_cell_and_effect"
            ],
            "simulation_seed": sensitivity["simulation_seed"],
            "effect_grid_latent_partial_correlation": sensitivity[
                "effect_grid_latent_partial_correlation"
            ],
            "primary_statistic": sensitivity["primary_statistic"],
            "one_sided_alpha": sensitivity["one_sided_alpha"],
            "target_power": sensitivity["target_power"],
            "planning_assumptions": sensitivity["planning_assumptions"],
            "frontier": frontier,
            "selected_power_curve": selected["power_curve"],
            "minimum_qualified_design_power_curve": minimum_qualified[
                "power_curve"
            ],
            "interpretation": (
                "DESIGN_SENSITIVITY_ONLY; NOT AN EFFECT-SIZE FORECAST AND NOT POWER "
                "CALIBRATED_FROM_A_TRACK_C_OUTCOME"
            ),
        },
        "selection_rule": {
            "cloud_rule": "REQUIRE_EMPIRICALLY_QUALIFIED_FULL_SUPPORT_N24",
            "behavior_rule": "REQUIRE_REFERENCE_RMS_RELATIVE_SE_AT_MOST_0.15",
            "budget_rule": "PLANNING_UPPER_MUST_NOT_EXCEED_65_USD",
            "ranking_rule": (
                "MINIMIZE_GRID_MDE_THEN_MAXIMIZE_BLOCK_COUNT_THEN_MINIMIZE_COST"
            ),
            "outcome_adaptive_changes": "FORBIDDEN",
        },
        "future_modal_budget": sensitivity["future_modal_budget"],
        "proposed_design": {
            "selected": True,
            "block_count": 20,
            "minimum_qualified_blocks": 16,
            "minimum_qualified_blocks_per_rollout_depth": 4,
            "failed_blocks_replaced_or_added": False,
            "cloud_samples_per_node_per_half": 24,
            "map_halves_per_block": 2,
            "map_nodes_per_half": len(NODES),
            "map_forwards_per_block": 2 * len(NODES) * 24,
            "behavior_episodes_per_block": 24,
            "rollout_depth_levels": list(DEPTH_LEVELS),
            "rollout_depth_counts": selected["cost_forecast"]["depth_counts"],
            "cost_forecast": selected["cost_forecast"],
            "minimum_grid_latent_partial_correlation_at_80pct_power": selected[
                "minimum_grid_latent_partial_correlation_at_80pct_power"
            ],
            "minimum_qualified_grid_latent_partial_correlation_at_80pct_power": (
                minimum_qualified[
                    "minimum_grid_latent_partial_correlation_at_80pct_power"
                ]
            ),
        },
        "sensitivity_limits": [
            (
                "Only the n=24 historical full-support grid point has one complete "
                "all-layer pass; n=8/12/16 fail rates are diagnostic, not block-level "
                "qualification-probability estimates."
            ),
            (
                "The selected 20/24/24 design reaches 80% simulated sensitivity only "
                "at a latent partial correlation of 0.80 on the frozen grid."
            ),
            (
                "The minimum-qualified 16/24/24 analysis has still weaker grid "
                "sensitivity; small or moderate effects are not reliably detectable."
            ),
            (
                "Noise parameters are declared planning assumptions, not estimates "
                "from a Track C coupling outcome."
            ),
            (
                "Cost extrapolation uses historical fresh-B average cost per forward; "
                "future provider/runtime variance is represented only by 25% contingency."
            ),
        ],
        "terminal_recommendation": "REVIEW2_READY_FOR_HUMAN_AUTHORIZATION",
        "model_calls": 0,
        "track_c_outcome_read": False,
        "modal_called": False,
        "gpu_allocated": False,
    }
def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Review2ValidationError(f"JSON root must be an object: {path}")
    return value


def validate_candidate_package(review2_root: Path) -> dict[str, Any]:
    """Validate the tracked Review 2 package without consulting ignored data."""

    review2_root = review2_root.resolve()
    errors: list[str] = []
    for filename in REQUIRED_FILES:
        if not (review2_root / filename).is_file():
            errors.append(f"missing required file: {filename}")
    tests_root = review2_root / "tests"
    if not tests_root.is_dir() or not any(tests_root.glob("test_*.py")):
        errors.append("missing targeted tests")
    if errors:
        return {"schema_version": SCHEMA_VERSION, "status": "FAIL", "errors": errors}

    try:
        lock = _load_json(review2_root / "track_c_estimand_lock_candidate.json")
        sensitivity = _load_json(review2_root / "track_c_sensitivity_and_cost.json")
    except (OSError, json.JSONDecodeError, Review2ValidationError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "errors": [str(exc)],
        }

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(lock.get("terminal_review2_state") in TERMINAL_STATES, "invalid terminal state")
    require(lock.get("terminal_review2_state") == "REVIEW2_READY_FOR_HUMAN_AUTHORIZATION", "candidate is not ready for human authorization")
    require(lock.get("model_target", {}).get("repository") == MODEL_REPOSITORY, "model repository drift")
    require(lock.get("model_target", {}).get("revision") == MODEL_REVISION, "model revision drift")
    require(lock.get("block_construction", {}).get("template") == FROZEN_TEMPLATE, "template drift")
    require(tuple(lock.get("representation_feature", {}).get("layers", [])) == FROZEN_LAYERS, "layer drift")
    require(lock.get("experimental_unit") == "ONE_INDEPENDENT_FRESH_NATURALITY_SQUARE_BLOCK", "experimental-unit drift")
    require(lock.get("prospective_design", {}).get("planned_blocks") == 20, "planned-block drift")
    require(lock.get("prospective_design", {}).get("cloud_samples_per_node_per_half") == 24, "cloud-size drift")
    require(lock.get("prospective_design", {}).get("behavior_episodes_per_block") == 24, "behavior-support drift")
    require(lock.get("block_construction", {}).get("template_variants") == "FORBIDDEN", "template variants opened")
    require(lock.get("block_construction", {}).get("map_estimation_and_behavior_evaluation") == "FULLY_DISJOINT", "map/behavior disjointness drift")
    require(lock.get("representation_feature", {}).get("gauge_rule") == "DELTA_AND_SIGMA_MUST_SHARE_THE_SAME_TRAINING_HALF_SOURCE_GAUGE", "source-gauge rule drift")
    require(lock.get("representation_feature", {}).get("unrelated_gauge_matrix_multiplication") == "FORBIDDEN", "cross-gauge multiplication opened")
    require(lock.get("behavioral_target", {}).get("episodes_per_block") == 24, "behavioral target support drift")
    require(lock.get("nuisance_model", {}).get("columns") == ["INTERCEPT", "ROLLOUT_DEPTH", "BLOCK_LEVEL_MEAN_PATH_AVERAGED_MARGIN"], "nuisance model drift")
    require(lock.get("primary_statistic", {}).get("formula") == "T = 1 - SSE_full_LOBO / SSE_nuisance_LOBO", "primary statistic drift")
    require(lock.get("null_test", {}).get("permutations") == 99_999, "permutation-count drift")
    require(lock.get("null_test", {}).get("seed") == PERMUTATION_SEED, "permutation-seed drift")
    require(lock.get("primary_test", {}).get("multiplicity") == 1, "primary multiplicity must equal one")
    require(lock.get("fresh_distribution_validity_gates", {}).get("minimum_qualified_blocks") == 16, "minimum-qualified-block drift")
    require(lock.get("fresh_distribution_validity_gates", {}).get("minimum_qualified_blocks_per_rollout_depth") == 4, "per-depth minimum drift")
    require(lock.get("authority", {}).get("track_c_execution") == "FORBIDDEN", "Track C execution opened")
    require(lock.get("authority", {}).get("execution_authorization_created") is False, "execution authorization created")
    require(lock.get("authority", {}).get("modal_called") is False, "Modal call recorded")
    require(lock.get("authority", {}).get("gpu_allocated") is False, "GPU allocation recorded")
    require(lock.get("authority", {}).get("model_downloaded_or_loaded") is False, "model load recorded")
    require(lock.get("authority", {}).get("model_forward_performed") is False, "model forward recorded")
    require(lock.get("authority", {}).get("activation_collection_performed") is False, "activation collection recorded")
    require(lock.get("authority", {}).get("track_c_outcome_inspected") is False, "Track C outcome inspection recorded")
    require(sensitivity.get("model_calls") == 0, "sensitivity file records model calls")
    require(sensitivity.get("track_c_outcome_read") is False, "Track C outcome was read")
    require(sensitivity.get("modal_called") is False, "sensitivity file records a Modal call")
    require(sensitivity.get("gpu_allocated") is False, "sensitivity file records GPU allocation")
    require(sensitivity.get("terminal_recommendation") == "REVIEW2_READY_FOR_HUMAN_AUTHORIZATION", "sensitivity terminal recommendation drift")
    require(sensitivity.get("proposed_design", {}).get("selected") is True, "no proposed design selected")
    proposal = sensitivity.get("proposed_design", {})
    require(
        (
            proposal.get("block_count"),
            proposal.get("cloud_samples_per_node_per_half"),
            proposal.get("behavior_episodes_per_block"),
        )
        == (20, 24, 24),
        "proposed design drift",
    )
    expected_cost = forward_and_cost_forecast(20, 24, 24)
    observed_cost = proposal.get("cost_forecast", {})
    for key in (
        "map_activation_forwards",
        "behavior_forwards",
        "total_scientific_forwards",
    ):
        require(observed_cost.get(key) == expected_cost[key], f"cost count drift: {key}")
    require(
        math.isclose(
            float(observed_cost.get("planning_upper_usd", math.inf)),
            expected_cost["planning_upper_usd"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "planning upper cost drift",
    )
    downsampling = sensitivity.get("existing_b_fixed_grid_downsampling", {})
    require(downsampling.get("subsample_schedule_sha256") == "78569ffac2a3cc7826482cf3e3f51e77a4a71413ac34baab843fadfd17ff44f3", "downsampling schedule drift")
    require(
        [row.get("samples_per_node_per_half") for row in downsampling.get("grid", [])]
        == list(GRID_CLOUD_SIZES),
        "downsampling grid drift",
    )
    require(
        [row.get("all_layer_gate_pass_count") for row in downsampling.get("grid", [])]
        == [2, 4, 3, 1],
        "downsampling result drift",
    )
    provenance = sensitivity.get("source_provenance", {})
    require(provenance.get("qualification_result_sha256") == "3edb6ce5bedb7c9ec2df93a319b25c8253010460fb14375949ae2869f7bae464", "B qualification provenance drift")
    require(provenance.get("response_or_logit_files_read") is False, "response/logit input recorded")
    budget = float(sensitivity.get("future_modal_budget", {}).get("hard_ceiling_usd", -1.0))
    require(sensitivity.get("future_modal_budget", {}).get("allocation_created") is False, "future budget allocation was created")
    upper_cost = float(sensitivity.get("proposed_design", {}).get("cost_forecast", {}).get("planning_upper_usd", math.inf))
    require(upper_cost <= budget, "proposed design exceeds declared future Modal budget")
    validator_source = (review2_root / "track_c_review2_validator.py").read_text(
        encoding="utf-8"
    )
    require(
        re.search(
            r"^\s*(?:from|import)\s+(?:modal|torch|transformers|huggingface_hub)\b",
            validator_source,
            flags=re.MULTILINE,
        )
        is None,
        "validator contains a forbidden execution-capable import",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "terminal_review2_state": lock.get("terminal_review2_state"),
        "package_files": {
            filename: sha256_file(review2_root / filename)
            for filename in REQUIRED_FILES
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review2-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="candidate package directory",
    )
    parser.add_argument(
        "--downsample-source",
        type=Path,
        help="existing fresh_square_operator directory; never a model directory",
    )
    parser.add_argument("--replicates", type=int, default=24)
    parser.add_argument(
        "--simulate-frontier",
        action="store_true",
        help="print deterministic model-free sensitivity and cost frontier",
    )
    parser.add_argument(
        "--build-sensitivity-and-cost",
        action="store_true",
        help="combine existing-B downsampling with the model-free planning frontier",
    )
    parser.add_argument("--simulations", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.build_sensitivity_and_cost:
        if args.downsample_source is None or args.simulate_frontier:
            raise Review2ValidationError(
                "combined output requires --downsample-source and excludes --simulate-frontier"
            )
        result = build_sensitivity_and_cost_payload(
            args.downsample_source,
            replicates=args.replicates,
            simulations=args.simulations,
        )
    elif args.downsample_source is not None and args.simulate_frontier:
        raise Review2ValidationError("choose downsampling or sensitivity, not both")
    elif args.downsample_source is not None:
        result = run_fixed_grid_downsampling(
            args.downsample_source,
            replicates=args.replicates,
        )
    elif args.simulate_frontier:
        result = simulate_sensitivity_frontier(simulations=args.simulations)
    else:
        result = validate_candidate_package(args.review2_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
