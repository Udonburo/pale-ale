#!/usr/bin/env python3
"""Run Gate12C-1 equal-rank compressed-overlap associator audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import inspect_gate12c_associator_feasibility as gate12c0


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12c_compressed_overlap_associator_v1"
METHOD_ID = "gate12c_compressed_overlap_associator_v1"
RUN_MODE = "gate12a_residual_bearing_explicit_triangle_equal_rank_alpha_v1"
RAW_OVERLAP_MODE = gate12c0.RAW_OVERLAP_MODE
TRANSPORT_RECONSTRUCTION_MODE = gate12c0.TRANSPORT_RECONSTRUCTION_MODE
STABLE_CUT_MODE = gate12c0.STABLE_CUT_MODE
ORDINARY_NULL_MODE = gate12c0.ORDINARY_NULL_MODE
NO_COMPRESSION_NULL_MODE = "same_path_rank_r_compression_null_v1"
GAUGE_MODE = "deterministic_node_signed_permutation_gauge_v1"
ORIENTATION_NULL_MODE = "cycle_shared_spectrum_preserving_operator_null_v1"
ORIENTATION_ORTHOGONAL_GENERATOR = "sha256_counter_box_muller_qr_sign_normalized_v1"
ORIENTATION_SEED_ENCODING = "canonical_json_utf8_no_insignificant_whitespace_v1"

DEFAULT_TAU_OVERLAP_SV_MIN = gate12c0.DEFAULT_TAU_OVERLAP_SV_MIN
DEFAULT_TAU_OVERLAP_SINGULAR_VALUE_ABS_ERROR = gate12c0.DEFAULT_TAU_OVERLAP_SV_ABS_ERROR
DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO = gate12c0.DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO
DEFAULT_TAU_ORDINARY_ASSOCIATOR_FRO = gate12c0.DEFAULT_TAU_ORDINARY_ASSOCIATOR_FRO
DEFAULT_TAU_NO_COMPRESSION_ASSOCIATOR_FRO = 1.0e-10
DEFAULT_TAU_SPLIT_REL = gate12c0.DEFAULT_TAU_SPLIT_REL
DEFAULT_TAU_GAUGE_OPERATOR_COVARIANCE_FRO = 1.0e-8
DEFAULT_TAU_GAUGE_SCALAR_DELTA_ABS = 1.0e-10
DEFAULT_EPSILON = gate12c0.DEFAULT_EPSILON

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_REGISTRY = "triangle_associator_registry.jsonl"
DEFAULT_ARRAYS = "triangle_associator_arrays.npz"
DEFAULT_CYCLE_SUMMARY = "cycle_associator_summary.jsonl"
DEFAULT_COMPRESSION_SWEEP = "compression_sweep_summary.csv"
DEFAULT_GAUGE_SUMMARY = "gauge_stability_summary.json"
DEFAULT_ORIENTATION_NULL_SUMMARY = "spectral_orientation_null_summary.jsonl"
DEFAULT_STATUS = "gate12c_status.json"
DEFAULT_READ = "gate12c_read.md"
DEFAULT_CHECKSUMS = "checksums.json"

OUTPUT_FILES = (
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    DEFAULT_ARRAYS,
    DEFAULT_CYCLE_SUMMARY,
    DEFAULT_COMPRESSION_SWEEP,
    DEFAULT_GAUGE_SUMMARY,
    DEFAULT_ORIENTATION_NULL_SUMMARY,
    DEFAULT_STATUS,
    DEFAULT_READ,
)

REGISTRY_FIELDNAMES = (
    "probe_id",
    "cycle_id",
    "canonical_base_node_id",
    "evaluation_root_node_id",
    "root_rotation_index",
    "ordered_node_id_path",
    "ordered_edge_id_path",
    "ordered_relation_kind_path",
    "cycle_rank",
    "compression_rank_q",
    "left_inner_split_gap_rel",
    "right_inner_split_gap_rel",
    "left_cut_status",
    "right_cut_status",
    "truncation_status",
    "ordinary_associator_fro",
    "no_compression_associator_fro",
    "compressed_overlap_associator_fro",
    "compressed_overlap_associator_rel",
    "compressed_overlap_closure_left_fro",
    "compressed_overlap_closure_right_fro",
    "compressed_overlap_closure_gap_abs",
    "gate12a_holonomy_residual_fro",
    "edge_compatibility_gap_max",
    "source_sample_block_id",
    "source_block_status",
    "measurement_status",
    "control_status",
    "aggregation_eligible",
    "gauge_operator_covariance_fro",
    "gauge_scalar_delta_abs",
    "gauge_cut_status_preserved",
    "gauge_scalar_status",
    "orientation_null_status",
    "orientation_null_excess_status",
    "orientation_null_requested_draw_count",
    "orientation_null_valid_draw_count",
    "orientation_null_invalid_cut_count",
    "orientation_null_attempt_count",
    "orientation_null_median",
    "orientation_null_mad",
    "orientation_null_mean",
    "orientation_null_std",
    "orientation_null_empirical_p_upper",
    "orientation_null_robust_z",
    "orientation_null_scale_degenerate",
    "operator_array_index",
)

CYCLE_SUMMARY_FIELDNAMES = (
    "cycle_id",
    "cycle_rank",
    "eligible_root_q_count",
    "stable_both_active_count",
    "compressed_overlap_associator_root_rms",
    "compressed_overlap_associator_root_max",
    "compressed_overlap_associator_root_spread",
    "ordinary_associator_max_fro",
    "no_compression_associator_max_fro",
    "gauge_stable_row_count",
    "orientation_null_complete_row_count",
    "aggregation_eligible_row_count",
)

COMPRESSION_SWEEP_FIELDNAMES = (
    "compression_rank_q",
    "row_count",
    "measured_row_count",
    "aggregation_eligible_row_count",
    "compressed_overlap_associator_fro_mean",
    "compressed_overlap_associator_fro_max",
)

MEASUREMENT_NOT_EVALUATED = "not_evaluated"
MEASUREMENT_MEASURED = "measured"
MEASUREMENT_INVALID_INPUT = "invalid_input"

CONTROL_NOT_EVALUATED = "not_evaluated"
CONTROL_PASS = "pass"
CONTROL_FAIL = "fail"
CONTROL_INCOMPLETE = "incomplete"

ORIENTATION_NULL_NOT_EVALUATED = "not_evaluated"
ORIENTATION_NULL_COMPLETE = "complete"
ORIENTATION_NULL_INSUFFICIENT = "insufficient_valid_draws"
ORIENTATION_NULL_INVALID_INPUT = "invalid_input"

ORIENTATION_EXCESS_NOT_EVALUATED = "not_evaluated"
ORIENTATION_EXCESS_DESCRIPTIVE = "descriptive_only"
ORIENTATION_EXCESS_SCALE_DEGENERATE = "scale_degenerate"

GAUGE_SCALAR_PASS = "pass"
GAUGE_SCALAR_FAIL = "fail"
GAUGE_SCALAR_NOT_COMPARABLE = "not_comparable"
GAUGE_SCALAR_NOT_EVALUATED = "not_evaluated"

TRUNCATION_STABLE_BOTH = "stable_both_active"
TRUNCATION_NEAR_LEFT = "near_degenerate_left"
TRUNCATION_NEAR_RIGHT = "near_degenerate_right"
TRUNCATION_NEAR_BOTH = "near_degenerate_both"
TRUNCATION_INACTIVE = "compression_inactive"
TRUNCATION_UNDEFINED = "undefined_input"


class Gate12CContractError(RuntimeError):
    """Raised when a Gate12C-1 contract or implementation failure is detected."""


@dataclass(frozen=True)
class Tolerances:
    tau_overlap_sv_min: float = DEFAULT_TAU_OVERLAP_SV_MIN
    tau_overlap_singular_value_abs_error: float = DEFAULT_TAU_OVERLAP_SINGULAR_VALUE_ABS_ERROR
    tau_transport_reconstruction_fro: float = DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO
    tau_ordinary_associator_fro: float = DEFAULT_TAU_ORDINARY_ASSOCIATOR_FRO
    tau_no_compression_associator_fro: float = DEFAULT_TAU_NO_COMPRESSION_ASSOCIATOR_FRO
    tau_split_rel: float = DEFAULT_TAU_SPLIT_REL
    tau_gauge_operator_covariance_fro: float = DEFAULT_TAU_GAUGE_OPERATOR_COVARIANCE_FRO
    tau_gauge_scalar_delta_abs: float = DEFAULT_TAU_GAUGE_SCALAR_DELTA_ABS
    epsilon: float = DEFAULT_EPSILON

    def as_dict(self) -> Dict[str, float]:
        return {
            "tau_overlap_sv_min": float(self.tau_overlap_sv_min),
            "tau_overlap_singular_value_abs_error": float(
                self.tau_overlap_singular_value_abs_error
            ),
            "tau_transport_reconstruction_fro": float(self.tau_transport_reconstruction_fro),
            "tau_ordinary_associator_fro": float(self.tau_ordinary_associator_fro),
            "tau_no_compression_associator_fro": float(
                self.tau_no_compression_associator_fro
            ),
            "tau_split_rel": float(self.tau_split_rel),
            "tau_gauge_operator_covariance_fro": float(
                self.tau_gauge_operator_covariance_fro
            ),
            "tau_gauge_scalar_delta_abs": float(self.tau_gauge_scalar_delta_abs),
            "epsilon": float(self.epsilon),
        }


@dataclass
class NullAccumulator:
    requested_draw_count: int
    valid_values: List[float]
    invalid_cut_count: int = 0
    attempt_count: int = 0

    @property
    def complete(self) -> bool:
        return len(self.valid_values) >= self.requested_draw_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gate12C-1 equal-rank compressed-overlap associator audit "
            "over Gate12A artifacts."
        )
    )
    parser.add_argument("--gate12a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--orientation-null-seed", required=True)
    parser.add_argument("--orientation-null-requested-draw-count", required=True, type=int)
    parser.add_argument("--orientation-null-max-attempt-count", required=True, type=int)
    parser.add_argument("--tau-overlap-sv-min", type=float, default=DEFAULT_TAU_OVERLAP_SV_MIN)
    parser.add_argument(
        "--tau-overlap-singular-value-abs-error",
        type=float,
        default=DEFAULT_TAU_OVERLAP_SINGULAR_VALUE_ABS_ERROR,
    )
    parser.add_argument(
        "--tau-transport-reconstruction-fro",
        type=float,
        default=DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO,
    )
    parser.add_argument(
        "--tau-ordinary-associator-fro",
        type=float,
        default=DEFAULT_TAU_ORDINARY_ASSOCIATOR_FRO,
    )
    parser.add_argument(
        "--tau-no-compression-associator-fro",
        type=float,
        default=DEFAULT_TAU_NO_COMPRESSION_ASSOCIATOR_FRO,
    )
    parser.add_argument("--tau-split-rel", type=float, default=DEFAULT_TAU_SPLIT_REL)
    parser.add_argument(
        "--tau-gauge-operator-covariance-fro",
        type=float,
        default=DEFAULT_TAU_GAUGE_OPERATOR_COVARIANCE_FRO,
    )
    parser.add_argument(
        "--tau-gauge-scalar-delta-abs",
        type=float,
        default=DEFAULT_TAU_GAUGE_SCALAR_DELTA_ABS,
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser.parse_args(argv)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return json_ready(float(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON outputs must not contain NaN or infinity")
        return value
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_ready(dict(row)), ensure_ascii=False, allow_nan=False) + "\n"
            )


def csv_value(value: Any) -> Any:
    ready = json_ready(value)
    if isinstance(ready, (dict, list, tuple)):
        return json.dumps(ready, ensure_ascii=False, sort_keys=True)
    if ready is None:
        return ""
    return ready


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def current_git_commit() -> str:
    return gate12c0.current_git_commit()


def required_source_checksums(gate12a_dir: Path) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for name in gate12c0.REQUIRED_FILES:
        path = gate12a_dir / name
        if path.exists():
            checksums[name] = sha256_file(path)
    return checksums


def verify_source_unchanged(gate12a_dir: Path, before: Mapping[str, str]) -> None:
    after = required_source_checksums(gate12a_dir)
    if dict(before) != after:
        raise Gate12CContractError("Gate12A source artifact files changed during Gate12C-1 run")


def canonical_orientation_seed_bytes(
    *,
    orientation_null_seed: str,
    cycle_id: str,
    edge_id: str,
    draw_index: int,
    left_or_right_orientation_label: str,
) -> bytes:
    payload = [
        "gate12c1_orientation_null_v1",
        str(orientation_null_seed),
        str(cycle_id),
        str(edge_id),
        int(draw_index),
        str(left_or_right_orientation_label),
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_counter_stream(key: bytes, byte_count: int) -> bytes:
    chunks: List[bytes] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < int(byte_count):
        chunks.append(hashlib.sha256(key + counter.to_bytes(8, "big")).digest())
        counter += 1
    return b"".join(chunks)[:byte_count]


def _uniforms_from_counter_stream(key: bytes, count: int) -> List[float]:
    stream = sha256_counter_stream(key, int(count) * 8)
    uniforms: List[float] = []
    denominator = float(1 << 64)
    for index in range(int(count)):
        raw = stream[index * 8 : (index + 1) * 8]
        value = int.from_bytes(raw, "big", signed=False)
        uniforms.append((float(value) + 0.5) / denominator)
    return uniforms


def normal_matrix_from_seed(seed_bytes: bytes, rank: int) -> np.ndarray:
    digest = hashlib.sha256(seed_bytes).digest()
    value_count = int(rank) * int(rank)
    pair_count = (value_count + 1) // 2
    uniforms = _uniforms_from_counter_stream(digest, pair_count * 2)
    normals: List[float] = []
    for pair_index in range(pair_count):
        u1 = max(uniforms[2 * pair_index], np.finfo(float).tiny)
        u2 = uniforms[2 * pair_index + 1]
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        normals.append(radius * math.cos(angle))
        if len(normals) < value_count:
            normals.append(radius * math.sin(angle))
    return np.asarray(normals, dtype=np.float64).reshape((int(rank), int(rank)))


def deterministic_orthogonal_matrix(seed_bytes: bytes, rank: int) -> np.ndarray:
    z_matrix = normal_matrix_from_seed(seed_bytes, int(rank))
    q_matrix, r_matrix = np.linalg.qr(z_matrix)
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return np.asarray(q_matrix * signs, dtype=np.float64)


def orientation_matrix(
    *,
    orientation_null_seed: str,
    cycle_id: str,
    edge_id: str,
    draw_index: int,
    label: str,
    rank: int,
) -> np.ndarray:
    seed_bytes = canonical_orientation_seed_bytes(
        orientation_null_seed=orientation_null_seed,
        cycle_id=cycle_id,
        edge_id=edge_id,
        draw_index=int(draw_index),
        left_or_right_orientation_label=label,
    )
    return deterministic_orthogonal_matrix(seed_bytes, int(rank))


def canonical_gauge_seed_bytes(*, node_id: str, rank: int) -> bytes:
    payload = ["gate12c1_gauge_signed_permutation_v1", str(node_id), int(rank)]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def deterministic_signed_permutation(node_id: str, rank: int) -> np.ndarray:
    key = hashlib.sha256(canonical_gauge_seed_bytes(node_id=node_id, rank=int(rank))).digest()
    stream = sha256_counter_stream(key, int(rank) * 9)
    keyed_indices = []
    for index in range(int(rank)):
        key_value = int.from_bytes(stream[index * 8 : (index + 1) * 8], "big")
        keyed_indices.append((key_value, index))
    permutation = [index for _key, index in sorted(keyed_indices)]
    sign_offset = int(rank) * 8
    matrix = np.zeros((int(rank), int(rank)), dtype=np.float64)
    for col_index, row_index in enumerate(permutation):
        sign = 1.0 if stream[sign_offset + col_index] % 2 == 0 else -1.0
        matrix[row_index, col_index] = sign
    return matrix


def best_rank_approximation(matrix: np.ndarray, q: int) -> np.ndarray:
    u_matrix, singular_values, vt_matrix = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64),
        full_matrices=False,
    )
    if int(q) <= 0 or int(q) > singular_values.shape[0]:
        raise ValueError(f"compression rank q out of range: q={q}")
    return np.asarray(
        u_matrix[:, : int(q)]
        @ np.diag(singular_values[: int(q)])
        @ vt_matrix[: int(q), :],
        dtype=np.float64,
    )


def compressed_overlap_operators(
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    q: int,
    epsilon: float,
) -> Dict[str, Any]:
    left_operator = best_rank_approximation(m2 @ m1, int(q)) @ m0
    right_operator = m2 @ best_rank_approximation(m1 @ m0, int(q))
    associator = np.asarray(left_operator - right_operator, dtype=np.float64)
    assoc_fro = float(np.linalg.norm(associator, ord="fro"))
    left_fro = float(np.linalg.norm(left_operator, ord="fro"))
    right_fro = float(np.linalg.norm(right_operator, ord="fro"))
    rel = assoc_fro / (math.sqrt(2.0 * (left_fro**2 + right_fro**2)) + float(epsilon))
    identity = np.eye(left_operator.shape[0], dtype=np.float64)
    closure_left = float(np.linalg.norm(left_operator - identity, ord="fro"))
    closure_right = float(np.linalg.norm(right_operator - identity, ord="fro"))
    return {
        "left_operator": np.asarray(left_operator, dtype=np.float64),
        "right_operator": np.asarray(right_operator, dtype=np.float64),
        "associator_operator": associator,
        "associator_fro": float(assoc_fro),
        "associator_rel": float(rel),
        "closure_left_fro": float(closure_left),
        "closure_right_fro": float(closure_right),
        "closure_gap_abs": float(abs(closure_left - closure_right)),
    }


def ordinary_associator_fro(m0: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> float:
    return float(np.linalg.norm((m2 @ m1) @ m0 - m2 @ (m1 @ m0), ord="fro"))


def split_gap_and_status(matrix: np.ndarray, *, q: int, tolerances: Tolerances) -> Tuple[float, str]:
    gap = gate12c0.split_gap_rel(matrix, q=int(q), epsilon=float(tolerances.epsilon))
    status = "stable" if gap > float(tolerances.tau_split_rel) else "near_degenerate"
    return float(gap), status


def truncation_status(left_status: str, right_status: str) -> str:
    left_stable = left_status == "stable"
    right_stable = right_status == "stable"
    if left_stable and right_stable:
        return TRUNCATION_STABLE_BOTH
    if left_stable and not right_stable:
        return TRUNCATION_NEAR_RIGHT
    if not left_stable and right_stable:
        return TRUNCATION_NEAR_LEFT
    return TRUNCATION_NEAR_BOTH


def pad_operator(operator: np.ndarray | None, *, rank: int, r_max: int) -> np.ndarray:
    padded = np.zeros((int(r_max), int(r_max)), dtype=np.float64)
    if operator is not None:
        padded[: int(rank), : int(rank)] = np.asarray(operator, dtype=np.float64)
    return padded


def edge_compatibility_gap_max(edges: Sequence[Mapping[str, Any]]) -> float:
    values: List[float] = []
    for edge in edges:
        raw = edge.get("compatibility_gap_fro")
        if raw is not None:
            values.append(float(raw))
    return float(max(values) if values else 0.0)


def source_block_status(node_ids: Sequence[str]) -> Tuple[str, str]:
    sample_ids: List[str] = []
    for node_id in node_ids:
        match = re.match(r"^(sample_\d{6})(?:$|[:/_-])", str(node_id))
        if not match:
            return "mixed_or_undefined", "mixed_or_undefined"
        sample_ids.append(match.group(1))
    if len(set(sample_ids)) == 1:
        return sample_ids[0], "single_sample"
    return "mixed_or_undefined", "mixed_or_undefined"


def null_edge_matrix(
    *,
    singular_values: np.ndarray,
    orientation_null_seed: str,
    cycle_id: str,
    edge_id: str,
    draw_index: int,
    rank: int,
) -> np.ndarray:
    left = orientation_matrix(
        orientation_null_seed=orientation_null_seed,
        cycle_id=cycle_id,
        edge_id=edge_id,
        draw_index=int(draw_index),
        label="left",
        rank=int(rank),
    )
    right = orientation_matrix(
        orientation_null_seed=orientation_null_seed,
        cycle_id=cycle_id,
        edge_id=edge_id,
        draw_index=int(draw_index),
        label="right",
        rank=int(rank),
    )
    sigma = np.diag(np.asarray(singular_values[: int(rank)], dtype=np.float64))
    return np.asarray(left @ sigma @ right.T, dtype=np.float64)


def summarize_null_values(
    *,
    observed_value: float | None,
    accumulator: NullAccumulator,
    epsilon: float,
) -> Dict[str, Any]:
    values = np.asarray(accumulator.valid_values, dtype=np.float64)
    complete = int(values.shape[0]) >= int(accumulator.requested_draw_count)
    status = ORIENTATION_NULL_COMPLETE if complete else ORIENTATION_NULL_INSUFFICIENT
    if values.shape[0] == 0:
        return {
            "orientation_null_status": status,
            "orientation_null_median": None,
            "orientation_null_mad": None,
            "orientation_null_mean": None,
            "orientation_null_std": None,
            "orientation_null_empirical_p_upper": None,
            "orientation_null_robust_z": None,
            "orientation_null_scale_degenerate": None,
            "orientation_null_excess_status": ORIENTATION_EXCESS_NOT_EVALUATED,
        }

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    mean = float(np.mean(values))
    std = float(math.sqrt(float(np.mean((values - mean) ** 2))))
    scale_degenerate = bool(mad <= float(epsilon))

    p_upper = None
    robust_z = None
    excess_status = ORIENTATION_EXCESS_NOT_EVALUATED
    if complete and observed_value is not None:
        p_upper = float(
            (1 + int(np.sum(values >= float(observed_value)))) / (1 + int(values.shape[0]))
        )
        if scale_degenerate:
            robust_z = None
            excess_status = ORIENTATION_EXCESS_SCALE_DEGENERATE
        else:
            robust_z = float((float(observed_value) - median) / (1.4826 * mad + float(epsilon)))
            excess_status = ORIENTATION_EXCESS_DESCRIPTIVE

    return {
        "orientation_null_status": status,
        "orientation_null_median": median,
        "orientation_null_mad": mad,
        "orientation_null_mean": mean,
        "orientation_null_std": std,
        "orientation_null_empirical_p_upper": p_upper,
        "orientation_null_robust_z": robust_z,
        "orientation_null_scale_degenerate": scale_degenerate,
        "orientation_null_excess_status": excess_status,
    }


def orientation_null_accumulators_for_cycle(
    *,
    cycle_id: str,
    ordered_edges: Sequence[Mapping[str, Any]],
    edge_reconstructions: Mapping[str, gate12c0.EdgeReconstruction],
    rank: int,
    rows: Sequence[Dict[str, Any]],
    orientation_null_seed: str,
    requested_draw_count: int,
    max_attempt_count: int,
    tolerances: Tolerances,
) -> Dict[int, NullAccumulator]:
    row_accumulators = {
        int(row["operator_array_index"]): NullAccumulator(
            requested_draw_count=int(requested_draw_count),
            valid_values=[],
        )
        for row in rows
    }
    if not rows:
        return row_accumulators

    canonical_edge_ids = [str(edge["edge_id"]) for edge in ordered_edges]
    canonical_singular_values = {
        edge_id: np.asarray(
            edge_reconstructions[edge_id].reconstructed_singular_values[: int(rank)],
            dtype=np.float64,
        )
        for edge_id in canonical_edge_ids
    }
    rows_by_root_q = {
        (int(row["root_rotation_index"]), int(row["compression_rank_q"])): row for row in rows
    }

    for draw_index in range(int(max_attempt_count)):
        if all(acc.complete for acc in row_accumulators.values()):
            break
        null_edges = {
            edge_id: null_edge_matrix(
                singular_values=canonical_singular_values[edge_id],
                orientation_null_seed=orientation_null_seed,
                cycle_id=cycle_id,
                edge_id=edge_id,
                draw_index=draw_index,
                rank=int(rank),
            )
            for edge_id in canonical_edge_ids
        }
        canonical_matrices = [null_edges[edge_id] for edge_id in canonical_edge_ids]
        for root_rotation_index in range(3):
            root_matrices = gate12c0.rotate_three(canonical_matrices, root_rotation_index)
            m0, m1, m2 = root_matrices
            left_inner = m2 @ m1
            right_inner = m1 @ m0
            for q in range(1, int(rank)):
                row = rows_by_root_q.get((root_rotation_index, q))
                if row is None:
                    continue
                row_index = int(row["operator_array_index"])
                accumulator = row_accumulators[row_index]
                if accumulator.complete:
                    continue
                accumulator.attempt_count = int(draw_index) + 1
                left_gap, left_status = split_gap_and_status(
                    left_inner,
                    q=q,
                    tolerances=tolerances,
                )
                right_gap, right_status = split_gap_and_status(
                    right_inner,
                    q=q,
                    tolerances=tolerances,
                )
                if left_status != "stable" or right_status != "stable":
                    accumulator.invalid_cut_count += 1
                    continue
                measurement = compressed_overlap_operators(
                    m0,
                    m1,
                    m2,
                    q=q,
                    epsilon=float(tolerances.epsilon),
                )
                accumulator.valid_values.append(float(measurement["associator_fro"]))
    for accumulator in row_accumulators.values():
        if accumulator.attempt_count == 0:
            accumulator.attempt_count = int(max_attempt_count)
    return row_accumulators


def build_status_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def counts_for(key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            value = str(row.get(key))
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "row_count": int(len(rows)),
        "measurement_status_counts": counts_for("measurement_status"),
        "control_status_counts": counts_for("control_status"),
        "aggregation_eligible_row_count": int(
            sum(1 for row in rows if bool(row.get("aggregation_eligible")))
        ),
        "orientation_null_status_counts": counts_for("orientation_null_status"),
        "orientation_null_excess_status_counts": counts_for("orientation_null_excess_status"),
    }


def build_cycle_summaries(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cycle_id"]), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for cycle_id in sorted(grouped):
        cycle_rows = grouped[cycle_id]
        measured_values = [
            float(row["compressed_overlap_associator_fro"])
            for row in cycle_rows
            if row.get("compressed_overlap_associator_fro") is not None
        ]
        rms = None
        maximum = None
        spread = None
        if measured_values:
            arr = np.asarray(measured_values, dtype=np.float64)
            rms = float(math.sqrt(float(np.mean(arr**2))))
            maximum = float(np.max(arr))
            spread = float(np.max(arr) - np.min(arr))
        summaries.append(
            {
                "cycle_id": cycle_id,
                "cycle_rank": int(cycle_rows[0]["cycle_rank"]),
                "eligible_root_q_count": int(len(cycle_rows)),
                "stable_both_active_count": int(
                    sum(1 for row in cycle_rows if row["truncation_status"] == TRUNCATION_STABLE_BOTH)
                ),
                "compressed_overlap_associator_root_rms": rms,
                "compressed_overlap_associator_root_max": maximum,
                "compressed_overlap_associator_root_spread": spread,
                "ordinary_associator_max_fro": float(
                    max(float(row["ordinary_associator_fro"]) for row in cycle_rows)
                ),
                "no_compression_associator_max_fro": float(
                    max(float(row["no_compression_associator_fro"]) for row in cycle_rows)
                ),
                "gauge_stable_row_count": int(
                    sum(1 for row in cycle_rows if row["gauge_scalar_status"] == GAUGE_SCALAR_PASS)
                ),
                "orientation_null_complete_row_count": int(
                    sum(
                        1
                        for row in cycle_rows
                        if row["orientation_null_status"] == ORIENTATION_NULL_COMPLETE
                    )
                ),
                "aggregation_eligible_row_count": int(
                    sum(1 for row in cycle_rows if bool(row["aggregation_eligible"]))
                ),
            }
        )
    return summaries


def build_compression_sweep(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_q: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_q.setdefault(int(row["compression_rank_q"]), []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for q in sorted(by_q):
        q_rows = by_q[q]
        measured = [
            float(row["compressed_overlap_associator_fro"])
            for row in q_rows
            if row.get("compressed_overlap_associator_fro") is not None
        ]
        summary_rows.append(
            {
                "compression_rank_q": int(q),
                "row_count": int(len(q_rows)),
                "measured_row_count": int(len(measured)),
                "aggregation_eligible_row_count": int(
                    sum(1 for row in q_rows if bool(row.get("aggregation_eligible")))
                ),
                "compressed_overlap_associator_fro_mean": float(np.mean(measured))
                if measured
                else None,
                "compressed_overlap_associator_fro_max": float(np.max(measured))
                if measured
                else None,
            }
        )
    return summary_rows


def build_gauge_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    comparable = [
        row
        for row in rows
        if row.get("gauge_operator_covariance_fro") is not None
        and row.get("gauge_scalar_delta_abs") is not None
    ]
    return {
        "gauge_mode": GAUGE_MODE,
        "row_count": int(len(rows)),
        "comparable_row_count": int(len(comparable)),
        "gauge_cut_status_preserved_count": int(
            sum(1 for row in rows if bool(row.get("gauge_cut_status_preserved")))
        ),
        "gauge_scalar_pass_count": int(
            sum(1 for row in rows if row.get("gauge_scalar_status") == GAUGE_SCALAR_PASS)
        ),
        "gauge_scalar_fail_count": int(
            sum(1 for row in rows if row.get("gauge_scalar_status") == GAUGE_SCALAR_FAIL)
        ),
        "gauge_operator_covariance_fro_max": float(
            max(float(row["gauge_operator_covariance_fro"]) for row in comparable)
        )
        if comparable
        else None,
        "gauge_scalar_delta_abs_max": float(
            max(float(row["gauge_scalar_delta_abs"]) for row in comparable)
        )
        if comparable
        else None,
    }


def build_readme(*, status: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Gate12C-1 Compressed-Overlap Associator Read",
            "",
            "This is an implementation-only Gate12C-1 artifact read.",
            "",
            "It measures compressed-overlap parenthesization sensitivity on Gate12A-defined "
            "residual-bearing explicit equal-rank triangles.",
            "",
            "It does not run model inference, regenerate Gate12A artifacts, consume Gate12B "
            "overlays, add rectangular support, emit physical-claim terminology, or define a "
            "scientific null-excess threshold.",
            "",
            "## Status",
            "",
            f"- process_status: `{status['process_status']}`",
            f"- row_count: `{status['counts']['row_count']}`",
            f"- aggregation_eligible_row_count: "
            f"`{status['counts']['aggregation_eligible_row_count']}`",
            "",
            "## Determinism",
            "",
            f"- orientation_null_mode: `{manifest['orientation_null_mode']}`",
            f"- orientation_null_orthogonal_generator: "
            f"`{manifest['orientation_null_orthogonal_generator']}`",
            f"- orientation_seed_encoding: `{manifest['orientation_seed_encoding']}`",
            f"- gauge_mode: `{manifest['gauge_mode']}`",
            "",
            "Row-level null p-values and robust z-scores are descriptive telemetry only.",
        ]
    ) + "\n"


def build_checksums(out_dir: Path) -> Dict[str, str]:
    return {name: sha256_file(out_dir / name) for name in OUTPUT_FILES}


def write_failure_status(out_dir: Path, exc: BaseException) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / DEFAULT_STATUS,
            {
                "schema_version": SCHEMA_VERSION,
                "method_id": METHOD_ID,
                "process_status": "fail",
                "failure_type": exc.__class__.__name__,
                "failure_message": str(exc),
            },
        )
    except Exception:
        return


def run_gate12c_compressed_overlap_associator(
    *,
    gate12a_dir: Path,
    out_dir: Path,
    orientation_null_seed: str,
    orientation_null_requested_draw_count: int,
    orientation_null_max_attempt_count: int,
    tolerances: Tolerances | None = None,
) -> Dict[str, Any]:
    tolerances = tolerances or Tolerances()
    if int(orientation_null_requested_draw_count) <= 0:
        raise ValueError("orientation_null_requested_draw_count must be > 0")
    if int(orientation_null_max_attempt_count) < int(orientation_null_requested_draw_count):
        raise ValueError(
            "orientation_null_max_attempt_count must be >= "
            "orientation_null_requested_draw_count"
        )

    gate12a_dir = Path(gate12a_dir)
    out_dir = Path(out_dir)
    gate12c0.validate_output_directory(gate12a_dir=gate12a_dir, out_dir=out_dir)
    source_before = required_source_checksums(gate12a_dir)

    artifacts = gate12c0.load_gate12a_artifacts(gate12a_dir)
    edge_reconstructions, edge_diagnostics = gate12c0.reconstruct_edges(
        artifacts=artifacts,
        tau_overlap_sv_min=float(tolerances.tau_overlap_sv_min),
        tau_overlap_sv_abs_error=float(tolerances.tau_overlap_singular_value_abs_error),
        tau_transport_reconstruction_fro=float(tolerances.tau_transport_reconstruction_fro),
    )
    if int(edge_diagnostics["failed_edge_reconstruction_count"]) > 0:
        raise Gate12CContractError("Gate12C-1 edge reconstruction validation failed")

    rows: List[Dict[str, Any]] = []
    left_arrays: List[np.ndarray] = []
    right_arrays: List[np.ndarray] = []
    associator_arrays: List[np.ndarray] = []
    cycle_contexts: Dict[str, Dict[str, Any]] = {}

    sorted_cycles = sorted(artifacts.cycle_rows, key=lambda row: str(row.get("cycle_id") or ""))
    for cycle in sorted_cycles:
        gate12c0.require_keys(
            cycle,
            ("cycle_id", "base_node_id", "edge_id_path", "node_id_path"),
            "explicit_triangle_cycle_registry row",
        )
        cycle_id = str(cycle["cycle_id"])
        ordered_edges = gate12c0.reconstruct_ordered_edges(cycle=cycle, edge_map=artifacts.edge_map)
        relation_path = [str(edge["relation_kind"]) for edge in ordered_edges]
        if sum(1 for kind in relation_path if kind == "residual_chord") <= 0:
            continue

        holonomy = artifacts.holonomy_map.get(cycle_id, {})
        if str(holonomy.get("holonomy_status") or "missing") != "defined":
            continue
        node_ids = [str(node_id) for node_id in list(cycle["node_id_path"])[:3]]
        if any(node_id not in artifacts.node_map for node_id in node_ids):
            continue
        node_ranks = [int(artifacts.node_map[node_id].projector_rank) for node_id in node_ids]
        common_rank = node_ranks[0] if len(set(node_ranks)) == 1 else 0
        if common_rank < 2:
            continue
        transport_cases = [str(edge["transport_case"]) for edge in ordered_edges]
        if any(case != "equal_rank_orthogonal" for case in transport_cases):
            continue

        ordered_edge_ids = [str(edge["edge_id"]) for edge in ordered_edges]
        edge_matrices: List[np.ndarray] = []
        for edge_id in ordered_edge_ids:
            reconstruction = edge_reconstructions[edge_id]
            if reconstruction.overlap_matrix is None:
                raise Gate12CContractError(
                    f"eligible cycle {cycle_id} has undefined reconstructed overlap {edge_id}"
                )
            matrix = np.asarray(reconstruction.overlap_matrix, dtype=np.float64)
            if matrix.shape != (common_rank, common_rank):
                raise Gate12CContractError(
                    f"eligible cycle {cycle_id} edge {edge_id} overlap shape {matrix.shape} "
                    f"does not match common rank {common_rank}"
                )
            edge_matrices.append(matrix)

        source_sample_block_id, source_block = source_block_status(node_ids)
        cycle_rows: List[Dict[str, Any]] = []
        cycle_contexts[cycle_id] = {
            "ordered_edges": ordered_edges,
            "rank": int(common_rank),
            "row_indices": [],
        }

        for root_rotation_index in range(3):
            root_nodes = gate12c0.rotate_three(node_ids, root_rotation_index)
            root_edges = gate12c0.rotate_three(ordered_edges, root_rotation_index)
            root_edge_ids = [str(edge["edge_id"]) for edge in root_edges]
            root_relation_path = [str(edge["relation_kind"]) for edge in root_edges]
            matrices = gate12c0.rotate_three(edge_matrices, root_rotation_index)
            m0, m1, m2 = matrices
            ordinary_fro = ordinary_associator_fro(m0, m1, m2)
            if ordinary_fro > float(tolerances.tau_ordinary_associator_fro):
                raise Gate12CContractError(
                    f"ordinary associativity null failed for cycle {cycle_id}"
                )
            no_compression = compressed_overlap_operators(
                m0,
                m1,
                m2,
                q=int(common_rank),
                epsilon=float(tolerances.epsilon),
            )
            no_compression_fro = float(no_compression["associator_fro"])
            if no_compression_fro > float(tolerances.tau_no_compression_associator_fro):
                raise Gate12CContractError(
                    f"no-compression null failed for cycle {cycle_id}"
                )

            gauges = {
                node_id: deterministic_signed_permutation(node_id, int(common_rank))
                for node_id in node_ids
            }
            transformed_matrices = []
            for matrix, edge in zip(matrices, root_edges):
                source_id = str(edge["source_node_id"])
                target_id = str(edge["target_node_id"])
                transformed_matrices.append(gauges[target_id].T @ matrix @ gauges[source_id])
            gm0, gm1, gm2 = transformed_matrices

            for q in range(1, int(common_rank)):
                left_inner = m2 @ m1
                right_inner = m1 @ m0
                left_gap, left_status = split_gap_and_status(
                    left_inner,
                    q=q,
                    tolerances=tolerances,
                )
                right_gap, right_status = split_gap_and_status(
                    right_inner,
                    q=q,
                    tolerances=tolerances,
                )
                trunc_status = truncation_status(left_status, right_status)
                both_stable = trunc_status == TRUNCATION_STABLE_BOTH

                measurement = None
                if both_stable:
                    measurement = compressed_overlap_operators(
                        m0,
                        m1,
                        m2,
                        q=q,
                        epsilon=float(tolerances.epsilon),
                    )

                gauge_left_gap, gauge_left_status = split_gap_and_status(
                    gm2 @ gm1,
                    q=q,
                    tolerances=tolerances,
                )
                gauge_right_gap, gauge_right_status = split_gap_and_status(
                    gm1 @ gm0,
                    q=q,
                    tolerances=tolerances,
                )
                gauge_both_stable = (
                    gauge_left_status == "stable" and gauge_right_status == "stable"
                )
                gauge_cut_preserved = bool(both_stable and gauge_both_stable)
                gauge_operator_delta = None
                gauge_scalar_delta = None
                gauge_scalar_status = (
                    GAUGE_SCALAR_NOT_COMPARABLE if both_stable else GAUGE_SCALAR_NOT_EVALUATED
                )
                if measurement is not None and gauge_both_stable:
                    transformed_measurement = compressed_overlap_operators(
                        gm0,
                        gm1,
                        gm2,
                        q=q,
                        epsilon=float(tolerances.epsilon),
                    )
                    root_gauge = gauges[root_nodes[0]]
                    expected = root_gauge.T @ measurement["associator_operator"] @ root_gauge
                    gauge_operator_delta = float(
                        np.linalg.norm(
                            transformed_measurement["associator_operator"] - expected,
                            ord="fro",
                        )
                    )
                    gauge_scalar_delta = float(
                        abs(
                            float(transformed_measurement["associator_fro"])
                            - float(measurement["associator_fro"])
                        )
                    )
                    gauge_scalar_status = GAUGE_SCALAR_PASS
                    if (
                        gauge_operator_delta
                        > float(tolerances.tau_gauge_operator_covariance_fro)
                        or gauge_scalar_delta > float(tolerances.tau_gauge_scalar_delta_abs)
                    ):
                        gauge_scalar_status = GAUGE_SCALAR_FAIL
                        raise Gate12CContractError(
                            f"gauge covariance failed for cycle {cycle_id}"
                        )

                row_index = len(rows)
                row = {
                    "probe_id": f"gate12c1_probe:{row_index:06d}",
                    "cycle_id": cycle_id,
                    "canonical_base_node_id": str(cycle["base_node_id"]),
                    "evaluation_root_node_id": root_nodes[0],
                    "root_rotation_index": int(root_rotation_index),
                    "ordered_node_id_path": root_nodes + [root_nodes[0]],
                    "ordered_edge_id_path": root_edge_ids,
                    "ordered_relation_kind_path": root_relation_path,
                    "cycle_rank": int(common_rank),
                    "compression_rank_q": int(q),
                    "left_inner_split_gap_rel": float(left_gap),
                    "right_inner_split_gap_rel": float(right_gap),
                    "left_cut_status": left_status,
                    "right_cut_status": right_status,
                    "truncation_status": trunc_status,
                    "ordinary_associator_fro": float(ordinary_fro),
                    "no_compression_associator_fro": float(no_compression_fro),
                    "compressed_overlap_associator_fro": float(measurement["associator_fro"])
                    if measurement is not None
                    else None,
                    "compressed_overlap_associator_rel": float(measurement["associator_rel"])
                    if measurement is not None
                    else None,
                    "compressed_overlap_closure_left_fro": float(
                        measurement["closure_left_fro"]
                    )
                    if measurement is not None
                    else None,
                    "compressed_overlap_closure_right_fro": float(
                        measurement["closure_right_fro"]
                    )
                    if measurement is not None
                    else None,
                    "compressed_overlap_closure_gap_abs": float(
                        measurement["closure_gap_abs"]
                    )
                    if measurement is not None
                    else None,
                    "gate12a_holonomy_residual_fro": float(
                        holonomy.get("holonomy_residual_fro", 0.0)
                    ),
                    "edge_compatibility_gap_max": edge_compatibility_gap_max(root_edges),
                    "source_sample_block_id": source_sample_block_id,
                    "source_block_status": source_block,
                    "measurement_status": MEASUREMENT_MEASURED
                    if measurement is not None
                    else MEASUREMENT_NOT_EVALUATED,
                    "control_status": CONTROL_INCOMPLETE,
                    "aggregation_eligible": False,
                    "gauge_operator_covariance_fro": gauge_operator_delta,
                    "gauge_scalar_delta_abs": gauge_scalar_delta,
                    "gauge_cut_status_preserved": bool(gauge_cut_preserved),
                    "gauge_scalar_status": gauge_scalar_status,
                    "orientation_null_status": ORIENTATION_NULL_NOT_EVALUATED,
                    "orientation_null_excess_status": ORIENTATION_EXCESS_NOT_EVALUATED,
                    "orientation_null_requested_draw_count": int(
                        orientation_null_requested_draw_count
                    ),
                    "orientation_null_valid_draw_count": 0,
                    "orientation_null_invalid_cut_count": 0,
                    "orientation_null_attempt_count": 0,
                    "orientation_null_median": None,
                    "orientation_null_mad": None,
                    "orientation_null_mean": None,
                    "orientation_null_std": None,
                    "orientation_null_empirical_p_upper": None,
                    "orientation_null_robust_z": None,
                    "orientation_null_scale_degenerate": None,
                    "operator_array_index": int(row_index),
                }
                rows.append(row)
                cycle_rows.append(row)
                cycle_contexts[cycle_id]["row_indices"].append(row_index)
                left_arrays.append(
                    pad_operator(
                        measurement["left_operator"] if measurement is not None else None,
                        rank=int(common_rank),
                        r_max=int(artifacts.r_max),
                    )
                )
                right_arrays.append(
                    pad_operator(
                        measurement["right_operator"] if measurement is not None else None,
                        rank=int(common_rank),
                        r_max=int(artifacts.r_max),
                    )
                )
                associator_arrays.append(
                    pad_operator(
                        measurement["associator_operator"] if measurement is not None else None,
                        rank=int(common_rank),
                        r_max=int(artifacts.r_max),
                    )
                )

        cycle_contexts[cycle_id]["rows"] = cycle_rows

    for cycle_id in sorted(cycle_contexts):
        context = cycle_contexts[cycle_id]
        cycle_rows = [rows[index] for index in context["row_indices"]]
        accumulators = orientation_null_accumulators_for_cycle(
            cycle_id=cycle_id,
            ordered_edges=context["ordered_edges"],
            edge_reconstructions=edge_reconstructions,
            rank=int(context["rank"]),
            rows=cycle_rows,
            orientation_null_seed=str(orientation_null_seed),
            requested_draw_count=int(orientation_null_requested_draw_count),
            max_attempt_count=int(orientation_null_max_attempt_count),
            tolerances=tolerances,
        )
        for row_index, accumulator in accumulators.items():
            row = rows[int(row_index)]
            observed = row["compressed_overlap_associator_fro"]
            summary = summarize_null_values(
                observed_value=float(observed) if observed is not None else None,
                accumulator=accumulator,
                epsilon=float(tolerances.epsilon),
            )
            row.update(summary)
            row["orientation_null_valid_draw_count"] = int(len(accumulator.valid_values))
            row["orientation_null_invalid_cut_count"] = int(accumulator.invalid_cut_count)
            row["orientation_null_attempt_count"] = int(accumulator.attempt_count)
            row["aggregation_eligible"] = bool(
                row["measurement_status"] == MEASUREMENT_MEASURED
                and row["truncation_status"] == TRUNCATION_STABLE_BOTH
                and row["gauge_cut_status_preserved"]
                and row["gauge_scalar_status"] == GAUGE_SCALAR_PASS
                and row["orientation_null_status"] == ORIENTATION_NULL_COMPLETE
            )
            row["control_status"] = (
                CONTROL_PASS if row["aggregation_eligible"] else CONTROL_INCOMPLETE
            )

    status_counts = build_status_counts(rows)
    status_payload = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "process_status": "pass",
        "counts": status_counts,
        "measurement_counts": {
            "row_count": status_counts["row_count"],
            "measured_row_count": status_counts["measurement_status_counts"].get(
                MEASUREMENT_MEASURED,
                0,
            ),
        },
        "control_counts": status_counts["control_status_counts"],
        "aggregation_eligibility_counts": {
            "aggregation_eligible_row_count": status_counts["aggregation_eligible_row_count"]
        },
        "orientation_null_descriptive_statuses": status_counts[
            "orientation_null_excess_status_counts"
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "run_mode": RUN_MODE,
        "raw_overlap_mode": RAW_OVERLAP_MODE,
        "transport_reconstruction_mode": TRANSPORT_RECONSTRUCTION_MODE,
        "stable_cut_mode": STABLE_CUT_MODE,
        "ordinary_null_mode": ORDINARY_NULL_MODE,
        "no_compression_null_mode": NO_COMPRESSION_NULL_MODE,
        "gauge_mode": GAUGE_MODE,
        "orientation_null_mode": ORIENTATION_NULL_MODE,
        "orientation_null_seed": str(orientation_null_seed),
        "orientation_null_requested_draw_count": int(orientation_null_requested_draw_count),
        "orientation_null_max_attempt_count": int(orientation_null_max_attempt_count),
        "orientation_null_orthogonal_generator": ORIENTATION_ORTHOGONAL_GENERATOR,
        "orientation_seed_encoding": ORIENTATION_SEED_ENCODING,
        "tolerances": tolerances.as_dict(),
        "source_gate12a_manifest_path": repo_relative_or_posix(
            gate12a_dir / gate12c0.DEFAULT_MANIFEST
        ),
        "source_gate12a_run_id": str(artifacts.manifest.get("run_id") or ""),
        "source_gate12a_schema_version": str(artifacts.manifest.get("schema_version") or ""),
        "source_gate12a_code_git_commit": str(
            artifacts.manifest.get("code_git_commit") or ""
        ),
        "paths": {
            DEFAULT_REGISTRY: repo_relative_or_posix(out_dir / DEFAULT_REGISTRY),
            DEFAULT_ARRAYS: repo_relative_or_posix(out_dir / DEFAULT_ARRAYS),
            DEFAULT_CYCLE_SUMMARY: repo_relative_or_posix(out_dir / DEFAULT_CYCLE_SUMMARY),
            DEFAULT_COMPRESSION_SWEEP: repo_relative_or_posix(
                out_dir / DEFAULT_COMPRESSION_SWEEP
            ),
            DEFAULT_GAUGE_SUMMARY: repo_relative_or_posix(out_dir / DEFAULT_GAUGE_SUMMARY),
            DEFAULT_ORIENTATION_NULL_SUMMARY: repo_relative_or_posix(
                out_dir / DEFAULT_ORIENTATION_NULL_SUMMARY
            ),
            DEFAULT_STATUS: repo_relative_or_posix(out_dir / DEFAULT_STATUS),
            DEFAULT_READ: repo_relative_or_posix(out_dir / DEFAULT_READ),
        },
        "boundary": {
            "implementation_only": True,
            "synthetic_fixture_only_for_tests": True,
            "gate12b_overlay_used": False,
            "type_iii_claim_authorized": False,
            "rectangular_rank_mismatch_supported": False,
            "scientific_null_excess_threshold": None,
            "gate12a_or_gate12b_semantics_changed": False,
        },
    }

    cycle_summaries = build_cycle_summaries(rows)
    compression_sweep = build_compression_sweep(rows)
    gauge_summary = build_gauge_summary(rows)
    null_summary_rows = [
        {
            "probe_id": row["probe_id"],
            "cycle_id": row["cycle_id"],
            "root_rotation_index": row["root_rotation_index"],
            "compression_rank_q": row["compression_rank_q"],
            "orientation_null_requested_draw_count": row[
                "orientation_null_requested_draw_count"
            ],
            "orientation_null_valid_draw_count": row["orientation_null_valid_draw_count"],
            "orientation_null_invalid_cut_count": row["orientation_null_invalid_cut_count"],
            "orientation_null_attempt_count": row["orientation_null_attempt_count"],
            "orientation_null_status": row["orientation_null_status"],
            "orientation_null_median": row["orientation_null_median"],
            "orientation_null_mad": row["orientation_null_mad"],
            "orientation_null_mean": row["orientation_null_mean"],
            "orientation_null_std": row["orientation_null_std"],
            "orientation_null_empirical_p_upper": row[
                "orientation_null_empirical_p_upper"
            ],
            "orientation_null_robust_z": row["orientation_null_robust_z"],
            "orientation_null_scale_degenerate": row[
                "orientation_null_scale_degenerate"
            ],
            "orientation_null_excess_status": row["orientation_null_excess_status"],
        }
        for row in rows
    ]

    write_json(out_dir / DEFAULT_MANIFEST, manifest)
    write_jsonl(out_dir / DEFAULT_REGISTRY, rows)
    if left_arrays:
        left_array_payload = np.asarray(left_arrays, dtype=np.float64)
        right_array_payload = np.asarray(right_arrays, dtype=np.float64)
        associator_array_payload = np.asarray(associator_arrays, dtype=np.float64)
    else:
        empty_shape = (0, int(artifacts.r_max), int(artifacts.r_max))
        left_array_payload = np.zeros(empty_shape, dtype=np.float64)
        right_array_payload = np.zeros(empty_shape, dtype=np.float64)
        associator_array_payload = np.zeros(empty_shape, dtype=np.float64)

    np.savez(
        out_dir / DEFAULT_ARRAYS,
        compressed_overlap_left_operator=left_array_payload,
        compressed_overlap_right_operator=right_array_payload,
        compressed_overlap_associator_operator=associator_array_payload,
    )
    write_jsonl(out_dir / DEFAULT_CYCLE_SUMMARY, cycle_summaries)
    write_csv(out_dir / DEFAULT_COMPRESSION_SWEEP, COMPRESSION_SWEEP_FIELDNAMES, compression_sweep)
    write_json(out_dir / DEFAULT_GAUGE_SUMMARY, gauge_summary)
    write_jsonl(out_dir / DEFAULT_ORIENTATION_NULL_SUMMARY, null_summary_rows)
    write_json(out_dir / DEFAULT_STATUS, status_payload)
    write_text(out_dir / DEFAULT_READ, build_readme(status=status_payload, manifest=manifest))
    write_json(out_dir / DEFAULT_CHECKSUMS, build_checksums(out_dir))

    verify_source_unchanged(gate12a_dir, source_before)
    return {
        "manifest": manifest,
        "registry_rows": rows,
        "cycle_summaries": cycle_summaries,
        "compression_sweep": compression_sweep,
        "gauge_summary": gauge_summary,
        "orientation_null_summary_rows": null_summary_rows,
        "status": status_payload,
        "edge_diagnostics": edge_diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tolerances = Tolerances(
        tau_overlap_sv_min=float(args.tau_overlap_sv_min),
        tau_overlap_singular_value_abs_error=float(
            args.tau_overlap_singular_value_abs_error
        ),
        tau_transport_reconstruction_fro=float(args.tau_transport_reconstruction_fro),
        tau_ordinary_associator_fro=float(args.tau_ordinary_associator_fro),
        tau_no_compression_associator_fro=float(args.tau_no_compression_associator_fro),
        tau_split_rel=float(args.tau_split_rel),
        tau_gauge_operator_covariance_fro=float(args.tau_gauge_operator_covariance_fro),
        tau_gauge_scalar_delta_abs=float(args.tau_gauge_scalar_delta_abs),
        epsilon=float(args.epsilon),
    )
    out_dir = Path(args.out_dir)
    try:
        run_gate12c_compressed_overlap_associator(
            gate12a_dir=Path(args.gate12a_dir),
            out_dir=out_dir,
            orientation_null_seed=str(args.orientation_null_seed),
            orientation_null_requested_draw_count=int(
                args.orientation_null_requested_draw_count
            ),
            orientation_null_max_attempt_count=int(args.orientation_null_max_attempt_count),
            tolerances=tolerances,
        )
    except Exception as exc:
        write_failure_status(out_dir, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
