#!/usr/bin/env python3
"""Inspect Gate12C-0 associator feasibility over Gate12A artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12c_associator_feasibility_preflight_v1"
METHOD_ID = "gate12c_associator_feasibility_preflight_v1"
PREFLIGHT_MODE = "gate12a_residual_bearing_explicit_triangle_equal_rank_preflight_v1"
RAW_OVERLAP_MODE = "basis_factor_transpose_overlap_v1"
TRANSPORT_RECONSTRUCTION_MODE = "gate12a_polar_overlap_v1"
STABLE_CUT_MODE = "relative_svd_split_gap_v1"
ORDINARY_NULL_MODE = "float64_matrix_associativity_null_v1"

DEFAULT_TAU_OVERLAP_SV_MIN = 1.0e-8
DEFAULT_TAU_OVERLAP_SV_ABS_ERROR = 1.0e-8
DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO = 1.0e-8
DEFAULT_TAU_ORDINARY_ASSOCIATOR_FRO = 1.0e-10
DEFAULT_TAU_SPLIT_REL = 1.0e-3
DEFAULT_EPSILON = 1.0e-12

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_NODE_REGISTRY = "node_local_object_registry.jsonl"
DEFAULT_NODE_ARRAYS = "node_local_object_arrays.npz"
DEFAULT_TRANSPORT_REGISTRY = "transport_relation_registry.jsonl"
DEFAULT_TRANSPORT_ARRAYS = "transport_operator_arrays.npz"
DEFAULT_CYCLE_REGISTRY = "explicit_triangle_cycle_registry.jsonl"
DEFAULT_HOLONOMY_REGISTRY = "triangle_holonomy_registry.jsonl"

DEFAULT_PREFLIGHT_JSON = "gate12c_feasibility_preflight.json"
DEFAULT_CYCLE_CENSUS = "gate12c_feasibility_cycle_census.csv"
DEFAULT_CUT_CENSUS = "gate12c_feasibility_cut_census.jsonl"
DEFAULT_READ = "gate12c_feasibility_read.md"
DEFAULT_CHECKSUMS = "checksums.json"

CONTRACT_PASS = "pass"
CONTRACT_MISSING_ARTIFACT = "fail_missing_artifact"
CONTRACT_RECONSTRUCTION_MISMATCH = "fail_reconstruction_mismatch"
CONTRACT_ORDINARY_ASSOCIATIVITY_NULL = "fail_ordinary_associativity_null"

EMPIRICAL_PASS_DECLARED_MINIMUM = "pass_declared_minimum"
EMPIRICAL_FAIL_BELOW_DECLARED_MINIMUM = "fail_below_declared_minimum"
EMPIRICAL_FAIL_NO_NONTRIVIAL_EQUAL_RANK_CYCLE = "fail_no_nontrivial_equal_rank_cycle"

REQUIRED_FILES = (
    DEFAULT_MANIFEST,
    DEFAULT_NODE_REGISTRY,
    DEFAULT_NODE_ARRAYS,
    DEFAULT_TRANSPORT_REGISTRY,
    DEFAULT_TRANSPORT_ARRAYS,
    DEFAULT_CYCLE_REGISTRY,
    DEFAULT_HOLONOMY_REGISTRY,
)
REQUIRED_NODE_ARRAYS = ("basis_factor", "rank_active")
REQUIRED_TRANSPORT_ARRAYS = (
    "transport_matrix_local",
    "overlap_singular_values",
    "active_rank",
)

CYCLE_CENSUS_FIELDNAMES = (
    "cycle_id",
    "base_node_id",
    "node_id_path",
    "edge_id_path",
    "ordered_edge_id_path",
    "ordered_relation_kind_path",
    "residual_chord_count",
    "holonomy_status",
    "transport_case_signature",
    "node_rank_signature",
    "common_rank",
    "defined_equal_rank_triangle",
    "nontrivial_equal_rank_eligible",
    "nontrivial_q_count",
)


class MissingGate12CArtifactError(RuntimeError):
    """Raised internally when a required Gate12A artifact is absent."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__("missing required Gate12A artifacts: " + ", ".join(self.missing))


@dataclass(frozen=True)
class NodeLocalObject:
    node_id: str
    basis_array_index: int
    projector_rank: int
    local_object_status: str
    basis_factor: np.ndarray

    @property
    def active_basis(self) -> np.ndarray:
        return np.asarray(self.basis_factor[:, : self.projector_rank], dtype=np.float64)

    @property
    def is_defined_positive_rank(self) -> bool:
        return self.local_object_status == "defined" and self.projector_rank > 0


@dataclass(frozen=True)
class EdgeReconstruction:
    edge_id: str
    source_node_id: str
    target_node_id: str
    overlap_matrix: np.ndarray | None
    reconstructed_transport: np.ndarray
    reconstructed_singular_values: np.ndarray
    reconstructed_active_rank: int
    singular_value_max_abs_error: float
    transport_reconstruction_fro_error: float
    active_rank_abs_error: int
    failed: bool


@dataclass(frozen=True)
class Gate12AArtifacts:
    manifest: Dict[str, Any]
    node_rows: List[Dict[str, Any]]
    transport_rows: List[Dict[str, Any]]
    cycle_rows: List[Dict[str, Any]]
    holonomy_rows: List[Dict[str, Any]]
    node_map: Dict[str, NodeLocalObject]
    edge_map: Dict[str, Dict[str, Any]]
    holonomy_map: Dict[str, Dict[str, Any]]
    node_basis_factor: np.ndarray
    node_rank_active: np.ndarray
    transport_matrix_local: np.ndarray
    overlap_singular_values: np.ndarray
    transport_active_rank: np.ndarray
    r_max: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gate12C-0 feasibility preflight over Gate12A-defined "
            "residual-bearing explicit triangles."
        )
    )
    parser.add_argument("--gate12a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-eligible-cycles", required=True, type=int)
    parser.add_argument("--tau-overlap-sv-min", type=float, default=DEFAULT_TAU_OVERLAP_SV_MIN)
    parser.add_argument(
        "--tau-overlap-sv-abs-error",
        type=float,
        default=DEFAULT_TAU_OVERLAP_SV_ABS_ERROR,
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
    parser.add_argument("--tau-split-rel", type=float, default=DEFAULT_TAU_SPLIT_REL)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


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


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def path_is_relative_to(*, child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_directory(*, gate12a_dir: Path, out_dir: Path) -> None:
    source_dir = Path(gate12a_dir).resolve(strict=False)
    target_dir = Path(out_dir).resolve(strict=False)
    if target_dir == source_dir:
        raise ValueError(
            "Gate12C out_dir must not be the same directory as gate12a_dir; "
            "the Gate12A source artifact directory is read-only input."
        )
    if path_is_relative_to(child=target_dir, parent=source_dir):
        raise ValueError(
            "Gate12C out_dir must not be inside gate12a_dir; choose a separate "
            "output directory so the Gate12A source artifact directory remains read-only input."
        )


def require_keys(row: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def load_npz_arrays(path: Path, required_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    with np.load(path) as handle:
        missing = [key for key in required_keys if key not in handle.files]
        if missing:
            raise MissingGate12CArtifactError([f"{path.name}::{key}" for key in missing])
        return {key: np.asarray(handle[key]) for key in required_keys}


def load_gate12a_artifacts(gate12a_dir: Path) -> Gate12AArtifacts:
    missing_files = [name for name in REQUIRED_FILES if not (gate12a_dir / name).exists()]
    if missing_files:
        raise MissingGate12CArtifactError(missing_files)

    manifest = read_json(gate12a_dir / DEFAULT_MANIFEST)
    node_rows = read_jsonl(gate12a_dir / DEFAULT_NODE_REGISTRY)
    transport_rows = read_jsonl(gate12a_dir / DEFAULT_TRANSPORT_REGISTRY)
    cycle_rows = read_jsonl(gate12a_dir / DEFAULT_CYCLE_REGISTRY)
    holonomy_rows = read_jsonl(gate12a_dir / DEFAULT_HOLONOMY_REGISTRY)

    node_arrays = load_npz_arrays(gate12a_dir / DEFAULT_NODE_ARRAYS, REQUIRED_NODE_ARRAYS)
    transport_arrays = load_npz_arrays(
        gate12a_dir / DEFAULT_TRANSPORT_ARRAYS,
        REQUIRED_TRANSPORT_ARRAYS,
    )

    basis_factor = np.asarray(node_arrays["basis_factor"], dtype=np.float64)
    rank_active = np.asarray(node_arrays["rank_active"], dtype=np.int64)
    if basis_factor.ndim != 3:
        raise ValueError("node basis_factor must have shape [N, d_model, r_max]")
    if rank_active.ndim != 1 or rank_active.shape[0] != basis_factor.shape[0]:
        raise ValueError("node rank_active must have shape [N]")
    _n_nodes, _d_model, r_max = basis_factor.shape

    transport_matrix_local = np.asarray(
        transport_arrays["transport_matrix_local"],
        dtype=np.float64,
    )
    overlap_singular_values = np.asarray(
        transport_arrays["overlap_singular_values"],
        dtype=np.float64,
    )
    active_rank = np.asarray(transport_arrays["active_rank"], dtype=np.int64)
    if transport_matrix_local.ndim != 3 or transport_matrix_local.shape[1:] != (r_max, r_max):
        raise ValueError("transport_matrix_local must have shape [E, r_max, r_max]")
    if overlap_singular_values.ndim != 2 or overlap_singular_values.shape[1] != r_max:
        raise ValueError("overlap_singular_values must have shape [E, r_max]")
    if active_rank.ndim != 1 or active_rank.shape[0] != transport_matrix_local.shape[0]:
        raise ValueError("active_rank must have shape [E]")
    if overlap_singular_values.shape[0] != transport_matrix_local.shape[0]:
        raise ValueError("overlap_singular_values and transport_matrix_local row counts must match")

    node_map: Dict[str, NodeLocalObject] = {}
    for row in node_rows:
        require_keys(
            row,
            ("node_id", "basis_array_index", "projector_rank", "local_object_status"),
            "node_local_object_registry row",
        )
        node_id = str(row["node_id"])
        basis_index = int(row["basis_array_index"])
        if basis_index < 0 or basis_index >= basis_factor.shape[0]:
            raise ValueError(f"basis_array_index out of range for node {node_id}: {basis_index}")
        projector_rank = int(row["projector_rank"])
        if projector_rank != int(rank_active[basis_index]):
            raise ValueError(
                f"projector_rank mismatch for node {node_id}: "
                f"registry={projector_rank} arrays={int(rank_active[basis_index])}"
            )
        if projector_rank < 0 or projector_rank > r_max:
            raise ValueError(f"projector_rank out of range for node {node_id}: {projector_rank}")
        if node_id in node_map:
            raise ValueError(f"duplicate node_id in node registry: {node_id}")
        node_map[node_id] = NodeLocalObject(
            node_id=node_id,
            basis_array_index=basis_index,
            projector_rank=projector_rank,
            local_object_status=str(row["local_object_status"]),
            basis_factor=np.asarray(basis_factor[basis_index], dtype=np.float64),
        )

    edge_map: Dict[str, Dict[str, Any]] = {}
    for row in transport_rows:
        require_keys(
            row,
            (
                "edge_id",
                "source_node_id",
                "target_node_id",
                "relation_kind",
                "source_rank",
                "target_rank",
                "transport_case",
                "operator_array_index",
            ),
            "transport_relation_registry row",
        )
        edge_id = str(row["edge_id"])
        operator_index = int(row["operator_array_index"])
        if operator_index < 0 or operator_index >= transport_matrix_local.shape[0]:
            raise ValueError(f"operator_array_index out of range for edge {edge_id}: {operator_index}")
        if edge_id in edge_map:
            raise ValueError(f"duplicate edge_id in transport registry: {edge_id}")
        edge_map[edge_id] = dict(row)

    holonomy_map: Dict[str, Dict[str, Any]] = {}
    for row in holonomy_rows:
        require_keys(row, ("cycle_id", "base_node_id", "holonomy_status"), "triangle_holonomy_registry row")
        cycle_id = str(row["cycle_id"])
        if cycle_id in holonomy_map:
            raise ValueError(f"duplicate cycle_id in holonomy registry: {cycle_id}")
        holonomy_map[cycle_id] = dict(row)

    return Gate12AArtifacts(
        manifest=manifest,
        node_rows=node_rows,
        transport_rows=transport_rows,
        cycle_rows=cycle_rows,
        holonomy_rows=holonomy_rows,
        node_map=node_map,
        edge_map=edge_map,
        holonomy_map=holonomy_map,
        node_basis_factor=basis_factor,
        node_rank_active=rank_active,
        transport_matrix_local=transport_matrix_local,
        overlap_singular_values=overlap_singular_values,
        transport_active_rank=active_rank,
        r_max=int(r_max),
    )


def reconstruct_gate12a_transport(
    *,
    source: NodeLocalObject,
    target: NodeLocalObject,
    r_max: int,
    tau_overlap_sv_min: float,
) -> Tuple[np.ndarray | None, np.ndarray, np.ndarray, int]:
    singular_padded = np.zeros((r_max,), dtype=np.float64)
    transport_matrix = np.zeros((r_max, r_max), dtype=np.float64)

    if not source.is_defined_positive_rank or not target.is_defined_positive_rank:
        return None, transport_matrix, singular_padded, 0

    overlap = np.asarray(target.active_basis.T @ source.active_basis, dtype=np.float64)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(overlap, full_matrices=False)
    singular_padded[: singular_values.shape[0]] = singular_values
    active_rank = int(np.sum(singular_values > tau_overlap_sv_min))
    if active_rank <= 0:
        return overlap, transport_matrix, singular_padded, 0

    transport_active = np.asarray(
        u_matrix[:, :active_rank] @ vt_matrix[:active_rank, :],
        dtype=np.float64,
    )
    transport_matrix[: target.projector_rank, : source.projector_rank] = transport_active
    return overlap, transport_matrix, singular_padded, active_rank


def reconstruct_edges(
    *,
    artifacts: Gate12AArtifacts,
    tau_overlap_sv_min: float,
    tau_overlap_sv_abs_error: float,
    tau_transport_reconstruction_fro: float,
) -> Tuple[Dict[str, EdgeReconstruction], Dict[str, Any]]:
    reconstructions: Dict[str, EdgeReconstruction] = {}
    reconstructed_edge_count = 0
    failed_count = 0
    max_sv_error = 0.0
    max_transport_error = 0.0
    max_active_rank_abs_error = 0

    for row in artifacts.transport_rows:
        edge_id = str(row["edge_id"])
        source_node_id = str(row["source_node_id"])
        target_node_id = str(row["target_node_id"])
        if source_node_id not in artifacts.node_map:
            raise ValueError(f"edge {edge_id} references unknown source node: {source_node_id}")
        if target_node_id not in artifacts.node_map:
            raise ValueError(f"edge {edge_id} references unknown target node: {target_node_id}")
        source = artifacts.node_map[source_node_id]
        target = artifacts.node_map[target_node_id]
        operator_index = int(row["operator_array_index"])

        overlap, transport, singular_values, active_rank = reconstruct_gate12a_transport(
            source=source,
            target=target,
            r_max=artifacts.r_max,
            tau_overlap_sv_min=float(tau_overlap_sv_min),
        )
        if source.is_defined_positive_rank and target.is_defined_positive_rank:
            reconstructed_edge_count += 1

        stored_sv = np.asarray(artifacts.overlap_singular_values[operator_index], dtype=np.float64)
        stored_transport = np.asarray(artifacts.transport_matrix_local[operator_index], dtype=np.float64)
        stored_active_rank = int(artifacts.transport_active_rank[operator_index])
        sv_error = float(np.max(np.abs(singular_values - stored_sv))) if stored_sv.size else 0.0
        transport_error = float(np.linalg.norm(transport - stored_transport, ord="fro"))
        active_rank_abs_error = abs(int(active_rank) - stored_active_rank)
        failed = (
            sv_error > float(tau_overlap_sv_abs_error)
            or transport_error > float(tau_transport_reconstruction_fro)
            or active_rank_abs_error > 0
        )
        if failed:
            failed_count += 1
        max_sv_error = max(max_sv_error, sv_error)
        max_transport_error = max(max_transport_error, transport_error)
        max_active_rank_abs_error = max(max_active_rank_abs_error, active_rank_abs_error)

        reconstructions[edge_id] = EdgeReconstruction(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            overlap_matrix=overlap,
            reconstructed_transport=transport,
            reconstructed_singular_values=singular_values,
            reconstructed_active_rank=int(active_rank),
            singular_value_max_abs_error=sv_error,
            transport_reconstruction_fro_error=transport_error,
            active_rank_abs_error=active_rank_abs_error,
            failed=failed,
        )

    diagnostics = {
        "reconstructed_edge_count": int(reconstructed_edge_count),
        "failed_edge_reconstruction_count": int(failed_count),
        "overlap_singular_value_max_abs_error": float(max_sv_error),
        "transport_reconstruction_max_fro_error": float(max_transport_error),
        "active_rank_max_abs_error": int(max_active_rank_abs_error),
    }
    return reconstructions, diagnostics


def reconstruct_ordered_edges(
    *,
    cycle: Mapping[str, Any],
    edge_map: Mapping[str, Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    cycle_edge_ids = {str(edge_id) for edge_id in cycle["edge_id_path"]}
    node_path = [str(node_id) for node_id in cycle["node_id_path"]]
    if len(node_path) < 4:
        raise ValueError(f"triangle cycle {cycle['cycle_id']} must expose a closed node_id_path")

    ordered_edges: List[Mapping[str, Any]] = []
    for source_node_id, target_node_id in zip(node_path[:3], node_path[1:4]):
        matches = [
            edge
            for edge_id, edge in edge_map.items()
            if edge_id in cycle_edge_ids
            and str(edge["source_node_id"]) == source_node_id
            and str(edge["target_node_id"]) == target_node_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "triangle cycle cannot be reconstructed from edge_id_path "
                f"for cycle {cycle['cycle_id']}"
            )
        ordered_edges.append(matches[0])
    return ordered_edges


def rotate_three(items: Sequence[Any], rotation_index: int) -> List[Any]:
    prefix = list(items[:3])
    return prefix[rotation_index:3] + prefix[:rotation_index]


def split_gap_rel(matrix: np.ndarray, *, q: int, epsilon: float) -> float:
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if q <= 0 or q >= singular_values.shape[0]:
        raise ValueError(f"split rank q must satisfy 1 <= q < rank, got q={q}")
    denominator = max(float(singular_values[0]), float(epsilon))
    return float((float(singular_values[q - 1]) - float(singular_values[q])) / denominator)


def inspect_cycles_and_cuts(
    *,
    artifacts: Gate12AArtifacts,
    edge_reconstructions: Mapping[str, EdgeReconstruction],
    tau_split_rel: float,
    tau_ordinary_associator_fro: float,
    epsilon: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cycle_census_rows: List[Dict[str, Any]] = []
    cut_rows: List[Dict[str, Any]] = []
    ordinary_max_fro = 0.0
    ordinary_failed_count = 0
    total_residual_bearing = 0
    defined_equal_rank_count = 0
    rank_counts = {1: 0, 2: 0, 3: 0}
    rank_ge_4_count = 0
    max_common_rank = 0
    eligible_cycle_ids: set[str] = set()
    stable_cycle_ids: set[str] = set()

    sorted_cycles = sorted(artifacts.cycle_rows, key=lambda row: str(row.get("cycle_id") or ""))
    for cycle in sorted_cycles:
        require_keys(
            cycle,
            ("cycle_id", "base_node_id", "edge_id_path", "node_id_path"),
            "explicit_triangle_cycle_registry row",
        )
        cycle_id = str(cycle["cycle_id"])
        ordered_edges = reconstruct_ordered_edges(cycle=cycle, edge_map=artifacts.edge_map)
        relation_path = [str(edge["relation_kind"]) for edge in ordered_edges]
        residual_chord_count = sum(1 for kind in relation_path if kind == "residual_chord")
        if residual_chord_count <= 0:
            continue

        total_residual_bearing += 1
        holonomy = artifacts.holonomy_map.get(cycle_id, {})
        holonomy_status = str(holonomy.get("holonomy_status") or "missing")
        node_ids = [str(node_id) for node_id in list(cycle["node_id_path"])[:3]]
        node_ranks = [
            int(artifacts.node_map[node_id].projector_rank) if node_id in artifacts.node_map else 0
            for node_id in node_ids
        ]
        transport_cases = [str(edge["transport_case"]) for edge in ordered_edges]
        common_rank = node_ranks[0] if len(set(node_ranks)) == 1 and node_ranks[0] > 0 else 0
        equal_rank_transport = all(case == "equal_rank_orthogonal" for case in transport_cases)
        defined_equal_rank = (
            holonomy_status == "defined"
            and equal_rank_transport
            and common_rank > 0
        )
        eligible = bool(defined_equal_rank and common_rank >= 2)
        if defined_equal_rank:
            defined_equal_rank_count += 1
            max_common_rank = max(max_common_rank, common_rank)
            if common_rank in rank_counts:
                rank_counts[common_rank] += 1
            elif common_rank >= 4:
                rank_ge_4_count += 1
        if eligible:
            eligible_cycle_ids.add(cycle_id)

        ordered_edge_ids = [str(edge["edge_id"]) for edge in ordered_edges]
        cycle_census_rows.append(
            {
                "cycle_id": cycle_id,
                "base_node_id": str(cycle["base_node_id"]),
                "node_id_path": [str(node_id) for node_id in cycle["node_id_path"]],
                "edge_id_path": [str(edge_id) for edge_id in cycle["edge_id_path"]],
                "ordered_edge_id_path": ordered_edge_ids,
                "ordered_relation_kind_path": relation_path,
                "residual_chord_count": int(residual_chord_count),
                "holonomy_status": holonomy_status,
                "transport_case_signature": "|".join(transport_cases),
                "node_rank_signature": ">".join(str(rank) for rank in node_ranks),
                "common_rank": int(common_rank),
                "defined_equal_rank_triangle": bool(defined_equal_rank),
                "nontrivial_equal_rank_eligible": bool(eligible),
                "nontrivial_q_count": int(max(common_rank - 1, 0) if eligible else 0),
            }
        )

        if not eligible:
            continue

        edge_matrices: List[np.ndarray] = []
        for edge_id in ordered_edge_ids:
            reconstruction = edge_reconstructions[edge_id]
            if reconstruction.overlap_matrix is None:
                raise ValueError(f"eligible cycle {cycle_id} has undefined reconstructed overlap for edge {edge_id}")
            matrix = np.asarray(reconstruction.overlap_matrix, dtype=np.float64)
            if matrix.shape != (common_rank, common_rank):
                raise ValueError(
                    f"eligible cycle {cycle_id} edge {edge_id} overlap shape "
                    f"{matrix.shape} does not match common rank {common_rank}"
                )
            edge_matrices.append(matrix)

        for root_rotation_index in range(3):
            root_nodes = rotate_three(node_ids, root_rotation_index)
            root_edges = rotate_three(ordered_edges, root_rotation_index)
            root_edge_ids = [str(edge["edge_id"]) for edge in root_edges]
            root_relation_path = [str(edge["relation_kind"]) for edge in root_edges]
            matrices = rotate_three(edge_matrices, root_rotation_index)
            m0, m1, m2 = matrices
            ordinary_left = (m2 @ m1) @ m0
            ordinary_right = m2 @ (m1 @ m0)
            ordinary_fro = float(np.linalg.norm(ordinary_left - ordinary_right, ord="fro"))
            ordinary_max_fro = max(ordinary_max_fro, ordinary_fro)
            if ordinary_fro > float(tau_ordinary_associator_fro):
                ordinary_failed_count += 1

            left_inner = m2 @ m1
            right_inner = m1 @ m0
            for q in range(1, common_rank):
                left_gap = split_gap_rel(left_inner, q=q, epsilon=float(epsilon))
                right_gap = split_gap_rel(right_inner, q=q, epsilon=float(epsilon))
                left_status = "stable" if left_gap > float(tau_split_rel) else "near_degenerate"
                right_status = "stable" if right_gap > float(tau_split_rel) else "near_degenerate"
                both_stable = left_status == "stable" and right_status == "stable"
                if both_stable:
                    stable_cycle_ids.add(cycle_id)
                cut_rows.append(
                    {
                        "probe_id": f"gate12c_probe:{len(cut_rows):06d}",
                        "probe_configuration_index": int(len(cut_rows)),
                        "cycle_id": cycle_id,
                        "canonical_base_node_id": str(cycle["base_node_id"]),
                        "evaluation_root_node_id": root_nodes[0],
                        "root_rotation_index": int(root_rotation_index),
                        "ordered_node_id_path": root_nodes + [root_nodes[0]],
                        "ordered_edge_id_path": root_edge_ids,
                        "ordered_relation_kind_path": root_relation_path,
                        "common_rank": int(common_rank),
                        "q": int(q),
                        "left_inner_split_gap_rel": float(left_gap),
                        "right_inner_split_gap_rel": float(right_gap),
                        "left_cut_status": left_status,
                        "right_cut_status": right_status,
                        "both_inner_cut_status": "stable_both" if both_stable else "near_degenerate",
                        "ordinary_associator_fro": float(ordinary_fro),
                        "promotable_to_gate12c1_alpha": bool(both_stable),
                    }
                )

    stable_both_count = sum(
        1
        for row in cut_rows
        if row["left_cut_status"] == "stable" and row["right_cut_status"] == "stable"
    )
    near_left_count = sum(1 for row in cut_rows if row["left_cut_status"] == "near_degenerate")
    near_right_count = sum(1 for row in cut_rows if row["right_cut_status"] == "near_degenerate")
    near_both_count = sum(
        1
        for row in cut_rows
        if row["left_cut_status"] == "near_degenerate"
        and row["right_cut_status"] == "near_degenerate"
    )
    cycle_census = {
        "total_gate12a_residual_bearing_explicit_triangle_count": int(total_residual_bearing),
        "defined_equal_rank_triangle_count": int(defined_equal_rank_count),
        "common_rank_1_triangle_count": int(rank_counts[1]),
        "common_rank_2_triangle_count": int(rank_counts[2]),
        "common_rank_3_triangle_count": int(rank_counts[3]),
        "common_rank_ge_4_triangle_count": int(rank_ge_4_count),
        "max_common_equal_rank": int(max_common_rank),
        "eligible_equal_rank_common_rank_ge_2_cycle_count": int(len(eligible_cycle_ids)),
    }
    cut_census = {
        "probe_configuration_count": int(len(cut_rows)),
        "stable_both_inner_cut_count": int(stable_both_count),
        "near_degenerate_left_cut_count": int(near_left_count),
        "near_degenerate_right_cut_count": int(near_right_count),
        "near_degenerate_both_cut_count": int(near_both_count),
        "eligible_cycle_count_with_at_least_one_stable_q": int(len(stable_cycle_ids)),
    }
    ordinary_null = {
        "ordinary_associator_max_fro": float(ordinary_max_fro),
        "ordinary_associator_failed_count": int(ordinary_failed_count),
    }
    return cycle_census_rows, cut_rows, cycle_census, cut_census, ordinary_null


def empirical_surface_status(*, cut_census: Mapping[str, Any], min_eligible_cycles: int) -> str:
    stable_cycle_count = int(cut_census["eligible_cycle_count_with_at_least_one_stable_q"])
    if int(cut_census.get("probe_configuration_count", 0)) == 0:
        return EMPIRICAL_FAIL_NO_NONTRIVIAL_EQUAL_RANK_CYCLE
    if stable_cycle_count >= int(min_eligible_cycles):
        return EMPIRICAL_PASS_DECLARED_MINIMUM
    return EMPIRICAL_FAIL_BELOW_DECLARED_MINIMUM


def contract_status_for(
    *,
    edge_diagnostics: Mapping[str, Any],
    ordinary_null: Mapping[str, Any],
) -> str:
    if int(edge_diagnostics["failed_edge_reconstruction_count"]) > 0:
        return CONTRACT_RECONSTRUCTION_MISMATCH
    if int(ordinary_null["ordinary_associator_failed_count"]) > 0:
        return CONTRACT_ORDINARY_ASSOCIATIVITY_NULL
    return CONTRACT_PASS


def empty_cycle_census() -> Dict[str, Any]:
    return {
        "total_gate12a_residual_bearing_explicit_triangle_count": 0,
        "defined_equal_rank_triangle_count": 0,
        "common_rank_1_triangle_count": 0,
        "common_rank_2_triangle_count": 0,
        "common_rank_3_triangle_count": 0,
        "common_rank_ge_4_triangle_count": 0,
        "max_common_equal_rank": 0,
        "eligible_equal_rank_common_rank_ge_2_cycle_count": 0,
    }


def empty_cut_census() -> Dict[str, Any]:
    return {
        "probe_configuration_count": 0,
        "stable_both_inner_cut_count": 0,
        "near_degenerate_left_cut_count": 0,
        "near_degenerate_right_cut_count": 0,
        "near_degenerate_both_cut_count": 0,
        "eligible_cycle_count_with_at_least_one_stable_q": 0,
    }


def empty_edge_diagnostics() -> Dict[str, Any]:
    return {
        "reconstructed_edge_count": 0,
        "failed_edge_reconstruction_count": 0,
        "overlap_singular_value_max_abs_error": 0.0,
        "transport_reconstruction_max_fro_error": 0.0,
        "active_rank_max_abs_error": 0,
    }


def build_preflight_payload(
    *,
    contract_feasibility_status: str,
    empirical_status: str,
    min_eligible_cycles: int,
    edge_diagnostics: Mapping[str, Any],
    cycle_census: Mapping[str, Any],
    cut_census: Mapping[str, Any],
    ordinary_null: Mapping[str, Any],
    tolerances: Mapping[str, Any],
    missing_required_artifacts: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "gate12c_phase": "Gate12C-0",
        "preflight_mode": PREFLIGHT_MODE,
        "raw_overlap_mode": RAW_OVERLAP_MODE,
        "transport_reconstruction_mode": TRANSPORT_RECONSTRUCTION_MODE,
        "stable_cut_mode": STABLE_CUT_MODE,
        "ordinary_null_mode": ORDINARY_NULL_MODE,
        "contract_feasibility_status": contract_feasibility_status,
        "empirical_surface_status": empirical_status,
        "min_eligible_cycles": int(min_eligible_cycles),
        "missing_required_artifacts": list(missing_required_artifacts or []),
        "tolerances": dict(tolerances),
        "edge_reconstruction_diagnostics": dict(edge_diagnostics),
        "cycle_census": dict(cycle_census),
        "cut_census": dict(cut_census),
        "ordinary_associativity_null": dict(ordinary_null),
        "reading_boundary": {
            "input_surface": "Gate12A-defined residual-bearing explicit triangles",
            "gate12c1_implemented": False,
            "compressed_associator_values_emitted": False,
            "rectangular_rank_mismatch_associators_deferred": True,
            "gate12a_or_gate12b_semantics_changed": False,
            "real_empirical_feasibility_claim": False,
        },
    }


def build_readme(
    *,
    source_manifest: Mapping[str, Any],
    preflight_payload: Mapping[str, Any],
) -> str:
    cycle_census = preflight_payload["cycle_census"]
    cut_census = preflight_payload["cut_census"]
    edge_diagnostics = preflight_payload["edge_reconstruction_diagnostics"]
    ordinary_null = preflight_payload["ordinary_associativity_null"]
    lines = [
        "# Gate12C Feasibility Preflight Read",
        "",
        "This is a read-only Gate12C-0 feasibility preflight over Gate12A-defined residual-bearing explicit triangles.",
        "It does not implement Gate12C-1, emit compressed associator values, consume Gate12B overlays, or change Gate12A/Gate12B semantics.",
        "",
        "## Source",
        "",
        f"- source Gate12A run: `{source_manifest.get('run_id', '')}`",
        f"- source Gate12A code commit: `{source_manifest.get('code_git_commit', '')}`",
        "",
        "## Status",
        "",
        f"- contract_feasibility_status: `{preflight_payload['contract_feasibility_status']}`",
        f"- empirical_surface_status: `{preflight_payload['empirical_surface_status']}`",
        f"- min_eligible_cycles: `{preflight_payload['min_eligible_cycles']}`",
        "",
        "## Edge Reconstruction",
        "",
        f"- reconstructed_edge_count: `{edge_diagnostics['reconstructed_edge_count']}`",
        f"- failed_edge_reconstruction_count: `{edge_diagnostics['failed_edge_reconstruction_count']}`",
        f"- overlap_singular_value_max_abs_error: `{edge_diagnostics['overlap_singular_value_max_abs_error']}`",
        f"- transport_reconstruction_max_fro_error: `{edge_diagnostics['transport_reconstruction_max_fro_error']}`",
        "",
        "## Cycle Census",
        "",
        f"- total_gate12a_residual_bearing_explicit_triangle_count: `{cycle_census['total_gate12a_residual_bearing_explicit_triangle_count']}`",
        f"- defined_equal_rank_triangle_count: `{cycle_census['defined_equal_rank_triangle_count']}`",
        f"- common_rank_1_triangle_count: `{cycle_census['common_rank_1_triangle_count']}`",
        f"- common_rank_2_triangle_count: `{cycle_census['common_rank_2_triangle_count']}`",
        f"- common_rank_3_triangle_count: `{cycle_census['common_rank_3_triangle_count']}`",
        f"- common_rank_ge_4_triangle_count: `{cycle_census['common_rank_ge_4_triangle_count']}`",
        "",
        "## Stable-Cut Census",
        "",
        f"- probe_configuration_count: `{cut_census['probe_configuration_count']}`",
        f"- stable_both_inner_cut_count: `{cut_census['stable_both_inner_cut_count']}`",
        f"- near_degenerate_left_cut_count: `{cut_census['near_degenerate_left_cut_count']}`",
        f"- near_degenerate_right_cut_count: `{cut_census['near_degenerate_right_cut_count']}`",
        f"- near_degenerate_both_cut_count: `{cut_census['near_degenerate_both_cut_count']}`",
        f"- eligible_cycle_count_with_at_least_one_stable_q: `{cut_census['eligible_cycle_count_with_at_least_one_stable_q']}`",
        "",
        "## Ordinary Associativity Null",
        "",
        f"- ordinary_associator_max_fro: `{ordinary_null['ordinary_associator_max_fro']}`",
        f"- ordinary_associator_failed_count: `{ordinary_null['ordinary_associator_failed_count']}`",
        "",
        "Empirical Gate12C-1 feasibility remains unknown until this preflight is run on real Gate12A artifact directories.",
    ]
    missing = list(preflight_payload.get("missing_required_artifacts") or [])
    if missing:
        lines.extend(["", "## Missing Required Artifacts", ""])
        lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines) + "\n"


def build_checksums(out_dir: Path, included_files: Sequence[str]) -> Dict[str, str]:
    return {name: sha256_file(out_dir / name) for name in included_files}


def write_outputs(
    *,
    gate12a_dir: Path,
    out_dir: Path,
    source_manifest: Mapping[str, Any],
    preflight_payload: Mapping[str, Any],
    cycle_rows: Sequence[Mapping[str, Any]],
    cut_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / DEFAULT_MANIFEST
    preflight_json_path = out_dir / DEFAULT_PREFLIGHT_JSON
    cycle_census_path = out_dir / DEFAULT_CYCLE_CENSUS
    cut_census_path = out_dir / DEFAULT_CUT_CENSUS
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_json(preflight_json_path, preflight_payload)
    write_csv(cycle_census_path, CYCLE_CENSUS_FIELDNAMES, cycle_rows)
    write_jsonl(cut_census_path, cut_rows)
    write_text(read_path, build_readme(source_manifest=source_manifest, preflight_payload=preflight_payload))

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "preflight_mode": PREFLIGHT_MODE,
        "raw_overlap_mode": RAW_OVERLAP_MODE,
        "transport_reconstruction_mode": TRANSPORT_RECONSTRUCTION_MODE,
        "stable_cut_mode": STABLE_CUT_MODE,
        "ordinary_null_mode": ORDINARY_NULL_MODE,
        "source_gate12a_manifest_path": repo_relative_or_posix(gate12a_dir / DEFAULT_MANIFEST),
        "source_gate12a_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate12a_schema_version": str(source_manifest.get("schema_version") or ""),
        "source_gate12a_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "min_eligible_cycles": int(preflight_payload["min_eligible_cycles"]),
        "tolerances": dict(preflight_payload["tolerances"]),
        "contract_feasibility_status": str(preflight_payload["contract_feasibility_status"]),
        "empirical_surface_status": str(preflight_payload["empirical_surface_status"]),
        "paths": {
            DEFAULT_PREFLIGHT_JSON: repo_relative_or_posix(preflight_json_path),
            DEFAULT_CYCLE_CENSUS: repo_relative_or_posix(cycle_census_path),
            DEFAULT_CUT_CENSUS: repo_relative_or_posix(cut_census_path),
            DEFAULT_READ: repo_relative_or_posix(read_path),
        },
        "status": {
            "contract_feasibility_status": str(preflight_payload["contract_feasibility_status"]),
            "empirical_surface_status": str(preflight_payload["empirical_surface_status"]),
        },
    }
    write_json(manifest_path, manifest)
    included_files = (
        DEFAULT_MANIFEST,
        DEFAULT_PREFLIGHT_JSON,
        DEFAULT_CYCLE_CENSUS,
        DEFAULT_CUT_CENSUS,
        DEFAULT_READ,
    )
    write_json(checksums_path, build_checksums(out_dir, included_files))

    return {
        "manifest": manifest,
        "preflight": dict(preflight_payload),
        "cycle_rows": list(cycle_rows),
        "cut_rows": list(cut_rows),
    }


def build_tolerances(
    *,
    tau_overlap_sv_min: float,
    tau_overlap_sv_abs_error: float,
    tau_transport_reconstruction_fro: float,
    tau_ordinary_associator_fro: float,
    tau_split_rel: float,
    epsilon: float,
) -> Dict[str, float]:
    return {
        "tau_overlap_sv_min": float(tau_overlap_sv_min),
        "tau_overlap_singular_value_abs_error": float(tau_overlap_sv_abs_error),
        "tau_transport_reconstruction_fro": float(tau_transport_reconstruction_fro),
        "tau_ordinary_associator_fro": float(tau_ordinary_associator_fro),
        "tau_split_rel": float(tau_split_rel),
        "epsilon": float(epsilon),
    }


def run_associator_feasibility_preflight(
    *,
    gate12a_dir: Path,
    out_dir: Path,
    min_eligible_cycles: int,
    tau_overlap_sv_min: float = DEFAULT_TAU_OVERLAP_SV_MIN,
    tau_overlap_sv_abs_error: float = DEFAULT_TAU_OVERLAP_SV_ABS_ERROR,
    tau_transport_reconstruction_fro: float = DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO,
    tau_ordinary_associator_fro: float = DEFAULT_TAU_ORDINARY_ASSOCIATOR_FRO,
    tau_split_rel: float = DEFAULT_TAU_SPLIT_REL,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, Any]:
    if int(min_eligible_cycles) < 0:
        raise ValueError("min_eligible_cycles must be nonnegative")
    gate12a_dir = Path(gate12a_dir)
    out_dir = Path(out_dir)
    validate_output_directory(gate12a_dir=gate12a_dir, out_dir=out_dir)
    tolerances = build_tolerances(
        tau_overlap_sv_min=float(tau_overlap_sv_min),
        tau_overlap_sv_abs_error=float(tau_overlap_sv_abs_error),
        tau_transport_reconstruction_fro=float(tau_transport_reconstruction_fro),
        tau_ordinary_associator_fro=float(tau_ordinary_associator_fro),
        tau_split_rel=float(tau_split_rel),
        epsilon=float(epsilon),
    )

    try:
        artifacts = load_gate12a_artifacts(gate12a_dir)
    except MissingGate12CArtifactError as exc:
        source_manifest: Dict[str, Any] = {}
        manifest_path = gate12a_dir / DEFAULT_MANIFEST
        if manifest_path.exists():
            source_manifest = read_json(manifest_path)
        cut_census = empty_cut_census()
        preflight_payload = build_preflight_payload(
            contract_feasibility_status=CONTRACT_MISSING_ARTIFACT,
            empirical_status=EMPIRICAL_FAIL_NO_NONTRIVIAL_EQUAL_RANK_CYCLE,
            min_eligible_cycles=int(min_eligible_cycles),
            edge_diagnostics=empty_edge_diagnostics(),
            cycle_census=empty_cycle_census(),
            cut_census=cut_census,
            ordinary_null={
                "ordinary_associator_max_fro": 0.0,
                "ordinary_associator_failed_count": 0,
            },
            tolerances=tolerances,
            missing_required_artifacts=exc.missing,
        )
        return write_outputs(
            gate12a_dir=gate12a_dir,
            out_dir=out_dir,
            source_manifest=source_manifest,
            preflight_payload=preflight_payload,
            cycle_rows=[],
            cut_rows=[],
        )

    edge_reconstructions, edge_diagnostics = reconstruct_edges(
        artifacts=artifacts,
        tau_overlap_sv_min=float(tau_overlap_sv_min),
        tau_overlap_sv_abs_error=float(tau_overlap_sv_abs_error),
        tau_transport_reconstruction_fro=float(tau_transport_reconstruction_fro),
    )
    cycle_rows, cut_rows, cycle_census, cut_census, ordinary_null = inspect_cycles_and_cuts(
        artifacts=artifacts,
        edge_reconstructions=edge_reconstructions,
        tau_split_rel=float(tau_split_rel),
        tau_ordinary_associator_fro=float(tau_ordinary_associator_fro),
        epsilon=float(epsilon),
    )
    contract_status = contract_status_for(
        edge_diagnostics=edge_diagnostics,
        ordinary_null=ordinary_null,
    )
    empirical_status = empirical_surface_status(
        cut_census=cut_census,
        min_eligible_cycles=int(min_eligible_cycles),
    )
    preflight_payload = build_preflight_payload(
        contract_feasibility_status=contract_status,
        empirical_status=empirical_status,
        min_eligible_cycles=int(min_eligible_cycles),
        edge_diagnostics=edge_diagnostics,
        cycle_census=cycle_census,
        cut_census=cut_census,
        ordinary_null=ordinary_null,
        tolerances=tolerances,
    )
    return write_outputs(
        gate12a_dir=gate12a_dir,
        out_dir=out_dir,
        source_manifest=artifacts.manifest,
        preflight_payload=preflight_payload,
        cycle_rows=cycle_rows,
        cut_rows=cut_rows,
    )


def main() -> int:
    args = parse_args()
    result = run_associator_feasibility_preflight(
        gate12a_dir=Path(args.gate12a_dir),
        out_dir=Path(args.out_dir),
        min_eligible_cycles=int(args.min_eligible_cycles),
        tau_overlap_sv_min=float(args.tau_overlap_sv_min),
        tau_overlap_sv_abs_error=float(args.tau_overlap_sv_abs_error),
        tau_transport_reconstruction_fro=float(args.tau_transport_reconstruction_fro),
        tau_ordinary_associator_fro=float(args.tau_ordinary_associator_fro),
        tau_split_rel=float(args.tau_split_rel),
        epsilon=float(args.epsilon),
    )
    contract_status = str(result["preflight"]["contract_feasibility_status"])
    return 0 if contract_status == CONTRACT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
