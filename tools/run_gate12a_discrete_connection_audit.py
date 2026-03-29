#!/usr/bin/env python3
"""Run the Gate12A discrete-connection audit over fixed upstream artifacts."""

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

SCHEMA_VERSION = "gate12a_discrete_connection_v1"
METHOD_ID = "gate12a_discrete_connection_v1"
GRAPH_OBJECT_POLICY = "flat_artifact_only_v1"
LOCAL_OBJECT_MODE = "projector_factor_public_basis_aux_v1"
TRANSPORT_OPERATOR_MODE = "polar_overlap_v1"
RANK_MISMATCH_MODE = "svd_partial_isometry_v1"
CYCLE_MODE = "explicit_triangle_only_v1"
COMPATIBILITY_MODE = "singular_spectrum_basis_invariant_v1"
HOLONOMY_MODE = "triangle_equal_rank_orthogonal_fro_residual_v1"
RELATION_SEED_MODE = "explicit_edge_seed_v1"

DEFAULT_TAU_OVERLAP_SV_MIN = 1.0e-8
DEFAULT_TAU_TRANSPORT_GAP_FRO = 1.0e-6
DEFAULT_TAU_HOLONOMY_RESIDUAL_FRO = 1.0e-6

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_NODE_REGISTRY = "node_local_object_registry.jsonl"
DEFAULT_NODE_ARRAYS = "node_local_object_arrays.npz"
DEFAULT_TRANSPORT_REGISTRY = "transport_relation_registry.jsonl"
DEFAULT_TRANSPORT_ARRAYS = "transport_operator_arrays.npz"
DEFAULT_TRIANGLE_REGISTRY = "explicit_triangle_cycle_registry.jsonl"
DEFAULT_HOLONOMY_REGISTRY = "triangle_holonomy_registry.jsonl"
DEFAULT_HOLONOMY_ARRAYS = "triangle_holonomy_arrays.npz"
DEFAULT_STATUS = "gate12a_discrete_connection_status.json"
DEFAULT_POLICY_COMPARE = "gate12a_discrete_connection_policy_compare.csv"
DEFAULT_READ = "gate12a_discrete_connection_read.md"
DEFAULT_CHECKSUMS = "checksums.json"

LOCAL_OBJECT_STATUS_VALUES = {
    "defined",
    "undefined_rank_zero",
    "undefined_aux_basis_missing",
}
RELATION_KIND_VALUES = {"trusted_tree", "residual_chord"}
CYCLE_STATUS = "admissible_explicit_triangle"


@dataclass(frozen=True)
class NodeLocalObject:
    node_id: str
    node_label: str
    basis_array_index: int
    projector_rank: int
    local_object_status: str
    basis_factor: np.ndarray

    @property
    def active_basis(self) -> np.ndarray:
        return np.asarray(self.basis_factor[:, : self.projector_rank], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first Gate12A discrete-connection audit from one node-local-object "
            "artifact family plus one explicit relation-seed artifact family."
        )
    )
    parser.add_argument("--node-artifact-dir", required=True)
    parser.add_argument("--relation-seed-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tau-overlap-sv-min",
        type=float,
        default=DEFAULT_TAU_OVERLAP_SV_MIN,
    )
    parser.add_argument(
        "--tau-transport-gap-fro",
        type=float,
        default=DEFAULT_TAU_TRANSPORT_GAP_FRO,
    )
    parser.add_argument(
        "--tau-holonomy-residual-fro",
        type=float,
        default=DEFAULT_TAU_HOLONOMY_RESIDUAL_FRO,
    )
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


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


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def require_keys(row: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def stack_or_empty(matrices: Sequence[np.ndarray], shape: Tuple[int, ...], *, dtype: np.dtype[Any]) -> np.ndarray:
    if not matrices:
        return np.zeros(shape, dtype=dtype)
    return np.asarray(matrices, dtype=dtype)


def load_node_family(
    node_artifact_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, NodeLocalObject]]:
    manifest_path = node_artifact_dir / DEFAULT_MANIFEST
    registry_path = node_artifact_dir / DEFAULT_NODE_REGISTRY
    arrays_path = node_artifact_dir / DEFAULT_NODE_ARRAYS

    manifest = read_json(manifest_path)
    require_keys(manifest, ("run_id", "schema_version"), "node local object manifest")
    registry_rows = read_jsonl(registry_path)
    with np.load(arrays_path) as npz_handle:
        basis_factor = np.asarray(npz_handle["basis_factor"], dtype=np.float64)
        rank_active = np.asarray(npz_handle["rank_active"], dtype=np.int64)

    if basis_factor.ndim != 3:
        raise ValueError("node basis_factor must have shape [N, d_model, r_max]")
    if rank_active.ndim != 1 or rank_active.shape[0] != basis_factor.shape[0]:
        raise ValueError("node rank_active must have shape [N]")

    sorted_input_rows = sorted(registry_rows, key=lambda row: str(row.get("node_id") or ""))
    normalized_rows: List[Dict[str, Any]] = []
    output_basis_rows: List[np.ndarray] = []
    output_ranks: List[int] = []
    node_map: Dict[str, NodeLocalObject] = {}

    for output_index, row in enumerate(sorted_input_rows):
        require_keys(
            row,
            ("node_id", "node_label", "basis_array_index", "projector_rank", "local_object_status"),
            "node_local_object_registry row",
        )
        node_id = str(row["node_id"])
        source_index = int(row["basis_array_index"])
        local_object_status = str(row["local_object_status"])
        if local_object_status not in LOCAL_OBJECT_STATUS_VALUES:
            raise ValueError(f"unsupported local_object_status for node {node_id}: {local_object_status}")
        if source_index < 0 or source_index >= basis_factor.shape[0]:
            raise ValueError(f"basis_array_index out of range for node {node_id}: {source_index}")
        basis_row = np.asarray(basis_factor[source_index], dtype=np.float64)
        rank_row = int(rank_active[source_index])
        projector_rank = int(row["projector_rank"])
        if projector_rank != rank_row:
            raise ValueError(
                f"projector_rank mismatch for node {node_id}: registry={projector_rank} arrays={rank_row}"
            )
        if local_object_status != "defined":
            if rank_row != 0:
                raise ValueError(
                    f"non-defined node {node_id} must not expose positive rank: "
                    f"status={local_object_status} rank={rank_row}"
                )
            if np.any(np.abs(basis_row) > 0.0):
                raise ValueError(
                    f"non-defined node {node_id} must not expose nonzero auxiliary basis rows"
                )
        normalized_row = {
            "node_id": node_id,
            "node_label": str(row["node_label"]),
            "basis_array_index": output_index,
            "projector_rank": rank_row,
            "local_object_status": local_object_status,
        }
        normalized_rows.append(normalized_row)
        output_basis_rows.append(basis_row)
        output_ranks.append(rank_row)
        node_map[node_id] = NodeLocalObject(
            node_id=node_id,
            node_label=str(row["node_label"]),
            basis_array_index=output_index,
            projector_rank=rank_row,
            local_object_status=local_object_status,
            basis_factor=basis_row,
        )

    arrays_out = {
        "basis_factor": np.asarray(output_basis_rows, dtype=np.float64),
        "rank_active": np.asarray(output_ranks, dtype=np.int64),
    }
    return manifest, normalized_rows, arrays_out, node_map


def load_relation_seed_family(
    relation_seed_dir: Path,
    node_map: Mapping[str, NodeLocalObject],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = relation_seed_dir / DEFAULT_MANIFEST
    registry_path = relation_seed_dir / "explicit_relation_seed_registry.jsonl"

    manifest = read_json(manifest_path)
    require_keys(manifest, ("run_id", "schema_version", "relation_seed_mode"), "relation-seed manifest")
    if str(manifest.get("relation_seed_mode") or "") != RELATION_SEED_MODE:
        raise ValueError("relation-seed manifest must declare relation_seed_mode = explicit_edge_seed_v1")

    registry_rows = read_jsonl(registry_path)
    sorted_rows = sorted(registry_rows, key=lambda row: str(row.get("edge_id") or ""))
    normalized_rows: List[Dict[str, Any]] = []
    for row in sorted_rows:
        require_keys(
            row,
            ("edge_id", "source_node_id", "target_node_id", "relation_kind", "anchor_qualified", "anchor_relation_id"),
            "explicit_relation_seed_registry row",
        )
        edge_id = str(row["edge_id"])
        source_node_id = str(row["source_node_id"])
        target_node_id = str(row["target_node_id"])
        relation_kind = str(row["relation_kind"])
        if relation_kind not in RELATION_KIND_VALUES:
            raise ValueError(f"unsupported relation_kind for edge {edge_id}: {relation_kind}")
        if source_node_id not in node_map:
            raise ValueError(f"edge {edge_id} references unknown source node: {source_node_id}")
        if target_node_id not in node_map:
            raise ValueError(f"edge {edge_id} references unknown target node: {target_node_id}")
        normalized_rows.append(
            {
                "edge_id": edge_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "relation_kind": relation_kind,
                "anchor_qualified": boolish(row["anchor_qualified"]),
                "anchor_relation_id": str(row["anchor_relation_id"] or ""),
            }
        )
    return manifest, normalized_rows


def compute_transport_operator(
    source: NodeLocalObject,
    target: NodeLocalObject,
    *,
    tau_overlap_sv_min: float,
    tau_transport_gap_fro: float,
    r_max: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, int]:
    source_rank = int(source.projector_rank)
    target_rank = int(target.projector_rank)
    singular_padded = np.zeros((r_max,), dtype=np.float64)
    transport_matrix = np.zeros((r_max, r_max), dtype=np.float64)

    if source.local_object_status != "defined" or target.local_object_status != "defined":
        return (
            {
                "overlap_rank": 0,
                "transport_case": "undefined_zero_overlap",
                "compatibility_gap_fro": None,
                "transport_level_compatibility_status": "undefined",
            },
            transport_matrix,
            singular_padded,
            0,
        )

    if source_rank <= 0 or target_rank <= 0:
        return (
            {
                "overlap_rank": 0,
                "transport_case": "undefined_zero_overlap",
                "compatibility_gap_fro": None,
                "transport_level_compatibility_status": "undefined",
            },
            transport_matrix,
            singular_padded,
            0,
        )

    overlap = np.asarray(target.active_basis.T @ source.active_basis, dtype=np.float64)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(overlap, full_matrices=False)
    singular_padded[: singular_values.shape[0]] = singular_values
    active_rank = int(np.sum(singular_values > tau_overlap_sv_min))
    if active_rank <= 0:
        return (
            {
                "overlap_rank": 0,
                "transport_case": "undefined_zero_overlap",
                "compatibility_gap_fro": None,
                "transport_level_compatibility_status": "undefined",
            },
            transport_matrix,
            singular_padded,
            0,
        )

    transport_active = np.asarray(u_matrix[:, :active_rank] @ vt_matrix[:active_rank, :], dtype=np.float64)
    transport_matrix[:target_rank, :source_rank] = transport_active
    transport_case = (
        "equal_rank_orthogonal"
        if source_rank == target_rank == active_rank
        else "rank_mismatch_partial_isometry"
    )
    compatibility_gap = float(np.linalg.norm(1.0 - singular_values[:active_rank]))
    compatibility_status = "compatible" if compatibility_gap <= tau_transport_gap_fro else "incompatible"
    return (
        {
            "overlap_rank": active_rank,
            "transport_case": transport_case,
            "compatibility_gap_fro": compatibility_gap,
            "transport_level_compatibility_status": compatibility_status,
        },
        transport_matrix,
        singular_padded,
        active_rank,
    )


def build_transport_relations(
    relation_rows: Sequence[Dict[str, Any]],
    node_map: Mapping[str, NodeLocalObject],
    *,
    tau_overlap_sv_min: float,
    tau_transport_gap_fro: float,
    r_max: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Dict[str, Any]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    transport_rows: List[Dict[str, Any]] = []
    transport_matrix_rows: List[np.ndarray] = []
    singular_value_rows: List[np.ndarray] = []
    active_ranks: List[int] = []
    edge_row_map: Dict[str, Dict[str, Any]] = {}
    outgoing: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for operator_index, relation in enumerate(relation_rows):
        source = node_map[relation["source_node_id"]]
        target = node_map[relation["target_node_id"]]
        metrics, matrix, singular_values, active_rank = compute_transport_operator(
            source,
            target,
            tau_overlap_sv_min=tau_overlap_sv_min,
            tau_transport_gap_fro=tau_transport_gap_fro,
            r_max=r_max,
        )
        row = {
            "edge_id": relation["edge_id"],
            "source_node_id": source.node_id,
            "target_node_id": target.node_id,
            "relation_kind": relation["relation_kind"],
            "anchor_qualified": bool(relation["anchor_qualified"]),
            "anchor_relation_id": str(relation["anchor_relation_id"] or ""),
            "source_rank": int(source.projector_rank),
            "target_rank": int(target.projector_rank),
            "overlap_rank": int(metrics["overlap_rank"]),
            "transport_case": metrics["transport_case"],
            "operator_array_index": operator_index,
            "compatibility_gap_fro": metrics["compatibility_gap_fro"],
            "transport_level_compatibility_status": metrics["transport_level_compatibility_status"],
        }
        transport_rows.append(row)
        transport_matrix_rows.append(matrix)
        singular_value_rows.append(singular_values)
        active_ranks.append(active_rank)
        edge_row_map[row["edge_id"]] = row
        outgoing.setdefault((source.node_id, target.node_id), []).append(row)

    for key in outgoing:
        outgoing[key] = sorted(outgoing[key], key=lambda row: str(row["edge_id"]))

    arrays = {
        "transport_matrix_local": stack_or_empty(
            transport_matrix_rows,
            (0, r_max, r_max),
            dtype=np.float64,
        ),
        "overlap_singular_values": stack_or_empty(
            singular_value_rows,
            (0, r_max),
            dtype=np.float64,
        ),
        "active_rank": np.asarray(active_ranks, dtype=np.int64) if active_ranks else np.zeros((0,), dtype=np.int64),
    }
    return transport_rows, arrays, edge_row_map, outgoing


def rotate_cycle_to_base(
    node_cycle: Sequence[str],
    edge_cycle: Sequence[str],
    base_node_id: str,
) -> Tuple[List[str], List[str]]:
    for index in range(3):
        if node_cycle[index] == base_node_id:
            rotated_nodes = list(node_cycle[index:3]) + list(node_cycle[:index]) + [base_node_id]
            rotated_edges = list(edge_cycle[index:3]) + list(edge_cycle[:index])
            return rotated_nodes, rotated_edges
    raise ValueError(f"base node {base_node_id} is not present in cycle {node_cycle}")


def build_explicit_triangle_cycles(
    transport_rows: Sequence[Dict[str, Any]],
    outgoing: Mapping[Tuple[str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    edges_sorted = sorted(transport_rows, key=lambda row: str(row["edge_id"]))
    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for row in edges_sorted:
        adjacency.setdefault(str(row["source_node_id"]), []).append(row)
    for source_node_id in adjacency:
        adjacency[source_node_id] = sorted(adjacency[source_node_id], key=lambda row: str(row["edge_id"]))

    cycles: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Tuple[str, str, str]]] = set()

    for first in edges_sorted:
        n0 = str(first["source_node_id"])
        n1 = str(first["target_node_id"])
        if n0 == n1:
            continue
        for second in adjacency.get(n1, []):
            if second["edge_id"] == first["edge_id"]:
                continue
            n2 = str(second["target_node_id"])
            if len({n0, n1, n2}) != 3:
                continue
            for third in outgoing.get((n2, n0), []):
                if len({first["edge_id"], second["edge_id"], third["edge_id"]}) != 3:
                    continue
                if "residual_chord" not in {
                    str(first["relation_kind"]),
                    str(second["relation_kind"]),
                    str(third["relation_kind"]),
                }:
                    continue
                base_node_id = min(n0, n1, n2)
                node_path, edge_path = rotate_cycle_to_base(
                    (n0, n1, n2),
                    (str(first["edge_id"]), str(second["edge_id"]), str(third["edge_id"])),
                    base_node_id,
                )
                edge_path_sorted = sorted(edge_path)
                key = (base_node_id, tuple(edge_path_sorted))
                if key in seen:
                    continue
                seen.add(key)
                cycles.append(
                    {
                        "cycle_id": f"triangle:{len(cycles):06d}",
                        "base_node_id": base_node_id,
                        "edge_id_path": edge_path_sorted,
                        "node_id_path": node_path,
                        "cycle_length": 3,
                        "cycle_status": CYCLE_STATUS,
                    }
                )
    return cycles


def build_triangle_holonomies(
    cycle_rows: Sequence[Dict[str, Any]],
    edge_row_map: Mapping[str, Dict[str, Any]],
    transport_matrix_local: np.ndarray,
    node_map: Mapping[str, NodeLocalObject],
    *,
    r_max: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    holonomy_rows: List[Dict[str, Any]] = []
    holonomy_matrix_rows: List[np.ndarray] = []

    for cycle in cycle_rows:
        cycle_edge_ids = {str(edge_id) for edge_id in cycle["edge_id_path"]}
        traversal_pairs = list(zip(cycle["node_id_path"][:3], cycle["node_id_path"][1:4]))
        edge_rows: List[Dict[str, Any]] = []
        for source_node_id, target_node_id in traversal_pairs:
            matches = [
                row
                for edge_id, row in edge_row_map.items()
                if edge_id in cycle_edge_ids
                and str(row["source_node_id"]) == str(source_node_id)
                and str(row["target_node_id"]) == str(target_node_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "triangle cycle cannot be reconstructed from lexicographic edge_id_path "
                    f"for cycle {cycle['cycle_id']}"
                )
            edge_rows.append(matches[0])
        holonomy_matrix = np.zeros((r_max, r_max), dtype=np.float64)
        cycle_nodes = list(cycle["node_id_path"][:3])
        node_ranks = [int(node_map[node_id].projector_rank) for node_id in cycle_nodes]
        transport_cases = [str(row["transport_case"]) for row in edge_rows]

        if any(case != "equal_rank_orthogonal" for case in transport_cases):
            holonomy_status = "equal_rank_required"
            holonomy_rank = 0
            holonomy_residual = None
        else:
            common_rank = node_ranks[0] if len(set(node_ranks)) == 1 else 0
            if common_rank <= 0:
                holonomy_status = "rank_chain_undefined"
                holonomy_rank = 0
                holonomy_residual = None
            else:
                local_maps = []
                for edge_row in edge_rows:
                    matrix = np.asarray(
                        transport_matrix_local[int(edge_row["operator_array_index"])],
                        dtype=np.float64,
                    )
                    local_maps.append(matrix[:common_rank, :common_rank])
                holonomy_product = local_maps[2] @ local_maps[1] @ local_maps[0]
                holonomy_matrix[:common_rank, :common_rank] = holonomy_product
                holonomy_status = "defined"
                holonomy_rank = int(common_rank)
                holonomy_residual = float(
                    np.linalg.norm(
                        holonomy_product - np.eye(common_rank, dtype=np.float64),
                        ord="fro",
                    )
                )

        holonomy_rows.append(
            {
                "cycle_id": cycle["cycle_id"],
                "base_node_id": cycle["base_node_id"],
                "holonomy_rank": holonomy_rank,
                "holonomy_residual_fro": holonomy_residual,
                "holonomy_status": holonomy_status,
            }
        )
        holonomy_matrix_rows.append(holonomy_matrix)

    arrays = {
        "holonomy_matrix_local": stack_or_empty(
            holonomy_matrix_rows,
            (0, r_max, r_max),
            dtype=np.float64,
        ),
    }
    return holonomy_rows, arrays


def build_readme(
    *,
    node_manifest: Mapping[str, Any],
    relation_manifest: Mapping[str, Any],
    zero_overlap_count: int,
    defined_holonomy_count: int,
    defined_holonomy_within_threshold_count: int,
    tau_transport_gap_fro: float,
    tau_holonomy_residual_fro: float,
    triangle_count: int,
) -> str:
    lines = [
        "# Gate12A Discrete Connection Read",
        "",
        "Gate12A accepted exactly two upstream artifact families:",
        f"- node local object family: `{node_manifest.get('run_id', '')}`",
        f"- explicit relation-seed family: `{relation_manifest.get('run_id', '')}`",
        "",
        "The public implementation surface remained flat-artifact only.",
        "The transport operator law was overlap/SVD based with square-padded source-to-target local maps.",
        (
            "The basis-invariant compatibility judgment was "
            "`compatibility_gap_fro := ||I_k - Sigma_k||_F` with the compatible/incompatible split "
            f"taken at tau_transport_gap_fro = `{tau_transport_gap_fro}`."
        ),
        "Cycle search was triangle-only; no generic loop mining was performed.",
        (
            "Defined holonomy residuals were reported numerically and counted against "
            f"tau_holonomy_residual_fro = `{tau_holonomy_residual_fro}`."
        ),
        "",
        f"- zero-overlap edges emitted: `{zero_overlap_count}`",
        f"- explicit triangle cycles emitted: `{triangle_count}`",
        f"- defined holonomy rows emitted: `{defined_holonomy_count}`",
        f"- defined holonomy rows at or below threshold: `{defined_holonomy_within_threshold_count}`",
    ]
    return "\n".join(lines) + "\n"


def build_checksums(out_dir: Path, included_files: Sequence[str]) -> Dict[str, str]:
    return {name: sha256_file(out_dir / name) for name in included_files}


def run_discrete_connection_audit(
    *,
    node_artifact_dir: Path,
    relation_seed_dir: Path,
    out_dir: Path,
    tau_overlap_sv_min: float = DEFAULT_TAU_OVERLAP_SV_MIN,
    tau_transport_gap_fro: float = DEFAULT_TAU_TRANSPORT_GAP_FRO,
    tau_holonomy_residual_fro: float = DEFAULT_TAU_HOLONOMY_RESIDUAL_FRO,
) -> Dict[str, Any]:
    node_artifact_dir = Path(node_artifact_dir)
    relation_seed_dir = Path(relation_seed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_manifest, node_rows, node_arrays, node_map = load_node_family(node_artifact_dir)
    relation_manifest, relation_rows = load_relation_seed_family(relation_seed_dir, node_map)

    basis_factor = np.asarray(node_arrays["basis_factor"], dtype=np.float64)
    _n_nodes, _d_model, r_max = basis_factor.shape

    transport_rows, transport_arrays, edge_row_map, outgoing = build_transport_relations(
        relation_rows,
        node_map,
        tau_overlap_sv_min=tau_overlap_sv_min,
        tau_transport_gap_fro=tau_transport_gap_fro,
        r_max=r_max,
    )
    cycle_rows = build_explicit_triangle_cycles(transport_rows, outgoing)
    holonomy_rows, holonomy_arrays = build_triangle_holonomies(
        cycle_rows,
        edge_row_map,
        np.asarray(transport_arrays["transport_matrix_local"], dtype=np.float64),
        node_map,
        r_max=r_max,
    )

    manifest_path = out_dir / DEFAULT_MANIFEST
    node_registry_path = out_dir / DEFAULT_NODE_REGISTRY
    node_arrays_path = out_dir / DEFAULT_NODE_ARRAYS
    transport_registry_path = out_dir / DEFAULT_TRANSPORT_REGISTRY
    transport_arrays_path = out_dir / DEFAULT_TRANSPORT_ARRAYS
    triangle_registry_path = out_dir / DEFAULT_TRIANGLE_REGISTRY
    holonomy_registry_path = out_dir / DEFAULT_HOLONOMY_REGISTRY
    holonomy_arrays_path = out_dir / DEFAULT_HOLONOMY_ARRAYS
    status_path = out_dir / DEFAULT_STATUS
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_jsonl(node_registry_path, node_rows)
    np.savez(
        node_arrays_path,
        basis_factor=np.asarray(node_arrays["basis_factor"], dtype=np.float64),
        rank_active=np.asarray(node_arrays["rank_active"], dtype=np.int64),
    )
    write_jsonl(transport_registry_path, transport_rows)
    np.savez(
        transport_arrays_path,
        transport_matrix_local=np.asarray(transport_arrays["transport_matrix_local"], dtype=np.float64),
        overlap_singular_values=np.asarray(transport_arrays["overlap_singular_values"], dtype=np.float64),
        active_rank=np.asarray(transport_arrays["active_rank"], dtype=np.int64),
    )
    write_jsonl(triangle_registry_path, cycle_rows)
    write_jsonl(holonomy_registry_path, holonomy_rows)
    np.savez(
        holonomy_arrays_path,
        holonomy_matrix_local=np.asarray(holonomy_arrays["holonomy_matrix_local"], dtype=np.float64),
    )

    zero_overlap_count = sum(
        1 for row in transport_rows if str(row["transport_case"]) == "undefined_zero_overlap"
    )
    defined_holonomy_count = sum(
        1 for row in holonomy_rows if str(row["holonomy_status"]) == "defined"
    )
    defined_holonomy_within_threshold_count = sum(
        1
        for row in holonomy_rows
        if str(row["holonomy_status"]) == "defined"
        and float(row["holonomy_residual_fro"]) <= tau_holonomy_residual_fro
    )

    status_payload = {
        "graph_object_policy_status": "flat_artifact_only",
        "transport_operator_surface_status": "defined",
        "basis_invariant_compatibility_status": "defined",
        "triangle_holonomy_scope_status": "explicit_triangle_only",
        "triangle_holonomy_status": "defined_or_equal_rank_required_or_rank_chain_undefined_only",
        "node_count": len(node_rows),
        "transport_relation_count": len(transport_rows),
        "explicit_triangle_cycle_count": len(cycle_rows),
        "defined_triangle_holonomy_count": defined_holonomy_count,
        "defined_triangle_holonomy_within_threshold_count": defined_holonomy_within_threshold_count,
    }
    write_json(status_path, status_payload)

    write_csv(
        policy_compare_path,
        (
            "run_id",
            "graph_object_policy_status",
            "transport_operator_surface_status",
            "basis_invariant_compatibility_status",
            "triangle_holonomy_scope_status",
            "triangle_holonomy_status",
            "node_count",
            "transport_relation_count",
            "explicit_triangle_cycle_count",
            "defined_triangle_holonomy_count",
            "defined_triangle_holonomy_within_threshold_count",
        ),
        [{"run_id": out_dir.name, **status_payload}],
    )
    write_text(
        read_path,
        build_readme(
            node_manifest=node_manifest,
            relation_manifest=relation_manifest,
            zero_overlap_count=zero_overlap_count,
            defined_holonomy_count=defined_holonomy_count,
            defined_holonomy_within_threshold_count=defined_holonomy_within_threshold_count,
            tau_transport_gap_fro=float(tau_transport_gap_fro),
            tau_holonomy_residual_fro=float(tau_holonomy_residual_fro),
            triangle_count=len(cycle_rows),
        ),
    )

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "graph_object_policy": GRAPH_OBJECT_POLICY,
        "local_object_mode": LOCAL_OBJECT_MODE,
        "transport_operator_mode": TRANSPORT_OPERATOR_MODE,
        "rank_mismatch_mode": RANK_MISMATCH_MODE,
        "cycle_mode": CYCLE_MODE,
        "compatibility_mode": COMPATIBILITY_MODE,
        "holonomy_mode": HOLONOMY_MODE,
        "tau_overlap_sv_min": float(tau_overlap_sv_min),
        "tau_transport_gap_fro": float(tau_transport_gap_fro),
        "tau_holonomy_residual_fro": float(tau_holonomy_residual_fro),
        "input_manifest_refs": {
            "node_local_object_family": {
                "manifest_path": repo_relative_or_posix(node_artifact_dir / DEFAULT_MANIFEST),
                "run_id": str(node_manifest.get("run_id") or ""),
                "schema_version": str(node_manifest.get("schema_version") or ""),
                "code_git_commit": str(node_manifest.get("code_git_commit") or ""),
            },
            "explicit_relation_seed_family": {
                "manifest_path": repo_relative_or_posix(relation_seed_dir / DEFAULT_MANIFEST),
                "run_id": str(relation_manifest.get("run_id") or ""),
                "schema_version": str(relation_manifest.get("schema_version") or ""),
                "relation_seed_mode": str(relation_manifest.get("relation_seed_mode") or ""),
            },
        },
        "paths": {
            DEFAULT_NODE_REGISTRY: repo_relative_or_posix(node_registry_path),
            DEFAULT_NODE_ARRAYS: repo_relative_or_posix(node_arrays_path),
            DEFAULT_TRANSPORT_REGISTRY: repo_relative_or_posix(transport_registry_path),
            DEFAULT_TRANSPORT_ARRAYS: repo_relative_or_posix(transport_arrays_path),
            DEFAULT_TRIANGLE_REGISTRY: repo_relative_or_posix(triangle_registry_path),
            DEFAULT_HOLONOMY_REGISTRY: repo_relative_or_posix(holonomy_registry_path),
            DEFAULT_HOLONOMY_ARRAYS: repo_relative_or_posix(holonomy_arrays_path),
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_POLICY_COMPARE: repo_relative_or_posix(policy_compare_path),
            DEFAULT_READ: repo_relative_or_posix(read_path),
        },
    }
    write_json(manifest_path, manifest)

    write_json(
        checksums_path,
        build_checksums(
            out_dir,
            (
                DEFAULT_MANIFEST,
                DEFAULT_NODE_REGISTRY,
                DEFAULT_NODE_ARRAYS,
                DEFAULT_TRANSPORT_REGISTRY,
                DEFAULT_TRANSPORT_ARRAYS,
                DEFAULT_TRIANGLE_REGISTRY,
                DEFAULT_HOLONOMY_REGISTRY,
                DEFAULT_HOLONOMY_ARRAYS,
                DEFAULT_STATUS,
                DEFAULT_POLICY_COMPARE,
                DEFAULT_READ,
            ),
        ),
    )

    return {
        "manifest": manifest,
        "status": status_payload,
        "transport_rows": transport_rows,
        "cycle_rows": cycle_rows,
        "holonomy_rows": holonomy_rows,
    }


def main() -> int:
    args = parse_args()
    run_discrete_connection_audit(
        node_artifact_dir=Path(args.node_artifact_dir),
        relation_seed_dir=Path(args.relation_seed_dir),
        out_dir=Path(args.out_dir),
        tau_overlap_sv_min=float(args.tau_overlap_sv_min),
        tau_transport_gap_fro=float(args.tau_transport_gap_fro),
        tau_holonomy_residual_fro=float(args.tau_holonomy_residual_fro),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
