#!/usr/bin/env python3
"""Run a Gate9A graph-gauge consumer on an existing Gate8 execution bundle."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import build_gate6_native_local_span as gate6_builder
import run_gate7_progression_anisotropic_consumer_v3 as gate7c_consumer


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9a_graph_gauge_consumer_v1"
METHOD_ID = "gate9a_graph_gauge_consumer_v1"
PROJECTOR_PUBLIC_KIND = "factorized_projector_public_v1"
CYCLE_TOKEN_SELECTOR = "terminal_token_state_v1"
NODE_TYPES = (
    "token_state",
    "support_chunk",
    "conflict_chunk",
    "answer_state",
)
EDGE_TYPES = (
    "temporal_transition",
    "support_anchor",
    "conflict_anchor",
    "answer_projection",
    "quietness_pair",
)
PRIMARY_OBSERVABLES = (
    "edge_transport_defect",
    "small_cycle_holonomy",
    "anchor_conditioned_closure",
)

DEFAULT_NODE_REGISTRY = "node_registry.jsonl"
DEFAULT_EDGE_REGISTRY = "edge_transport_registry.jsonl"
DEFAULT_CYCLE_REGISTRY = "small_cycle_holonomy.jsonl"
DEFAULT_ANCHOR_CLOSURE = "anchor_conditioned_closure.jsonl"
DEFAULT_EDGE_SUMMARY = "edge_transport_by_type.csv"
DEFAULT_CYCLE_SUMMARY = "cycle_summary_by_cell.csv"
DEFAULT_ANCHOR_SUMMARY = "anchor_conditioned_closure_by_cell.csv"
DEFAULT_REPORT = "gate9a_failure_surface.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


@dataclass
class LocalObject:
    node_id: str
    node_type: str
    execution_sample_id: int
    benchmark_sample_id: str
    cell_id: str
    world_id: str
    world_type: str
    answer_target_type: str
    quietness_pair_id: str
    rendering_family_id: str
    basis: np.ndarray
    singular_values: np.ndarray
    rank_local: int
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate9A graph-gauge object-level diagnostics from an existing Gate8 "
            "candidate execution bundle without reopening the Gate8 court."
        )
    )
    parser.add_argument("--gate8-execution-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


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


def read_jsonl_if_exists(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    return read_jsonl(path)


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


def build_anchor_object(anchor_triplet_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not anchor_triplet_rows:
        raise ValueError("anchor triplets are empty")

    columns: List[np.ndarray] = []
    for row in anchor_triplet_rows:
        for key in gate6_builder.RAW_KEYS:
            raw = gate6_builder.load_raw_vector(row, key)
            normalized, _norm = gate6_builder.normalize_raw_vector(raw)
            columns.append(normalized)
    construction = np.stack(columns, axis=1)
    u_matrix, singular_values, _vt = np.linalg.svd(construction, full_matrices=False)

    basis = np.zeros((construction.shape[0], gate6_builder.LOCAL_DIM_MAX), dtype=np.float64)
    singular_values_padded = np.zeros(gate6_builder.LOCAL_DIM_MAX, dtype=np.float64)
    take = min(gate6_builder.LOCAL_DIM_MAX, singular_values.shape[0])
    singular_values_padded[:take] = singular_values[:take]
    sigma_1 = float(singular_values_padded[0]) if singular_values_padded.size else 0.0
    rank_cutoff = max(gate6_builder.TAU_RANK_ABS, gate6_builder.TAU_RANK_REL * sigma_1)
    rank_local = int(np.sum(singular_values_padded >= rank_cutoff))
    for axis_idx in range(rank_local):
        fixed_column, _flipped, _anchor_index = gate6_builder.sign_fix_column(u_matrix[:, axis_idx])
        basis[:, axis_idx] = fixed_column

    return {
        "basis": basis,
        "singular_values": singular_values_padded,
        "rank_local": rank_local,
        "n_anchor_steps": len(anchor_triplet_rows),
        "n_anchor_columns": len(columns),
    }


def build_projector(basis: np.ndarray, rank_local: int) -> Optional[np.ndarray]:
    if rank_local <= 0:
        return None
    basis_slice = np.asarray(basis[:, :rank_local], dtype=np.float64)
    return basis_slice @ basis_slice.T


def projector_gap(
    current_basis: np.ndarray,
    current_rank: int,
    next_basis: np.ndarray,
    next_rank: int,
) -> Optional[float]:
    if current_rank <= 0 or next_rank <= 0:
        return None
    current_slice = np.asarray(current_basis[:, :current_rank], dtype=np.float64)
    next_slice = np.asarray(next_basis[:, :next_rank], dtype=np.float64)
    overlap_matrix = current_slice.T @ next_slice
    overlap_trace = float(np.sum(np.square(overlap_matrix)))
    denom = float(max(1, min(current_rank, next_rank)))
    return float(np.clip(1.0 - (overlap_trace / denom), 0.0, 1.0))


def coverage_ratio(
    anchor_basis: np.ndarray,
    anchor_rank: int,
    target_basis: np.ndarray,
    target_rank: int,
) -> Optional[float]:
    if anchor_rank <= 0 or target_rank <= 0:
        return None
    anchor_slice = np.asarray(anchor_basis[:, :anchor_rank], dtype=np.float64)
    target_slice = np.asarray(target_basis[:, :target_rank], dtype=np.float64)
    overlap = anchor_slice.T @ target_slice
    value = float(np.sum(np.square(overlap)) / float(max(1, target_rank)))
    return float(np.clip(value, 0.0, 1.0))


def restricted_factor(
    target_basis: np.ndarray,
    target_rank: int,
    anchor_basis: np.ndarray,
    anchor_rank: int,
) -> Tuple[Optional[np.ndarray], int]:
    anchor_projector = build_projector(anchor_basis, anchor_rank)
    if anchor_projector is None or target_rank <= 0:
        return None, 0
    target_slice = np.asarray(target_basis[:, :target_rank], dtype=np.float64)
    projected = anchor_projector @ target_slice
    if not np.isfinite(projected).all():
        return None, 0
    u_matrix, singular_values, _vt = np.linalg.svd(projected, full_matrices=False)
    sigma_1 = float(singular_values[0]) if singular_values.size else 0.0
    if not np.isfinite(sigma_1) or sigma_1 <= gate6_builder.TAU_RANK_ABS:
        return None, 0
    rank_cutoff = max(gate6_builder.TAU_RANK_ABS, gate6_builder.TAU_RANK_REL * sigma_1)
    rank_local = int(np.sum(singular_values >= rank_cutoff))
    if rank_local <= 0:
        return None, 0
    factor = np.zeros((projected.shape[0], gate6_builder.LOCAL_DIM_MAX), dtype=np.float64)
    for axis_idx in range(min(rank_local, gate6_builder.LOCAL_DIM_MAX)):
        fixed_column, _flipped, _anchor_index = gate6_builder.sign_fix_column(u_matrix[:, axis_idx])
        factor[:, axis_idx] = fixed_column
    return factor, min(rank_local, gate6_builder.LOCAL_DIM_MAX)


def build_transport(
    source_basis: np.ndarray,
    source_rank: int,
    target_basis: np.ndarray,
    target_rank: int,
) -> Dict[str, Any]:
    if source_rank <= 0:
        return {
            "edge_outcome": "invalid_source_rank",
            "transport_mode": "",
            "edge_transport_defect": None,
            "overlap_ratio": None,
            "coord_isometry": None,
        }
    if target_rank <= 0:
        return {
            "edge_outcome": "invalid_target_rank",
            "transport_mode": "",
            "edge_transport_defect": None,
            "overlap_ratio": None,
            "coord_isometry": None,
        }
    source_slice = np.asarray(source_basis[:, :source_rank], dtype=np.float64)
    target_slice = np.asarray(target_basis[:, :target_rank], dtype=np.float64)
    overlap = target_slice.T @ source_slice
    if not np.isfinite(overlap).all():
        return {
            "edge_outcome": "invalid_overlap",
            "transport_mode": "",
            "edge_transport_defect": None,
            "overlap_ratio": None,
            "coord_isometry": None,
        }
    u_matrix, singular_values, vt_matrix = np.linalg.svd(overlap, full_matrices=False)
    coord_isometry = u_matrix @ vt_matrix
    overlap_ratio = float(
        np.clip(
            np.sum(np.square(singular_values)) / float(max(1, min(source_rank, target_rank))),
            0.0,
            1.0,
        )
    )
    return {
        "edge_outcome": "none",
        "transport_mode": (
            "orthogonal_equal_rank"
            if source_rank == target_rank
            else "partial_isometry_rank_mismatch"
        ),
        "edge_transport_defect": float(np.clip(1.0 - overlap_ratio, 0.0, 1.0)),
        "overlap_ratio": overlap_ratio,
        "coord_isometry": coord_isometry,
    }


def build_local_object_row(local_object: LocalObject) -> Dict[str, Any]:
    return {
        "node_id": local_object.node_id,
        "node_type": local_object.node_type,
        "execution_sample_id": local_object.execution_sample_id,
        "benchmark_sample_id": local_object.benchmark_sample_id,
        "cell_id": local_object.cell_id,
        "world_id": local_object.world_id,
        "world_type": local_object.world_type,
        "answer_target_type": local_object.answer_target_type,
        "quietness_pair_id": local_object.quietness_pair_id,
        "rendering_family_id": local_object.rendering_family_id,
        "projector_public_kind": PROJECTOR_PUBLIC_KIND,
        "ambient_dim": int(local_object.basis.shape[0]),
        "rank_local": int(local_object.rank_local),
        "projector_trace": int(local_object.rank_local),
        "singular_values": [float(x) for x in np.asarray(local_object.singular_values, dtype=np.float64)],
        "metadata": local_object.metadata,
    }


def make_node_id(execution_sample_id: int, suffix: str) -> str:
    return f"sample_{execution_sample_id:06d}:{suffix}"


def load_gate6_arrays(gate6_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    step_rows = gate7c_consumer.load_rows(gate6_dir / gate7c_consumer.DEFAULT_STEP_INDEX)
    with np.load(gate6_dir / gate7c_consumer.DEFAULT_ARRAYS) as npz_handle:
        arrays = {
            "basis": np.asarray(npz_handle["basis"], dtype=np.float64),
            "coords_local": np.asarray(npz_handle["coords_local"], dtype=np.float64),
            "singular_values": np.asarray(npz_handle["singular_values"], dtype=np.float64),
            "rank_local": np.asarray(npz_handle["rank_local"], dtype=np.int64),
        }
    return step_rows, arrays


def build_token_local_object(
    registry_row: Dict[str, Any],
    step_row: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
) -> LocalObject:
    array_row_index = int(step_row["array_row_index"])
    return LocalObject(
        node_id=make_node_id(
            int(registry_row["execution_sample_id"]),
            f"token_state_{int(step_row['step']):04d}",
        ),
        node_type="token_state",
        execution_sample_id=int(registry_row["execution_sample_id"]),
        benchmark_sample_id=str(registry_row["benchmark_sample_id"]),
        cell_id=str(registry_row["cell_id"]),
        world_id=str(registry_row["world_id"]),
        world_type=str(registry_row["world_type"]),
        answer_target_type=str(registry_row["answer_target_type"]),
        quietness_pair_id=str(registry_row.get("quietness_pair_id") or ""),
        rendering_family_id=str(registry_row.get("rendering_family_id") or ""),
        basis=np.asarray(arrays["basis"][array_row_index], dtype=np.float64),
        singular_values=np.asarray(arrays["singular_values"][array_row_index], dtype=np.float64),
        rank_local=int(arrays["rank_local"][array_row_index]),
        metadata={
            "step": int(step_row["step"]),
            "token_text": str(step_row["token_text"]),
            "label_token": int(step_row["label_token"]),
            "flags_compact": str(step_row["flags_compact"]),
            "array_row_index": array_row_index,
        },
    )


def build_anchor_local_object(
    registry_row: Dict[str, Any],
    node_type: str,
    node_suffix: str,
    triplet_rows: Sequence[Dict[str, Any]],
    extra_meta: Dict[str, Any],
) -> LocalObject:
    anchor_object = build_anchor_object(triplet_rows)
    return LocalObject(
        node_id=make_node_id(int(registry_row["execution_sample_id"]), node_suffix),
        node_type=node_type,
        execution_sample_id=int(registry_row["execution_sample_id"]),
        benchmark_sample_id=str(registry_row["benchmark_sample_id"]),
        cell_id=str(registry_row["cell_id"]),
        world_id=str(registry_row["world_id"]),
        world_type=str(registry_row["world_type"]),
        answer_target_type=str(registry_row["answer_target_type"]),
        quietness_pair_id=str(registry_row.get("quietness_pair_id") or ""),
        rendering_family_id=str(registry_row.get("rendering_family_id") or ""),
        basis=np.asarray(anchor_object["basis"], dtype=np.float64),
        singular_values=np.asarray(anchor_object["singular_values"], dtype=np.float64),
        rank_local=int(anchor_object["rank_local"]),
        metadata={
            "n_anchor_steps": int(anchor_object["n_anchor_steps"]),
            "n_anchor_columns": int(anchor_object["n_anchor_columns"]),
            **extra_meta,
        },
    )


def build_edge_row(
    edge_id: str,
    edge_type: str,
    source_object: LocalObject,
    target_object: LocalObject,
    metadata: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    transport = build_transport(
        source_basis=source_object.basis,
        source_rank=source_object.rank_local,
        target_basis=target_object.basis,
        target_rank=target_object.rank_local,
    )
    row = {
        "edge_id": edge_id,
        "edge_type": edge_type,
        "source_node_id": source_object.node_id,
        "target_node_id": target_object.node_id,
        "execution_sample_id": source_object.execution_sample_id,
        "benchmark_sample_id": source_object.benchmark_sample_id,
        "cell_id": source_object.cell_id,
        "world_id": source_object.world_id,
        "world_type": source_object.world_type,
        "answer_target_type": source_object.answer_target_type,
        "quietness_pair_id": source_object.quietness_pair_id,
        "rendering_family_id": source_object.rendering_family_id,
        "source_node_type": source_object.node_type,
        "target_node_type": target_object.node_type,
        "source_rank": int(source_object.rank_local),
        "target_rank": int(target_object.rank_local),
        "edge_outcome": transport["edge_outcome"],
        "transport_mode": transport["transport_mode"],
        "edge_transport_defect": transport["edge_transport_defect"],
        "overlap_ratio": transport["overlap_ratio"],
        "metadata": metadata,
    }
    return row, transport


def compute_cycle_holonomy(
    root_object: LocalObject,
    edge_rows: Sequence[Dict[str, Any]],
    edge_transport_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if root_object.rank_local <= 0:
        return {
            "cycle_outcome": "invalid_root_rank",
            "holonomy_defect": None,
            "holonomy_trace": None,
        }
    composite = np.eye(root_object.rank_local, dtype=np.float64)
    for edge_row in edge_rows:
        transport = edge_transport_map[edge_row["edge_id"]]
        if transport["edge_outcome"] != "none":
            return {
                "cycle_outcome": f"edge_failure:{transport['edge_outcome']}",
                "holonomy_defect": None,
                "holonomy_trace": None,
            }
        coord_isometry = transport["coord_isometry"]
        if coord_isometry is None:
            return {
                "cycle_outcome": "missing_coord_isometry",
                "holonomy_defect": None,
                "holonomy_trace": None,
            }
        composite = np.asarray(coord_isometry, dtype=np.float64) @ composite
    identity = np.eye(root_object.rank_local, dtype=np.float64)
    defect = float(np.linalg.norm(composite - identity, ord="fro") / math.sqrt(float(root_object.rank_local)))
    return {
        "cycle_outcome": "none",
        "holonomy_defect": defect,
        "holonomy_trace": float(np.trace(composite)),
    }


def compute_anchor_conditioned_closure(
    anchor_object: LocalObject,
    answer_object: LocalObject,
    token_object: LocalObject,
) -> Dict[str, Any]:
    answer_coverage = coverage_ratio(
        anchor_basis=anchor_object.basis,
        anchor_rank=anchor_object.rank_local,
        target_basis=answer_object.basis,
        target_rank=answer_object.rank_local,
    )
    token_coverage = coverage_ratio(
        anchor_basis=anchor_object.basis,
        anchor_rank=anchor_object.rank_local,
        target_basis=token_object.basis,
        target_rank=token_object.rank_local,
    )
    restricted_answer_basis, restricted_answer_rank = restricted_factor(
        target_basis=answer_object.basis,
        target_rank=answer_object.rank_local,
        anchor_basis=anchor_object.basis,
        anchor_rank=anchor_object.rank_local,
    )
    restricted_token_basis, restricted_token_rank = restricted_factor(
        target_basis=token_object.basis,
        target_rank=token_object.rank_local,
        anchor_basis=anchor_object.basis,
        anchor_rank=anchor_object.rank_local,
    )
    if restricted_answer_basis is None or restricted_answer_rank <= 0:
        return {
            "closure_outcome": "insufficient_answer_anchor_overlap",
            "anchor_answer_coverage": answer_coverage,
            "anchor_token_coverage": token_coverage,
            "answer_conditioned_rank": restricted_answer_rank,
            "token_conditioned_rank": restricted_token_rank,
            "anchor_conditioned_closure_defect": None,
        }
    if restricted_token_basis is None or restricted_token_rank <= 0:
        return {
            "closure_outcome": "insufficient_token_anchor_overlap",
            "anchor_answer_coverage": answer_coverage,
            "anchor_token_coverage": token_coverage,
            "answer_conditioned_rank": restricted_answer_rank,
            "token_conditioned_rank": restricted_token_rank,
            "anchor_conditioned_closure_defect": None,
        }
    closure_defect = projector_gap(
        current_basis=restricted_answer_basis,
        current_rank=restricted_answer_rank,
        next_basis=restricted_token_basis,
        next_rank=restricted_token_rank,
    )
    return {
        "closure_outcome": "none" if closure_defect is not None else "invalid_conditioned_projector_gap",
        "anchor_answer_coverage": answer_coverage,
        "anchor_token_coverage": token_coverage,
        "answer_conditioned_rank": restricted_answer_rank,
        "token_conditioned_rank": restricted_token_rank,
        "anchor_conditioned_closure_defect": closure_defect,
    }


def summarize_edges(edge_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        grouped[(str(row["edge_type"]), str(row["edge_outcome"]))].append(row)
    out_rows: List[Dict[str, Any]] = []
    for edge_type, edge_outcome in sorted(grouped):
        rows = grouped[(edge_type, edge_outcome)]
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "edge_type": edge_type,
                "edge_outcome": edge_outcome,
                "n_edges": len(rows),
                "mean_edge_transport_defect": None if not defects else float(sum(defects) / len(defects)),
                "max_edge_transport_defect": None if not defects else float(max(defects)),
            }
        )
    return out_rows


def summarize_cycles(cycle_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in cycle_rows:
        grouped[(str(row["cell_id"]), str(row["cycle_type"]), str(row["cycle_outcome"]))].append(row)
    out_rows: List[Dict[str, Any]] = []
    for cell_id, cycle_type, cycle_outcome in sorted(grouped):
        rows = grouped[(cell_id, cycle_type, cycle_outcome)]
        defects = [
            float(row["holonomy_defect"])
            for row in rows
            if row["holonomy_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_id": cell_id,
                "cycle_type": cycle_type,
                "cycle_outcome": cycle_outcome,
                "n_cycles": len(rows),
                "mean_holonomy_defect": None if not defects else float(sum(defects) / len(defects)),
                "max_holonomy_defect": None if not defects else float(max(defects)),
            }
        )
    return out_rows


def summarize_anchor_closure(anchor_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        grouped[(str(row["cell_id"]), str(row["anchor_kind"]), str(row["closure_outcome"]))].append(row)
    out_rows: List[Dict[str, Any]] = []
    for cell_id, anchor_kind, closure_outcome in sorted(grouped):
        rows = grouped[(cell_id, anchor_kind, closure_outcome)]
        defects = [
            float(row["anchor_conditioned_closure_defect"])
            for row in rows
            if row["anchor_conditioned_closure_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_id": cell_id,
                "anchor_kind": anchor_kind,
                "closure_outcome": closure_outcome,
                "n_rows": len(rows),
                "mean_anchor_conditioned_closure_defect": (
                    None if not defects else float(sum(defects) / len(defects))
                ),
                "max_anchor_conditioned_closure_defect": None if not defects else float(max(defects)),
            }
        )
    return out_rows


def build_report(
    run_id: str,
    source_execution_manifest: Dict[str, Any],
    node_rows: Sequence[Dict[str, Any]],
    edge_summary_rows: Sequence[Dict[str, Any]],
    cycle_summary_rows: Sequence[Dict[str, Any]],
    anchor_summary_rows: Sequence[Dict[str, Any]],
) -> str:
    node_counts = Counter(str(row["node_type"]) for row in node_rows)
    lines = [
        "# Gate9A Graph-Gauge Failure Surface",
        "",
        f"run_id: {run_id}",
        f"source_gate8_run_id: {source_execution_manifest.get('run_id', '')}",
        f"source_rendering_family_id: {source_execution_manifest.get('rendering_family_id', '')}",
        f"source_code_git_commit: {source_execution_manifest.get('code_git_commit', '')}",
        f"projector_public_kind: {PROJECTOR_PUBLIC_KIND}",
        f"cycle_token_selector: {CYCLE_TOKEN_SELECTOR}",
        "",
        "## Node Counts",
        "",
    ]
    for node_type in NODE_TYPES:
        lines.append(f"- {node_type}: {node_counts.get(node_type, 0)}")

    lines.extend(
        [
            "",
            "## Edge Failure Surface",
            "",
            "| edge_type | edge_outcome | n_edges | mean_edge_transport_defect | max_edge_transport_defect |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in edge_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["edge_type"]),
                    str(row["edge_outcome"]),
                    str(row["n_edges"]),
                    ""
                    if row["mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['mean_edge_transport_defect']):.6f}",
                    ""
                    if row["max_edge_transport_defect"] in (None, "")
                    else f"{float(row['max_edge_transport_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Small-Cycle Holonomy",
            "",
            "| cell_id | cycle_type | cycle_outcome | n_cycles | mean_holonomy_defect | max_holonomy_defect |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in cycle_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["cycle_type"]),
                    str(row["cycle_outcome"]),
                    str(row["n_cycles"]),
                    ""
                    if row["mean_holonomy_defect"] in (None, "")
                    else f"{float(row['mean_holonomy_defect']):.6f}",
                    ""
                    if row["max_holonomy_defect"] in (None, "")
                    else f"{float(row['max_holonomy_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Anchor-Conditioned Closure",
            "",
            "| cell_id | anchor_kind | closure_outcome | n_rows | mean_anchor_conditioned_closure_defect | max_anchor_conditioned_closure_defect |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in anchor_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["anchor_kind"]),
                    str(row["closure_outcome"]),
                    str(row["n_rows"]),
                    ""
                    if row["mean_anchor_conditioned_closure_defect"] in (None, "")
                    else f"{float(row['mean_anchor_conditioned_closure_defect']):.6f}",
                    ""
                    if row["max_anchor_conditioned_closure_defect"] in (None, "")
                    else f"{float(row['max_anchor_conditioned_closure_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Discipline",
            "",
            "- explicit cycle closure only; no implicit return legs",
            "- projector remains the public primitive; basis remains auxiliary for transport computation",
            "- failure enums are emitted before any aggregate interpretation",
            "- distributed_incompatibility remains the primary proving ground for this line",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_dir = Path(args.gate8_execution_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_manifest = read_json(source_dir / "manifest.json")
    sample_registry_rows = read_jsonl(source_dir / "sample_registry.jsonl")
    quietness_pair_rows = read_jsonl(source_dir / "quietness_pairs.jsonl")
    gate6_dir = source_dir / "gate6_native"
    step_rows, arrays = load_gate6_arrays(gate6_dir)
    registry_by_execution_id = {
        int(row["execution_sample_id"]): row for row in sample_registry_rows
    }
    registry_by_benchmark_id = {
        str(row["benchmark_sample_id"]): row for row in sample_registry_rows
    }
    sample_steps: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        sample_steps[int(row["sample_id"])].append(row)
    for rows in sample_steps.values():
        rows.sort(key=lambda row: int(row["step"]))

    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    anchor_closure_rows: List[Dict[str, Any]] = []
    local_objects: Dict[str, LocalObject] = {}
    edge_transport_map: Dict[str, Dict[str, Any]] = {}
    answer_node_ids_by_sample: Dict[int, str] = {}
    support_node_ids_by_sample: Dict[int, str] = {}
    conflict_node_ids_by_sample: Dict[int, str] = {}
    terminal_token_ids_by_sample: Dict[int, str] = {}

    for execution_sample_id in sorted(registry_by_execution_id):
        registry_row = registry_by_execution_id[execution_sample_id]
        sample_dir = source_dir / "samples" / f"sample_{execution_sample_id:06d}"
        token_step_rows = sample_steps.get(execution_sample_id, [])
        terminal_token_row = token_step_rows[-1] if token_step_rows else None

        for step_row in token_step_rows:
            token_object = build_token_local_object(registry_row, step_row, arrays)
            local_objects[token_object.node_id] = token_object
            node_rows.append(build_local_object_row(token_object))
        if terminal_token_row is not None:
            terminal_token_ids_by_sample[execution_sample_id] = make_node_id(
                execution_sample_id,
                f"token_state_{int(terminal_token_row['step']):04d}",
            )

        answer_triplet_rows = read_jsonl(sample_dir / "triplets.ndjson")
        answer_object = build_anchor_local_object(
            registry_row=registry_row,
            node_type="answer_state",
            node_suffix="answer_state",
            triplet_rows=answer_triplet_rows,
            extra_meta={"source_triplets": "triplets.ndjson"},
        )
        local_objects[answer_object.node_id] = answer_object
        node_rows.append(build_local_object_row(answer_object))
        answer_node_ids_by_sample[execution_sample_id] = answer_object.node_id

        support_triplet_rows = read_jsonl_if_exists(sample_dir / "support_anchor_triplets.ndjson")
        support_object: Optional[LocalObject] = None
        if support_triplet_rows is not None:
            support_object = build_anchor_local_object(
                registry_row=registry_row,
                node_type="support_chunk",
                node_suffix="support_chunk",
                triplet_rows=support_triplet_rows,
                extra_meta={"source_triplets": "support_anchor_triplets.ndjson"},
            )
            local_objects[support_object.node_id] = support_object
            node_rows.append(build_local_object_row(support_object))
            support_node_ids_by_sample[execution_sample_id] = support_object.node_id

        conflict_triplet_rows = read_jsonl_if_exists(sample_dir / "conflict_anchor_triplets.ndjson")
        conflict_object: Optional[LocalObject] = None
        if conflict_triplet_rows is not None:
            conflict_object = build_anchor_local_object(
                registry_row=registry_row,
                node_type="conflict_chunk",
                node_suffix="conflict_chunk",
                triplet_rows=conflict_triplet_rows,
                extra_meta={"source_triplets": "conflict_anchor_triplets.ndjson"},
            )
            local_objects[conflict_object.node_id] = conflict_object
            node_rows.append(build_local_object_row(conflict_object))
            conflict_node_ids_by_sample[execution_sample_id] = conflict_object.node_id

        for index in range(len(token_step_rows) - 1):
            source_step = token_step_rows[index]
            target_step = token_step_rows[index + 1]
            source_node_id = make_node_id(execution_sample_id, f"token_state_{int(source_step['step']):04d}")
            target_node_id = make_node_id(execution_sample_id, f"token_state_{int(target_step['step']):04d}")
            edge_id = f"edge:{source_node_id}->{target_node_id}:temporal_transition"
            row, transport = build_edge_row(
                edge_id=edge_id,
                edge_type="temporal_transition",
                source_object=local_objects[source_node_id],
                target_object=local_objects[target_node_id],
                metadata={"direction": "forward"},
            )
            edge_rows.append(row)
            edge_transport_map[edge_id] = transport

        terminal_token_id = terminal_token_ids_by_sample.get(execution_sample_id)
        if terminal_token_id:
            edge_specs = [
                ("answer_projection", answer_object.node_id, terminal_token_id, {"direction": "answer_to_terminal_token"}),
                ("answer_projection", terminal_token_id, answer_object.node_id, {"direction": "terminal_token_to_answer"}),
            ]
            if support_object is not None:
                edge_specs.extend(
                    [
                        ("support_anchor", support_object.node_id, answer_object.node_id, {"direction": "support_to_answer"}),
                        ("support_anchor", answer_object.node_id, support_object.node_id, {"direction": "answer_to_support"}),
                        ("support_anchor", terminal_token_id, support_object.node_id, {"direction": "terminal_token_to_support"}),
                        ("support_anchor", support_object.node_id, terminal_token_id, {"direction": "support_to_terminal_token"}),
                    ]
                )
            if conflict_object is not None:
                edge_specs.extend(
                    [
                        ("conflict_anchor", conflict_object.node_id, answer_object.node_id, {"direction": "conflict_to_answer"}),
                        ("conflict_anchor", answer_object.node_id, conflict_object.node_id, {"direction": "answer_to_conflict"}),
                        ("conflict_anchor", terminal_token_id, conflict_object.node_id, {"direction": "terminal_token_to_conflict"}),
                        ("conflict_anchor", conflict_object.node_id, terminal_token_id, {"direction": "conflict_to_terminal_token"}),
                    ]
                )
            for edge_type, source_node_id, target_node_id, metadata in edge_specs:
                edge_id = f"edge:{source_node_id}->{target_node_id}:{edge_type}"
                row, transport = build_edge_row(
                    edge_id=edge_id,
                    edge_type=edge_type,
                    source_object=local_objects[source_node_id],
                    target_object=local_objects[target_node_id],
                    metadata=metadata,
                )
                edge_rows.append(row)
                edge_transport_map[edge_id] = transport

            cycle_specs: List[Tuple[str, str, List[str], LocalObject]] = []
            if support_object is not None:
                cycle_specs.append(
                    (
                        "support_answer_terminal_token_cycle",
                        support_object.node_id,
                        [
                            f"edge:{support_object.node_id}->{answer_object.node_id}:support_anchor",
                            f"edge:{answer_object.node_id}->{terminal_token_id}:answer_projection",
                            f"edge:{terminal_token_id}->{support_object.node_id}:support_anchor",
                        ],
                        support_object,
                    )
                )
            else:
                cycle_rows.append(
                    {
                        "cycle_id": f"cycle:{execution_sample_id}:support_answer_terminal_token_cycle",
                        "cycle_type": "support_answer_terminal_token_cycle",
                        "root_node_id": "",
                        "execution_sample_id": execution_sample_id,
                        "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                        "cell_id": str(registry_row["cell_id"]),
                        "world_id": str(registry_row["world_id"]),
                        "world_type": str(registry_row["world_type"]),
                        "answer_target_type": str(registry_row["answer_target_type"]),
                        "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                        "cycle_outcome": "missing_support_anchor",
                        "holonomy_defect": None,
                        "holonomy_trace": None,
                        "edge_ids": [],
                        "metadata": {},
                    }
                )
            if conflict_object is not None:
                cycle_specs.append(
                    (
                        "conflict_answer_terminal_token_cycle",
                        conflict_object.node_id,
                        [
                            f"edge:{conflict_object.node_id}->{answer_object.node_id}:conflict_anchor",
                            f"edge:{answer_object.node_id}->{terminal_token_id}:answer_projection",
                            f"edge:{terminal_token_id}->{conflict_object.node_id}:conflict_anchor",
                        ],
                        conflict_object,
                    )
                )
            else:
                cycle_rows.append(
                    {
                        "cycle_id": f"cycle:{execution_sample_id}:conflict_answer_terminal_token_cycle",
                        "cycle_type": "conflict_answer_terminal_token_cycle",
                        "root_node_id": "",
                        "execution_sample_id": execution_sample_id,
                        "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                        "cell_id": str(registry_row["cell_id"]),
                        "world_id": str(registry_row["world_id"]),
                        "world_type": str(registry_row["world_type"]),
                        "answer_target_type": str(registry_row["answer_target_type"]),
                        "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                        "cycle_outcome": "missing_conflict_anchor",
                        "holonomy_defect": None,
                        "holonomy_trace": None,
                        "edge_ids": [],
                        "metadata": {},
                    }
                )
            for cycle_type, root_node_id, edge_ids, root_object in cycle_specs:
                cycle_edge_rows: List[Dict[str, Any]] = []
                missing_edge_id = ""
                for edge_id in edge_ids:
                    matching = next((row for row in edge_rows if row["edge_id"] == edge_id), None)
                    if matching is None:
                        missing_edge_id = edge_id
                        break
                    cycle_edge_rows.append(matching)
                if missing_edge_id:
                    cycle_rows.append(
                        {
                            "cycle_id": f"cycle:{execution_sample_id}:{cycle_type}",
                            "cycle_type": cycle_type,
                            "root_node_id": root_node_id,
                            "execution_sample_id": execution_sample_id,
                            "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                            "cell_id": str(registry_row["cell_id"]),
                            "world_id": str(registry_row["world_id"]),
                            "world_type": str(registry_row["world_type"]),
                            "answer_target_type": str(registry_row["answer_target_type"]),
                            "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                            "cycle_outcome": "missing_cycle_edge",
                            "holonomy_defect": None,
                            "holonomy_trace": None,
                            "edge_ids": edge_ids,
                            "metadata": {"missing_edge_id": missing_edge_id},
                        }
                    )
                    continue
                cycle_metrics = compute_cycle_holonomy(
                    root_object=root_object,
                    edge_rows=cycle_edge_rows,
                    edge_transport_map=edge_transport_map,
                )
                cycle_rows.append(
                    {
                        "cycle_id": f"cycle:{execution_sample_id}:{cycle_type}",
                        "cycle_type": cycle_type,
                        "root_node_id": root_node_id,
                        "execution_sample_id": execution_sample_id,
                        "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                        "cell_id": str(registry_row["cell_id"]),
                        "world_id": str(registry_row["world_id"]),
                        "world_type": str(registry_row["world_type"]),
                        "answer_target_type": str(registry_row["answer_target_type"]),
                        "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                        "cycle_outcome": cycle_metrics["cycle_outcome"],
                        "holonomy_defect": cycle_metrics["holonomy_defect"],
                        "holonomy_trace": cycle_metrics["holonomy_trace"],
                        "edge_ids": edge_ids,
                        "metadata": {"explicit_cycle": True},
                    }
                )

            for anchor_kind, anchor_object in (("support", support_object), ("conflict", conflict_object)):
                if anchor_object is None:
                    anchor_closure_rows.append(
                        {
                            "closure_id": f"closure:{execution_sample_id}:{anchor_kind}",
                            "anchor_kind": anchor_kind,
                            "anchor_node_id": "",
                            "answer_node_id": answer_object.node_id,
                            "token_node_id": terminal_token_id,
                            "execution_sample_id": execution_sample_id,
                            "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                            "cell_id": str(registry_row["cell_id"]),
                            "world_id": str(registry_row["world_id"]),
                            "world_type": str(registry_row["world_type"]),
                            "answer_target_type": str(registry_row["answer_target_type"]),
                            "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                            "closure_outcome": f"missing_{anchor_kind}_anchor",
                            "anchor_answer_coverage": None,
                            "anchor_token_coverage": None,
                            "answer_conditioned_rank": None,
                            "token_conditioned_rank": None,
                            "anchor_conditioned_closure_defect": None,
                        }
                    )
                    continue
                closure_metrics = compute_anchor_conditioned_closure(
                    anchor_object=anchor_object,
                    answer_object=answer_object,
                    token_object=local_objects[terminal_token_id],
                )
                anchor_closure_rows.append(
                    {
                        "closure_id": f"closure:{execution_sample_id}:{anchor_kind}",
                        "anchor_kind": anchor_kind,
                        "anchor_node_id": anchor_object.node_id,
                        "answer_node_id": answer_object.node_id,
                        "token_node_id": terminal_token_id,
                        "execution_sample_id": execution_sample_id,
                        "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                        "cell_id": str(registry_row["cell_id"]),
                        "world_id": str(registry_row["world_id"]),
                        "world_type": str(registry_row["world_type"]),
                        "answer_target_type": str(registry_row["answer_target_type"]),
                        "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                        "closure_outcome": closure_metrics["closure_outcome"],
                        "anchor_answer_coverage": closure_metrics["anchor_answer_coverage"],
                        "anchor_token_coverage": closure_metrics["anchor_token_coverage"],
                        "answer_conditioned_rank": closure_metrics["answer_conditioned_rank"],
                        "token_conditioned_rank": closure_metrics["token_conditioned_rank"],
                        "anchor_conditioned_closure_defect": closure_metrics["anchor_conditioned_closure_defect"],
                    }
                )
        else:
            for cycle_type, root_node_id in (
                ("support_answer_terminal_token_cycle", "" if support_object is None else support_object.node_id),
                ("conflict_answer_terminal_token_cycle", "" if conflict_object is None else conflict_object.node_id),
            ):
                cycle_rows.append(
                    {
                        "cycle_id": f"cycle:{execution_sample_id}:{cycle_type}",
                        "cycle_type": cycle_type,
                        "root_node_id": root_node_id,
                        "execution_sample_id": execution_sample_id,
                        "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                        "cell_id": str(registry_row["cell_id"]),
                        "world_id": str(registry_row["world_id"]),
                        "world_type": str(registry_row["world_type"]),
                        "answer_target_type": str(registry_row["answer_target_type"]),
                        "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                        "cycle_outcome": "missing_terminal_token",
                        "holonomy_defect": None,
                        "holonomy_trace": None,
                        "edge_ids": [],
                        "metadata": {},
                    }
                )
            for anchor_kind, anchor_object in (("support", support_object), ("conflict", conflict_object)):
                anchor_closure_rows.append(
                    {
                        "closure_id": f"closure:{execution_sample_id}:{anchor_kind}",
                        "anchor_kind": anchor_kind,
                        "anchor_node_id": "" if anchor_object is None else anchor_object.node_id,
                        "answer_node_id": answer_object.node_id,
                        "token_node_id": "",
                        "execution_sample_id": execution_sample_id,
                        "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                        "cell_id": str(registry_row["cell_id"]),
                        "world_id": str(registry_row["world_id"]),
                        "world_type": str(registry_row["world_type"]),
                        "answer_target_type": str(registry_row["answer_target_type"]),
                        "rendering_family_id": str(registry_row.get("rendering_family_id") or ""),
                        "closure_outcome": "missing_terminal_token",
                        "anchor_answer_coverage": None,
                        "anchor_token_coverage": None,
                        "answer_conditioned_rank": None,
                        "token_conditioned_rank": None,
                        "anchor_conditioned_closure_defect": None,
                    }
                )

    for pair_row in quietness_pair_rows:
        clean_registry_row = registry_by_benchmark_id[str(pair_row["clean_benchmark_sample_id"])]
        noisy_registry_row = registry_by_benchmark_id[str(pair_row["surface_noisy_benchmark_sample_id"])]
        clean_sample_id = int(clean_registry_row["execution_sample_id"])
        noisy_sample_id = int(noisy_registry_row["execution_sample_id"])
        clean_answer_id = answer_node_ids_by_sample[clean_sample_id]
        noisy_answer_id = answer_node_ids_by_sample[noisy_sample_id]
        pair_id = str(pair_row["quietness_pair_id"])
        for source_node_id, target_node_id, direction in (
            (clean_answer_id, noisy_answer_id, "clean_to_surface_noisy"),
            (noisy_answer_id, clean_answer_id, "surface_noisy_to_clean"),
        ):
            edge_id = f"edge:{source_node_id}->{target_node_id}:quietness_pair"
            row, transport = build_edge_row(
                edge_id=edge_id,
                edge_type="quietness_pair",
                source_object=local_objects[source_node_id],
                target_object=local_objects[target_node_id],
                metadata={"direction": direction, "quietness_pair_id": pair_id},
            )
            edge_rows.append(row)
            edge_transport_map[edge_id] = transport

    edge_summary_rows = summarize_edges(edge_rows)
    cycle_summary_rows = summarize_cycles(cycle_rows)
    anchor_summary_rows = summarize_anchor_closure(anchor_closure_rows)

    node_registry_path = out_dir / DEFAULT_NODE_REGISTRY
    edge_registry_path = out_dir / DEFAULT_EDGE_REGISTRY
    cycle_registry_path = out_dir / DEFAULT_CYCLE_REGISTRY
    anchor_closure_path = out_dir / DEFAULT_ANCHOR_CLOSURE
    edge_summary_path = out_dir / DEFAULT_EDGE_SUMMARY
    cycle_summary_path = out_dir / DEFAULT_CYCLE_SUMMARY
    anchor_summary_path = out_dir / DEFAULT_ANCHOR_SUMMARY
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_jsonl(node_registry_path, node_rows)
    write_jsonl(edge_registry_path, edge_rows)
    write_jsonl(cycle_registry_path, cycle_rows)
    write_jsonl(anchor_closure_path, anchor_closure_rows)
    write_csv(
        edge_summary_path,
        ("edge_type", "edge_outcome", "n_edges", "mean_edge_transport_defect", "max_edge_transport_defect"),
        edge_summary_rows,
    )
    write_csv(
        cycle_summary_path,
        ("cell_id", "cycle_type", "cycle_outcome", "n_cycles", "mean_holonomy_defect", "max_holonomy_defect"),
        cycle_summary_rows,
    )
    write_csv(
        anchor_summary_path,
        (
            "cell_id",
            "anchor_kind",
            "closure_outcome",
            "n_rows",
            "mean_anchor_conditioned_closure_defect",
            "max_anchor_conditioned_closure_defect",
        ),
        anchor_summary_rows,
    )
    write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_execution_manifest=source_manifest,
            node_rows=node_rows,
            edge_summary_rows=edge_summary_rows,
            cycle_summary_rows=cycle_summary_rows,
            anchor_summary_rows=anchor_summary_rows,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": current_git_commit(),
        "source_gate8_execution_dir": repo_relative_or_posix(source_dir),
        "source_gate8_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate8_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_rendering_family_id": str(source_manifest.get("rendering_family_id") or ""),
        "projector_public_kind": PROJECTOR_PUBLIC_KIND,
        "cycle_token_selector": CYCLE_TOKEN_SELECTOR,
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "primary_observables": list(PRIMARY_OBSERVABLES),
        "aggregation_ban_inherited": True,
        "quietness_pairing_rule": str(source_manifest.get("quietness_pairing_rule") or ""),
        "paths": {
            DEFAULT_NODE_REGISTRY: repo_relative_or_posix(node_registry_path),
            DEFAULT_EDGE_REGISTRY: repo_relative_or_posix(edge_registry_path),
            DEFAULT_CYCLE_REGISTRY: repo_relative_or_posix(cycle_registry_path),
            DEFAULT_ANCHOR_CLOSURE: repo_relative_or_posix(anchor_closure_path),
            DEFAULT_EDGE_SUMMARY: repo_relative_or_posix(edge_summary_path),
            DEFAULT_CYCLE_SUMMARY: repo_relative_or_posix(cycle_summary_path),
            DEFAULT_ANCHOR_SUMMARY: repo_relative_or_posix(anchor_summary_path),
            DEFAULT_REPORT: repo_relative_or_posix(report_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_NODE_REGISTRY: sha256_file(node_registry_path),
            DEFAULT_EDGE_REGISTRY: sha256_file(edge_registry_path),
            DEFAULT_CYCLE_REGISTRY: sha256_file(cycle_registry_path),
            DEFAULT_ANCHOR_CLOSURE: sha256_file(anchor_closure_path),
            DEFAULT_EDGE_SUMMARY: sha256_file(edge_summary_path),
            DEFAULT_CYCLE_SUMMARY: sha256_file(cycle_summary_path),
            DEFAULT_ANCHOR_SUMMARY: sha256_file(anchor_summary_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
