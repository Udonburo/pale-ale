"""Descriptive Gate12 artifact census with explicit convention blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ACTIVE_TOOLS = REPO_ROOT / "tools"
if str(ACTIVE_TOOLS) not in sys.path:
    sys.path.insert(0, str(ACTIVE_TOOLS))

import inspect_gate12c_associator_feasibility as gate12c  # noqa: E402

SCHEMA_VERSION = "gate13_candidate_topology_census_v0.1.1"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_quantiles(values: Iterable[float]) -> Dict[str, float | None]:
    array = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def graph_convention_declaration() -> Dict[str, Any]:
    return {
        "status": "PARTIAL_AUTHORITY_REVIEW1_BLOCKER",
        "authoritative_source": "tools/run_gate12a_discrete_connection_audit.py",
        "authoritative": {
            "graph_kind": (
                "directed provenance-bearing edge registry; the active manifest "
                "declares graph_object_policy=flat_artifact_only_v1"
            ),
            "edge_identity": "edge_id; duplicate edge_id rejected by active loader",
            "deduplication": (
                "transport rows are retained individually; explicit triangles are "
                "deduplicated by lexicographic base node plus sorted edge-id triple"
            ),
            "orientation": (
                "node_id_path gives directed traversal; stored edge_id_path is "
                "lexicographically sorted and must be reconstructed against node_id_path"
            ),
            "reciprocal_edges": "distinct directed rows when separately present",
            "parallel_edges": "retained as distinct edge_id rows",
            "self_loops": "retained by the edge registry if present; excluded by triangle construction",
            "registered_cycle_kind": (
                "directed length-3 cycles with three distinct nodes and edges and "
                "at least one residual_chord"
            ),
            "registered_cycle_basepoint": "lexicographically minimum node_id",
            "cycle_mode": "explicit_triangle_only_v1",
        },
        "unresolved": [
            "single connected-component convention for general census",
            "beta_1 formula for the directed provenance multigraph",
            "deterministic spanning-tree rule",
            "general fundamental-cycle orientation",
            "general cycle-vector field",
            "loop-independence definition",
            "shared-base definition beyond registered triangle base_node_id",
            "shared-vertex independent-loop-pair definition",
        ],
        "decision": (
            "registered-triangle and raw edge census may be reported; general beta_1, "
            "fundamental-cycle, and independent-loop metrics remain null until Review 1"
        ),
    }


def historical_scalar_candidates() -> list[Dict[str, Any]]:
    return [
        {
            "candidate_id": "gate12a_triangle_holonomy_residual_fro",
            "field": "holonomy_residual_fro",
            "artifact": "triangle_holonomy_registry.jsonl",
            "schema": "gate12a_discrete_connection_v1",
            "mode": "triangle_equal_rank_orthogonal_fro_residual_v1",
            "definition": "||T_cycle - I||_F",
            "normalization": "none",
            "source": "tools/run_gate12a_discrete_connection_audit.py:579-587",
        },
        {
            "candidate_id": "gate12c1_compressed_overlap_associator_fro",
            "field": "compressed_overlap_associator_fro",
            "artifact": "triangle_associator_registry.jsonl",
            "schema": "gate12c_compressed_overlap_associator_v1",
            "definition": "||Q_q(M2 M1) M0 - M2 Q_q(M1 M0)||_F",
            "normalization": "none",
            "source": "tools/run_gate12c_compressed_overlap_associator.py:486-513",
        },
        {
            "candidate_id": "gate12c1_compressed_overlap_closure_pair",
            "fields": [
                "compressed_overlap_closure_left_fro",
                "compressed_overlap_closure_right_fro",
                "compressed_overlap_closure_gap_abs",
            ],
            "artifact": "triangle_associator_registry.jsonl",
            "schema": "gate12c_compressed_overlap_associator_v1",
            "definition": "||L-I||_F, ||R-I||_F, and their absolute gap",
            "normalization": "none",
            "source": "tools/run_gate12c_compressed_overlap_associator.py:501-512",
        },
    ]


def overlap_metrics(matrix: np.ndarray, *, tolerance: float) -> Dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(value, compute_uv=False)
    numerical_rank = int(np.sum(singular > float(tolerance)))
    sigma_min = float(singular[-1]) if singular.size else 0.0
    sigma_max = float(singular[0]) if singular.size else 0.0
    condition = float(np.inf if sigma_min <= 0.0 else sigma_max / sigma_min)
    return {
        "rank": numerical_rank,
        "sigma_min": sigma_min,
        "condition": condition,
        "full_rank": numerical_rank == min(value.shape),
        "square_full_rank": value.shape[0] == value.shape[1] == numerical_rank,
    }


def census_source(source_dir: Path) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    artifacts = gate12c.load_gate12a_artifacts(source_dir)
    tolerance = float(artifacts.manifest.get("tau_overlap_sv_min") or 1.0e-8)
    reconstructions, diagnostics = gate12c.reconstruct_edges(
        artifacts=artifacts,
        tau_overlap_sv_min=tolerance,
        tau_overlap_sv_abs_error=gate12c.DEFAULT_TAU_OVERLAP_SV_ABS_ERROR,
        tau_transport_reconstruction_fro=gate12c.DEFAULT_TAU_TRANSPORT_RECONSTRUCTION_FRO,
    )

    endpoint_counts = Counter(
        (str(row["source_node_id"]), str(row["target_node_id"]))
        for row in artifacts.transport_rows
    )
    endpoint_set = set(endpoint_counts)
    reciprocal_edge_count = sum(
        count
        for (source, target), count in endpoint_counts.items()
        if (target, source) in endpoint_set
    )
    parallel_group_count = sum(1 for count in endpoint_counts.values() if count > 1)
    self_loop_count = sum(
        count for (source, target), count in endpoint_counts.items() if source == target
    )

    edge_sigma_min: list[float] = []
    edge_condition: list[float] = []
    square_full_rank_count = 0
    raw_defined_count = 0
    reverse_transpose_errors: list[float] = []
    for edge_id, reconstruction in reconstructions.items():
        overlap = reconstruction.overlap_matrix
        if overlap is None:
            continue
        raw_defined_count += 1
        metrics = overlap_metrics(overlap, tolerance=tolerance)
        edge_sigma_min.append(metrics["sigma_min"])
        edge_condition.append(metrics["condition"])
        square_full_rank_count += int(metrics["square_full_rank"])

        edge = artifacts.edge_map[edge_id]
        reverse_ids = [
            other_id
            for other_id, other in artifacts.edge_map.items()
            if str(other["source_node_id"]) == str(edge["target_node_id"])
            and str(other["target_node_id"]) == str(edge["source_node_id"])
        ]
        for reverse_id in reverse_ids:
            reverse = reconstructions[reverse_id].overlap_matrix
            if reverse is not None:
                reverse_transpose_errors.append(
                    float(np.linalg.norm(reverse - overlap.T, ord="fro"))
                )

    path_sigma_min: list[float] = []
    path_condition: list[float] = []
    registered_full_rank_path_count = 0
    reconstructable_triangle_count = 0
    for cycle in artifacts.cycle_rows:
        try:
            ordered = gate12c.reconstruct_ordered_edges(
                cycle=cycle,
                edge_map=artifacts.edge_map,
            )
        except ValueError:
            continue
        matrices: list[np.ndarray] = []
        common_shape: tuple[int, int] | None = None
        for edge in ordered:
            reconstruction = reconstructions[str(edge["edge_id"])]
            if reconstruction.overlap_matrix is None:
                matrices = []
                break
            matrix = np.asarray(reconstruction.overlap_matrix, dtype=np.float64)
            if common_shape is None:
                common_shape = matrix.shape
            if matrix.shape != common_shape or matrix.shape[0] != matrix.shape[1]:
                matrices = []
                break
            matrices.append(matrix)
        if len(matrices) != 3:
            continue
        reconstructable_triangle_count += 1
        product = matrices[2] @ matrices[1] @ matrices[0]
        metrics = overlap_metrics(product, tolerance=tolerance)
        path_sigma_min.append(metrics["sigma_min"])
        path_condition.append(metrics["condition"])
        registered_full_rank_path_count += int(metrics["square_full_rank"])

    base_counts = Counter(str(row["base_node_id"]) for row in artifacts.cycle_rows)
    shared_base_registered_pair_count = sum(
        count * (count - 1) // 2 for count in base_counts.values()
    )
    defined_holonomy_count = sum(
        str(row.get("holonomy_status") or "") == "defined"
        for row in artifacts.holonomy_rows
    )

    return {
        "source_dir": str(source_dir),
        "run_id": str(artifacts.manifest.get("run_id") or ""),
        "schema_version": str(artifacts.manifest.get("schema_version") or ""),
        "code_git_commit": str(artifacts.manifest.get("code_git_commit") or ""),
        "graph_object_policy": str(artifacts.manifest.get("graph_object_policy") or ""),
        "cycle_mode": str(artifacts.manifest.get("cycle_mode") or ""),
        "tau_overlap_sv_min": tolerance,
        "manifest_sha256": sha256_file(source_dir / gate12c.DEFAULT_MANIFEST),
        "node_count": len(artifacts.node_rows),
        "directed_edge_count": len(artifacts.transport_rows),
        "unique_directed_endpoint_pair_count": len(endpoint_counts),
        "parallel_directed_endpoint_group_count": parallel_group_count,
        "self_loop_edge_count": self_loop_count,
        "edge_count_with_reciprocal_endpoint": reciprocal_edge_count,
        "raw_overlap_defined_edge_count": raw_defined_count,
        "square_full_rank_raw_edge_count": square_full_rank_count,
        "edge_sigma_min": finite_quantiles(edge_sigma_min),
        "edge_condition": finite_quantiles(edge_condition),
        "reverse_transpose_integrity": {
            "status": "SCHEMA_CHECK_ONLY",
            "comparison_count": len(reverse_transpose_errors),
            "max_fro_error": (
                max(reverse_transpose_errors) if reverse_transpose_errors else None
            ),
        },
        "registered_triangle_count": len(artifacts.cycle_rows),
        "reconstructable_equal_shape_triangle_count": reconstructable_triangle_count,
        "registered_full_rank_path_count": registered_full_rank_path_count,
        "defined_triangle_holonomy_count": defined_holonomy_count,
        "shared_base_registered_triangle_pair_count": shared_base_registered_pair_count,
        "registered_path_sigma_min": finite_quantiles(path_sigma_min),
        "registered_path_condition": finite_quantiles(path_condition),
        "edge_reconstruction_diagnostics": diagnostics,
        "general_topology": {
            "connected_components": None,
            "beta_1": None,
            "fundamental_cycles": None,
            "shared_vertex_independent_loop_pairs": None,
            "status": "UNAVAILABLE_PENDING_GRAPH_CONVENTION",
        },
    }


def census_from_case_manifest(case_manifest_path: Path) -> Dict[str, Any]:
    case_manifest_path = Path(case_manifest_path)
    case_manifest = read_json(case_manifest_path)
    cases = list(case_manifest.get("cases") or [])
    if not cases:
        raise ValueError("case manifest contains no cases")
    rows = [
        census_source(Path(str(case["source_gate12a_dir"])))
        for case in sorted(cases, key=lambda item: int(item["case_order"]))
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "B0_PARTIAL_CENSUS_GRAPH_CONVENTION_BLOCKER",
        "model_forward_count": 0,
        "source_case_manifest": str(case_manifest_path),
        "source_case_manifest_sha256": sha256_file(case_manifest_path),
        "source_run_count": len(rows),
        "graph_convention_declaration": graph_convention_declaration(),
        "historical_scalar_candidates": historical_scalar_candidates(),
        "b2_legacy_scalar_status": "REVIEW1_SELECTION_REQUIRED_BEFORE_B2",
        "aggregate": {
            "node_count": sum(int(row["node_count"]) for row in rows),
            "directed_edge_count": sum(int(row["directed_edge_count"]) for row in rows),
            "registered_triangle_count": sum(
                int(row["registered_triangle_count"]) for row in rows
            ),
            "defined_triangle_holonomy_count": sum(
                int(row["defined_triangle_holonomy_count"]) for row in rows
            ),
            "general_beta_1_status": "UNAVAILABLE_PENDING_GRAPH_CONVENTION",
            "track_b_pass": False,
        },
        "runs": rows,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = census_from_case_manifest(args.case_manifest)
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_run_count": report["source_run_count"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

