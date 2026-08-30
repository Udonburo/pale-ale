"""Generic path-pair operator packet with fail-closed polar semantics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

QUALIFIED = "QUALIFIED"
UNQUALIFIED = "UNQUALIFIED"

RANK_DEFICIENT = "RANK_DEFICIENT"
ILL_CONDITIONED = "ILL_CONDITIONED"
DIMENSION_MISMATCH = "DIMENSION_MISMATCH"


def as_square_matrix(value: np.ndarray, *, context: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{context} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} must contain only finite values")
    return matrix


def compose_path(
    edges: Sequence[np.ndarray],
    *,
    identity_rank: int | None = None,
) -> np.ndarray:
    """Compose edges listed in traversal order."""
    if not edges:
        if identity_rank is None or int(identity_rank) <= 0:
            raise ValueError("identity_rank must be positive for an empty path")
        return np.eye(int(identity_rank), dtype=np.float64)

    matrices = [
        as_square_matrix(edge, context=f"edge[{index}]")
        for index, edge in enumerate(edges)
    ]
    rank = matrices[0].shape[0]
    if any(matrix.shape != (rank, rank) for matrix in matrices):
        raise ValueError(DIMENSION_MISMATCH)
    product = np.eye(rank, dtype=np.float64)
    for matrix in matrices:
        product = matrix @ product
    return np.asarray(product, dtype=np.float64)


def matrix_metrics(matrix: np.ndarray, *, rank_tolerance: float) -> Dict[str, Any]:
    value = as_square_matrix(matrix, context="matrix")
    singular_values = np.linalg.svd(value, compute_uv=False)
    numerical_rank = int(np.sum(singular_values > float(rank_tolerance)))
    sigma_min = float(singular_values[-1])
    sigma_max = float(singular_values[0])
    condition = float(np.inf if sigma_min <= 0.0 else sigma_max / sigma_min)
    return {
        "rank": numerical_rank,
        "singular_values": singular_values,
        "sigma_min": sigma_min,
        "condition": condition,
    }


def polar_record(
    matrix: np.ndarray,
    *,
    rank_tolerance: float,
    condition_ceiling: float,
) -> Dict[str, Any]:
    """Return unique PSD factor always and orthogonal factor only when qualified."""
    value = as_square_matrix(matrix, context="polar matrix")
    u_matrix, singular_values, vt_matrix = np.linalg.svd(value, full_matrices=False)
    positive = vt_matrix.T @ np.diag(singular_values) @ vt_matrix
    metrics = matrix_metrics(value, rank_tolerance=rank_tolerance)
    record: Dict[str, Any] = {
        "status": QUALIFIED,
        "S": np.asarray(positive, dtype=np.float64),
        "rank": int(metrics["rank"]),
        "singular_values": np.asarray(metrics["singular_values"], dtype=np.float64),
        "sigma_min": float(metrics["sigma_min"]),
        "condition": float(metrics["condition"]),
    }
    if int(metrics["rank"]) != value.shape[0]:
        record["status"] = UNQUALIFIED
        record["rejection_reason"] = RANK_DEFICIENT
        return record
    if float(metrics["condition"]) > float(condition_ceiling):
        record["status"] = UNQUALIFIED
        record["rejection_reason"] = ILL_CONDITIONED
        return record

    orthogonal = np.asarray(u_matrix @ vt_matrix, dtype=np.float64)
    record["O"] = orthogonal
    record["reconstruction_error_fro"] = float(
        np.linalg.norm(value - orthogonal @ positive, ord="fro")
    )
    record["orthogonality_error_fro"] = float(
        np.linalg.norm(
            orthogonal.T @ orthogonal - np.eye(value.shape[0], dtype=np.float64),
            ord="fro",
        )
    )
    return record


def edgewise_record(
    path_p_edges: Sequence[np.ndarray],
    path_q_edges: Sequence[np.ndarray],
    *,
    rank: int,
    rank_tolerance: float,
    condition_ceiling: float,
) -> Dict[str, Any]:
    def qualify_path(edges: Sequence[np.ndarray]) -> Dict[str, Any]:
        if not edges:
            return {
                "status": QUALIFIED,
                "per_edge_status": [],
                "per_edge_factors": [],
                "Q": np.eye(rank, dtype=np.float64),
            }
        records = [
            polar_record(
                edge,
                rank_tolerance=rank_tolerance,
                condition_ceiling=condition_ceiling,
            )
            for edge in edges
        ]
        statuses = [str(record["status"]) for record in records]
        result: Dict[str, Any] = {
            "status": QUALIFIED if all(status == QUALIFIED for status in statuses) else UNQUALIFIED,
            "per_edge_status": statuses,
            "per_edge_factors": [
                record["O"] if record["status"] == QUALIFIED else None
                for record in records
            ],
        }
        if result["status"] == UNQUALIFIED:
            result["rejection_reason"] = next(
                str(record["rejection_reason"])
                for record in records
                if record["status"] == UNQUALIFIED
            )
            return result
        result["Q"] = compose_path(
            [np.asarray(record["O"], dtype=np.float64) for record in records],
            identity_rank=rank,
        )
        return result

    p_record = qualify_path(path_p_edges)
    q_record = qualify_path(path_q_edges)
    result: Dict[str, Any] = {
        "status": (
            QUALIFIED
            if p_record["status"] == QUALIFIED and q_record["status"] == QUALIFIED
            else UNQUALIFIED
        ),
        "path_p": p_record,
        "path_q": q_record,
    }
    if result["status"] == UNQUALIFIED:
        failing = p_record if p_record["status"] == UNQUALIFIED else q_record
        result["rejection_reason"] = failing["rejection_reason"]
        return result
    q_p = np.asarray(p_record["Q"], dtype=np.float64)
    q_q = np.asarray(q_record["Q"], dtype=np.float64)
    result["Q_p"] = q_p
    result["Q_q"] = q_q
    result["H_edge"] = q_q.T @ q_p
    return result


def build_operator_packet(
    *,
    path_p_edges: Sequence[np.ndarray],
    path_q_edges: Sequence[np.ndarray],
    source_node: str,
    target_node: str,
    path_p_id: str,
    path_q_id: str,
    topology_id: str,
    identity_rank: int | None = None,
    rank_tolerance: float = 1.0e-10,
    condition_ceiling: float = 1.0e12,
) -> Dict[str, Any]:
    p_product = compose_path(path_p_edges, identity_rank=identity_rank)
    q_product = compose_path(path_q_edges, identity_rank=identity_rank)
    if p_product.shape != q_product.shape:
        raise ValueError(DIMENSION_MISMATCH)
    rank = p_product.shape[0]
    delta = p_product - q_product
    p_metrics = matrix_metrics(p_product, rank_tolerance=rank_tolerance)
    q_metrics = matrix_metrics(q_product, rank_tolerance=rank_tolerance)

    raw = {
        "P_p": p_product,
        "P_q": q_product,
        "Delta_pq": delta,
        "rank_p": p_metrics["rank"],
        "rank_q": q_metrics["rank"],
        "singular_values_p": p_metrics["singular_values"],
        "singular_values_q": q_metrics["singular_values"],
        "sigma_min_p": p_metrics["sigma_min"],
        "sigma_min_q": q_metrics["sigma_min"],
        "condition_p": p_metrics["condition"],
        "condition_q": q_metrics["condition"],
    }

    p_polar = polar_record(
        p_product,
        rank_tolerance=rank_tolerance,
        condition_ceiling=condition_ceiling,
    )
    q_polar = polar_record(
        q_product,
        rank_tolerance=rank_tolerance,
        condition_ceiling=condition_ceiling,
    )
    path_polar: Dict[str, Any] = {
        "status": (
            QUALIFIED
            if p_polar["status"] == QUALIFIED and q_polar["status"] == QUALIFIED
            else UNQUALIFIED
        ),
        "S_p": p_polar["S"],
        "S_q": q_polar["S"],
        "path_p_status": p_polar["status"],
        "path_q_status": q_polar["status"],
    }
    if path_polar["status"] == QUALIFIED:
        o_p = np.asarray(p_polar["O"], dtype=np.float64)
        o_q = np.asarray(q_polar["O"], dtype=np.float64)
        path_polar["O_p_path"] = o_p
        path_polar["O_q_path"] = o_q
        path_polar["H_path"] = o_q.T @ o_p
    else:
        failing = p_polar if p_polar["status"] == UNQUALIFIED else q_polar
        path_polar["rejection_reason"] = failing["rejection_reason"]

    edge_polar = edgewise_record(
        path_p_edges,
        path_q_edges,
        rank=rank,
        rank_tolerance=rank_tolerance,
        condition_ceiling=condition_ceiling,
    )

    return {
        "schema_version": "gate13_candidate_operator_packet_v0.1.1",
        "identity": {
            "source_node": str(source_node),
            "target_node": str(target_node),
            "path_p": str(path_p_id),
            "path_q": str(path_q_id),
            "topology_id": str(topology_id),
        },
        "raw": raw,
        "path_polar": path_polar,
        "edge_polar": edge_polar,
    }


def orthogonal_eigenangles(matrix: np.ndarray) -> np.ndarray:
    value = as_square_matrix(matrix, context="orthogonal matrix")
    angles = np.angle(np.linalg.eigvals(value))
    return np.asarray(sorted(float(angle) for angle in angles), dtype=np.float64)


def shadow_distance(matrix: np.ndarray) -> float:
    value = as_square_matrix(matrix, context="shadow matrix")
    return float(np.linalg.norm(value - np.eye(value.shape[0]), ord="fro"))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

