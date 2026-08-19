"""Split-half frame alignment and packet-disagreement diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def validate_frame(frame: np.ndarray, *, context: str) -> np.ndarray:
    value = np.asarray(frame, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{context} must be rank-2")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{context} must be finite")
    gram_error = np.linalg.norm(
        value.T @ value - np.eye(value.shape[1], dtype=np.float64),
        ord="fro",
    )
    if gram_error > 1.0e-8:
        raise ValueError(f"{context} columns must be orthonormal")
    return value


def align_frame(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, Any]:
    """Align candidate to reference by right orthogonal Procrustes."""
    ref = validate_frame(reference, context="reference")
    cand = validate_frame(candidate, context="candidate")
    if ref.shape != cand.shape:
        raise ValueError("frame shapes must match")
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(
        cand.T @ ref,
        full_matrices=False,
    )
    gauge = u_matrix @ vt_matrix
    aligned = cand @ gauge
    return {
        "aligned": aligned,
        "gauge": gauge,
        "pre_error_fro": float(np.linalg.norm(cand - ref, ord="fro")),
        "post_error_fro": float(np.linalg.norm(aligned - ref, ord="fro")),
    }


def packet_disagreement(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> Dict[str, Any]:
    first_raw = first["raw"]
    second_raw = second["raw"]
    result: Dict[str, Any] = {
        "P_p_fro": float(
            np.linalg.norm(
                np.asarray(first_raw["P_p"]) - np.asarray(second_raw["P_p"]),
                ord="fro",
            )
        ),
        "P_q_fro": float(
            np.linalg.norm(
                np.asarray(first_raw["P_q"]) - np.asarray(second_raw["P_q"]),
                ord="fro",
            )
        ),
        "Delta_pq_fro": float(
            np.linalg.norm(
                np.asarray(first_raw["Delta_pq"])
                - np.asarray(second_raw["Delta_pq"]),
                ord="fro",
            )
        ),
        "singular_values_p_l2": float(
            np.linalg.norm(
                np.asarray(first_raw["singular_values_p"])
                - np.asarray(second_raw["singular_values_p"])
            )
        ),
        "singular_values_q_l2": float(
            np.linalg.norm(
                np.asarray(first_raw["singular_values_q"])
                - np.asarray(second_raw["singular_values_q"])
            )
        ),
        "path_qualification_agreement": (
            first["path_polar"]["status"] == second["path_polar"]["status"]
        ),
        "edge_qualification_agreement": (
            first["edge_polar"]["status"] == second["edge_polar"]["status"]
        ),
    }
    if (
        first["path_polar"]["status"] == "QUALIFIED"
        and second["path_polar"]["status"] == "QUALIFIED"
    ):
        result["H_path_fro"] = float(
            np.linalg.norm(
                np.asarray(first["path_polar"]["H_path"])
                - np.asarray(second["path_polar"]["H_path"]),
                ord="fro",
            )
        )
    if (
        first["edge_polar"]["status"] == "QUALIFIED"
        and second["edge_polar"]["status"] == "QUALIFIED"
    ):
        result["H_edge_fro"] = float(
            np.linalg.norm(
                np.asarray(first["edge_polar"]["H_edge"])
                - np.asarray(second["edge_polar"]["H_edge"]),
                ord="fro",
            )
        )
    return result

