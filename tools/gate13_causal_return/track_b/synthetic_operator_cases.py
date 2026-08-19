"""Run deterministic Track B canonical operator qualification cases I-V."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .operator_core import (
    QUALIFIED,
    RANK_DEFICIENT,
    UNQUALIFIED,
    build_operator_packet,
    json_ready,
    orthogonal_eigenangles,
    shadow_distance,
)
from .split_half_floor import align_frame, packet_disagreement

ATOL = 1.0e-10


def rotation(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_synthetic_qualification() -> Dict[str, Any]:
    identity = np.eye(2, dtype=np.float64)

    case_i = build_operator_packet(
        path_p_edges=[identity],
        path_q_edges=[],
        identity_rank=2,
        source_node="A",
        target_node="A",
        path_p_id="identity_edge",
        path_q_id="identity_path",
        topology_id="case_i",
    )
    require(np.linalg.norm(case_i["raw"]["Delta_pq"], ord="fro") <= ATOL, "case I raw")
    require(case_i["path_polar"]["status"] == QUALIFIED, "case I path status")
    require(case_i["edge_polar"]["status"] == QUALIFIED, "case I edge status")

    broken_square = build_operator_packet(
        path_p_edges=[np.diag([1.0, -1.0])],
        path_q_edges=[identity],
        source_node="A",
        target_node="D",
        path_p_id="broken_transition_path",
        path_q_id="reference_transition_path",
        topology_id="broken_square_positive_control",
    )
    exact_delta = float(np.linalg.norm(case_i["raw"]["Delta_pq"], ord="fro"))
    broken_delta = float(
        np.linalg.norm(broken_square["raw"]["Delta_pq"], ord="fro")
    )
    require(exact_delta <= ATOL, "exact-square control is not closed")
    require(broken_delta > exact_delta + 1.0, "broken-square sensitivity absent")

    p_a = np.diag([2.0, 1.0])
    theta = math.acos(0.75)
    p_b = rotation(theta)
    shadow_a = shadow_distance(p_a)
    shadow_b = shadow_distance(p_b)
    require(abs(shadow_a - shadow_b) <= ATOL, "case II shadow mismatch")
    singular_a = np.linalg.svd(p_a, compute_uv=False)
    singular_b = np.linalg.svd(p_b, compute_uv=False)
    require(not np.allclose(singular_a, singular_b, atol=ATOL), "case II collision absent")

    projection = np.diag([1.0, 0.0])
    hidden_action = np.diag([1.0, 2.0])
    vector = np.asarray([0.0, 1.0])
    require(np.allclose(projection @ hidden_action @ vector, projection @ vector), "case III projected")
    require(not np.allclose(hidden_action @ vector, vector), "case III lifted")

    same_twist_1 = build_operator_packet(
        path_p_edges=[np.diag([1.0, 2.0])],
        path_q_edges=[np.diag([2.0, 3.0])],
        source_node="A",
        target_node="D",
        path_p_id="loss_1",
        path_q_id="loss_2",
        topology_id="case_iv_same_twist",
    )
    require(
        np.allclose(
            same_twist_1["path_polar"]["O_p_path"],
            same_twist_1["path_polar"]["O_q_path"],
            atol=ATOL,
        ),
        "case IV same twist",
    )
    require(
        not np.allclose(
            same_twist_1["path_polar"]["S_p"],
            same_twist_1["path_polar"]["S_q"],
            atol=ATOL,
        ),
        "case IV different loss",
    )

    shared_loss = np.diag([2.0, 1.0])
    different_twist = build_operator_packet(
        path_p_edges=[shared_loss],
        path_q_edges=[rotation(math.pi / 2.0) @ shared_loss],
        source_node="A",
        target_node="D",
        path_p_id="twist_1",
        path_q_id="twist_2",
        topology_id="case_iv_different_twist",
    )
    require(
        np.allclose(
            different_twist["path_polar"]["S_p"],
            different_twist["path_polar"]["S_q"],
            atol=ATOL,
        ),
        "case IV same loss",
    )
    require(
        not np.allclose(
            different_twist["path_polar"]["O_p_path"],
            different_twist["path_polar"]["O_q_path"],
            atol=ATOL,
        ),
        "case IV different twist",
    )

    singular = np.diag([1.0, 0.0])
    case_v = build_operator_packet(
        path_p_edges=[singular],
        path_q_edges=[identity],
        source_node="A",
        target_node="D",
        path_p_id="singular",
        path_q_id="identity",
        topology_id="case_v",
    )
    require(case_v["path_polar"]["status"] == UNQUALIFIED, "case V path status")
    require(case_v["path_polar"]["rejection_reason"] == RANK_DEFICIENT, "case V reason")
    require("O_p_path" not in case_v["path_polar"], "case V O must be absent")
    require("H_path" not in case_v["path_polar"], "case V H path must be absent")
    require(case_v["edge_polar"]["status"] == UNQUALIFIED, "case V edge status")
    require("H_edge" not in case_v["edge_polar"], "case V H edge must be absent")

    gauge_a = rotation(0.31)
    gauge_d = rotation(-0.47)
    transformed_p = gauge_d.T @ p_a @ gauge_a
    transformed_q = gauge_d.T @ p_b @ gauge_a
    base_delta = p_a - p_b
    transformed_delta = transformed_p - transformed_q
    require(
        np.allclose(transformed_delta, gauge_d.T @ base_delta @ gauge_a, atol=ATOL),
        "gauge covariance",
    )
    require(
        np.allclose(
            np.linalg.svd(base_delta, compute_uv=False),
            np.linalg.svd(transformed_delta, compute_uv=False),
            atol=ATOL,
        ),
        "gauge-invariant delta spectrum",
    )

    frame = np.eye(4, 2, dtype=np.float64)
    frame_gauge = rotation(0.63)
    alignment = align_frame(frame, frame @ frame_gauge)
    require(alignment["post_error_fro"] <= ATOL, "split-half alignment")

    packet_a = build_operator_packet(
        path_p_edges=[p_a],
        path_q_edges=[p_b],
        source_node="A",
        target_node="D",
        path_p_id="p",
        path_q_id="q",
        topology_id="split",
    )
    packet_b = build_operator_packet(
        path_p_edges=[p_a],
        path_q_edges=[p_b],
        source_node="A",
        target_node="D",
        path_p_id="p",
        path_q_id="q",
        topology_id="split",
    )
    split_disagreement = packet_disagreement(packet_a, packet_b)
    require(split_disagreement["Delta_pq_fro"] <= ATOL, "split packet stability")

    return {
        "schema_version": "gate13_candidate_synthetic_operator_qualification_v0.1.1",
        "status": "PASS_SYNTHETIC_OPERATOR_QUALIFICATION",
        "model_forward_count": 0,
        "numeric_tolerance": ATOL,
        "cases": {
            "I_identity_return": {
                "status": "PASS",
                "delta_fro": float(np.linalg.norm(case_i["raw"]["Delta_pq"], ord="fro")),
            },
            "II_scalar_shadow_collision": {
                "status": "PASS",
                "shadow_a": shadow_a,
                "shadow_b": shadow_b,
                "singular_values_a": singular_a,
                "singular_values_b": singular_b,
                "orthogonal_angles_b": orthogonal_eigenangles(p_b),
            },
            "III_projection_hidden_return": {"status": "PASS"},
            "IV_loss_twist_separation": {"status": "PASS"},
            "V_singular_edge_rejection": {
                "status": "PASS",
                "path_status": case_v["path_polar"]["status"],
                "edge_status": case_v["edge_polar"]["status"],
                "reason": case_v["path_polar"]["rejection_reason"],
            },
        },
        "gauge_covariance": "PASS",
        "broken_square_positive_control": {
            "status": "PASS",
            "scope": "B1_SYNTHETIC_ONLY",
            "exact_square_delta_fro": exact_delta,
            "broken_square_delta_fro": broken_delta,
            "historical_artifacts_modified": False,
        },
        "split_half_alignment": {
            "status": "PASS",
            "pre_error_fro": alignment["pre_error_fro"],
            "post_error_fro": alignment["post_error_fro"],
            "packet_disagreement": split_disagreement,
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_synthetic_qualification()
    write_json(args.out, report)
    print(json.dumps(json_ready(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
