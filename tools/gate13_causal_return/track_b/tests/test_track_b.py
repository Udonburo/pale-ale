from __future__ import annotations

import math
import unittest

import numpy as np

from tools.gate13_causal_return.track_b.operator_core import (
    QUALIFIED,
    RANK_DEFICIENT,
    UNQUALIFIED,
    build_operator_packet,
    compose_path,
    shadow_distance,
)
from tools.gate13_causal_return.track_b.split_half_floor import align_frame
from tools.gate13_causal_return.track_b.synthetic_operator_cases import (
    run_synthetic_qualification,
)


class OperatorCoreTests(unittest.TestCase):
    def test_path_order_and_loop_identity_special_case(self) -> None:
        first = np.diag([2.0, 1.0])
        second = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        self.assertTrue(
            np.allclose(compose_path([first, second]), second @ first)
        )
        packet = build_operator_packet(
            path_p_edges=[first, second],
            path_q_edges=[],
            identity_rank=2,
            source_node="A",
            target_node="A",
            path_p_id="gamma",
            path_q_id="identity",
            topology_id="loop",
        )
        self.assertEqual(packet["identity"]["path_q"], "identity")
        self.assertTrue(np.allclose(packet["raw"]["P_q"], np.eye(2)))

    def test_rank_deficient_packet_omits_twist(self) -> None:
        packet = build_operator_packet(
            path_p_edges=[np.diag([1.0, 0.0])],
            path_q_edges=[np.eye(2)],
            source_node="A",
            target_node="D",
            path_p_id="p",
            path_q_id="q",
            topology_id="singular",
        )
        self.assertEqual(packet["path_polar"]["status"], UNQUALIFIED)
        self.assertEqual(
            packet["path_polar"]["rejection_reason"], RANK_DEFICIENT
        )
        self.assertNotIn("O_p_path", packet["path_polar"])
        self.assertNotIn("H_path", packet["path_polar"])
        self.assertNotIn("H_edge", packet["edge_polar"])

    def test_scalar_shadow_collision(self) -> None:
        p_a = np.diag([2.0, 1.0])
        p_b = np.asarray(
            [
                [0.75, -math.sqrt(1.0 - 0.75**2)],
                [math.sqrt(1.0 - 0.75**2), 0.75],
            ]
        )
        self.assertAlmostEqual(shadow_distance(p_a), shadow_distance(p_b))
        self.assertFalse(
            np.allclose(
                np.linalg.svd(p_a, compute_uv=False),
                np.linalg.svd(p_b, compute_uv=False),
            )
        )

    def test_full_rank_packet_qualifies(self) -> None:
        packet = build_operator_packet(
            path_p_edges=[np.diag([2.0, 1.0])],
            path_q_edges=[np.eye(2)],
            source_node="A",
            target_node="D",
            path_p_id="p",
            path_q_id="q",
            topology_id="full",
        )
        self.assertEqual(packet["path_polar"]["status"], QUALIFIED)
        self.assertEqual(packet["edge_polar"]["status"], QUALIFIED)
        self.assertIn("H_path", packet["path_polar"])
        self.assertIn("H_edge", packet["edge_polar"])


class SplitHalfTests(unittest.TestCase):
    def test_procrustes_removes_frame_gauge(self) -> None:
        reference = np.eye(5, 2)
        theta = 0.71
        gauge = np.asarray(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        result = align_frame(reference, reference @ gauge)
        self.assertLess(result["post_error_fro"], 1.0e-10)


class SyntheticQualificationTests(unittest.TestCase):
    def test_all_canonical_cases_pass(self) -> None:
        report = run_synthetic_qualification()
        self.assertEqual(
            report["status"], "PASS_SYNTHETIC_OPERATOR_QUALIFICATION"
        )
        self.assertEqual(report["model_forward_count"], 0)
        self.assertTrue(
            all(row["status"] == "PASS" for row in report["cases"].values())
        )
        control = report["broken_square_positive_control"]
        self.assertEqual(control["status"], "PASS")
        self.assertEqual(control["scope"], "B1_SYNTHETIC_ONLY")
        self.assertEqual(control["exact_square_delta_fro"], 0.0)
        self.assertGreater(control["broken_square_delta_fro"], 1.0)
        self.assertFalse(control["historical_artifacts_modified"])


if __name__ == "__main__":
    unittest.main()
