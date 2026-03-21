#!/usr/bin/env python3
"""Regression tests for Gate9A graph-gauge consumer helpers."""

import unittest

import numpy as np

import run_gate9a_graph_gauge_consumer as gate9a


def make_local_object(node_id: str, node_type: str, basis: np.ndarray, rank_local: int) -> gate9a.LocalObject:
    singular_values = np.zeros((3,), dtype=np.float64)
    singular_values[:rank_local] = 1.0
    return gate9a.LocalObject(
        node_id=node_id,
        node_type=node_type,
        execution_sample_id=1,
        benchmark_sample_id="bench_1",
        cell_id="distributed_incompatibility",
        world_id="w0",
        world_type="genealogy",
        answer_target_type="conflict_following_wrong_answer",
        quietness_pair_id="",
        rendering_family_id="archive_v1",
        basis=basis,
        singular_values=singular_values,
        rank_local=rank_local,
        metadata={},
    )


class RunGate9AGraphGaugeConsumerTest(unittest.TestCase):
    def test_build_transport_zero_defect_for_identical_projectors(self) -> None:
        basis = np.zeros((4, 3), dtype=np.float64)
        basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        transport = gate9a.build_transport(basis, 1, basis, 1)
        self.assertEqual(transport["edge_outcome"], "none")
        self.assertEqual(transport["transport_mode"], "orthogonal_equal_rank")
        self.assertAlmostEqual(float(transport["edge_transport_defect"]), 0.0, places=10)
        self.assertAlmostEqual(float(transport["overlap_ratio"]), 1.0, places=10)

    def test_cycle_holonomy_zero_for_identity_cycle(self) -> None:
        basis = np.zeros((4, 3), dtype=np.float64)
        basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        support = make_local_object("support", "support_chunk", basis, 1)
        answer = make_local_object("answer", "answer_state", basis, 1)
        token = make_local_object("token", "token_state", basis, 1)

        edge_rows = []
        edge_transport_map = {}
        for edge_id, source, target, edge_type in (
            ("e1", support, answer, "support_anchor"),
            ("e2", answer, token, "answer_projection"),
            ("e3", token, support, "support_anchor"),
        ):
            row, transport = gate9a.build_edge_row(edge_id, edge_type, source, target, {})
            edge_rows.append(row)
            edge_transport_map[edge_id] = transport

        cycle = gate9a.compute_cycle_holonomy(support, edge_rows, edge_transport_map)
        self.assertEqual(cycle["cycle_outcome"], "none")
        self.assertAlmostEqual(float(cycle["holonomy_defect"]), 0.0, places=10)
        self.assertAlmostEqual(float(cycle["holonomy_trace"]), 1.0, places=10)

    def test_anchor_conditioned_closure_zero_when_anchor_matches(self) -> None:
        basis = np.zeros((4, 3), dtype=np.float64)
        basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        anchor = make_local_object("support", "support_chunk", basis, 1)
        answer = make_local_object("answer", "answer_state", basis, 1)
        token = make_local_object("token", "token_state", basis, 1)

        closure = gate9a.compute_anchor_conditioned_closure(anchor, answer, token)
        self.assertEqual(closure["closure_outcome"], "none")
        self.assertAlmostEqual(float(closure["anchor_answer_coverage"]), 1.0, places=10)
        self.assertAlmostEqual(float(closure["anchor_token_coverage"]), 1.0, places=10)
        self.assertAlmostEqual(float(closure["anchor_conditioned_closure_defect"]), 0.0, places=10)

    def test_anchor_conditioned_closure_reports_insufficient_overlap(self) -> None:
        anchor_basis = np.zeros((4, 3), dtype=np.float64)
        answer_basis = np.zeros((4, 3), dtype=np.float64)
        token_basis = np.zeros((4, 3), dtype=np.float64)
        anchor_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        answer_basis[:, 0] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        token_basis[:, 0] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

        anchor = make_local_object("support", "support_chunk", anchor_basis, 1)
        answer = make_local_object("answer", "answer_state", answer_basis, 1)
        token = make_local_object("token", "token_state", token_basis, 1)

        closure = gate9a.compute_anchor_conditioned_closure(anchor, answer, token)
        self.assertEqual(closure["closure_outcome"], "insufficient_answer_anchor_overlap")
        self.assertAlmostEqual(float(closure["anchor_answer_coverage"]), 0.0, places=10)
        self.assertAlmostEqual(float(closure["anchor_token_coverage"]), 0.0, places=10)
        self.assertIsNone(closure["anchor_conditioned_closure_defect"])


if __name__ == "__main__":
    unittest.main()
