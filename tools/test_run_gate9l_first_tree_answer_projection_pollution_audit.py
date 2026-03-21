#!/usr/bin/env python3
"""Regression tests for Gate9L first-tree answer-projection pollution helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9l_first_tree_answer_projection_pollution_audit as gate9l


def make_node(node_id: str, sample_id: int) -> dict:
    return {
        "node_id": node_id,
        "execution_sample_id": sample_id,
    }


def make_edge(
    edge_id: str,
    edge_type: str,
    source: str,
    target: str,
    sample_id: int,
    cell_id: str,
    defect: float = 0.1,
) -> dict:
    return {
        "edge_id": edge_id,
        "edge_type": edge_type,
        "source_node_id": source,
        "target_node_id": target,
        "execution_sample_id": sample_id,
        "benchmark_sample_id": f"bench_{sample_id}",
        "cell_id": cell_id,
        "world_id": f"world_{sample_id}",
        "world_type": "type",
        "answer_target_type": "consistent_answer",
        "edge_transport_defect": defect,
    }


class RunGate9LFirstTreeAnswerProjectionPollutionAuditTest(unittest.TestCase):
    def test_first_tree_build_and_cleaner_pollution(self) -> None:
        node_rows = [
            make_node("c_tok", 1),
            make_node("c_sup", 1),
            make_node("c_ans", 1),
            make_node("d_tok", 2),
            make_node("d_sup", 2),
            make_node("d_ans", 2),
            make_node("d_conf", 2),
        ]
        edge_rows = [
            make_edge("e1", "temporal_transition", "c_tok", "c_sup", 1, "clean_support"),
            make_edge("e2", "support_anchor", "c_sup", "c_ans", 1, "clean_support"),
            make_edge("e3", "answer_projection", "c_ans", "c_tok", 1, "clean_support"),
            make_edge("e4", "temporal_transition", "d_tok", "d_sup", 2, "distributed_incompatibility"),
            make_edge("e5", "support_anchor", "d_sup", "d_ans", 2, "distributed_incompatibility"),
            make_edge("e6", "answer_projection", "d_ans", "d_tok", 2, "distributed_incompatibility"),
            make_edge("e7", "conflict_anchor", "d_conf", "d_ans", 2, "distributed_incompatibility"),
        ]
        tree_rows, residual_rows = gate9l.build_first_tree_and_residual_rows(node_rows, edge_rows)
        payload = gate9l.build_status_payload(tree_rows, residual_rows)
        self.assertEqual(payload["trusted_forest_build_status"], "built")
        self.assertEqual(payload["trusted_forest_cycle_free_status"], "clear")
        self.assertEqual(payload["cleaner_answer_projection_residual_pollution_status"], "triggered")
        self.assertEqual(payload["residual_cleaner_pollution_source_status"], "answer_projection_only")
        self.assertEqual(payload["conflict_residual_chord_bridge_status"], "clear")
        self.assertEqual(payload["next_named_blocker"], "cleaner_answer_projection_residual_pollution")

    def test_cycle_free_status_triggers_when_trusted_edges_skip(self) -> None:
        tree_rows = [
            {"tree_edge_selected": True},
            {"tree_edge_selected": False},
        ]
        residual_rows = []
        payload = gate9l.build_status_payload(tree_rows, residual_rows)
        self.assertEqual(payload["trusted_forest_cycle_free_status"], "triggered")

    def test_summarize_residual_by_cell(self) -> None:
        rows = [
            {
                "cell_id": "clean_support",
                "edge_type": "answer_projection",
                "edge_transport_defect": 0.2,
                "sample_component_count": 1,
            },
            {
                "cell_id": "clean_support",
                "edge_type": "answer_projection",
                "edge_transport_defect": 0.4,
                "sample_component_count": 1,
            },
        ]
        summary = gate9l.summarize_residual_by_cell(rows)
        self.assertEqual(summary[0]["n_edges"], 2)
        self.assertAlmostEqual(summary[0]["mean_edge_transport_defect"], 0.3)


if __name__ == "__main__":
    unittest.main()
