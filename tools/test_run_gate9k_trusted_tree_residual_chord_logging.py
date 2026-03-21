#!/usr/bin/env python3
"""Regression tests for Gate9K trusted-tree / residual-chord helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9k_trusted_tree_residual_chord_logging as gate9k


def make_edge(edge_type: str, *, cell_id: str = "clean_support", defect: float = 0.1) -> dict:
    return {
        "edge_id": f"{edge_type}:{cell_id}",
        "edge_type": edge_type,
        "execution_sample_id": 1,
        "benchmark_sample_id": "bench",
        "cell_id": cell_id,
        "world_id": "world",
        "world_type": "type",
        "answer_target_type": "consistent_answer",
        "quietness_pair_id": "",
        "rendering_family_id": "transcript_v1",
        "source_node_type": "token_state",
        "target_node_type": "answer_state",
        "edge_outcome": "none",
        "edge_transport_defect": defect,
        "transport_mode": "mode",
    }


class RunGate9KTrustedTreeResidualChordLoggingTest(unittest.TestCase):
    def test_role_for_edge_type(self) -> None:
        self.assertEqual(gate9k.role_for_edge_type("temporal_transition")[0], "trusted_tree_candidate")
        self.assertEqual(gate9k.role_for_edge_type("support_anchor")[0], "trusted_tree_candidate")
        self.assertEqual(gate9k.role_for_edge_type("conflict_anchor")[0], "residual_chord_candidate")
        self.assertEqual(gate9k.role_for_edge_type("answer_projection")[0], "residual_chord_candidate")
        self.assertEqual(gate9k.role_for_edge_type("quietness_pair")[0], "excluded_nonstructural")

    def test_build_registry_rows_classifies_edges(self) -> None:
        rows = gate9k.build_registry_rows(
            [
                make_edge("temporal_transition"),
                make_edge("conflict_anchor", cell_id="distributed_incompatibility"),
                make_edge("quietness_pair", cell_id="surface_noisy_clean"),
            ]
        )
        self.assertEqual(rows[0]["decomposition_role"], "trusted_tree_candidate")
        self.assertEqual(rows[1]["decomposition_role"], "residual_chord_candidate")
        self.assertEqual(rows[2]["decomposition_role"], "excluded_nonstructural")

    def test_build_status_payload_enforces_non_promotion(self) -> None:
        registry_rows = gate9k.build_registry_rows(
            [
                make_edge("temporal_transition"),
                make_edge("support_anchor"),
                make_edge("conflict_anchor", cell_id="distributed_incompatibility"),
                make_edge("answer_projection", cell_id="distributed_incompatibility"),
            ]
        )
        payload = gate9k.build_status_payload(
            registry_rows,
            {"support_anchor_cleaner_dominance_status": "triggered"},
            {
                "distributed_underactivation_status": "triggered",
                "distributed_consistent_branch_status": "underactivated",
            },
        )
        self.assertEqual(payload["trusted_tree_candidate_edge_count"], 2)
        self.assertEqual(payload["residual_chord_candidate_edge_count"], 2)
        self.assertEqual(payload["scalar_masking_violation_status"], "clear")
        self.assertEqual(payload["operator_admission_non_promotion_status"], "enforced")
        self.assertEqual(payload["decomposition_hypothesis_execution_status"], "not_yet_executed")


if __name__ == "__main__":
    unittest.main()
