#!/usr/bin/env python3
"""Regression tests for Gate9M cleaner answer-projection pollution helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9m_cleaner_answer_projection_pollution_audit as gate9m


def make_residual(
    edge_id: str,
    cell_id: str,
    answer_target_type: str = "consistent_answer",
    defect: float = 0.2,
) -> dict:
    return {
        "edge_id": edge_id,
        "edge_type": "answer_projection",
        "execution_sample_id": 1,
        "benchmark_sample_id": "bench",
        "cell_id": cell_id,
        "world_id": "world",
        "world_type": "type",
        "answer_target_type": answer_target_type,
        "edge_transport_defect": defect,
    }


class RunGate9MCleanerAnswerProjectionPollutionAuditTest(unittest.TestCase):
    def test_role_coupling_when_cleaner_edges_are_support_only(self) -> None:
        residual_rows = [
            make_residual("clean_a", "clean_support"),
            make_residual("clean_b", "surface_noisy_clean"),
            make_residual("conf_a", "direct_contradiction"),
            make_residual("conf_b", "distributed_incompatibility", "unsupported_bridge_answer"),
        ]
        registry_rows = gate9m.build_registry_rows(
            residual_rows,
            support_cycle_answer_projection_edges={"clean_a", "clean_b", "conf_a", "conf_b"},
            conflict_cycle_answer_projection_edges={"conf_a", "conf_b"},
        )
        payload = gate9m.build_status_payload(registry_rows)
        self.assertEqual(payload["structural_return_leg_pollution_status"], "triggered")
        self.assertEqual(payload["policy_mixing_pollution_status"], "triggered")
        self.assertEqual(payload["split_policy_conflict_bridge_preservation_status"], "clear")
        self.assertEqual(payload["removing_cleaner_answer_projection_breaks_closure_doctrine_status"], "triggered")
        self.assertEqual(payload["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_collect_cycle_answer_projection_sets(self) -> None:
        support, conflict = gate9m.collect_cycle_answer_projection_sets(
            [
                {
                    "cycle_outcome": "none",
                    "cycle_type": "support_answer_terminal_token_cycle",
                    "edge_ids": ["a:answer_projection", "x:support_anchor"],
                },
                {
                    "cycle_outcome": "none",
                    "cycle_type": "conflict_answer_terminal_token_cycle",
                    "edge_ids": ["b:answer_projection", "y:conflict_anchor"],
                },
            ]
        )
        self.assertEqual(support, {"a:answer_projection"})
        self.assertEqual(conflict, {"b:answer_projection"})

    def test_policy_summary_counts_split_roles(self) -> None:
        rows = gate9m.build_registry_rows(
            [
                make_residual("clean_a", "clean_support"),
                make_residual("conf_a", "direct_contradiction"),
            ],
            support_cycle_answer_projection_edges={"clean_a", "conf_a"},
            conflict_cycle_answer_projection_edges={"conf_a"},
        )
        summary = gate9m.summarize_policy_compare(rows)
        self.assertEqual(len(summary), 2)


if __name__ == "__main__":
    unittest.main()
