#!/usr/bin/env python3
"""Regression tests for Gate9N cleaner answer-projection role-coupling separation audit helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9n_cleaner_answer_projection_role_coupling_audit as gate9n


def make_gate9m_row(
    edge_id: str,
    cell_id: str,
    cell_class: str,
    split_policy_role: str,
    in_support_cycle: bool = False,
    in_conflict_cycle: bool = False,
    defect: float = 0.4,
) -> dict:
    return {
        "edge_id": edge_id,
        "execution_sample_id": 1,
        "benchmark_sample_id": "bench",
        "cell_id": cell_id,
        "cell_class": cell_class,
        "world_id": "world",
        "world_type": "genealogy",
        "answer_target_type": "consistent_answer",
        "edge_transport_defect": defect,
        "baseline_residual_role": "residual_chord_candidate",
        "split_policy_role": split_policy_role,
        "participates_in_support_cycle": in_support_cycle,
        "participates_in_conflict_cycle": in_conflict_cycle,
        "structural_return_leg_candidate": cell_class == "cleaner" and in_support_cycle,
        "policy_mixing_candidate": cell_class == "cleaner" and in_support_cycle,
    }


class RunGate9NCleanerAnswerProjectionRoleCouplingAuditTest(unittest.TestCase):
    def test_role_coupling_registry_classifies_edges(self) -> None:
        """Cleaner edges → auxiliary_only, conflict edges → residual_only."""
        source_rows = [
            make_gate9m_row("c1", "clean_support", "cleaner", "closure_return_leg_auxiliary", in_support_cycle=True),
            make_gate9m_row("c2", "surface_noisy_clean", "cleaner", "closure_return_leg_auxiliary", in_support_cycle=True),
            make_gate9m_row("x1", "direct_contradiction", "conflict", "residual_chord_candidate", in_support_cycle=True, in_conflict_cycle=True),
            make_gate9m_row("x2", "distributed_incompatibility", "conflict", "residual_chord_candidate", in_support_cycle=True, in_conflict_cycle=True),
        ]
        registry = gate9n.build_role_coupling_registry(source_rows)
        self.assertEqual(len(registry), 4)

        cleaner_rows = [r for r in registry if r["cell_class"] == "cleaner"]
        conflict_rows = [r for r in registry if r["cell_class"] == "conflict"]
        for r in cleaner_rows:
            self.assertEqual(r["role_coupling_class"], "auxiliary_only")
            self.assertEqual(r["declared_role"], "closure_return_leg_auxiliary")
            self.assertTrue(r["role_coupling_separable"])
        for r in conflict_rows:
            self.assertEqual(r["role_coupling_class"], "residual_only")
            self.assertEqual(r["declared_role"], "residual_chord_candidate")
            self.assertTrue(r["role_coupling_separable"])

    def test_status_payload_separable_case(self) -> None:
        """When split cleanly separates, role_coupling_separability_status is 'separable'."""
        source_rows = [
            make_gate9m_row("c1", "clean_support", "cleaner", "closure_return_leg_auxiliary", in_support_cycle=True),
            make_gate9m_row("c2", "surface_noisy_clean", "cleaner", "closure_return_leg_auxiliary", in_support_cycle=True),
            make_gate9m_row("x1", "direct_contradiction", "conflict", "residual_chord_candidate", in_support_cycle=True, in_conflict_cycle=True),
            make_gate9m_row("x2", "distributed_incompatibility", "conflict", "residual_chord_candidate", in_support_cycle=True, in_conflict_cycle=True),
        ]
        registry = gate9n.build_role_coupling_registry(source_rows)
        status = gate9n.build_status_payload(registry, {})

        self.assertEqual(status["baseline_cleaner_residual_answer_projection_edge_count"], 2)
        self.assertEqual(status["declared_split_cleaner_auxiliary_answer_projection_edge_count"], 2)
        self.assertEqual(status["declared_split_conflict_residual_answer_projection_edge_count"], 2)
        self.assertEqual(status["conflict_bridge_preservation_status"], "clear")
        self.assertEqual(status["closure_doctrine_preservation_status"], "clear")
        self.assertEqual(status["cleaner_pollution_reduction_status"], "reduced")
        self.assertEqual(status["role_coupling_separability_status"], "separable")
        self.assertEqual(status["scalar_masking_violation_status"], "denied")
        self.assertEqual(status["undeclared_role_surgery_required_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "")

    def test_status_payload_coupled_when_bridge_degrades(self) -> None:
        """When conflict edges lack cycle participation, bridge degrades → coupled."""
        source_rows = [
            make_gate9m_row("c1", "clean_support", "cleaner", "closure_return_leg_auxiliary", in_support_cycle=True),
            # Conflict edge with no cycle participation → bridge degraded
            make_gate9m_row("x1", "direct_contradiction", "conflict", "residual_chord_candidate", in_support_cycle=False, in_conflict_cycle=False),
        ]
        registry = gate9n.build_role_coupling_registry(source_rows)
        status = gate9n.build_status_payload(registry, {})

        self.assertEqual(status["conflict_bridge_preservation_status"], "denied")
        self.assertEqual(status["role_coupling_separability_status"], "coupled")
        self.assertEqual(status["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_status_payload_coupled_when_closure_breaks(self) -> None:
        """When cleaner auxiliary edges have no support cycle, closure breaks → coupled."""
        source_rows = [
            # Cleaner edge moved to auxiliary but not in support cycle
            make_gate9m_row("c1", "clean_support", "cleaner", "closure_return_leg_auxiliary", in_support_cycle=False),
            make_gate9m_row("x1", "direct_contradiction", "conflict", "residual_chord_candidate", in_support_cycle=True, in_conflict_cycle=True),
        ]
        registry = gate9n.build_role_coupling_registry(source_rows)
        status = gate9n.build_status_payload(registry, {})

        self.assertEqual(status["closure_doctrine_preservation_status"], "denied")
        self.assertEqual(status["role_coupling_separability_status"], "coupled")
        self.assertEqual(status["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_policy_compare_table_structure(self) -> None:
        """Policy compare table has correct grouping and counts."""
        source_rows = [
            make_gate9m_row("c1", "clean_support", "cleaner", "closure_return_leg_auxiliary", defect=0.5),
            make_gate9m_row("x1", "direct_contradiction", "conflict", "residual_chord_candidate", defect=0.3),
        ]
        registry = gate9n.build_role_coupling_registry(source_rows)
        compare = gate9n.build_policy_compare(registry)
        # Baseline: both under residual_chord_candidate
        # Declared: cleaner→auxiliary, conflict→residual
        self.assertTrue(len(compare) >= 2)
        roles_seen = {(r["cell_class"], r["role"]) for r in compare}
        self.assertIn(("cleaner", "closure_return_leg_auxiliary"), roles_seen)
        self.assertIn(("conflict", "residual_chord_candidate"), roles_seen)


if __name__ == "__main__":
    unittest.main()
