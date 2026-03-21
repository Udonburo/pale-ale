#!/usr/bin/env python3
"""Regression tests for Gate9Q post-adoption integration helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9q_post_adoption_integration as gate9q


def make_gate9p_row(
    edge_id: str,
    cell_id: str,
    cell_class: str,
    baseline_residual_role: str = "residual_chord_candidate",
    declared_role: str = "residual_chord_candidate",
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
        "baseline_residual_role": baseline_residual_role,
        "declared_role": declared_role,
        "role_coupling_class": "auxiliary_only" if declared_role == "closure_return_leg_auxiliary" else "residual_only",
        "role_coupling_separable": True,
        "baseline_blocks_bypass": cell_class == "cleaner" and baseline_residual_role == "residual_chord_candidate",
        "declared_split_blocks_bypass": cell_class == "cleaner" and declared_role == "residual_chord_candidate",
        "requires_historical_reinterpretation": False,
        "requires_doctrine_scope_change": False,
        "weakens_audit_lane_boundary": False,
        "requires_hidden_role_surgery": False,
    }


GATE9P_STATUS_ADOPT = {
    "adoption_worthiness_status": "adoption_worthy",
    "mainline_comparability_preservation_status": "clear",
    "audit_lane_boundary_preservation_status": "clear",
    "operator_admission_non_promotion_status": "confirmed",
    "historical_reinterpretation_required_status": "denied",
    "doctrine_scope_change_required_status": "denied",
    "adopt_candidate_status": "clear",
    "defer_candidate_status": "no_surviving_blocker",
    "judgment_outcome_status": "adopt",
    "next_named_blocker": "",
}


class RunGate9QPostAdoptionIntegrationTest(unittest.TestCase):
    def test_integrated_when_adopt_and_no_falsifiers(self) -> None:
        """Adopt judgment + no falsifiers → integrated."""
        source_rows = [
            make_gate9p_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary", defect=0.5),
            make_gate9p_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate", defect=0.3),
        ]
        registry = gate9q.build_integration_registry(source_rows, "adopt")
        status = gate9q.build_status_payload(registry, GATE9P_STATUS_ADOPT)

        self.assertEqual(status["forward_basis_adoption_status"], "adopted")
        self.assertEqual(status["mainline_memory_update_status"], "updated")
        self.assertEqual(status["operator_admission_still_denied_status"], "confirmed")
        self.assertEqual(status["retroactive_reinterpretation_forbidden_status"], "confirmed")
        self.assertEqual(status["broader_tree_settlement_unresolved_status"], "confirmed")
        self.assertEqual(status["historical_lane_preservation_status"], "clear")
        self.assertEqual(status["integration_scope_preservation_status"], "clear")
        self.assertEqual(status["post_adoption_integration_readiness_status"], "ready")
        self.assertEqual(status["integration_outcome_status"], "integrated")
        self.assertEqual(status["next_named_blocker"], "")

    def test_blocked_when_defer(self) -> None:
        """Defer judgment → not integrated."""
        source_rows = [
            make_gate9p_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary"),
        ]
        gate9p_defer = dict(GATE9P_STATUS_ADOPT)
        gate9p_defer["judgment_outcome_status"] = "defer"

        registry = gate9q.build_integration_registry(source_rows, "defer")
        status = gate9q.build_status_payload(registry, gate9p_defer)

        self.assertEqual(status["forward_basis_adoption_status"], "deferred")
        self.assertEqual(status["integration_outcome_status"], "blocked")
        self.assertEqual(status["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_registry_forward_basis_role_on_adopt(self) -> None:
        """On adopt, forward_basis_role = declared_role; historical_role = baseline."""
        source_rows = [
            make_gate9p_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary"),
        ]
        registry = gate9q.build_integration_registry(source_rows, "adopt")
        row = registry[0]
        self.assertEqual(row["historical_role"], "residual_chord_candidate")
        self.assertEqual(row["forward_basis_role"], "closure_return_leg_auxiliary")
        self.assertTrue(row["role_changed_by_adoption"])
        self.assertFalse(row["requires_retroactive_reinterpretation"])

    def test_registry_forward_basis_role_on_defer(self) -> None:
        """On defer, forward_basis_role = baseline_role (unchanged)."""
        source_rows = [
            make_gate9p_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary"),
        ]
        registry = gate9q.build_integration_registry(source_rows, "defer")
        row = registry[0]
        self.assertEqual(row["historical_role"], "residual_chord_candidate")
        self.assertEqual(row["forward_basis_role"], "residual_chord_candidate")
        self.assertFalse(row["role_changed_by_adoption"])

    def test_policy_compare_structure(self) -> None:
        """Policy compare groups by cell_class × historical_role × forward_basis_role."""
        source_rows = [
            make_gate9p_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary", defect=0.5),
            make_gate9p_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate", defect=0.3),
        ]
        registry = gate9q.build_integration_registry(source_rows, "adopt")
        compare = gate9q.build_policy_compare(registry)
        self.assertEqual(len(compare), 2)
        cleaner_row = [r for r in compare if r["cell_class"] == "cleaner"][0]
        self.assertEqual(cleaner_row["n_role_changed_by_adoption"], 1)
        conflict_row = [r for r in compare if r["cell_class"] == "conflict"][0]
        self.assertEqual(conflict_row["n_role_changed_by_adoption"], 0)


if __name__ == "__main__":
    unittest.main()
