#!/usr/bin/env python3
"""Regression tests for Gate10A trusted-tree generalization eligibility helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate10a_trusted_tree_generalization_eligibility as gate10a


def make_gate9q_row(
    edge_id: str,
    cell_class: str,
    historical_role: str = "residual_chord_candidate",
    forward_basis_role: str = "residual_chord_candidate",
    role_changed_by_adoption: bool = False,
    defect: float = 0.4,
    requires_retroactive_reinterpretation: bool = False,
    implies_operator_admission_open: bool = False,
    implies_broader_tree_settlement: bool = False,
    widens_doctrine: bool = False,
) -> dict:
    return {
        "edge_id": edge_id,
        "execution_sample_id": 1,
        "benchmark_sample_id": "bench",
        "cell_id": "cell",
        "cell_class": cell_class,
        "world_id": "world",
        "world_type": "genealogy",
        "answer_target_type": "consistent_answer",
        "edge_transport_defect": defect,
        "historical_role": historical_role,
        "forward_basis_role": forward_basis_role,
        "role_changed_by_adoption": role_changed_by_adoption,
        "requires_retroactive_reinterpretation": requires_retroactive_reinterpretation,
        "implies_operator_admission_open": implies_operator_admission_open,
        "implies_broader_tree_settlement": implies_broader_tree_settlement,
        "widens_doctrine": widens_doctrine,
    }


GATE9Q_STATUS_INTEGRATED = {
    "forward_basis_adoption_status": "adopted",
    "mainline_memory_update_status": "updated",
    "operator_admission_still_denied_status": "confirmed",
    "retroactive_reinterpretation_forbidden_status": "confirmed",
    "broader_tree_settlement_unresolved_status": "confirmed",
    "historical_lane_preservation_status": "clear",
    "integration_scope_preservation_status": "clear",
    "post_adoption_integration_readiness_status": "ready",
    "integration_outcome_status": "integrated",
    "next_named_blocker": "",
}


class RunGate10AEligibilityTest(unittest.TestCase):
    def test_eligible_when_integrated_baseline_and_no_pressures(self) -> None:
        rows = [
            make_gate9q_row(
                "c1",
                "cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
                defect=0.5,
            ),
            make_gate9q_row(
                "x1",
                "conflict",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
                defect=0.3,
            ),
        ]
        registry = gate10a.build_eligibility_registry(rows)
        status = gate10a.build_status_payload(registry, GATE9Q_STATUS_INTEGRATED)

        self.assertEqual(status["integrated_baseline_source_status"], "clear")
        self.assertEqual(status["forward_basis_adoption_preservation_status"], "clear")
        self.assertEqual(status["non_retroactive_memory_preservation_status"], "clear")
        self.assertEqual(status["operator_adjacent_rescue_pressure_status"], "clear")
        self.assertEqual(
            status["trusted_tree_semantics_broadening_pressure_status"], "clear"
        )
        self.assertEqual(status["broader_tree_settlement_non_promotion_status"], "clear")
        self.assertEqual(status["broader_candidate_eligibility_status"], "eligible")
        self.assertEqual(status["settlement_comparison_permission_status"], "permitted")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_eligible_when_forward_basis_preservation_fails(self) -> None:
        rows = [
            make_gate9q_row(
                "c1",
                "cleaner",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=True,
            )
        ]
        registry = gate10a.build_eligibility_registry(rows)
        status = gate10a.build_status_payload(registry, GATE9Q_STATUS_INTEGRATED)

        self.assertEqual(status["forward_basis_adoption_preservation_status"], "denied")
        self.assertEqual(status["broader_candidate_eligibility_status"], "not_yet_eligible")
        self.assertEqual(status["next_named_blocker"], "forward_basis_adoption_not_preserved")

    def test_not_yet_eligible_when_retroactive_pressure_appears(self) -> None:
        rows = [
            make_gate9q_row(
                "x1",
                "conflict",
                requires_retroactive_reinterpretation=True,
            )
        ]
        registry = gate10a.build_eligibility_registry(rows)
        status = gate10a.build_status_payload(registry, GATE9Q_STATUS_INTEGRATED)

        self.assertEqual(status["non_retroactive_memory_preservation_status"], "denied")
        self.assertEqual(status["broader_candidate_eligibility_status"], "not_yet_eligible")
        self.assertEqual(status["next_named_blocker"], "retroactive_reinterpretation_pressure")

    def test_not_yet_eligible_when_operator_pressure_appears(self) -> None:
        rows = [
            make_gate9q_row(
                "x1",
                "conflict",
                implies_operator_admission_open=True,
            )
        ]
        registry = gate10a.build_eligibility_registry(rows)
        status = gate10a.build_status_payload(registry, GATE9Q_STATUS_INTEGRATED)

        self.assertEqual(status["operator_adjacent_rescue_pressure_status"], "triggered")
        self.assertEqual(status["operator_admission_still_denied_status"], "violated")
        self.assertEqual(status["broader_candidate_eligibility_status"], "not_yet_eligible")
        self.assertEqual(status["next_named_blocker"], "operator_adjacent_rescue_pressure")

    def test_policy_compare_groups_baseline_and_opening_lane(self) -> None:
        rows = [
            make_gate9q_row(
                "c1",
                "cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
                defect=0.5,
            ),
            make_gate9q_row(
                "x1",
                "conflict",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
                defect=0.3,
            ),
        ]
        registry = gate10a.build_eligibility_registry(rows)
        compare = gate10a.build_policy_compare(registry)
        self.assertEqual(len(compare), 2)
        cleaner = [row for row in compare if row["cell_class"] == "cleaner"][0]
        self.assertEqual(cleaner["broader_candidate_class"], "adopted_split_baseline")
        self.assertEqual(cleaner["n_role_changed_by_adoption"], 1)
        conflict = [row for row in compare if row["cell_class"] == "conflict"][0]
        self.assertEqual(conflict["broader_candidate_class"], "broader_candidate_opening_lane")
        self.assertEqual(conflict["n_role_changed_by_adoption"], 0)


if __name__ == "__main__":
    unittest.main()
