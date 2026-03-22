#!/usr/bin/env python3
"""Regression tests for Gate10B trusted-tree settlement comparison helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate10b_trusted_tree_settlement_comparison as gate10b


def make_gate10a_row(
    edge_id: str,
    broader_candidate_class: str,
    cell_class: str,
    forward_basis_role: str,
    historical_role: str = "residual_chord_candidate",
    role_changed_by_adoption: bool = False,
    defect: float = 0.4,
    forward_basis_adoption_preserved: bool = True,
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
        "broader_candidate_class": broader_candidate_class,
        "forward_basis_adoption_preserved": forward_basis_adoption_preserved,
        "requires_retroactive_reinterpretation": requires_retroactive_reinterpretation,
        "implies_operator_admission_open": implies_operator_admission_open,
        "implies_broader_tree_settlement": implies_broader_tree_settlement,
        "widens_doctrine": widens_doctrine,
    }


GATE10A_STATUS_ELIGIBLE = {
    "integrated_baseline_source_status": "clear",
    "forward_basis_adoption_preservation_status": "clear",
    "non_retroactive_memory_preservation_status": "clear",
    "operator_adjacent_rescue_pressure_status": "clear",
    "trusted_tree_semantics_broadening_pressure_status": "clear",
    "broader_tree_settlement_non_promotion_status": "clear",
    "operator_admission_still_denied_status": "confirmed",
    "broader_candidate_eligibility_status": "eligible",
    "settlement_comparison_permission_status": "permitted",
    "next_named_blocker": "",
}


class RunGate10BSettlementComparisonTest(unittest.TestCase):
    def test_settled_when_all_checks_clear_and_candidate_adds_conflict_lane(self) -> None:
        rows = [
            make_gate10a_row(
                edge_id="baseline-1",
                broader_candidate_class=gate10b.BASELINE_LANE,
                cell_class="cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
                defect=0.6,
            ),
            make_gate10a_row(
                edge_id="candidate-1",
                broader_candidate_class=gate10b.CANDIDATE_LANE,
                cell_class="conflict",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
                defect=0.35,
            ),
        ]

        status = gate10b.build_status_payload(rows, GATE10A_STATUS_ELIGIBLE)

        self.assertEqual(status["forward_basis_baseline_preservation_status"], "clear")
        self.assertEqual(status["conflict_side_bridge_preservation_status"], "clear")
        self.assertEqual(status["decision_relevant_gain_beyond_baseline_status"], "present")
        self.assertEqual(status["comparison_outcome_status"], "settled")
        self.assertEqual(status["next_named_blocker"], "")

    def test_denied_when_conflict_side_bridge_degrades(self) -> None:
        rows = [
            make_gate10a_row(
                edge_id="baseline-1",
                broader_candidate_class=gate10b.BASELINE_LANE,
                cell_class="cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
            ),
            make_gate10a_row(
                edge_id="candidate-1",
                broader_candidate_class=gate10b.CANDIDATE_LANE,
                cell_class="cleaner",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
            ),
        ]

        status = gate10b.build_status_payload(rows, GATE10A_STATUS_ELIGIBLE)

        self.assertEqual(status["conflict_side_bridge_preservation_status"], "denied")
        self.assertEqual(status["comparison_outcome_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "conflict_side_bridge_degrades")

    def test_bounded_keep_when_gain_beyond_baseline_is_absent(self) -> None:
        rows = [
            make_gate10a_row(
                edge_id="shared-1",
                broader_candidate_class=gate10b.BASELINE_LANE,
                cell_class="cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
            ),
            make_gate10a_row(
                edge_id="shared-1",
                broader_candidate_class=gate10b.CANDIDATE_LANE,
                cell_class="conflict",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
            ),
        ]

        status = gate10b.build_status_payload(rows, GATE10A_STATUS_ELIGIBLE)

        self.assertEqual(status["decision_relevant_gain_beyond_baseline_status"], "absent")
        self.assertEqual(status["comparison_outcome_status"], "bounded keep")
        self.assertEqual(
            status["next_named_blocker"],
            "decision_relevant_gain_beyond_baseline_absent",
        )

    def test_deferred_when_gate10a_has_not_permitted_comparison(self) -> None:
        rows = [
            make_gate10a_row(
                edge_id="baseline-1",
                broader_candidate_class=gate10b.BASELINE_LANE,
                cell_class="cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
            ),
            make_gate10a_row(
                edge_id="candidate-1",
                broader_candidate_class=gate10b.CANDIDATE_LANE,
                cell_class="conflict",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
            ),
        ]
        gate10a_status = dict(GATE10A_STATUS_ELIGIBLE)
        gate10a_status["broader_candidate_eligibility_status"] = "not_yet_eligible"
        gate10a_status["settlement_comparison_permission_status"] = "withheld"
        gate10a_status["next_named_blocker"] = "retroactive_reinterpretation_pressure"

        status = gate10b.build_status_payload(rows, gate10a_status)

        self.assertEqual(status["comparison_outcome_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "retroactive_reinterpretation_pressure")

    def test_policy_compare_emits_exact_two_lanes(self) -> None:
        rows = [
            make_gate10a_row(
                edge_id="baseline-1",
                broader_candidate_class=gate10b.BASELINE_LANE,
                cell_class="cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
                defect=0.6,
            ),
            make_gate10a_row(
                edge_id="candidate-1",
                broader_candidate_class=gate10b.CANDIDATE_LANE,
                cell_class="conflict",
                forward_basis_role="residual_chord_candidate",
                role_changed_by_adoption=False,
                defect=0.35,
            ),
        ]

        compare = gate10b.build_policy_compare(rows)

        self.assertEqual(len(compare), 2)
        self.assertEqual(compare[0]["broader_candidate_class"], gate10b.BASELINE_LANE)
        self.assertEqual(compare[1]["broader_candidate_class"], gate10b.CANDIDATE_LANE)


if __name__ == "__main__":
    unittest.main()