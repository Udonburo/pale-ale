#!/usr/bin/env python3
"""Regression tests for Gate10C second settlement comparison helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate10c_second_settlement_comparison as gate10c


def make_gate10b_row(
    edge_id: str,
    broader_candidate_class: str,
    cell_id: str,
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
        "cell_id": cell_id,
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


GATE10B_STATUS_SETTLED = {
    "forward_basis_baseline_preservation_status": "clear",
    "conflict_side_bridge_preservation_status": "clear",
    "non_retroactive_memory_preservation_status": "clear",
    "operator_adjacent_rescue_pressure_status": "clear",
    "trusted_tree_semantics_broadening_pressure_status": "clear",
    "decision_relevant_gain_beyond_baseline_status": "present",
    "comparison_outcome_status": "settled",
    "operator_admission_still_denied_status": "confirmed",
    "broader_tree_settlement_non_promotion_status": "clear",
    "next_named_blocker": "",
}


class RunGate10CSecondSettlementComparisonTest(unittest.TestCase):
    def test_registry_extracts_only_baseline_and_distributed_incompatibility(self) -> None:
        source_rows = [
            make_gate10b_row(
                edge_id="baseline-1",
                broader_candidate_class=gate10c.BASELINE_LANE,
                cell_id="clean_support",
                cell_class="cleaner",
                forward_basis_role="closure_return_leg_auxiliary",
                role_changed_by_adoption=True,
            ),
            make_gate10b_row(
                edge_id="candidate-1",
                broader_candidate_class=gate10c.SOURCE_CANDIDATE_LANE,
                cell_id=gate10c.SECOND_CANDIDATE_CELL_ID,
                cell_class="conflict",
                forward_basis_role="residual_chord_candidate",
            ),
            make_gate10b_row(
                edge_id="out-of-scope-1",
                broader_candidate_class=gate10c.SOURCE_CANDIDATE_LANE,
                cell_id="direct_contradiction",
                cell_class="conflict",
                forward_basis_role="residual_chord_candidate",
            ),
        ]

        registry = gate10c.build_comparison_registry(source_rows)

        self.assertEqual(len(registry), 2)
        self.assertEqual(registry[0]["comparison_lane"], gate10c.BASELINE_LANE)
        self.assertEqual(registry[1]["comparison_lane"], gate10c.SECOND_CANDIDATE_LANE)
        self.assertEqual(registry[1]["second_candidate_declaration"], "declaratively_extracted")

    def test_settled_when_second_candidate_is_declared_and_checks_clear(self) -> None:
        rows = [
            {
                **make_gate10b_row(
                    edge_id="baseline-1",
                    broader_candidate_class=gate10c.BASELINE_LANE,
                    cell_id="clean_support",
                    cell_class="cleaner",
                    forward_basis_role="closure_return_leg_auxiliary",
                    role_changed_by_adoption=True,
                    defect=0.6,
                ),
                "comparison_lane": gate10c.BASELINE_LANE,
                "second_candidate_declaration": "not_applicable",
            },
            {
                **make_gate10b_row(
                    edge_id="candidate-1",
                    broader_candidate_class=gate10c.SOURCE_CANDIDATE_LANE,
                    cell_id=gate10c.SECOND_CANDIDATE_CELL_ID,
                    cell_class="conflict",
                    forward_basis_role="residual_chord_candidate",
                    defect=0.38,
                ),
                "comparison_lane": gate10c.SECOND_CANDIDATE_LANE,
                "second_candidate_declaration": "declaratively_extracted",
            },
        ]

        status = gate10c.build_status_payload(rows, GATE10B_STATUS_SETTLED)

        self.assertEqual(status["second_candidate_declaration_status"], "clear")
        self.assertEqual(status["gate10b_slice_non_retroactive_preservation_status"], "clear")
        self.assertEqual(status["decision_relevant_gain_beyond_baseline_status"], "present")
        self.assertEqual(status["comparison_outcome_status"], "settled")

    def test_denied_when_second_candidate_declaration_fails(self) -> None:
        rows = [
            {
                **make_gate10b_row(
                    edge_id="baseline-1",
                    broader_candidate_class=gate10c.BASELINE_LANE,
                    cell_id="clean_support",
                    cell_class="cleaner",
                    forward_basis_role="closure_return_leg_auxiliary",
                    role_changed_by_adoption=True,
                ),
                "comparison_lane": gate10c.BASELINE_LANE,
                "second_candidate_declaration": "not_applicable",
            },
            {
                **make_gate10b_row(
                    edge_id="candidate-1",
                    broader_candidate_class=gate10c.SOURCE_CANDIDATE_LANE,
                    cell_id="direct_contradiction",
                    cell_class="conflict",
                    forward_basis_role="residual_chord_candidate",
                ),
                "comparison_lane": gate10c.SECOND_CANDIDATE_LANE,
                "second_candidate_declaration": "declaratively_extracted",
            },
        ]

        status = gate10c.build_status_payload(rows, GATE10B_STATUS_SETTLED)

        self.assertEqual(status["second_candidate_declaration_status"], "denied")
        self.assertEqual(status["comparison_outcome_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "second_candidate_declaration_integrity_fails")

    def test_bounded_keep_when_gain_beyond_baseline_is_absent(self) -> None:
        rows = [
            {
                **make_gate10b_row(
                    edge_id="shared-1",
                    broader_candidate_class=gate10c.BASELINE_LANE,
                    cell_id="clean_support",
                    cell_class="cleaner",
                    forward_basis_role="closure_return_leg_auxiliary",
                    role_changed_by_adoption=True,
                ),
                "comparison_lane": gate10c.BASELINE_LANE,
                "second_candidate_declaration": "not_applicable",
            },
            {
                **make_gate10b_row(
                    edge_id="shared-1",
                    broader_candidate_class=gate10c.SOURCE_CANDIDATE_LANE,
                    cell_id=gate10c.SECOND_CANDIDATE_CELL_ID,
                    cell_class="conflict",
                    forward_basis_role="residual_chord_candidate",
                ),
                "comparison_lane": gate10c.SECOND_CANDIDATE_LANE,
                "second_candidate_declaration": "declaratively_extracted",
            },
        ]

        status = gate10c.build_status_payload(rows, GATE10B_STATUS_SETTLED)

        self.assertEqual(status["decision_relevant_gain_beyond_baseline_status"], "absent")
        self.assertEqual(status["comparison_outcome_status"], "bounded keep")
        self.assertEqual(status["next_named_blocker"], "decision_relevant_gain_beyond_baseline_absent")

    def test_deferred_when_gate10b_source_slice_is_not_preserved(self) -> None:
        rows = [
            {
                **make_gate10b_row(
                    edge_id="baseline-1",
                    broader_candidate_class=gate10c.BASELINE_LANE,
                    cell_id="clean_support",
                    cell_class="cleaner",
                    forward_basis_role="closure_return_leg_auxiliary",
                    role_changed_by_adoption=True,
                ),
                "comparison_lane": gate10c.BASELINE_LANE,
                "second_candidate_declaration": "not_applicable",
            },
            {
                **make_gate10b_row(
                    edge_id="candidate-1",
                    broader_candidate_class=gate10c.SOURCE_CANDIDATE_LANE,
                    cell_id=gate10c.SECOND_CANDIDATE_CELL_ID,
                    cell_class="conflict",
                    forward_basis_role="residual_chord_candidate",
                ),
                "comparison_lane": gate10c.SECOND_CANDIDATE_LANE,
                "second_candidate_declaration": "declaratively_extracted",
            },
        ]
        gate10b_status = dict(GATE10B_STATUS_SETTLED)
        gate10b_status["comparison_outcome_status"] = "denied"
        gate10b_status["next_named_blocker"] = "conflict_side_bridge_degrades"

        status = gate10c.build_status_payload(rows, gate10b_status)

        self.assertEqual(status["comparison_outcome_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "conflict_side_bridge_degrades")


if __name__ == "__main__":
    unittest.main()