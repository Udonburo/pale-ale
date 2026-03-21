#!/usr/bin/env python3
"""Regression tests for Gate9P declared-split adopt-or-defer judgment helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9p_declared_split_adopt_or_defer_judgment as gate9p


def make_gate9o_row(
    edge_id: str,
    cell_id: str,
    cell_class: str,
    baseline_residual_role: str = "residual_chord_candidate",
    declared_role: str = "residual_chord_candidate",
    role_coupling_class: str = "residual_only",
    baseline_blocks_bypass: bool = False,
    declared_split_blocks_bypass: bool = False,
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
        "role_coupling_class": role_coupling_class,
        "role_coupling_separable": True,
        "participates_in_support_cycle": True,
        "participates_in_conflict_cycle": cell_class == "conflict",
        "baseline_blocks_bypass": baseline_blocks_bypass,
        "declared_split_blocks_bypass": declared_split_blocks_bypass,
    }


GATE9O_STATUS_ADOPTION_WORTHY = {
    "baseline_bypass_readiness_status": "denied",
    "declared_split_bypass_readiness_status": "clear",
    "conflict_bridge_preservation_status": "clear",
    "closure_doctrine_preservation_status": "clear",
    "cleaner_pollution_reduction_status": "reduced",
    "decision_relevant_cleaner_pollution_reduction_status": "decision_relevant",
    "adoption_worthiness_status": "adoption_worthy",
    "operator_admission_non_promotion_status": "confirmed",
    "scalar_masking_violation_status": "denied",
    "next_named_blocker": "",
}


class RunGate9PDeclaredSplitAdoptOrDeferJudgmentTest(unittest.TestCase):
    def test_adopt_when_all_falsifiers_clear(self) -> None:
        """Adoption-worthy + no falsifiers + no deferral blocker → adopt."""
        source_rows = [
            make_gate9o_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only",
                          baseline_blocks_bypass=True,
                          declared_split_blocks_bypass=False),
            make_gate9o_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate"),
        ]
        registry = gate9p.build_adopt_or_defer_registry(source_rows)
        status = gate9p.build_status_payload(registry, GATE9O_STATUS_ADOPTION_WORTHY)

        self.assertEqual(status["adoption_worthiness_status"], "adoption_worthy")
        self.assertEqual(status["mainline_comparability_preservation_status"], "clear")
        self.assertEqual(status["audit_lane_boundary_preservation_status"], "clear")
        self.assertEqual(status["operator_admission_non_promotion_status"], "confirmed")
        self.assertEqual(status["historical_reinterpretation_required_status"], "denied")
        self.assertEqual(status["doctrine_scope_change_required_status"], "denied")
        self.assertEqual(status["adopt_candidate_status"], "clear")
        self.assertEqual(status["defer_candidate_status"], "no_surviving_blocker")
        self.assertEqual(status["judgment_outcome_status"], "adopt")
        self.assertEqual(status["next_named_blocker"], "")

    def test_defer_when_not_adoption_worthy(self) -> None:
        """Not adoption-worthy → defer regardless of falsifiers."""
        source_rows = [
            make_gate9o_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only"),
        ]
        gate9o_status_not_worthy = dict(GATE9O_STATUS_ADOPTION_WORTHY)
        gate9o_status_not_worthy["adoption_worthiness_status"] = "not_adoption_worthy"
        gate9o_status_not_worthy["next_named_blocker"] = "cleaner_answer_projection_role_coupling"

        registry = gate9p.build_adopt_or_defer_registry(source_rows)
        status = gate9p.build_status_payload(registry, gate9o_status_not_worthy)

        self.assertEqual(status["adopt_candidate_status"], "denied")
        self.assertEqual(status["defer_candidate_status"], "has_surviving_blocker")
        self.assertEqual(status["judgment_outcome_status"], "defer")
        self.assertEqual(status["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_defer_when_deferral_blocker_survives(self) -> None:
        """Adoption-worthy but deferral has surviving blocker → defer."""
        source_rows = [
            make_gate9o_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only"),
        ]
        gate9o_status_with_blocker = dict(GATE9O_STATUS_ADOPTION_WORTHY)
        gate9o_status_with_blocker["next_named_blocker"] = "cleaner_answer_projection_role_coupling"

        registry = gate9p.build_adopt_or_defer_registry(source_rows)
        status = gate9p.build_status_payload(registry, gate9o_status_with_blocker)

        self.assertEqual(status["adopt_candidate_status"], "clear")
        self.assertEqual(status["defer_candidate_status"], "has_surviving_blocker")
        self.assertEqual(status["judgment_outcome_status"], "defer")
        self.assertEqual(status["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_registry_falsifier_fields_all_false(self) -> None:
        """All falsifier fields are False for the normal declared split."""
        source_rows = [
            make_gate9o_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only"),
        ]
        registry = gate9p.build_adopt_or_defer_registry(source_rows)
        for row in registry:
            self.assertFalse(row["requires_historical_reinterpretation"])
            self.assertFalse(row["requires_doctrine_scope_change"])
            self.assertFalse(row["weakens_audit_lane_boundary"])
            self.assertFalse(row["requires_hidden_role_surgery"])

    def test_policy_compare_structure(self) -> None:
        """Policy compare groups by cell_class × baseline_role × declared_role."""
        source_rows = [
            make_gate9o_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only", defect=0.5),
            make_gate9o_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate", defect=0.3),
        ]
        registry = gate9p.build_adopt_or_defer_registry(source_rows)
        compare = gate9p.build_policy_compare(registry)
        self.assertEqual(len(compare), 2)
        for row in compare:
            self.assertEqual(row["n_requires_historical_reinterpretation"], 0)
            self.assertEqual(row["n_requires_doctrine_scope_change"], 0)
            self.assertEqual(row["n_weakens_audit_lane_boundary"], 0)
            self.assertEqual(row["n_requires_hidden_role_surgery"], 0)


if __name__ == "__main__":
    unittest.main()
