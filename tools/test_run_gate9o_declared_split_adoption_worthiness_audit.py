#!/usr/bin/env python3
"""Regression tests for Gate9O declared-split adoption-worthiness audit helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9o_declared_split_adoption_worthiness_audit as gate9o


def make_gate9n_row(
    edge_id: str,
    cell_id: str,
    cell_class: str,
    baseline_residual_role: str = "residual_chord_candidate",
    declared_role: str = "residual_chord_candidate",
    role_coupling_class: str = "residual_only",
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
        "baseline_residual_role": baseline_residual_role,
        "declared_role": declared_role,
        "role_coupling_class": role_coupling_class,
        "role_coupling_separable": True,
        "participates_in_support_cycle": in_support_cycle,
        "participates_in_conflict_cycle": in_conflict_cycle,
        "structural_return_leg_candidate": False,
        "policy_mixing_candidate": False,
    }


GATE9N_STATUS_SEPARABLE = {
    "conflict_bridge_preservation_status": "clear",
    "closure_doctrine_preservation_status": "clear",
    "cleaner_pollution_reduction_status": "reduced",
    "role_coupling_separability_status": "separable",
    "scalar_masking_violation_status": "denied",
    "undeclared_role_surgery_required_status": "denied",
    "next_named_blocker": "",
}


class RunGate9ODeclaredSplitAdoptionWorthinessAuditTest(unittest.TestCase):
    def test_adoption_worthy_when_split_clears_bypass(self) -> None:
        """Full separable split with bridge+closure preserved → adoption_worthy."""
        source_rows = [
            make_gate9n_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only",
                          in_support_cycle=True, defect=0.5),
            make_gate9n_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate",
                          in_support_cycle=True, in_conflict_cycle=True, defect=0.3),
        ]
        registry = gate9o.build_adoption_worthiness_registry(source_rows)
        status = gate9o.build_status_payload(registry, GATE9N_STATUS_SEPARABLE)

        self.assertEqual(status["baseline_bypass_readiness_status"], "denied")
        self.assertEqual(status["declared_split_bypass_readiness_status"], "clear")
        self.assertEqual(status["conflict_bridge_preservation_status"], "clear")
        self.assertEqual(status["closure_doctrine_preservation_status"], "clear")
        self.assertEqual(status["cleaner_pollution_reduction_status"], "reduced")
        self.assertEqual(status["decision_relevant_cleaner_pollution_reduction_status"], "decision_relevant")
        self.assertEqual(status["adoption_worthiness_status"], "adoption_worthy")
        self.assertEqual(status["operator_admission_non_promotion_status"], "confirmed")
        self.assertEqual(status["scalar_masking_violation_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_adoption_worthy_when_bypass_still_denied(self) -> None:
        """If cleaner edge remains as residual_chord_candidate, bypass still denied."""
        source_rows = [
            # Cleaner edge NOT moved to auxiliary → bypass still blocked
            make_gate9n_row("c1", "clean_support", "cleaner",
                          declared_role="residual_chord_candidate",
                          role_coupling_class="residual_only",
                          in_support_cycle=True),
            make_gate9n_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate",
                          in_support_cycle=True, in_conflict_cycle=True),
        ]
        gate9n_status_unchanged = dict(GATE9N_STATUS_SEPARABLE)
        gate9n_status_unchanged["cleaner_pollution_reduction_status"] = "unchanged"
        registry = gate9o.build_adoption_worthiness_registry(source_rows)
        status = gate9o.build_status_payload(registry, gate9n_status_unchanged)

        self.assertEqual(status["baseline_bypass_readiness_status"], "denied")
        self.assertEqual(status["declared_split_bypass_readiness_status"], "denied")
        self.assertEqual(status["decision_relevant_cleaner_pollution_reduction_status"], "not_decision_relevant")
        self.assertEqual(status["adoption_worthiness_status"], "not_adoption_worthy")
        self.assertEqual(status["next_named_blocker"], "cleaner_answer_projection_role_coupling")

    def test_not_adoption_worthy_when_bridge_degrades(self) -> None:
        """Even if bypass clears, bridge degradation → not adoption_worthy."""
        source_rows = [
            make_gate9n_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only",
                          in_support_cycle=True),
            make_gate9n_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate",
                          in_support_cycle=True, in_conflict_cycle=True),
        ]
        gate9n_status_bridge_broken = dict(GATE9N_STATUS_SEPARABLE)
        gate9n_status_bridge_broken["conflict_bridge_preservation_status"] = "denied"
        registry = gate9o.build_adoption_worthiness_registry(source_rows)
        status = gate9o.build_status_payload(registry, gate9n_status_bridge_broken)

        self.assertEqual(status["conflict_bridge_preservation_status"], "denied")
        self.assertEqual(status["adoption_worthiness_status"], "not_adoption_worthy")

    def test_registry_bypass_classification(self) -> None:
        """Verify baseline_blocks_bypass and declared_split_blocks_bypass fields."""
        source_rows = [
            make_gate9n_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only"),
            make_gate9n_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate"),
        ]
        registry = gate9o.build_adoption_worthiness_registry(source_rows)

        cleaner_row = [r for r in registry if r["cell_class"] == "cleaner"][0]
        self.assertTrue(cleaner_row["baseline_blocks_bypass"])
        self.assertFalse(cleaner_row["declared_split_blocks_bypass"])

        conflict_row = [r for r in registry if r["cell_class"] == "conflict"][0]
        self.assertFalse(conflict_row["baseline_blocks_bypass"])
        self.assertFalse(conflict_row["declared_split_blocks_bypass"])

    def test_policy_compare_structure(self) -> None:
        """Policy compare table groups by cell_class × baseline_role × declared_role."""
        source_rows = [
            make_gate9n_row("c1", "clean_support", "cleaner",
                          declared_role="closure_return_leg_auxiliary",
                          role_coupling_class="auxiliary_only", defect=0.5),
            make_gate9n_row("x1", "direct_contradiction", "conflict",
                          declared_role="residual_chord_candidate", defect=0.3),
        ]
        registry = gate9o.build_adoption_worthiness_registry(source_rows)
        compare = gate9o.build_policy_compare(registry)
        self.assertEqual(len(compare), 2)
        keys_seen = {(r["cell_class"], r["baseline_role"], r["declared_role"]) for r in compare}
        self.assertIn(("cleaner", "residual_chord_candidate", "closure_return_leg_auxiliary"), keys_seen)
        self.assertIn(("conflict", "residual_chord_candidate", "residual_chord_candidate"), keys_seen)


if __name__ == "__main__":
    unittest.main()
