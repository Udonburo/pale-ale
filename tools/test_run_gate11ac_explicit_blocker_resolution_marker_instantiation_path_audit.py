#!/usr/bin/env python3
"""Regression tests for Gate11AC explicit blocker-resolution marker instantiation-path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11ac_explicit_blocker_resolution_marker_instantiation_path_audit as gate11ac


def make_gate11ab_manifest(run_id: str = "gate11ab_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11ab_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    gate11d_not_yet_declared_state_preservation_status: str = "preserved",
    gate11e_path_defined_state_preservation_status: str = "preserved",
    gate11f_not_yet_admissible_state_preservation_status: str = "preserved",
    gate11g_naming_surface_preservation_status: str = "preserved",
    gate11h_not_yet_named_state_preservation_status: str = "preserved",
    gate11i_path_defined_state_preservation_status: str = "preserved",
    gate11j_not_yet_admissible_state_preservation_status: str = "preserved",
    gate11k_not_yet_present_state_preservation_status: str = "preserved",
    gate11l_path_defined_state_preservation_status: str = "preserved",
    gate11m_not_yet_present_state_preservation_status: str = "preserved",
    gate11n_residual_named_state_preservation_status: str = "preserved",
    gate11o_path_defined_state_preservation_status: str = "preserved",
    gate11p_not_yet_completed_state_preservation_status: str = "preserved",
    gate11q_surface_defined_state_preservation_status: str = "preserved",
    gate11r_not_yet_present_state_preservation_status: str = "preserved",
    gate11s_path_defined_state_preservation_status: str = "preserved",
    gate11t_not_yet_present_state_preservation_status: str = "preserved",
    gate11u_residual_named_state_preservation_status: str = "preserved",
    gate11v_path_defined_state_preservation_status: str = "preserved",
    gate11w_not_yet_completed_state_preservation_status: str = "preserved",
    gate11x_blocker_named_state_preservation_status: str = "preserved",
    gate11y_path_defined_state_preservation_status: str = "preserved",
    gate11z_not_yet_resolved_state_preservation_status: str = "preserved",
    gate11aa_surface_defined_state_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_blocker_resolution_marker_status: str = "absent",
    blocker_resolution_marker_singularity_status: str = "none",
    same_source_blocker_resolution_marker_binding_status: str = "not_explicit",
    blocker_resolution_marker_boundary_status: str = "confirmed",
    one_explicit_blocker_resolution_marker_status: str = "not_yet_present",
    next_named_blocker: str = "no_explicit_blocker_resolution_marker",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "gate11f_not_yet_admissible_state_preservation_status": gate11f_not_yet_admissible_state_preservation_status,
        "gate11g_naming_surface_preservation_status": gate11g_naming_surface_preservation_status,
        "gate11h_not_yet_named_state_preservation_status": gate11h_not_yet_named_state_preservation_status,
        "gate11i_path_defined_state_preservation_status": gate11i_path_defined_state_preservation_status,
        "gate11j_not_yet_admissible_state_preservation_status": gate11j_not_yet_admissible_state_preservation_status,
        "gate11k_not_yet_present_state_preservation_status": gate11k_not_yet_present_state_preservation_status,
        "gate11l_path_defined_state_preservation_status": gate11l_path_defined_state_preservation_status,
        "gate11m_not_yet_present_state_preservation_status": gate11m_not_yet_present_state_preservation_status,
        "gate11n_residual_named_state_preservation_status": gate11n_residual_named_state_preservation_status,
        "gate11o_path_defined_state_preservation_status": gate11o_path_defined_state_preservation_status,
        "gate11p_not_yet_completed_state_preservation_status": gate11p_not_yet_completed_state_preservation_status,
        "gate11q_surface_defined_state_preservation_status": gate11q_surface_defined_state_preservation_status,
        "gate11r_not_yet_present_state_preservation_status": gate11r_not_yet_present_state_preservation_status,
        "gate11s_path_defined_state_preservation_status": gate11s_path_defined_state_preservation_status,
        "gate11t_not_yet_present_state_preservation_status": gate11t_not_yet_present_state_preservation_status,
        "gate11u_residual_named_state_preservation_status": gate11u_residual_named_state_preservation_status,
        "gate11v_path_defined_state_preservation_status": gate11v_path_defined_state_preservation_status,
        "gate11w_not_yet_completed_state_preservation_status": gate11w_not_yet_completed_state_preservation_status,
        "gate11x_blocker_named_state_preservation_status": gate11x_blocker_named_state_preservation_status,
        "gate11y_path_defined_state_preservation_status": gate11y_path_defined_state_preservation_status,
        "gate11z_not_yet_resolved_state_preservation_status": gate11z_not_yet_resolved_state_preservation_status,
        "gate11aa_surface_defined_state_preservation_status": gate11aa_surface_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_blocker_resolution_marker_status": explicit_blocker_resolution_marker_status,
        "blocker_resolution_marker_singularity_status": blocker_resolution_marker_singularity_status,
        "same_source_blocker_resolution_marker_binding_status": same_source_blocker_resolution_marker_binding_status,
        "blocker_resolution_marker_boundary_status": blocker_resolution_marker_boundary_status,
        "one_explicit_blocker_resolution_marker_status": one_explicit_blocker_resolution_marker_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11ACExplicitBlockerResolutionMarkerInstantiationPathAuditTest(unittest.TestCase):
    def test_default_frozen_source_is_path_defined(self) -> None:
        status = gate11ac.build_status_payload(
            make_gate11ab_manifest(),
            make_gate11ab_status(),
            "Gate11AB preserves the marker-not-yet-present line with blocker no_explicit_blocker_resolution_marker.",
        )
        self.assertEqual(status["gate11ab_not_yet_present_state_preservation_status"], "preserved")
        self.assertEqual(status["missing_blocker_resolution_marker_instantiation_components_status"], "named")
        self.assertEqual(status["minimum_same_source_blocker_resolution_marker_instantiation_rule_status"], "defined")
        self.assertEqual(status["bounded_read_prefix_instantiation_requirement_status"], "defined")
        self.assertEqual(status["explicit_blocker_resolution_marker_instantiation_path_status"], "path_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_defined_when_rule_is_not_fixed(self) -> None:
        status = gate11ac.build_status_payload(
            make_gate11ab_manifest(),
            make_gate11ab_status(next_named_blocker="marker_instantiation_rule_not_fixed"),
            "Gate11AB preserves the line but does not fix the blocker-resolution marker path narrowly enough.",
        )
        self.assertEqual(status["missing_blocker_resolution_marker_instantiation_components_status"], "not_named")
        self.assertEqual(status["minimum_same_source_blocker_resolution_marker_instantiation_rule_status"], "not_defined")
        self.assertEqual(status["bounded_read_prefix_instantiation_requirement_status"], "not_defined")
        self.assertEqual(status["explicit_blocker_resolution_marker_instantiation_path_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "missing_blocker_resolution_marker_instantiation_components_not_named")

    def test_deferred_when_multiple_future_markers_compete(self) -> None:
        status = gate11ac.build_status_payload(
            make_gate11ab_manifest(),
            make_gate11ab_status(one_explicit_blocker_resolution_marker_status="deferred", next_named_blocker="multiple_candidate_markers"),
            "Gate11AB records competing future blocker-resolution markers.",
        )
        self.assertEqual(status["gate11ab_not_yet_present_state_preservation_status"], "deferred")
        self.assertEqual(status["missing_blocker_resolution_marker_instantiation_components_status"], "deferred")
        self.assertEqual(status["minimum_same_source_blocker_resolution_marker_instantiation_rule_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_instantiation_requirement_status"], "deferred")
        self.assertEqual(status["explicit_blocker_resolution_marker_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_future_blocker_markers")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11ac.build_status_payload(
            make_gate11ab_manifest(),
            make_gate11ab_status(blocker_resolution_marker_boundary_status="not_confirmed"),
            "Gate11AB breaks the blocker-resolution marker boundary.",
        )
        self.assertEqual(status["blocker_resolution_marker_boundary_status"], "denied")
        self.assertEqual(status["explicit_blocker_resolution_marker_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_marker_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11ac.build_status_payload(make_gate11ab_manifest(run_id=""), make_gate11ab_status(), "")
        self.assertEqual(status["missing_blocker_resolution_marker_instantiation_components_status"], "deferred")
        self.assertEqual(status["minimum_same_source_blocker_resolution_marker_instantiation_rule_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_instantiation_requirement_status"], "deferred")
        self.assertEqual(status["explicit_blocker_resolution_marker_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11ac.build_status_payload(
            make_gate11ab_manifest(),
            make_gate11ab_status(),
            "Gate11AB preserves the marker-not-yet-present line with blocker no_explicit_blocker_resolution_marker.",
        )
        registry = gate11ac.build_registry(make_gate11ab_manifest(), status)
        compare = gate11ac.build_policy_compare(registry)
        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11ab_run_id"], "gate11ab_run")
        self.assertEqual(compare[0]["explicit_blocker_resolution_marker_instantiation_path_status"], "path_defined")

    def test_report_for_path_defined_does_not_fall_back_to_not_yet_defined_sentence(self) -> None:
        status = gate11ac.build_status_payload(
            make_gate11ab_manifest(),
            make_gate11ab_status(),
            "Gate11AB preserves the marker-not-yet-present line with blocker no_explicit_blocker_resolution_marker.",
        )
        registry = gate11ac.build_registry(make_gate11ab_manifest(), status)
        compare = gate11ac.build_policy_compare(registry)
        report = gate11ac.build_report("gate11ac_run", make_gate11ab_manifest(), compare, status)
        self.assertIn("minimum same-source blocker-resolution marker-instantiation path is fixed narrowly enough", report)
        self.assertNotIn("is not yet fixed narrowly enough", report)


if __name__ == "__main__":
    unittest.main()
