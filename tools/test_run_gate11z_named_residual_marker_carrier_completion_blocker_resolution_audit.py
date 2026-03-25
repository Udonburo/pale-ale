#!/usr/bin/env python3
"""Regression tests for Gate11Z named residual marker-carrier completion blocker-resolution helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11z_named_residual_marker_carrier_completion_blocker_resolution_audit as gate11z


def make_gate11y_manifest(run_id: str = "gate11y_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11y_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_blocker_preservation_status: str = "preserved",
    minimum_blocker_resolution_rule_status: str = "defined",
    bounded_read_prefix_resolution_requirement_status: str = "defined",
    blocker_resolution_boundary_status: str = "confirmed",
    residual_marker_carrier_completion_blocker_resolution_path_status: str = "path_defined",
    next_named_blocker: str = "",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_blocker_preservation_status": named_blocker_preservation_status,
        "minimum_blocker_resolution_rule_status": minimum_blocker_resolution_rule_status,
        "bounded_read_prefix_resolution_requirement_status": bounded_read_prefix_resolution_requirement_status,
        "blocker_resolution_boundary_status": blocker_resolution_boundary_status,
        "residual_marker_carrier_completion_blocker_resolution_path_status": residual_marker_carrier_completion_blocker_resolution_path_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11ZNamedResidualMarkerCarrierCompletionBlockerResolutionAuditTest(unittest.TestCase):
    def test_not_yet_resolved_for_current_frozen_gate11y_source(self) -> None:
        status = gate11z.build_status_payload(
            make_gate11y_manifest(),
            make_gate11y_status(),
            "Gate11Y preserves the fixed blocker-resolution path but does not explicitly resolve the named blocker.",
        )

        self.assertEqual(status["gate11y_path_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["named_blocker_preservation_status"], "preserved")
        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["same_source_blocker_resolution_status"], "not_resolved")
        self.assertEqual(status["named_residual_marker_carrier_completion_blocker_resolution_status"], "not_yet_resolved")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_resolved_when_same_source_resolution_is_explicit(self) -> None:
        report_text = """
    residual_completion_blocker_resolution_marker_status: present
    residual_completion_later_source_id: runs/gate11q_future
    residual_completion_same_source_status: completed
    residual_completion_surface: one explicit residual completion marker
    residual_completion_surface: one explicit later_source_id or later_frozen_run_id
    residual_completion_surface: one marker and only one marker
    residual_completion_surface: one explicit same-source carrier-completion status marked completed
    residual_completion_surface: one bounded read-prefix declaration for the marker
    residual_completion_surface: one explicit admissible later-source presence marker
    residual_completion_surface: one declaration marker
    residual_completion_surface: one candidate id
    residual_completion_surface: one class
    residual_completion_surface: one explicit host-failure sentence
    residual_completion_surface: matched status, registry, and read surfaces
"""
        status = gate11z.build_status_payload(make_gate11y_manifest(), make_gate11y_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["same_source_blocker_resolution_status"], "resolved")
        self.assertEqual(status["named_residual_marker_carrier_completion_blocker_resolution_status"], "resolved")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_resolved_when_resolution_phrases_appear_only_in_path_prose(self) -> None:
        report_text = """
    The fixed path requires one explicit residual completion marker, one explicit later_source_id or later_frozen_run_id,
    one marker and only one marker, one explicit same-source carrier-completion status marked completed,
    one bounded read-prefix declaration for the marker, one explicit admissible later-source presence marker,
    one declaration marker, one candidate id, one class, one explicit host-failure sentence,
    and matched status, registry, and read surfaces before the blocker could count as resolved.
    """
        status = gate11z.build_status_payload(make_gate11y_manifest(), make_gate11y_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["same_source_blocker_resolution_status"], "not_resolved")
        self.assertEqual(status["named_residual_marker_carrier_completion_blocker_resolution_status"], "not_yet_resolved")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_deferred_when_multiple_candidate_resolutions_compete(self) -> None:
        report_text = """
    residual_completion_blocker_resolution_marker_status: present
    residual_completion_later_source_id: runs/gate11q_future_a
    residual_completion_later_frozen_run_id: runs/gate11q_future_b
"""
        status = gate11z.build_status_payload(make_gate11y_manifest(), make_gate11y_status(), report_text)

        self.assertEqual(status["same_source_blocker_resolution_status"], "deferred")
        self.assertEqual(status["named_residual_marker_carrier_completion_blocker_resolution_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_resolutions")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11z.build_status_payload(
            make_gate11y_manifest(),
            make_gate11y_status(blocker_resolution_boundary_status="not_confirmed"),
            "Gate11Y breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_boundary_status"], "denied")
        self.assertEqual(status["named_residual_marker_carrier_completion_blocker_resolution_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11z.build_status_payload(make_gate11y_manifest(run_id=""), make_gate11y_status(), "")

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "deferred")
        self.assertEqual(status["same_source_blocker_resolution_status"], "deferred")
        self.assertEqual(status["named_residual_marker_carrier_completion_blocker_resolution_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11z.build_status_payload(
            make_gate11y_manifest(),
            make_gate11y_status(),
            "Gate11Y preserves the fixed blocker-resolution path but does not explicitly resolve the named blocker.",
        )
        registry = gate11z.build_registry(make_gate11y_manifest(), status)
        compare = gate11z.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11y_run_id"], "gate11y_run")
        self.assertEqual(compare[0]["named_residual_marker_carrier_completion_blocker_resolution_status"], "not_yet_resolved")


if __name__ == "__main__":
    unittest.main()