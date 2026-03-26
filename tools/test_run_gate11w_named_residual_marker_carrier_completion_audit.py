#!/usr/bin/env python3
"""Regression tests for Gate11W named residual marker-carrier completion helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11w_named_residual_marker_carrier_completion_audit as gate11w


def make_gate11v_manifest(run_id: str = "gate11v_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11v_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_residual_marker_carrier_condition_preservation_status: str = "preserved",
    minimum_same_source_carrier_completion_rule_status: str = "defined",
    bounded_read_prefix_completion_requirement_status: str = "defined",
    carrier_completion_boundary_status: str = "confirmed",
    explicit_residual_completion_marker_carrier_completion_instantiation_path_status: str = "path_defined",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_residual_marker_carrier_condition_preservation_status": named_residual_marker_carrier_condition_preservation_status,
        "minimum_same_source_carrier_completion_rule_status": minimum_same_source_carrier_completion_rule_status,
        "bounded_read_prefix_completion_requirement_status": bounded_read_prefix_completion_requirement_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "explicit_residual_completion_marker_carrier_completion_instantiation_path_status": explicit_residual_completion_marker_carrier_completion_instantiation_path_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11WNamedResidualMarkerCarrierCompletionAuditTest(unittest.TestCase):
    def test_not_yet_completed_for_current_frozen_gate11v_source(self) -> None:
        status = gate11w.build_status_payload(
            make_gate11v_manifest(),
            make_gate11v_status(),
            "Gate11V preserves the fixed path but does not explicitly complete the named residual marker-carrier condition.",
        )

        self.assertEqual(status["gate11v_path_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_carrier_completion_marker_status"], "absent")
        self.assertEqual(status["same_source_carrier_completion_status"], "not_completed")
        self.assertEqual(status["named_residual_marker_carrier_completion_status"], "not_yet_completed")
        self.assertEqual(status["next_named_blocker"], "no_explicit_residual_completion_marker")

    def test_completed_when_marker_and_same_source_completion_are_explicit(self) -> None:
        report_text = """
    residual_completion_marker_status: present
    residual_completion_later_source_id: runs/gate11q_future
    residual_completion_same_source_status: completed
    residual_completion_surface: one explicit residual completion marker
    residual_completion_surface: one explicit later_source_id or later_frozen_run_id
    residual_completion_surface: one marker and only one marker
    residual_completion_surface: one explicit same-source path-attachment status
    residual_completion_surface: one bounded read-prefix declaration for the marker
    residual_completion_surface: one explicit admissible later-source presence marker
    residual_completion_surface: one declaration marker
    residual_completion_surface: one candidate id
    residual_completion_surface: one class
    residual_completion_surface: one explicit host-failure sentence
    residual_completion_surface: matched status, registry, and read surfaces
"""
        status = gate11w.build_status_payload(make_gate11v_manifest(), make_gate11v_status(), report_text)

        self.assertEqual(status["explicit_carrier_completion_marker_status"], "present")
        self.assertEqual(status["same_source_carrier_completion_status"], "completed")
        self.assertEqual(status["named_residual_marker_carrier_completion_status"], "completed")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_completed_when_completion_phrases_appear_only_in_path_definition_prose(self) -> None:
        report_text = """
    The minimum path requires one explicit residual completion marker and says the named residual marker-carrier condition is completed only after later evidence lands.
    The path also requires one explicit later_source_id or later_frozen_run_id, one marker and only one marker,
    one explicit same-source path-attachment status, one bounded read-prefix declaration for the marker,
    one explicit admissible later-source presence marker, one declaration marker, one candidate id, one class,
    one explicit host-failure sentence, and matched status, registry, and read surfaces.
    """
        status = gate11w.build_status_payload(make_gate11v_manifest(), make_gate11v_status(), report_text)

        self.assertEqual(status["explicit_carrier_completion_marker_status"], "absent")
        self.assertEqual(status["same_source_carrier_completion_status"], "not_completed")
        self.assertEqual(status["named_residual_marker_carrier_completion_status"], "not_yet_completed")
        self.assertEqual(status["next_named_blocker"], "no_explicit_residual_completion_marker")

    def test_deferred_when_multiple_candidate_carriers_compete(self) -> None:
        report_text = """
    residual_completion_marker_status: present
    residual_completion_later_source_id: runs/gate11q_future_a
    residual_completion_later_frozen_run_id: runs/gate11q_future_b
"""
        status = gate11w.build_status_payload(make_gate11v_manifest(), make_gate11v_status(), report_text)

        self.assertEqual(status["same_source_carrier_completion_status"], "deferred")
        self.assertEqual(status["named_residual_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_carriers")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11w.build_status_payload(
            make_gate11v_manifest(),
            make_gate11v_status(carrier_completion_boundary_status="not_confirmed"),
            "Gate11V breaks the carrier-completion boundary.",
        )

        self.assertEqual(status["carrier_completion_boundary_status"], "denied")
        self.assertEqual(status["named_residual_marker_carrier_completion_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "carrier_completion_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11w.build_status_payload(make_gate11v_manifest(run_id=""), make_gate11v_status(), "")

        self.assertEqual(status["explicit_carrier_completion_marker_status"], "deferred")
        self.assertEqual(status["same_source_carrier_completion_status"], "deferred")
        self.assertEqual(status["named_residual_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11w.build_status_payload(
            make_gate11v_manifest(),
            make_gate11v_status(),
            "Gate11V preserves the fixed path but does not explicitly complete the named residual marker-carrier condition.",
        )
        registry = gate11w.build_registry(make_gate11v_manifest(), status)
        compare = gate11w.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11v_run_id"], "gate11v_run")
        self.assertEqual(compare[0]["named_residual_marker_carrier_completion_status"], "not_yet_completed")


if __name__ == "__main__":
    unittest.main()