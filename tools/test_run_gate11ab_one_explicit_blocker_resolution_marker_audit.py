#!/usr/bin/env python3
"""Regression tests for Gate11AB one explicit blocker-resolution marker helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11ab_one_explicit_blocker_resolution_marker_audit as gate11ab


def make_gate11aa_manifest(run_id: str = "gate11aa_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11aa_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    bounded_blocker_resolution_marker_rows_status: str = "defined",
    same_source_binding_requirement_status: str = "defined",
    bounded_read_prefix_requirement_status: str = "defined",
    blocker_resolution_boundary_status: str = "confirmed",
    named_blocker_resolution_marker_surface_status: str = "surface_defined",
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
        "gate11y_path_defined_state_preservation_status": gate11y_path_defined_state_preservation_status,
        "gate11z_not_yet_resolved_state_preservation_status": gate11z_not_yet_resolved_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "bounded_blocker_resolution_marker_rows_status": bounded_blocker_resolution_marker_rows_status,
        "same_source_binding_requirement_status": same_source_binding_requirement_status,
        "bounded_read_prefix_requirement_status": bounded_read_prefix_requirement_status,
        "blocker_resolution_boundary_status": blocker_resolution_boundary_status,
        "named_blocker_resolution_marker_surface_status": named_blocker_resolution_marker_surface_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11ABOneExplicitBlockerResolutionMarkerAuditTest(unittest.TestCase):
    def test_not_yet_present_for_current_frozen_gate11aa_source(self) -> None:
        status = gate11ab.build_status_payload(
            make_gate11aa_manifest(),
            make_gate11aa_status(),
            "Gate11AA preserves the fixed marker surface but does not instantiate one explicit blocker-resolution marker.",
        )

        self.assertEqual(status["gate11aa_surface_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "none")
        self.assertEqual(status["same_source_blocker_resolution_marker_binding_status"], "not_explicit")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_present_when_one_explicit_marker_is_instantiated(self) -> None:
        report_text = """
    residual_completion_blocker_resolution_marker_status: present
    residual_completion_later_source_id: runs/gate11q_future
    residual_completion_same_source_status: completed
"""
        status = gate11ab.build_status_payload(make_gate11aa_manifest(), make_gate11aa_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "single")
        self.assertEqual(status["same_source_blocker_resolution_marker_binding_status"], "explicit")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_present_when_marker_appears_only_in_prose(self) -> None:
        report_text = """
    A later explicit blocker-resolution marker would need to be present and same-source bound before the audit could count it.
    This sentence is only path prose and not an actual marker row.
"""
        status = gate11ab.build_status_payload(make_gate11aa_manifest(), make_gate11aa_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_substatuses_do_not_advance_without_marker_presence(self) -> None:
        report_text = """
    residual_completion_later_source_id: runs/gate11q_future
    residual_completion_same_source_status: completed
"""
        status = gate11ab.build_status_payload(make_gate11aa_manifest(), make_gate11aa_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "none")
        self.assertEqual(status["same_source_blocker_resolution_marker_binding_status"], "not_explicit")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_deferred_when_multiple_candidate_markers_compete(self) -> None:
        report_text = """
    residual_completion_blocker_resolution_marker_status: present
    residual_completion_later_source_id: runs/gate11q_future_a
    residual_completion_later_frozen_run_id: runs/gate11q_future_b
"""
        status = gate11ab.build_status_payload(make_gate11aa_manifest(), make_gate11aa_status(), report_text)

        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "multiple")
        self.assertEqual(status["same_source_blocker_resolution_marker_binding_status"], "deferred")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_markers")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11ab.build_status_payload(
            make_gate11aa_manifest(),
            make_gate11aa_status(blocker_resolution_boundary_status="not_confirmed"),
            "Gate11AA breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_marker_boundary_status"], "denied")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_marker_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11ab.build_status_payload(make_gate11aa_manifest(run_id=""), make_gate11aa_status(), "")

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "deferred")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "deferred")
        self.assertEqual(status["same_source_blocker_resolution_marker_binding_status"], "deferred")
        self.assertEqual(status["one_explicit_blocker_resolution_marker_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11ab.build_status_payload(
            make_gate11aa_manifest(),
            make_gate11aa_status(),
            "Gate11AA preserves the fixed marker surface but does not instantiate one explicit blocker-resolution marker.",
        )
        registry = gate11ab.build_registry(make_gate11aa_manifest(), status)
        compare = gate11ab.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11aa_run_id"], "gate11aa_run")
        self.assertEqual(compare[0]["one_explicit_blocker_resolution_marker_status"], "not_yet_present")


if __name__ == "__main__":
    unittest.main()
