#!/usr/bin/env python3
"""Regression tests for Gate11R one explicit residual completion-marker helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11r_one_explicit_residual_completion_marker_audit as gate11r


def make_gate11q_manifest(run_id: str = "gate11q_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11q_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    bounded_marker_surface_rows_status: str = "defined",
    same_source_binding_requirement_status: str = "defined",
    bounded_read_prefix_requirement_status: str = "defined",
    residual_completion_boundary_status: str = "confirmed",
    named_residual_completion_marker_surface_status: str = "surface_defined",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "bounded_marker_surface_rows_status": bounded_marker_surface_rows_status,
        "same_source_binding_requirement_status": same_source_binding_requirement_status,
        "bounded_read_prefix_requirement_status": bounded_read_prefix_requirement_status,
        "residual_completion_boundary_status": residual_completion_boundary_status,
        "named_residual_completion_marker_surface_status": named_residual_completion_marker_surface_status,
        "next_named_blocker": next_named_blocker,
    }


def present_marker_report(later_source_id: str = "runs/example_source") -> str:
    surfaces = "\n".join(
        f"residual_completion_surface: {phrase}" for phrase in gate11r.REQUIRED_COMPLETION_SURFACES
    )
    return (
        "residual_completion_marker_status: present\n"
        f"residual_completion_later_source_id: {later_source_id}\n"
        "residual_completion_same_source_status: completed\n"
        f"{surfaces}\n"
    )


class RunGate11ROneExplicitResidualCompletionMarkerAuditTest(unittest.TestCase):
    def test_default_frozen_source_is_not_yet_present(self) -> None:
        status = gate11r.build_status_payload(
            make_gate11q_manifest(),
            make_gate11q_status(),
            "Gate11Q preserves the surface-defined line without instantiating a marker.",
        )

        self.assertEqual(status["gate11q_surface_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_residual_completion_marker_status"], "absent")
        self.assertEqual(status["residual_completion_marker_singularity_status"], "none")
        self.assertEqual(status["same_source_residual_completion_marker_binding_status"], "not_explicit")
        self.assertEqual(status["bounded_read_prefix_attachment_status"], "not_attached")
        self.assertEqual(status["one_explicit_residual_completion_marker_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_residual_completion_marker")

    def test_present_when_one_bounded_marker_is_explicitly_instantiated(self) -> None:
        status = gate11r.build_status_payload(
            make_gate11q_manifest(),
            make_gate11q_status(),
            present_marker_report(),
        )

        self.assertEqual(status["explicit_residual_completion_marker_status"], "present")
        self.assertEqual(status["residual_completion_marker_singularity_status"], "single")
        self.assertEqual(status["same_source_residual_completion_marker_binding_status"], "explicit")
        self.assertEqual(status["bounded_read_prefix_attachment_status"], "attached")
        self.assertEqual(status["residual_completion_marker_boundary_status"], "confirmed")
        self.assertEqual(status["one_explicit_residual_completion_marker_status"], "present")
        self.assertEqual(status["next_named_blocker"], "")

    def test_path_definition_prose_does_not_count_as_marker(self) -> None:
        status = gate11r.build_status_payload(
            make_gate11q_manifest(),
            make_gate11q_status(),
            "A future path could mention a residual completion marker and a later source in prose, but no bounded marker rows are instantiated.",
        )

        self.assertEqual(status["explicit_residual_completion_marker_status"], "absent")
        self.assertEqual(status["one_explicit_residual_completion_marker_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_residual_completion_marker")

    def test_deferred_when_multiple_candidate_markers_exist(self) -> None:
        report = present_marker_report("runs/source_a") + present_marker_report("runs/source_b")
        status = gate11r.build_status_payload(make_gate11q_manifest(), make_gate11q_status(), report)

        self.assertEqual(status["explicit_residual_completion_marker_status"], "present")
        self.assertEqual(status["residual_completion_marker_singularity_status"], "multiple")
        self.assertEqual(status["same_source_residual_completion_marker_binding_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_attachment_status"], "deferred")
        self.assertEqual(status["one_explicit_residual_completion_marker_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_markers")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11r.build_status_payload(
            make_gate11q_manifest(),
            make_gate11q_status(residual_completion_boundary_status="not_confirmed"),
            present_marker_report(),
        )

        self.assertEqual(status["residual_completion_marker_boundary_status"], "denied")
        self.assertEqual(status["one_explicit_residual_completion_marker_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "residual_completion_marker_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11r.build_status_payload(make_gate11q_manifest(run_id=""), make_gate11q_status(), "")

        self.assertEqual(status["explicit_residual_completion_marker_status"], "deferred")
        self.assertEqual(status["residual_completion_marker_singularity_status"], "deferred")
        self.assertEqual(status["same_source_residual_completion_marker_binding_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_attachment_status"], "deferred")
        self.assertEqual(status["one_explicit_residual_completion_marker_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11r.build_status_payload(
            make_gate11q_manifest(),
            make_gate11q_status(),
            present_marker_report(),
        )
        registry = gate11r.build_registry(make_gate11q_manifest(), status)
        compare = gate11r.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11q_run_id"], "gate11q_run")
        self.assertEqual(compare[0]["one_explicit_residual_completion_marker_status"], "present")


if __name__ == "__main__":
    unittest.main()