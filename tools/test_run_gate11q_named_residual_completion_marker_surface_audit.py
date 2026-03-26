#!/usr/bin/env python3
"""Regression tests for Gate11Q named residual completion-marker surface helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11q_named_residual_completion_marker_surface_audit as gate11q


def make_gate11p_manifest(run_id: str = "gate11p_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11p_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_residual_carrier_condition_preservation_status: str = "preserved",
    explicit_residual_completion_marker_status: str = "absent",
    same_source_residual_completion_status: str = "not_completed",
    residual_completion_boundary_status: str = "confirmed",
    named_residual_carrier_completion_status: str = "not_yet_completed",
    next_named_blocker: str = "no_explicit_residual_completion_marker",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_residual_carrier_condition_preservation_status": named_residual_carrier_condition_preservation_status,
        "explicit_residual_completion_marker_status": explicit_residual_completion_marker_status,
        "same_source_residual_completion_status": same_source_residual_completion_status,
        "residual_completion_boundary_status": residual_completion_boundary_status,
        "named_residual_carrier_completion_status": named_residual_carrier_completion_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11QNamedResidualCompletionMarkerSurfaceAuditTest(unittest.TestCase):
    def test_surface_defined_for_current_frozen_gate11p_source(self) -> None:
        status = gate11q.build_status_payload(
            make_gate11p_manifest(),
            make_gate11p_status(),
            "Gate11P preserves the fixed non-completion line with a bounded blocker.",
        )

        self.assertEqual(status["gate11p_not_yet_completed_state_preservation_status"], "preserved")
        self.assertEqual(status["bounded_marker_surface_rows_status"], "defined")
        self.assertEqual(status["same_source_binding_requirement_status"], "defined")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "defined")
        self.assertEqual(status["named_residual_completion_marker_surface_status"], "surface_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_defined_when_marker_surface_contract_is_not_fixed(self) -> None:
        status = gate11q.build_status_payload(
            make_gate11p_manifest(),
            make_gate11p_status(next_named_blocker="marker_surface_contract_not_fixed"),
            "Gate11P preserves the line but does not fix a bounded marker surface narrowly enough.",
        )

        self.assertEqual(status["bounded_marker_surface_rows_status"], "not_defined")
        self.assertEqual(status["same_source_binding_requirement_status"], "not_defined")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "not_defined")
        self.assertEqual(status["named_residual_completion_marker_surface_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "bounded_marker_surface_rows_not_fixed")

    def test_deferred_when_multiple_candidate_marker_surfaces_compete(self) -> None:
        status = gate11q.build_status_payload(
            make_gate11p_manifest(),
            make_gate11p_status(
                explicit_residual_completion_marker_status="deferred",
                same_source_residual_completion_status="deferred",
                named_residual_carrier_completion_status="deferred",
                next_named_blocker="multiple_later_sources",
            ),
            "Gate11P records competing marker candidates.",
        )

        self.assertEqual(status["bounded_marker_surface_rows_status"], "deferred")
        self.assertEqual(status["same_source_binding_requirement_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "deferred")
        self.assertEqual(status["named_residual_completion_marker_surface_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_marker_surfaces")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11q.build_status_payload(
            make_gate11p_manifest(),
            make_gate11p_status(residual_completion_boundary_status="not_confirmed"),
            "Gate11P breaks the residual completion boundary.",
        )

        self.assertEqual(status["residual_completion_boundary_status"], "denied")
        self.assertEqual(status["named_residual_completion_marker_surface_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "residual_completion_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11q.build_status_payload(make_gate11p_manifest(run_id=""), make_gate11p_status(), "")

        self.assertEqual(status["bounded_marker_surface_rows_status"], "deferred")
        self.assertEqual(status["same_source_binding_requirement_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "deferred")
        self.assertEqual(status["named_residual_completion_marker_surface_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11q.build_status_payload(
            make_gate11p_manifest(),
            make_gate11p_status(),
            "Gate11P preserves the fixed non-completion line with a bounded blocker.",
        )
        registry = gate11q.build_registry(make_gate11p_manifest(), status)
        compare = gate11q.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11p_run_id"], "gate11p_run")
        self.assertEqual(compare[0]["named_residual_completion_marker_surface_status"], "surface_defined")


if __name__ == "__main__":
    unittest.main()