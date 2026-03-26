#!/usr/bin/env python3
"""Regression tests for Gate11AA named blocker-resolution marker surface helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11aa_named_blocker_resolution_marker_surface_audit as gate11aa


def make_gate11z_manifest(run_id: str = "gate11z_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11z_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_blocker_preservation_status: str = "preserved",
    explicit_blocker_resolution_marker_status: str = "absent",
    same_source_blocker_resolution_status: str = "not_resolved",
    blocker_resolution_boundary_status: str = "confirmed",
    named_residual_marker_carrier_completion_blocker_resolution_status: str = "not_yet_resolved",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_blocker_preservation_status": named_blocker_preservation_status,
        "explicit_blocker_resolution_marker_status": explicit_blocker_resolution_marker_status,
        "same_source_blocker_resolution_status": same_source_blocker_resolution_status,
        "blocker_resolution_boundary_status": blocker_resolution_boundary_status,
        "named_residual_marker_carrier_completion_blocker_resolution_status": named_residual_marker_carrier_completion_blocker_resolution_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11AANamedBlockerResolutionMarkerSurfaceAuditTest(unittest.TestCase):
    def test_surface_defined_for_current_frozen_gate11z_source(self) -> None:
        status = gate11aa.build_status_payload(
            make_gate11z_manifest(),
            make_gate11z_status(),
            "Gate11Z preserves the fixed not-yet-resolved line and the blocker-resolution boundary.",
        )

        self.assertEqual(status["gate11z_not_yet_resolved_state_preservation_status"], "preserved")
        self.assertEqual(status["bounded_blocker_resolution_marker_rows_status"], "defined")
        self.assertEqual(status["same_source_binding_requirement_status"], "defined")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "defined")
        self.assertEqual(status["named_blocker_resolution_marker_surface_status"], "surface_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_defined_when_blocker_is_not_surface_fixable_here(self) -> None:
        status = gate11aa.build_status_payload(
            make_gate11z_manifest(),
            make_gate11z_status(next_named_blocker="named_blocker_not_preserved"),
            "Gate11Z does not fix a valid blocker-resolution marker surface under this blocker.",
        )

        self.assertEqual(status["bounded_blocker_resolution_marker_rows_status"], "not_defined")
        self.assertEqual(status["same_source_binding_requirement_status"], "not_defined")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "not_defined")
        self.assertEqual(status["named_blocker_resolution_marker_surface_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "bounded_blocker_resolution_marker_rows_not_fixed")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11aa.build_status_payload(
            make_gate11z_manifest(),
            make_gate11z_status(blocker_resolution_boundary_status="not_confirmed"),
            "Gate11Z breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_boundary_status"], "denied")
        self.assertEqual(status["named_blocker_resolution_marker_surface_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11aa.build_status_payload(make_gate11z_manifest(run_id=""), make_gate11z_status(), "")

        self.assertEqual(status["bounded_blocker_resolution_marker_rows_status"], "deferred")
        self.assertEqual(status["same_source_binding_requirement_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_requirement_status"], "deferred")
        self.assertEqual(status["named_blocker_resolution_marker_surface_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_denied_when_gate11z_not_yet_resolved_state_is_not_preserved(self) -> None:
        status = gate11aa.build_status_payload(
            make_gate11z_manifest(),
            make_gate11z_status(named_residual_marker_carrier_completion_blocker_resolution_status="resolved"),
            "Gate11Z no longer preserves the fixed not-yet-resolved line.",
        )

        self.assertEqual(status["gate11z_not_yet_resolved_state_preservation_status"], "not_preserved")
        self.assertEqual(status["named_blocker_resolution_marker_surface_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11z_not_yet_resolved_state_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11aa.build_status_payload(
            make_gate11z_manifest(),
            make_gate11z_status(),
            "Gate11Z preserves the fixed not-yet-resolved line and the blocker-resolution boundary.",
        )
        registry = gate11aa.build_registry(make_gate11z_manifest(), status)
        compare = gate11aa.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11z_run_id"], "gate11z_run")
        self.assertEqual(compare[0]["named_blocker_resolution_marker_surface_status"], "surface_defined")


if __name__ == "__main__":
    unittest.main()