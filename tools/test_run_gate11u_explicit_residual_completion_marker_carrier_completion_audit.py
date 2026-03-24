#!/usr/bin/env python3
"""Regression tests for Gate11U explicit residual completion-marker carrier-completion helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11u_explicit_residual_completion_marker_carrier_completion_audit as gate11u


def make_gate11t_manifest(run_id: str = "gate11t_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11t_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_residual_completion_marker_status: str = "absent",
    residual_completion_marker_singularity_status: str = "none",
    same_source_marker_path_attachment_status: str = "not_instantiated",
    residual_completion_marker_boundary_status: str = "confirmed",
    one_explicit_residual_completion_marker_path_instantiation_status: str = "not_yet_present",
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
        "gate11p_not_yet_completed_state_preservation_status": gate11p_not_yet_completed_state_preservation_status,
        "gate11q_surface_defined_state_preservation_status": gate11q_surface_defined_state_preservation_status,
        "gate11r_not_yet_present_state_preservation_status": gate11r_not_yet_present_state_preservation_status,
        "gate11s_path_defined_state_preservation_status": gate11s_path_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_residual_completion_marker_status": explicit_residual_completion_marker_status,
        "residual_completion_marker_singularity_status": residual_completion_marker_singularity_status,
        "same_source_marker_path_attachment_status": same_source_marker_path_attachment_status,
        "residual_completion_marker_boundary_status": residual_completion_marker_boundary_status,
        "one_explicit_residual_completion_marker_path_instantiation_status": one_explicit_residual_completion_marker_path_instantiation_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11UExplicitResidualCompletionMarkerCarrierCompletionAuditTest(unittest.TestCase):
    def test_default_frozen_source_is_residual_named(self) -> None:
        status = gate11u.build_status_payload(
            make_gate11t_manifest(),
            make_gate11t_status(),
            "Gate11T preserves the marker-not-yet-present line with blocker no_explicit_residual_completion_marker.",
        )

        self.assertEqual(status["gate11t_not_yet_present_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_marker_carrier_completion_status"], "missing")
        self.assertEqual(status["marker_singularity_carrier_completion_status"], "missing")
        self.assertEqual(status["same_source_path_attachment_carrier_completion_status"], "missing")
        self.assertEqual(status["explicit_residual_completion_marker_carrier_completion_status"], "residual_named")
        self.assertEqual(status["next_named_blocker"], "no_explicit_residual_completion_marker")

    def test_not_yet_named_when_residual_blocker_is_not_explicitly_named(self) -> None:
        status = gate11u.build_status_payload(
            make_gate11t_manifest(),
            make_gate11t_status(next_named_blocker="marker_carrier_condition_not_fixed"),
            "Gate11T preserves the line but does not explicitly name the remaining carrier condition.",
        )

        self.assertEqual(status["explicit_residual_completion_marker_carrier_completion_status"], "not_yet_named")
        self.assertEqual(status["next_named_blocker"], "no_residual_carrier_condition_explicitly_named")

    def test_deferred_when_multiple_candidate_markers_compete(self) -> None:
        status = gate11u.build_status_payload(
            make_gate11t_manifest(),
            make_gate11t_status(one_explicit_residual_completion_marker_path_instantiation_status="deferred", next_named_blocker="multiple_candidate_markers"),
            "Gate11T records competing candidate markers.",
        )

        self.assertEqual(status["gate11t_not_yet_present_state_preservation_status"], "deferred")
        self.assertEqual(status["explicit_residual_completion_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_markers")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11u.build_status_payload(
            make_gate11t_manifest(),
            make_gate11t_status(residual_completion_marker_boundary_status="not_confirmed"),
            "Gate11T breaks the marker boundary.",
        )

        self.assertEqual(status["carrier_completion_boundary_status"], "denied")
        self.assertEqual(status["explicit_residual_completion_marker_carrier_completion_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "carrier_completion_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11u.build_status_payload(make_gate11t_manifest(run_id=""), make_gate11t_status(), "")

        self.assertEqual(status["explicit_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["marker_singularity_carrier_completion_status"], "deferred")
        self.assertEqual(status["same_source_path_attachment_carrier_completion_status"], "deferred")
        self.assertEqual(status["explicit_residual_completion_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11u.build_status_payload(
            make_gate11t_manifest(),
            make_gate11t_status(),
            "Gate11T preserves the marker-not-yet-present line with blocker no_explicit_residual_completion_marker.",
        )
        registry = gate11u.build_registry(make_gate11t_manifest(), status)
        compare = gate11u.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11t_run_id"], "gate11t_run")
        self.assertEqual(compare[0]["explicit_residual_completion_marker_carrier_completion_status"], "residual_named")


if __name__ == "__main__":
    unittest.main()