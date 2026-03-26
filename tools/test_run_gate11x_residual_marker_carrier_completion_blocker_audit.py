#!/usr/bin/env python3
"""Regression tests for Gate11X residual marker-carrier completion blocker helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11x_residual_marker_carrier_completion_blocker_audit as gate11x


def make_gate11w_manifest(run_id: str = "gate11w_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11w_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_residual_marker_carrier_condition_preservation_status: str = "preserved",
    explicit_carrier_completion_marker_status: str = "absent",
    same_source_carrier_completion_status: str = "not_completed",
    carrier_completion_boundary_status: str = "confirmed",
    named_residual_marker_carrier_completion_status: str = "not_yet_completed",
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
        "gate11t_not_yet_present_state_preservation_status": gate11t_not_yet_present_state_preservation_status,
        "gate11u_residual_named_state_preservation_status": gate11u_residual_named_state_preservation_status,
        "gate11v_path_defined_state_preservation_status": gate11v_path_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_residual_marker_carrier_condition_preservation_status": named_residual_marker_carrier_condition_preservation_status,
        "explicit_carrier_completion_marker_status": explicit_carrier_completion_marker_status,
        "same_source_carrier_completion_status": same_source_carrier_completion_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "named_residual_marker_carrier_completion_status": named_residual_marker_carrier_completion_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11XResidualMarkerCarrierCompletionBlockerAuditTest(unittest.TestCase):
    def test_default_frozen_source_is_blocker_named(self) -> None:
        status = gate11x.build_status_payload(
            make_gate11w_manifest(),
            make_gate11w_status(),
            "Gate11W preserves the not-yet-completed line with blocker no_explicit_residual_completion_marker.",
        )

        self.assertEqual(status["gate11w_not_yet_completed_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_completion_marker_blocker_status"], "named")
        self.assertEqual(status["same_source_carrier_completion_blocker_status"], "named")
        self.assertEqual(status["residual_marker_carrier_completion_blocker_status"], "blocker_named")
        self.assertEqual(status["next_named_blocker"], "no_explicit_residual_completion_marker")

    def test_not_yet_named_when_residual_blocker_is_not_explicitly_named(self) -> None:
        status = gate11x.build_status_payload(
            make_gate11w_manifest(),
            make_gate11w_status(next_named_blocker="marker_carrier_blocker_not_fixed"),
            "Gate11W preserves the line but does not explicitly name the remaining blocker.",
        )

        self.assertEqual(status["residual_marker_carrier_completion_blocker_status"], "not_yet_named")
        self.assertEqual(status["next_named_blocker"], "no_residual_completion_blocker_explicitly_named")

    def test_deferred_when_multiple_candidate_carriers_compete(self) -> None:
        status = gate11x.build_status_payload(
            make_gate11w_manifest(),
            make_gate11w_status(
                same_source_carrier_completion_status="deferred",
                named_residual_marker_carrier_completion_status="deferred",
                next_named_blocker="multiple_candidate_carriers",
            ),
            "Gate11W records competing candidate carriers.",
        )

        self.assertEqual(status["gate11w_not_yet_completed_state_preservation_status"], "deferred")
        self.assertEqual(status["residual_marker_carrier_completion_blocker_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "upstream_completion_deferred")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11x.build_status_payload(
            make_gate11w_manifest(),
            make_gate11w_status(carrier_completion_boundary_status="not_confirmed"),
            "Gate11W breaks the carrier-completion boundary.",
        )

        self.assertEqual(status["carrier_completion_boundary_status"], "denied")
        self.assertEqual(status["residual_marker_carrier_completion_blocker_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "carrier_completion_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11x.build_status_payload(make_gate11w_manifest(run_id=""), make_gate11w_status(), "")

        self.assertEqual(status["explicit_completion_marker_blocker_status"], "deferred")
        self.assertEqual(status["same_source_carrier_completion_blocker_status"], "deferred")
        self.assertEqual(status["residual_marker_carrier_completion_blocker_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11x.build_status_payload(
            make_gate11w_manifest(),
            make_gate11w_status(),
            "Gate11W preserves the not-yet-completed line with blocker no_explicit_residual_completion_marker.",
        )
        registry = gate11x.build_registry(make_gate11w_manifest(), status)
        compare = gate11x.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11w_run_id"], "gate11w_run")
        self.assertEqual(compare[0]["residual_marker_carrier_completion_blocker_status"], "blocker_named")


if __name__ == "__main__":
    unittest.main()