#!/usr/bin/env python3
"""Regression tests for Gate11AT blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11at_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_audit as gate11at


def make_gate11as_manifest(run_id: str = "gate11as_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "0ac306b4e04f24e90929069ad652a6a09576dceb"}


def make_gate11as_status(
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
    gate11ab_not_yet_present_state_preservation_status: str = "preserved",
    gate11ac_path_defined_state_preservation_status: str = "preserved",
    gate11ad_not_yet_present_state_preservation_status: str = "preserved",
    gate11ae_residual_named_state_preservation_status: str = "preserved",
    gate11af_path_defined_state_preservation_status: str = "preserved",
    gate11ag_not_yet_completed_state_preservation_status: str = "preserved",
    gate11ah_blocker_named_state_preservation_status: str = "preserved",
    gate11ai_path_defined_state_preservation_status: str = "preserved",
    gate11aj_not_yet_resolved_state_preservation_status: str = "preserved",
    gate11ak_blocker_named_state_preservation_status: str = "preserved",
    gate11al_path_defined_state_preservation_status: str = "preserved",
    gate11am_not_yet_resolved_state_preservation_status: str = "preserved",
    gate11an_blocker_named_state_preservation_status: str = "preserved",
    gate11ao_path_defined_state_preservation_status: str = "preserved",
    gate11ap_not_yet_resolved_state_preservation_status: str = "preserved",
    gate11aq_blocker_named_state_preservation_status: str = "preserved",
    gate11ar_path_defined_state_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_blocker_preservation_status: str = "preserved",
    explicit_blocker_resolution_marker_status: str = "absent",
    same_source_blocker_resolution_status: str = "not_resolved",
    blocker_resolution_boundary_status: str = "confirmed",
    named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status: str = "not_yet_resolved",
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
        "gate11ab_not_yet_present_state_preservation_status": gate11ab_not_yet_present_state_preservation_status,
        "gate11ac_path_defined_state_preservation_status": gate11ac_path_defined_state_preservation_status,
        "gate11ad_not_yet_present_state_preservation_status": gate11ad_not_yet_present_state_preservation_status,
        "gate11ae_residual_named_state_preservation_status": gate11ae_residual_named_state_preservation_status,
        "gate11af_path_defined_state_preservation_status": gate11af_path_defined_state_preservation_status,
        "gate11ag_not_yet_completed_state_preservation_status": gate11ag_not_yet_completed_state_preservation_status,
        "gate11ah_blocker_named_state_preservation_status": gate11ah_blocker_named_state_preservation_status,
        "gate11ai_path_defined_state_preservation_status": gate11ai_path_defined_state_preservation_status,
        "gate11aj_not_yet_resolved_state_preservation_status": gate11aj_not_yet_resolved_state_preservation_status,
        "gate11ak_blocker_named_state_preservation_status": gate11ak_blocker_named_state_preservation_status,
        "gate11al_path_defined_state_preservation_status": gate11al_path_defined_state_preservation_status,
        "gate11am_not_yet_resolved_state_preservation_status": gate11am_not_yet_resolved_state_preservation_status,
        "gate11an_blocker_named_state_preservation_status": gate11an_blocker_named_state_preservation_status,
        "gate11ao_path_defined_state_preservation_status": gate11ao_path_defined_state_preservation_status,
        "gate11ap_not_yet_resolved_state_preservation_status": gate11ap_not_yet_resolved_state_preservation_status,
        "gate11aq_blocker_named_state_preservation_status": gate11aq_blocker_named_state_preservation_status,
        "gate11ar_path_defined_state_preservation_status": gate11ar_path_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_blocker_preservation_status": named_blocker_preservation_status,
        "explicit_blocker_resolution_marker_status": explicit_blocker_resolution_marker_status,
        "same_source_blocker_resolution_status": same_source_blocker_resolution_status,
        "blocker_resolution_boundary_status": blocker_resolution_boundary_status,
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status": named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11ATBlockerResolutionMarkerCarrierCompletionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerAuditTest(unittest.TestCase):
    def test_current_frozen_gate11as_source_is_blocker_named(self) -> None:
        status = gate11at.build_status_payload(
            make_gate11as_manifest(),
            make_gate11as_status(),
            "Gate11AS preserves the fixed not-yet-resolved line with blocker no_explicit_blocker_resolution_marker.",
        )

        self.assertEqual(status["gate11as_not_yet_resolved_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_blocker_resolution_marker_blocker_status"], "named")
        self.assertEqual(status["same_source_blocker_resolution_blocker_status"], "named")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "blocker_named",
        )
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_named_condition_not_preserved_is_denied(self) -> None:
        status = gate11at.build_status_payload(
            make_gate11as_manifest(),
            make_gate11as_status(named_blocker_preservation_status="not_preserved"),
            "Gate11AS no longer preserves the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution condition.",
        )

        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "denied",
        )
        self.assertEqual(
            status["next_named_blocker"],
            "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_not_preserved",
        )

    def test_boundary_break_causes_denied(self) -> None:
        status = gate11at.build_status_payload(
            make_gate11as_manifest(),
            make_gate11as_status(blocker_resolution_boundary_status="not_confirmed"),
            "Gate11AS breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_blocker_boundary_status"], "denied")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "denied",
        )
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_boundary_not_intact")

    def test_incomplete_controlling_source_causes_deferred(self) -> None:
        status = gate11at.build_status_payload(make_gate11as_manifest(run_id=""), make_gate11as_status(), "")

        self.assertEqual(status["explicit_blocker_resolution_marker_blocker_status"], "deferred")
        self.assertEqual(status["same_source_blocker_resolution_blocker_status"], "deferred")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "deferred",
        )
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_source_that_no_longer_supports_narrow_blocker_naming_is_not_yet_named(self) -> None:
        status = gate11at.build_status_payload(
            make_gate11as_manifest(),
            make_gate11as_status(
                explicit_blocker_resolution_marker_status="present",
                same_source_blocker_resolution_status="not_resolved",
                next_named_blocker="same_source_blocker_resolution_not_completed",
            ),
            "Gate11AS no longer supports the fixed no-explicit-marker blocker naming line.",
        )

        self.assertEqual(status["explicit_blocker_resolution_marker_blocker_status"], "not_named")
        self.assertEqual(status["same_source_blocker_resolution_blocker_status"], "named")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "not_yet_named",
        )
        self.assertEqual(
            status["next_named_blocker"],
            "no_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_explicitly_named",
        )

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11at.build_status_payload(
            make_gate11as_manifest(),
            make_gate11as_status(),
            "Gate11AS preserves the fixed not-yet-resolved line with blocker no_explicit_blocker_resolution_marker.",
        )
        registry = gate11at.build_registry(make_gate11as_manifest(), status)
        compare = gate11at.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11as_run_id"], "gate11as_run")
        self.assertEqual(
            compare[0][
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "blocker_named",
        )


if __name__ == "__main__":
    unittest.main()