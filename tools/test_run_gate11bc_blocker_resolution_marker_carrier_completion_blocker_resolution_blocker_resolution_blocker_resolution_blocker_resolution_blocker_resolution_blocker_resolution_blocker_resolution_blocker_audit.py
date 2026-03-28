#!/usr/bin/env python3
"""Regression tests for Gate11BC blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11bc_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_audit as gate11bc


def make_gate11bb_manifest(run_id: str = "gate11bb_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "dc9e56dc8ecc0a2957b0f77d85857615dd8a00d8"}


def make_gate11bb_status(**overrides: str) -> dict:
    status = {
        "gate10_closeout_preservation_status": "preserved",
        "gate11a_absence_result_preservation_status": "preserved",
        "gate11c_declaration_surface_preservation_status": "preserved",
        "gate11d_not_yet_declared_state_preservation_status": "preserved",
        "gate11e_path_defined_state_preservation_status": "preserved",
        "gate11f_not_yet_admissible_state_preservation_status": "preserved",
        "gate11g_naming_surface_preservation_status": "preserved",
        "gate11h_not_yet_named_state_preservation_status": "preserved",
        "gate11i_path_defined_state_preservation_status": "preserved",
        "gate11j_not_yet_admissible_state_preservation_status": "preserved",
        "gate11k_not_yet_present_state_preservation_status": "preserved",
        "gate11l_path_defined_state_preservation_status": "preserved",
        "gate11m_not_yet_present_state_preservation_status": "preserved",
        "gate11n_residual_named_state_preservation_status": "preserved",
        "gate11o_path_defined_state_preservation_status": "preserved",
        "gate11p_not_yet_completed_state_preservation_status": "preserved",
        "gate11q_surface_defined_state_preservation_status": "preserved",
        "gate11r_not_yet_present_state_preservation_status": "preserved",
        "gate11s_path_defined_state_preservation_status": "preserved",
        "gate11t_not_yet_present_state_preservation_status": "preserved",
        "gate11u_residual_named_state_preservation_status": "preserved",
        "gate11v_path_defined_state_preservation_status": "preserved",
        "gate11w_not_yet_completed_state_preservation_status": "preserved",
        "gate11x_blocker_named_state_preservation_status": "preserved",
        "gate11y_path_defined_state_preservation_status": "preserved",
        "gate11z_not_yet_resolved_state_preservation_status": "preserved",
        "gate11aa_surface_defined_state_preservation_status": "preserved",
        "gate11ab_not_yet_present_state_preservation_status": "preserved",
        "gate11ac_path_defined_state_preservation_status": "preserved",
        "gate11ad_not_yet_present_state_preservation_status": "preserved",
        "gate11ae_residual_named_state_preservation_status": "preserved",
        "gate11af_path_defined_state_preservation_status": "preserved",
        "gate11ag_not_yet_completed_state_preservation_status": "preserved",
        "gate11ah_blocker_named_state_preservation_status": "preserved",
        "gate11ai_path_defined_state_preservation_status": "preserved",
        "gate11aj_not_yet_resolved_state_preservation_status": "preserved",
        "gate11ak_blocker_named_state_preservation_status": "preserved",
        "gate11al_path_defined_state_preservation_status": "preserved",
        "gate11am_not_yet_resolved_state_preservation_status": "preserved",
        "gate11an_blocker_named_state_preservation_status": "preserved",
        "gate11ao_path_defined_state_preservation_status": "preserved",
        "gate11ap_not_yet_resolved_state_preservation_status": "preserved",
        "gate11aq_blocker_named_state_preservation_status": "preserved",
        "gate11ar_path_defined_state_preservation_status": "preserved",
        "gate11as_not_yet_resolved_state_preservation_status": "preserved",
        "gate11at_blocker_named_state_preservation_status": "preserved",
        "gate11au_path_defined_state_preservation_status": "preserved",
        "gate11av_not_yet_resolved_state_preservation_status": "preserved",
        "gate11aw_blocker_named_state_preservation_status": "preserved",
        "gate11ax_path_defined_state_preservation_status": "preserved",
        "gate11ay_not_yet_resolved_state_preservation_status": "preserved",
        "gate11az_blocker_named_state_preservation_status": "preserved",
        "gate11ba_path_defined_state_preservation_status": "preserved",
        "broader_trusted_tree_settlement_still_unearned_status": "confirmed",
        "operator_admission_still_denied_status": "confirmed",
        "retroactive_reinterpretation_forbidden_status": "confirmed",
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status": "preserved",
        "explicit_blocker_resolution_marker_status": "absent",
        "same_source_blocker_resolution_status": "not_resolved",
        "blocker_resolution_boundary_status": "confirmed",
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status": "not_yet_resolved",
        "next_named_blocker": "no_explicit_blocker_resolution_marker",
    }
    status.update(overrides)
    return status


class RunGate11BCBlockerResolutionMarkerCarrierCompletionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerAuditTest(
    unittest.TestCase
):
    def test_current_frozen_gate11bb_source_is_blocker_named(self) -> None:
        status = gate11bc.build_status_payload(
            make_gate11bb_manifest(),
            make_gate11bb_status(),
            "Gate11BB preserves the fixed not-yet-resolved line with blocker no_explicit_blocker_resolution_marker.",
        )

        self.assertEqual(status["gate11bb_not_yet_resolved_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_blocker_resolution_marker_blocker_status"], "named")
        self.assertEqual(status["same_source_blocker_resolution_blocker_status"], "named")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "blocker_named",
        )
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_named_blocker_not_preserved_is_denied(self) -> None:
        status = gate11bc.build_status_payload(
            make_gate11bb_manifest(),
            make_gate11bb_status(gate11az_blocker_named_state_preservation_status="not_preserved"),
            "Gate11BB no longer preserves the upstream named blocker state.",
        )

        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "denied",
        )
        self.assertEqual(status["next_named_blocker"], "gate11az_blocker_named_state_not_preserved")

    def test_boundary_break_causes_denied(self) -> None:
        status = gate11bc.build_status_payload(
            make_gate11bb_manifest(),
            make_gate11bb_status(blocker_resolution_boundary_status="not_confirmed"),
            "Gate11BB breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_blocker_boundary_status"], "denied")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "denied",
        )
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_boundary_not_intact")

    def test_incomplete_controlling_source_causes_deferred(self) -> None:
        status = gate11bc.build_status_payload(make_gate11bb_manifest(run_id=""), make_gate11bb_status(), "")

        self.assertEqual(status["explicit_blocker_resolution_marker_blocker_status"], "deferred")
        self.assertEqual(status["same_source_blocker_resolution_blocker_status"], "deferred")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "deferred",
        )
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_source_that_no_longer_supports_narrow_blocker_naming_is_not_yet_named(self) -> None:
        status = gate11bc.build_status_payload(
            make_gate11bb_manifest(),
            make_gate11bb_status(
                explicit_blocker_resolution_marker_status="present",
                same_source_blocker_resolution_status="not_resolved",
                next_named_blocker="same_source_blocker_resolution_not_completed",
            ),
            "Gate11BB no longer supports the fixed no-explicit-marker blocker naming line.",
        )

        self.assertEqual(status["explicit_blocker_resolution_marker_blocker_status"], "not_named")
        self.assertEqual(status["same_source_blocker_resolution_blocker_status"], "named")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "not_yet_named",
        )
        self.assertEqual(
            status["next_named_blocker"],
            "no_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_explicitly_named",
        )

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11bc.build_status_payload(
            make_gate11bb_manifest(),
            make_gate11bb_status(),
            "Gate11BB preserves the fixed not-yet-resolved line with blocker no_explicit_blocker_resolution_marker.",
        )
        registry = gate11bc.build_registry(make_gate11bb_manifest(), status)
        compare = gate11bc.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11bb_run_id"], "gate11bb_run")
        self.assertEqual(
            compare[0][
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status"
            ],
            "blocker_named",
        )


if __name__ == "__main__":
    unittest.main()