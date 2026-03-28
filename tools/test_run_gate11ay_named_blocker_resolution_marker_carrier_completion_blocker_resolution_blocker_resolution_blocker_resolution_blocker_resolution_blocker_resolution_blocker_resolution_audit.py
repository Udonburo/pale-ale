#!/usr/bin/env python3
"""Regression tests for Gate11AY named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11ay_named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_audit as gate11ay


def make_gate11ax_manifest(run_id: str = "gate11ax_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "74fe4396ed91f2d1483fadf3d0c9d31d357f5a91"}


def make_gate11ax_status(**overrides: str) -> dict:
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
        "broader_trusted_tree_settlement_still_unearned_status": "confirmed",
        "operator_admission_still_denied_status": "confirmed",
        "retroactive_reinterpretation_forbidden_status": "confirmed",
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_condition_preservation_status": "preserved",
        "minimum_same_source_blocker_resolution_rule_status": "defined",
        "bounded_read_prefix_resolution_requirement_status": "defined",
        "blocker_resolution_boundary_status": "confirmed",
        "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status": "path_defined",
        "next_named_blocker": "",
    }
    status.update(overrides)
    return status


class RunGate11AYNamedBlockerResolutionMarkerCarrierCompletionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionAuditTest(
    unittest.TestCase
):
    def test_not_yet_resolved_for_current_frozen_gate11ax_source(self) -> None:
        status = gate11ay.build_status_payload(
            make_gate11ax_manifest(),
            make_gate11ax_status(),
            "Gate11AX preserves the fixed blocker-resolution path but does not explicitly resolve the named blocker.",
        )

        self.assertEqual(status["gate11ax_path_defined_state_preservation_status"], "preserved")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_condition_preservation_status"
            ],
            "preserved",
        )
        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["same_source_blocker_resolution_status"], "not_resolved")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "not_yet_resolved",
        )
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_resolved_when_same_source_resolution_is_explicit(self) -> None:
        report_text = """
    blocker_resolution_marker_status: present
    later_source_id: runs/gate11ay_future
    same_source_blocker_resolution_status: resolved
    bounded_read_prefix_declaration_status: present
    residual_completion_marker_status: present
    admissible_later_source_presence_status: present
    declaration_marker_status: present
    candidate_id: gate11ay_candidate_001
    class: host_failure
    host_failure_sentence_status: present
    matched_status_registry_read_surfaces_status: matched
    residual_completion_surface: one explicit blocker-resolution marker
    residual_completion_surface: one explicit later-source identifier
    residual_completion_surface: one blocker-resolution marker and only one blocker-resolution marker
    residual_completion_surface: one explicit same-source blocker-resolution status marked resolved
    residual_completion_surface: one bounded read-prefix declaration for the blocker-resolution marker
    residual_completion_surface: repeated bounded residual_completion_surface rows for the required same-source elements
    residual_completion_surface: one explicit residual completion marker
    residual_completion_surface: one explicit admissible later-source presence marker
    residual_completion_surface: one declaration marker
    residual_completion_surface: one candidate id
    residual_completion_surface: one class
    residual_completion_surface: one explicit host-failure sentence
    residual_completion_surface: matched status, registry, and read surfaces
"""
        status = gate11ay.build_status_payload(make_gate11ax_manifest(), make_gate11ax_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["same_source_blocker_resolution_status"], "resolved")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "resolved",
        )
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_resolved_when_resolution_phrases_appear_only_in_path_prose(self) -> None:
        report_text = """
    the minimum honest path is fixed narrowly: one explicit blocker-resolution marker, one explicit later_source_id or later_frozen_run_id,
    one blocker-resolution marker and only one blocker-resolution marker, one explicit same-source blocker-resolution status marked resolved,
    one bounded read-prefix declaration for the blocker-resolution marker, repeated bounded residual_completion_surface rows for the required same-source elements,
    one explicit residual completion marker, one explicit admissible later-source presence marker, one declaration marker, one candidate id,
    one class, one explicit host-failure sentence, and matched status, registry, and read surfaces.
    """
        status = gate11ay.build_status_payload(make_gate11ax_manifest(), make_gate11ax_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["same_source_blocker_resolution_status"], "not_resolved")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "not_yet_resolved",
        )
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_deferred_when_multiple_candidate_later_source_identifiers_compete(self) -> None:
        report_text = """
    blocker_resolution_marker_status: present
    later_source_id: runs/gate11ay_future_a
    later_frozen_run_id: runs/gate11ay_future_b
"""
        status = gate11ay.build_status_payload(make_gate11ax_manifest(), make_gate11ax_status(), report_text)

        self.assertEqual(status["same_source_blocker_resolution_status"], "deferred")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "deferred",
        )
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_resolutions")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11ay.build_status_payload(
            make_gate11ax_manifest(),
            make_gate11ax_status(blocker_resolution_boundary_status="not_confirmed"),
            "Gate11AX breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_boundary_status"], "denied")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "denied",
        )
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11ay.build_status_payload(make_gate11ax_manifest(run_id=""), make_gate11ax_status(), "")

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "deferred")
        self.assertEqual(status["same_source_blocker_resolution_status"], "deferred")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "deferred",
        )
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11ay.build_status_payload(
            make_gate11ax_manifest(),
            make_gate11ax_status(),
            "Gate11AX preserves the fixed blocker-resolution path but does not explicitly resolve the named blocker.",
        )
        registry = gate11ay.build_registry(make_gate11ax_manifest(), status)
        compare = gate11ay.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11ax_run_id"], "gate11ax_run")
        self.assertEqual(
            compare[0][
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
            ],
            "not_yet_resolved",
        )


if __name__ == "__main__":
    unittest.main()