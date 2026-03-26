#!/usr/bin/env python3
"""Regression tests for Gate11AF explicit blocker-resolution marker carrier-completion instantiation-path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11af_explicit_blocker_resolution_marker_carrier_completion_instantiation_path_audit as gate11af


def make_gate11ae_manifest(run_id: str = "gate11ae_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11ae_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_marker_carrier_completion_status: str = "missing",
    marker_singularity_carrier_completion_status: str = "missing",
    same_source_path_attachment_carrier_completion_status: str = "missing",
    carrier_completion_boundary_status: str = "confirmed",
    explicit_blocker_resolution_marker_carrier_completion_status: str = "residual_named",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_marker_carrier_completion_status": explicit_marker_carrier_completion_status,
        "marker_singularity_carrier_completion_status": marker_singularity_carrier_completion_status,
        "same_source_path_attachment_carrier_completion_status": same_source_path_attachment_carrier_completion_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "explicit_blocker_resolution_marker_carrier_completion_status": explicit_blocker_resolution_marker_carrier_completion_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11AFExplicitBlockerResolutionMarkerCarrierCompletionInstantiationPathAuditTest(unittest.TestCase):
    def test_default_frozen_source_is_path_defined(self) -> None:
        status = gate11af.build_status_payload(
            make_gate11ae_manifest(),
            make_gate11ae_status(),
            "Gate11AE preserves the residual-named line with blocker no_explicit_blocker_resolution_marker.",
        )

        self.assertEqual(status["gate11ae_residual_named_state_preservation_status"], "preserved")
        self.assertEqual(status["named_residual_marker_carrier_condition_preservation_status"], "preserved")
        self.assertEqual(status["minimum_same_source_carrier_completion_rule_status"], "defined")
        self.assertEqual(status["bounded_read_prefix_completion_requirement_status"], "defined")
        self.assertEqual(status["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"], "path_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_defined_when_blocker_is_not_path_definable(self) -> None:
        status = gate11af.build_status_payload(
            make_gate11ae_manifest(),
            make_gate11ae_status(next_named_blocker="same_source_marker_path_not_instantiated"),
            "Gate11AE preserves the residual-named line but the remaining carrier path is not yet fixed narrowly enough.",
        )

        self.assertEqual(status["named_residual_marker_carrier_condition_preservation_status"], "preserved")
        self.assertEqual(status["minimum_same_source_carrier_completion_rule_status"], "not_defined")
        self.assertEqual(status["bounded_read_prefix_completion_requirement_status"], "not_defined")
        self.assertEqual(status["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "minimum_same_source_carrier_completion_rule_not_fixed")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11af.build_status_payload(
            make_gate11ae_manifest(),
            make_gate11ae_status(carrier_completion_boundary_status="denied"),
            "Gate11AE breaks the carrier-completion boundary.",
        )

        self.assertEqual(status["carrier_completion_boundary_status"], "denied")
        self.assertEqual(status["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "carrier_completion_boundary_not_intact")

    def test_denied_when_upstream_residual_named_state_is_not_preserved(self) -> None:
        status = gate11af.build_status_payload(
            make_gate11ae_manifest(),
            make_gate11ae_status(explicit_blocker_resolution_marker_carrier_completion_status="not_yet_named"),
            "Gate11AE no longer preserves the residual-named line.",
        )

        self.assertEqual(status["gate11ae_residual_named_state_preservation_status"], "not_preserved")
        self.assertEqual(status["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11ae_residual_named_state_not_preserved")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11af.build_status_payload(make_gate11ae_manifest(run_id=""), make_gate11ae_status(), "")

        self.assertEqual(status["named_residual_marker_carrier_condition_preservation_status"], "deferred")
        self.assertEqual(status["minimum_same_source_carrier_completion_rule_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_completion_requirement_status"], "deferred")
        self.assertEqual(status["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11af.build_status_payload(
            make_gate11ae_manifest(),
            make_gate11ae_status(),
            "Gate11AE preserves the residual-named line with blocker no_explicit_blocker_resolution_marker.",
        )
        registry = gate11af.build_registry(make_gate11ae_manifest(), status)
        compare = gate11af.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11ae_run_id"], "gate11ae_run")
        self.assertEqual(compare[0]["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"], "path_defined")


if __name__ == "__main__":
    unittest.main()