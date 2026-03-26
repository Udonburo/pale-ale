#!/usr/bin/env python3
"""Regression tests for Gate11AG named blocker-resolution marker carrier-completion helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11ag_named_blocker_resolution_marker_carrier_completion_audit as gate11ag


def make_gate11af_manifest(run_id: str = "gate11af_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11af_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_residual_marker_carrier_condition_preservation_status: str = "preserved",
    minimum_same_source_carrier_completion_rule_status: str = "defined",
    bounded_read_prefix_completion_requirement_status: str = "defined",
    carrier_completion_boundary_status: str = "confirmed",
    explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status: str = "path_defined",
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
        "gate11aa_surface_defined_state_preservation_status": gate11aa_surface_defined_state_preservation_status,
        "gate11ab_not_yet_present_state_preservation_status": gate11ab_not_yet_present_state_preservation_status,
        "gate11ac_path_defined_state_preservation_status": gate11ac_path_defined_state_preservation_status,
        "gate11ad_not_yet_present_state_preservation_status": gate11ad_not_yet_present_state_preservation_status,
        "gate11ae_residual_named_state_preservation_status": gate11ae_residual_named_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_residual_marker_carrier_condition_preservation_status": named_residual_marker_carrier_condition_preservation_status,
        "minimum_same_source_carrier_completion_rule_status": minimum_same_source_carrier_completion_rule_status,
        "bounded_read_prefix_completion_requirement_status": bounded_read_prefix_completion_requirement_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status": explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status,
        "next_named_blocker": next_named_blocker,
    }


def completed_report(later_source_id: str = "runs/gate11ag_future") -> str:
    surfaces = "\n".join(f"residual_completion_surface: {phrase}" for phrase in gate11ag.REQUIRED_COMPLETION_SURFACES)
    return (
        "residual_completion_marker_status: present\n"
        "residual_completion_blocker_resolution_marker_status: present\n"
        f"residual_completion_later_source_id: {later_source_id}\n"
        "residual_completion_same_source_status: completed\n"
        f"{surfaces}\n"
    )


class RunGate11AGNamedBlockerResolutionMarkerCarrierCompletionAuditTest(unittest.TestCase):
    def test_current_frozen_gate11af_source_is_not_yet_completed(self) -> None:
        status = gate11ag.build_status_payload(
            make_gate11af_manifest(),
            make_gate11af_status(),
            "Gate11AF preserves the fixed path but does not explicitly complete the named blocker-resolution marker carrier condition.",
        )

        self.assertEqual(status["gate11af_path_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "none")
        self.assertEqual(status["same_source_carrier_completion_status"], "not_completed")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_gate11af_path_prose_does_not_falsely_earn_completion(self) -> None:
        report_text = (
            "The fixed path says one explicit blocker-resolution marker, one explicit later_source_id or later_frozen_run_id, "
            "one blocker-resolution marker and only one blocker-resolution marker, and one explicit same-source path-attachment status marked completed."
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")
        self.assertEqual(status["next_named_blocker"], "no_explicit_blocker_resolution_marker")

    def test_hypothetical_prose_does_not_falsely_earn_completion(self) -> None:
        report_text = (
            "Example: if a later source were to carry one explicit blocker-resolution marker and matched status, registry, and read surfaces, "
            "the condition would count as completed."
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")

    def test_absent_marker_keeps_singularity_and_same_source_from_advancing(self) -> None:
        report_text = (
            "residual_completion_later_source_id: runs/gate11ag_future\n"
            "residual_completion_marker_status: present\n"
            "residual_completion_same_source_status: completed\n"
            + "\n".join(f"residual_completion_surface: {phrase}" for phrase in gate11ag.REQUIRED_COMPLETION_SURFACES)
            + "\n"
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "absent")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "none")
        self.assertEqual(status["same_source_carrier_completion_status"], "not_completed")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")

    def test_explicit_residual_marker_row_is_required_for_completed(self) -> None:
        report_text = (
            "residual_completion_blocker_resolution_marker_status: present\n"
            "residual_completion_later_source_id: runs/gate11ag_future\n"
            "residual_completion_same_source_status: completed\n"
            + "\n".join(f"residual_completion_surface: {phrase}" for phrase in gate11ag.REQUIRED_COMPLETION_SURFACES)
            + "\n"
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "single")
        self.assertEqual(status["same_source_carrier_completion_status"], "not_completed")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")

    def test_explicit_later_source_row_is_required_for_completed(self) -> None:
        report_text = (
            "residual_completion_marker_status: present\n"
            "residual_completion_blocker_resolution_marker_status: present\n"
            "residual_completion_same_source_status: completed\n"
            + "\n".join(f"residual_completion_surface: {phrase}" for phrase in gate11ag.REQUIRED_COMPLETION_SURFACES)
            + "\n"
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "single")
        self.assertEqual(status["same_source_carrier_completion_status"], "not_completed")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")
        self.assertEqual(status["next_named_blocker"], "same_source_carrier_completion_not_completed")

    def test_multiple_blocker_marker_rows_on_same_source_break_singularity(self) -> None:
        report_text = completed_report("runs/gate11ag_future") + (
            "residual_completion_marker_status: present\n"
            "residual_completion_blocker_resolution_marker_status: present\n"
            "residual_completion_later_source_id: runs/gate11ag_future\n"
            "residual_completion_same_source_status: completed\n"
            + "\n".join(f"residual_completion_surface: {phrase}" for phrase in gate11ag.REQUIRED_COMPLETION_SURFACES)
            + "\n"
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "multiple")
        self.assertEqual(status["same_source_carrier_completion_status"], "deferred")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_carriers")

    def test_completed_case_requires_bounded_explicit_rows_only(self) -> None:
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), completed_report())

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "present")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "single")
        self.assertEqual(status["same_source_carrier_completion_status"], "completed")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "completed")
        self.assertEqual(status["next_named_blocker"], "")

    def test_multiple_later_source_ids_cause_deferred(self) -> None:
        report_text = (
            "residual_completion_marker_status: present\n"
            "residual_completion_blocker_resolution_marker_status: present\n"
            "residual_completion_later_source_id: runs/gate11ag_future_a\n"
            "residual_completion_later_source_id: runs/gate11ag_future_b\n"
            "residual_completion_same_source_status: completed\n"
            + "\n".join(f"residual_completion_surface: {phrase}" for phrase in gate11ag.REQUIRED_COMPLETION_SURFACES)
            + "\n"
        )
        status = gate11ag.build_status_payload(make_gate11af_manifest(), make_gate11af_status(), report_text)

        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "single")
        self.assertEqual(status["same_source_carrier_completion_status"], "deferred")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_carriers")

    def test_boundary_failure_causes_denied(self) -> None:
        status = gate11ag.build_status_payload(
            make_gate11af_manifest(),
            make_gate11af_status(carrier_completion_boundary_status="denied"),
            completed_report(),
        )

        self.assertEqual(status["carrier_completion_boundary_status"], "denied")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "carrier_completion_boundary_not_intact")

    def test_incomplete_controlling_source_causes_deferred(self) -> None:
        status = gate11ag.build_status_payload(make_gate11af_manifest(run_id=""), make_gate11af_status(), "")

        self.assertEqual(status["explicit_blocker_resolution_marker_status"], "deferred")
        self.assertEqual(status["blocker_resolution_marker_singularity_status"], "deferred")
        self.assertEqual(status["same_source_carrier_completion_status"], "deferred")
        self.assertEqual(status["named_blocker_resolution_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11ag.build_status_payload(
            make_gate11af_manifest(),
            make_gate11af_status(),
            "Gate11AF preserves the fixed path but does not explicitly complete the named blocker-resolution marker carrier condition.",
        )
        registry = gate11ag.build_registry(make_gate11af_manifest(), status)
        compare = gate11ag.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11af_run_id"], "gate11af_run")
        self.assertEqual(compare[0]["named_blocker_resolution_marker_carrier_completion_status"], "not_yet_completed")


if __name__ == "__main__":
    unittest.main()