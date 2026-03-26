#!/usr/bin/env python3
"""Regression tests for Gate11N admissible later-source carrier-completion helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11n_admissible_later_source_carrier_completion_audit as gate11n


def make_gate11m_manifest(run_id: str = "gate11m_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11m_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_admissible_later_source_presence_marker_status: str = "absent",
    later_source_singularity_status: str = "none",
    same_source_fixed_gate11l_path_instantiation_status: str = "not_instantiated",
    admissibility_boundary_status: str = "confirmed",
    one_admissible_later_source_explicit_presence_path_instantiation_status: str = "not_yet_present",
    next_named_blocker: str = "no_explicit_admissible_later_source_presence_marker",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_admissible_later_source_presence_marker_status": explicit_admissible_later_source_presence_marker_status,
        "later_source_singularity_status": later_source_singularity_status,
        "same_source_fixed_gate11l_path_instantiation_status": same_source_fixed_gate11l_path_instantiation_status,
        "admissibility_boundary_status": admissibility_boundary_status,
        "one_admissible_later_source_explicit_presence_path_instantiation_status": one_admissible_later_source_explicit_presence_path_instantiation_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11NAdmissibleLaterSourceCarrierCompletionAuditTest(unittest.TestCase):
    def test_residual_named_for_current_frozen_gate11m_source(self) -> None:
        status = gate11n.build_status_payload(
            make_gate11m_manifest(),
            make_gate11m_status(),
            "Gate11M preserves the bounded line and names the next blocker.",
        )

        self.assertEqual(status["gate11m_not_yet_present_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_presence_marker_carrier_completion_status"], "missing")
        self.assertEqual(status["later_source_singularity_carrier_completion_status"], "missing")
        self.assertEqual(status["same_source_path_attachment_carrier_completion_status"], "missing")
        self.assertEqual(status["admissible_later_source_carrier_completion_status"], "residual_named")
        self.assertEqual(status["next_named_blocker"], "no_explicit_admissible_later_source_presence_marker")

    def test_residual_named_when_multiple_later_sources_are_the_named_blocker(self) -> None:
        status = gate11n.build_status_payload(
            make_gate11m_manifest(),
            make_gate11m_status(
                explicit_admissible_later_source_presence_marker_status="present",
                later_source_singularity_status="multiple",
                same_source_fixed_gate11l_path_instantiation_status="deferred",
                one_admissible_later_source_explicit_presence_path_instantiation_status="deferred",
                next_named_blocker="multiple_later_sources",
            ),
            "Gate11M records multiple later sources.",
        )

        self.assertEqual(status["explicit_presence_marker_carrier_completion_status"], "complete")
        self.assertEqual(status["later_source_singularity_carrier_completion_status"], "missing")
        self.assertEqual(status["admissible_later_source_carrier_completion_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11m_not_yet_present_state_not_preserved")

    def test_not_yet_named_when_no_residual_blocker_is_explicit(self) -> None:
        status = gate11n.build_status_payload(
            make_gate11m_manifest(),
            make_gate11m_status(next_named_blocker="carrier_condition_not_narrowly_named"),
            "Gate11M preserves the line but does not explicitly name the residual carrier condition.",
        )

        self.assertEqual(status["admissible_later_source_carrier_completion_status"], "not_yet_named")
        self.assertEqual(status["next_named_blocker"], "no_residual_carrier_condition_explicitly_named")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11n.build_status_payload(
            make_gate11m_manifest(),
            make_gate11m_status(admissibility_boundary_status="not_confirmed"),
            "Gate11M breaks the admissibility boundary.",
        )

        self.assertEqual(status["carrier_completion_boundary_status"], "denied")
        self.assertEqual(status["admissible_later_source_carrier_completion_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "carrier_completion_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11n.build_status_payload(make_gate11m_manifest(run_id=""), make_gate11m_status(), "")

        self.assertEqual(status["explicit_presence_marker_carrier_completion_status"], "deferred")
        self.assertEqual(status["later_source_singularity_carrier_completion_status"], "deferred")
        self.assertEqual(status["same_source_path_attachment_carrier_completion_status"], "deferred")
        self.assertEqual(status["admissible_later_source_carrier_completion_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11n.build_status_payload(
            make_gate11m_manifest(),
            make_gate11m_status(),
            "Gate11M preserves the bounded line and names the next blocker.",
        )
        registry = gate11n.build_registry(make_gate11m_manifest(), status)
        compare = gate11n.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11m_run_id"], "gate11m_run")
        self.assertEqual(compare[0]["admissible_later_source_carrier_completion_status"], "residual_named")


if __name__ == "__main__":
    unittest.main()