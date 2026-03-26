#!/usr/bin/env python3
"""Regression tests for Gate11O admissible later-source carrier-completion instantiation-path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11o_admissible_later_source_carrier_completion_instantiation_path_audit as gate11o


def make_gate11n_manifest(run_id: str = "gate11n_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11n_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_presence_marker_carrier_completion_status: str = "missing",
    later_source_singularity_carrier_completion_status: str = "missing",
    same_source_path_attachment_carrier_completion_status: str = "missing",
    carrier_completion_boundary_status: str = "confirmed",
    admissible_later_source_carrier_completion_status: str = "residual_named",
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
        "gate11m_not_yet_present_state_preservation_status": gate11m_not_yet_present_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_presence_marker_carrier_completion_status": explicit_presence_marker_carrier_completion_status,
        "later_source_singularity_carrier_completion_status": later_source_singularity_carrier_completion_status,
        "same_source_path_attachment_carrier_completion_status": same_source_path_attachment_carrier_completion_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "admissible_later_source_carrier_completion_status": admissible_later_source_carrier_completion_status,
        "next_named_blocker": next_named_blocker,
    }


class RunGate11OAdmissibleLaterSourceCarrierCompletionInstantiationPathAuditTest(unittest.TestCase):
    def test_path_defined_for_current_frozen_gate11n_source(self) -> None:
        status = gate11o.build_status_payload(
            make_gate11n_manifest(),
            make_gate11n_status(),
            "Gate11N preserves the named residual carrier condition under the bounded line.",
        )

        self.assertEqual(status["gate11n_residual_named_state_preservation_status"], "preserved")
        self.assertEqual(status["named_residual_carrier_condition_preservation_status"], "preserved")
        self.assertEqual(status["minimum_residual_carrier_completion_rule_status"], "defined")
        self.assertEqual(status["admissible_later_source_carrier_completion_instantiation_path_status"], "path_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_defined_when_named_residual_is_not_path_definable_here(self) -> None:
        status = gate11o.build_status_payload(
            make_gate11n_manifest(),
            make_gate11n_status(next_named_blocker="multiple_later_sources"),
            "Gate11N preserves a residual blocker, but this slice does not fix a worker-choice path.",
        )

        self.assertEqual(status["named_residual_carrier_condition_preservation_status"], "preserved")
        self.assertEqual(status["minimum_residual_carrier_completion_rule_status"], "not_defined")
        self.assertEqual(status["admissible_later_source_carrier_completion_instantiation_path_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "minimum_residual_carrier_completion_rule_not_fixed")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11o.build_status_payload(
            make_gate11n_manifest(),
            make_gate11n_status(carrier_completion_boundary_status="not_confirmed"),
            "Gate11N breaks the carrier-completion boundary.",
        )

        self.assertEqual(status["residual_completion_boundary_status"], "denied")
        self.assertEqual(status["admissible_later_source_carrier_completion_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "residual_completion_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11o.build_status_payload(make_gate11n_manifest(run_id=""), make_gate11n_status(), "")

        self.assertEqual(status["named_residual_carrier_condition_preservation_status"], "deferred")
        self.assertEqual(status["minimum_residual_carrier_completion_rule_status"], "deferred")
        self.assertEqual(status["admissible_later_source_carrier_completion_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_denied_when_gate11n_residual_named_state_is_not_preserved(self) -> None:
        status = gate11o.build_status_payload(
            make_gate11n_manifest(),
            make_gate11n_status(admissible_later_source_carrier_completion_status="not_yet_named"),
            "Gate11N no longer preserves the residual-named state.",
        )

        self.assertEqual(status["gate11n_residual_named_state_preservation_status"], "not_preserved")
        self.assertEqual(status["admissible_later_source_carrier_completion_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11n_residual_named_state_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11o.build_status_payload(
            make_gate11n_manifest(),
            make_gate11n_status(),
            "Gate11N preserves the named residual carrier condition under the bounded line.",
        )
        registry = gate11o.build_registry(make_gate11n_manifest(), status)
        compare = gate11o.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11n_run_id"], "gate11n_run")
        self.assertEqual(compare[0]["admissible_later_source_carrier_completion_instantiation_path_status"], "path_defined")


if __name__ == "__main__":
    unittest.main()