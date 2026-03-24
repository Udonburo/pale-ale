#!/usr/bin/env python3
"""Regression tests for Gate11L admissible later-source explicit-presence instantiation-path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11l_admissible_later_source_explicit_presence_instantiation_path_audit as gate11l


def make_gate11k_manifest(run_id: str = "gate11k_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11k_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_later_source_marker_status: str = "absent",
    later_source_singularity_status: str = "none",
    same_source_path_attachment_status: str = "not_attached",
    admissibility_boundary_status: str = "confirmed",
    one_admissible_later_source_explicit_presence_status: str = "not_yet_present",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_later_source_marker_status": explicit_later_source_marker_status,
        "later_source_singularity_status": later_source_singularity_status,
        "same_source_path_attachment_status": same_source_path_attachment_status,
        "admissibility_boundary_status": admissibility_boundary_status,
        "one_admissible_later_source_explicit_presence_status": one_admissible_later_source_explicit_presence_status,
    }


class RunGate11LAdmissibleLaterSourceExplicitPresenceInstantiationPathAuditTest(unittest.TestCase):
    def test_path_defined_for_current_frozen_not_yet_present_source(self) -> None:
        status = gate11l.build_status_payload(
            make_gate11k_manifest(),
            make_gate11k_status(),
            "Gate11K preserves a not-yet-present source with explicit missing presence components.",
        )

        self.assertEqual(status["gate11k_not_yet_present_state_preservation_status"], "preserved")
        self.assertEqual(status["missing_explicit_presence_component_naming_status"], "named")
        self.assertEqual(status["minimal_same_source_admissible_presence_instantiation_rule_status"], "defined")
        self.assertEqual(status["admissibility_boundary_status"], "confirmed")
        self.assertEqual(status["admissible_later_source_explicit_presence_instantiation_path_status"], "path_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_denied_when_admissibility_boundary_breaks(self) -> None:
        status = gate11l.build_status_payload(
            make_gate11k_manifest(),
            make_gate11k_status(admissibility_boundary_status="not_confirmed"),
            "Gate11K source breaks the admissibility boundary.",
        )

        self.assertEqual(status["admissibility_boundary_status"], "denied")
        self.assertEqual(status["minimal_same_source_admissible_presence_instantiation_rule_status"], "not_defined")
        self.assertEqual(status["admissible_later_source_explicit_presence_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "admissibility_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11l.build_status_payload(
            make_gate11k_manifest(run_id=""),
            make_gate11k_status(),
            "",
        )

        self.assertEqual(status["missing_explicit_presence_component_naming_status"], "deferred")
        self.assertEqual(status["minimal_same_source_admissible_presence_instantiation_rule_status"], "deferred")
        self.assertEqual(status["admissibility_boundary_status"], "deferred")
        self.assertEqual(status["admissible_later_source_explicit_presence_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_not_yet_defined_when_missing_components_are_not_explicitly_named(self) -> None:
        status = gate11l.build_status_payload(
            make_gate11k_manifest(),
            make_gate11k_status(explicit_later_source_marker_status="present"),
            "Gate11K source does not cleanly name the missing explicit-presence components.",
        )

        self.assertEqual(status["missing_explicit_presence_component_naming_status"], "not_named")
        self.assertEqual(status["admissible_later_source_explicit_presence_instantiation_path_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "missing_explicit_presence_components_not_named")

    def test_denied_when_source_is_already_present(self) -> None:
        status = gate11l.build_status_payload(
            make_gate11k_manifest(),
            make_gate11k_status(one_admissible_later_source_explicit_presence_status="present"),
            "Gate11K source has already moved past not_yet_present.",
        )

        self.assertEqual(status["gate11k_not_yet_present_state_preservation_status"], "not_preserved")
        self.assertEqual(status["admissible_later_source_explicit_presence_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11k_not_yet_present_state_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11l.build_status_payload(
            make_gate11k_manifest(),
            make_gate11k_status(),
            "Gate11K preserves a not-yet-present source with explicit missing presence components.",
        )
        registry = gate11l.build_registry(make_gate11k_manifest(), make_gate11k_status(), status)
        compare = gate11l.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11k_run_id"], "gate11k_run")
        self.assertEqual(compare[0]["admissible_later_source_explicit_presence_instantiation_path_status"], "path_defined")


if __name__ == "__main__":
    unittest.main()