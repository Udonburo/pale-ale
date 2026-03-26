#!/usr/bin/env python3
"""Regression tests for Gate11I later-source explicit-naming instantiation-path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11i_later_source_explicit_naming_instantiation_path_audit as gate11i


def make_gate11h_manifest(run_id: str = "gate11h_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11h_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    gate11d_not_yet_declared_state_preservation_status: str = "preserved",
    gate11e_path_defined_state_preservation_status: str = "preserved",
    gate11f_not_yet_admissible_state_preservation_status: str = "preserved",
    gate11g_naming_surface_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_later_source_marker_status: str = "absent",
    later_source_singularity_status: str = "none",
    full_path_attachment_status: str = "not_attached",
    anti_shortcut_boundary_status: str = "confirmed",
    one_later_source_explicit_naming_status: str = "not_yet_named",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "gate11f_not_yet_admissible_state_preservation_status": gate11f_not_yet_admissible_state_preservation_status,
        "gate11g_naming_surface_preservation_status": gate11g_naming_surface_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_later_source_marker_status": explicit_later_source_marker_status,
        "later_source_singularity_status": later_source_singularity_status,
        "full_path_attachment_status": full_path_attachment_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "one_later_source_explicit_naming_status": one_later_source_explicit_naming_status,
    }


class RunGate11ILaterSourceExplicitNamingInstantiationPathAuditTest(unittest.TestCase):
    def test_path_defined_for_current_frozen_not_yet_named_source(self) -> None:
        status = gate11i.build_status_payload(
            make_gate11h_manifest(),
            make_gate11h_status(),
            "Gate11H preserves a not-yet-named source with explicit missing naming components.",
        )

        self.assertEqual(status["gate11h_not_yet_named_state_preservation_status"], "preserved")
        self.assertEqual(status["missing_naming_component_naming_status"], "named")
        self.assertEqual(status["minimal_same_source_later_source_instantiation_rule_status"], "defined")
        self.assertEqual(status["anti_shortcut_boundary_status"], "confirmed")
        self.assertEqual(status["later_source_explicit_naming_instantiation_path_status"], "path_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_denied_when_anti_shortcut_boundary_breaks(self) -> None:
        status = gate11i.build_status_payload(
            make_gate11h_manifest(),
            make_gate11h_status(anti_shortcut_boundary_status="not_confirmed"),
            "Gate11H source breaks the anti-shortcut boundary.",
        )

        self.assertEqual(status["anti_shortcut_boundary_status"], "denied")
        self.assertEqual(status["minimal_same_source_later_source_instantiation_rule_status"], "not_defined")
        self.assertEqual(status["later_source_explicit_naming_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "anti_shortcut_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11i.build_status_payload(
            make_gate11h_manifest(run_id=""),
            make_gate11h_status(),
            "",
        )

        self.assertEqual(status["missing_naming_component_naming_status"], "deferred")
        self.assertEqual(status["minimal_same_source_later_source_instantiation_rule_status"], "deferred")
        self.assertEqual(status["anti_shortcut_boundary_status"], "deferred")
        self.assertEqual(status["later_source_explicit_naming_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_not_yet_defined_when_missing_components_are_not_explicitly_named(self) -> None:
        status = gate11i.build_status_payload(
            make_gate11h_manifest(),
            make_gate11h_status(explicit_later_source_marker_status="present"),
            "Gate11H source does not cleanly name the missing naming components.",
        )

        self.assertEqual(status["missing_naming_component_naming_status"], "not_named")
        self.assertEqual(status["later_source_explicit_naming_instantiation_path_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "missing_naming_components_not_explicitly_named")

    def test_denied_when_source_is_already_named(self) -> None:
        status = gate11i.build_status_payload(
            make_gate11h_manifest(),
            make_gate11h_status(one_later_source_explicit_naming_status="named"),
            "Gate11H source has already moved past not_yet_named.",
        )

        self.assertEqual(status["gate11h_not_yet_named_state_preservation_status"], "not_preserved")
        self.assertEqual(status["later_source_explicit_naming_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11h_not_yet_named_state_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11i.build_status_payload(
            make_gate11h_manifest(),
            make_gate11h_status(),
            "Gate11H preserves a not-yet-named source with explicit missing naming components.",
        )
        registry = gate11i.build_registry(make_gate11h_manifest(), make_gate11h_status(), status)
        compare = gate11i.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11h_run_id"], "gate11h_run")
        self.assertEqual(compare[0]["later_source_explicit_naming_instantiation_path_status"], "path_defined")


if __name__ == "__main__":
    unittest.main()