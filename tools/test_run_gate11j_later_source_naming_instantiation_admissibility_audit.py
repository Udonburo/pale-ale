#!/usr/bin/env python3
"""Regression tests for Gate11J later-source naming-instantiation admissibility helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11j_later_source_naming_instantiation_admissibility_audit as gate11j


def make_gate11i_manifest(run_id: str = "gate11i_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11i_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    gate11d_not_yet_declared_state_preservation_status: str = "preserved",
    gate11e_path_defined_state_preservation_status: str = "preserved",
    gate11f_not_yet_admissible_state_preservation_status: str = "preserved",
    gate11g_naming_surface_preservation_status: str = "preserved",
    gate11h_not_yet_named_state_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    missing_naming_component_naming_status: str = "named",
    minimal_same_source_later_source_instantiation_rule_status: str = "defined",
    anti_shortcut_boundary_status: str = "confirmed",
    later_source_explicit_naming_instantiation_path_status: str = "path_defined",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "missing_naming_component_naming_status": missing_naming_component_naming_status,
        "minimal_same_source_later_source_instantiation_rule_status": minimal_same_source_later_source_instantiation_rule_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_explicit_naming_instantiation_path_status": later_source_explicit_naming_instantiation_path_status,
    }


class RunGate11JLaterSourceNamingInstantiationAdmissibilityAuditTest(unittest.TestCase):
    def test_not_yet_admissible_for_current_frozen_path_defined_source(self) -> None:
        status = gate11j.build_status_payload(
            make_gate11i_manifest(),
            make_gate11i_status(),
            "Gate11I preserves a path-defined source but does not name any later source.",
        )

        self.assertEqual(status["gate11i_path_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["later_source_naming_status"], "absent")
        self.assertEqual(status["later_source_cardinality_status"], "none")
        self.assertEqual(status["same_source_path_attachment_status"], "not_attached")
        self.assertEqual(status["later_source_naming_instantiation_admissibility_status"], "not_yet_admissible")
        self.assertEqual(status["next_named_blocker"], "no_later_source_named")

    def test_instantiation_admissible_when_one_later_source_is_named_and_attached(self) -> None:
        report_text = """
one later source is explicitly named: runs/gate11k_future
- one explicit later_source_id or later_frozen_run_id
- one later source and only one later source
- one declaration marker
- one candidate id
- one class
- one explicit host-failure sentence
- matched status, registry, and read surfaces
"""
        status = gate11j.build_status_payload(
            make_gate11i_manifest(),
            make_gate11i_status(),
            report_text,
        )

        self.assertEqual(status["later_source_naming_status"], "present")
        self.assertEqual(status["later_source_cardinality_status"], "single")
        self.assertEqual(status["same_source_path_attachment_status"], "attached")
        self.assertEqual(status["later_source_naming_instantiation_admissibility_status"], "instantiation_admissible")
        self.assertEqual(status["next_named_blocker"], "")

    def test_deferred_when_multiple_later_sources_compete(self) -> None:
        report_text = """
one later source is explicitly named: runs/gate11k_future_a
one later frozen run is explicitly named: runs/gate11k_future_b
"""
        status = gate11j.build_status_payload(
            make_gate11i_manifest(),
            make_gate11i_status(),
            report_text,
        )

        self.assertEqual(status["later_source_cardinality_status"], "multiple")
        self.assertEqual(status["same_source_path_attachment_status"], "deferred")
        self.assertEqual(status["later_source_naming_instantiation_admissibility_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_later_sources")

    def test_denied_when_anti_shortcut_boundary_breaks(self) -> None:
        status = gate11j.build_status_payload(
            make_gate11i_manifest(),
            make_gate11i_status(anti_shortcut_boundary_status="not_confirmed"),
            "Gate11I source breaks the anti-shortcut boundary.",
        )

        self.assertEqual(status["anti_shortcut_boundary_status"], "denied")
        self.assertEqual(status["later_source_naming_instantiation_admissibility_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "anti_shortcut_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11j.build_status_payload(
            make_gate11i_manifest(run_id=""),
            make_gate11i_status(),
            "",
        )

        self.assertEqual(status["later_source_naming_status"], "deferred")
        self.assertEqual(status["later_source_cardinality_status"], "deferred")
        self.assertEqual(status["same_source_path_attachment_status"], "deferred")
        self.assertEqual(status["later_source_naming_instantiation_admissibility_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11j.build_status_payload(
            make_gate11i_manifest(),
            make_gate11i_status(),
            "Gate11I preserves a path-defined source but does not name any later source.",
        )
        registry = gate11j.build_registry(make_gate11i_manifest(), make_gate11i_status(), status)
        compare = gate11j.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11i_run_id"], "gate11i_run")
        self.assertEqual(compare[0]["later_source_naming_instantiation_admissibility_status"], "not_yet_admissible")


if __name__ == "__main__":
    unittest.main()