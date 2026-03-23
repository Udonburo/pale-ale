#!/usr/bin/env python3
"""Regression tests for Gate11E explicit-declaration instantiation-path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11e_explicit_declaration_instantiation_path_audit as gate11e


def make_gate11d_manifest(run_id: str = "gate11d_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11d_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    anti_inflation_boundary_status: str = "confirmed",
    bounded_line_insufficiency_explicit_declaration_marker_status: str = "absent",
    bounded_line_insufficiency_candidate_id_singularity_status: str = "absent",
    bounded_line_insufficiency_class_singularity_status: str = "none",
    bounded_line_host_failure_statement_status: str = "absent",
    one_bounded_line_insufficiency_explicit_declaration_status: str = "not_yet_declared",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "anti_inflation_boundary_status": anti_inflation_boundary_status,
        "bounded_line_insufficiency_explicit_declaration_marker_status": bounded_line_insufficiency_explicit_declaration_marker_status,
        "bounded_line_insufficiency_candidate_id_singularity_status": bounded_line_insufficiency_candidate_id_singularity_status,
        "bounded_line_insufficiency_class_singularity_status": bounded_line_insufficiency_class_singularity_status,
        "bounded_line_host_failure_statement_status": bounded_line_host_failure_statement_status,
        "one_bounded_line_insufficiency_explicit_declaration_status": one_bounded_line_insufficiency_explicit_declaration_status,
    }


class RunGate11EExplicitDeclarationInstantiationPathAuditTest(unittest.TestCase):
    def test_path_defined_for_current_frozen_not_yet_declared_source(self) -> None:
        status = gate11e.build_status_payload(
            make_gate11d_manifest(),
            make_gate11d_status(),
            "Gate11D preserves a not-yet-declared source with explicit missing components.",
        )

        self.assertEqual(status["gate11d_not_yet_declared_state_preservation_status"], "preserved")
        self.assertEqual(status["missing_surface_component_naming_status"], "named")
        self.assertEqual(status["minimal_later_source_instantiation_rule_status"], "defined")
        self.assertEqual(status["anti_shortcut_boundary_status"], "confirmed")
        self.assertEqual(status["explicit_declaration_instantiation_path_status"], "path_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_denied_when_anti_shortcut_boundary_breaks(self) -> None:
        status = gate11e.build_status_payload(
            make_gate11d_manifest(),
            make_gate11d_status(anti_inflation_boundary_status="not_confirmed"),
            "Gate11D source breaks the anti-shortcut boundary.",
        )

        self.assertEqual(status["anti_shortcut_boundary_status"], "denied")
        self.assertEqual(status["minimal_later_source_instantiation_rule_status"], "denied")
        self.assertEqual(status["explicit_declaration_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "anti_shortcut_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11e.build_status_payload(
            make_gate11d_manifest(run_id=""),
            make_gate11d_status(),
            "",
        )

        self.assertEqual(status["missing_surface_component_naming_status"], "deferred")
        self.assertEqual(status["minimal_later_source_instantiation_rule_status"], "deferred")
        self.assertEqual(status["anti_shortcut_boundary_status"], "deferred")
        self.assertEqual(status["explicit_declaration_instantiation_path_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_not_yet_defined_when_missing_components_are_not_explicitly_named(self) -> None:
        status = gate11e.build_status_payload(
            make_gate11d_manifest(),
            make_gate11d_status(
                bounded_line_insufficiency_explicit_declaration_marker_status="present",
                bounded_line_insufficiency_candidate_id_singularity_status="absent",
                bounded_line_insufficiency_class_singularity_status="none",
                bounded_line_host_failure_statement_status="absent",
            ),
            "Gate11D source does not cleanly name the current missing components.",
        )

        self.assertEqual(status["missing_surface_component_naming_status"], "not_yet_named")
        self.assertEqual(status["explicit_declaration_instantiation_path_status"], "not_yet_defined")
        self.assertEqual(status["next_named_blocker"], "missing_components_not_explicitly_named")

    def test_denied_when_source_is_already_declared(self) -> None:
        status = gate11e.build_status_payload(
            make_gate11d_manifest(),
            make_gate11d_status(one_bounded_line_insufficiency_explicit_declaration_status="declared"),
            "Gate11D source has already moved past not_yet_declared.",
        )

        self.assertEqual(status["gate11d_not_yet_declared_state_preservation_status"], "not_preserved")
        self.assertEqual(status["explicit_declaration_instantiation_path_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11d_not_yet_declared_state_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11e.build_status_payload(
            make_gate11d_manifest(),
            make_gate11d_status(),
            "Gate11D preserves a not-yet-declared source with explicit missing components.",
        )
        registry = gate11e.build_registry(make_gate11d_manifest(), make_gate11d_status(), status)
        compare = gate11e.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11d_run_id"], "gate11d_run")
        self.assertEqual(compare[0]["explicit_declaration_instantiation_path_status"], "path_defined")


if __name__ == "__main__":
    unittest.main()