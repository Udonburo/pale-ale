#!/usr/bin/env python3
"""Regression tests for Gate11K one admissible later-source explicit-presence helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11k_one_admissible_later_source_explicit_presence_audit as gate11k


def make_gate11j_manifest(run_id: str = "gate11j_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11j_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    gate11d_not_yet_declared_state_preservation_status: str = "preserved",
    gate11e_path_defined_state_preservation_status: str = "preserved",
    gate11f_not_yet_admissible_state_preservation_status: str = "preserved",
    gate11g_naming_surface_preservation_status: str = "preserved",
    gate11h_not_yet_named_state_preservation_status: str = "preserved",
    gate11i_path_defined_state_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    later_source_naming_status: str = "absent",
    later_source_cardinality_status: str = "none",
    same_source_path_attachment_status: str = "not_attached",
    anti_shortcut_boundary_status: str = "confirmed",
    later_source_naming_instantiation_admissibility_status: str = "not_yet_admissible",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "later_source_naming_status": later_source_naming_status,
        "later_source_cardinality_status": later_source_cardinality_status,
        "same_source_path_attachment_status": same_source_path_attachment_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_naming_instantiation_admissibility_status": later_source_naming_instantiation_admissibility_status,
    }


class RunGate11KOneAdmissibleLaterSourceExplicitPresenceAuditTest(unittest.TestCase):
    def test_not_yet_present_for_current_frozen_not_yet_admissible_source(self) -> None:
        status = gate11k.build_status_payload(
            make_gate11j_manifest(),
            make_gate11j_status(),
            "Gate11J preserves a not-yet-admissible source with no explicit later-source marker.",
        )

        self.assertEqual(status["gate11j_not_yet_admissible_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_later_source_marker_status"], "absent")
        self.assertEqual(status["later_source_singularity_status"], "none")
        self.assertEqual(status["same_source_path_attachment_status"], "not_attached")
        self.assertEqual(status["admissibility_boundary_status"], "confirmed")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_later_source_marker")

    def test_present_when_one_later_source_is_explicit_and_attached(self) -> None:
        report_text = """
one later source is explicitly named: runs/gate11l_future
- one explicit later_source_id or later_frozen_run_id
- one declaration marker
- one candidate id
- one class
- one explicit host-failure sentence
- matched status, registry, and read surfaces
"""
        status = gate11k.build_status_payload(
            make_gate11j_manifest(),
            make_gate11j_status(),
            report_text,
        )

        self.assertEqual(status["explicit_later_source_marker_status"], "present")
        self.assertEqual(status["later_source_singularity_status"], "single")
        self.assertEqual(status["same_source_path_attachment_status"], "attached")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_status"], "present")
        self.assertEqual(status["next_named_blocker"], "")

    def test_deferred_when_multiple_later_sources_compete(self) -> None:
        report_text = """
one later source is explicitly named: runs/gate11l_future_a
one later frozen run is explicitly named: runs/gate11l_future_b
"""
        status = gate11k.build_status_payload(
            make_gate11j_manifest(),
            make_gate11j_status(),
            report_text,
        )

        self.assertEqual(status["later_source_singularity_status"], "multiple")
        self.assertEqual(status["same_source_path_attachment_status"], "deferred")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_later_sources")

    def test_denied_when_admissibility_boundary_breaks(self) -> None:
        status = gate11k.build_status_payload(
            make_gate11j_manifest(),
            make_gate11j_status(anti_shortcut_boundary_status="not_confirmed"),
            "Gate11J source breaks the admissibility boundary.",
        )

        self.assertEqual(status["admissibility_boundary_status"], "denied")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "admissibility_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11k.build_status_payload(
            make_gate11j_manifest(run_id=""),
            make_gate11j_status(),
            "",
        )

        self.assertEqual(status["explicit_later_source_marker_status"], "deferred")
        self.assertEqual(status["later_source_singularity_status"], "deferred")
        self.assertEqual(status["same_source_path_attachment_status"], "deferred")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11k.build_status_payload(
            make_gate11j_manifest(),
            make_gate11j_status(),
            "Gate11J preserves a not-yet-admissible source with no explicit later-source marker.",
        )
        registry = gate11k.build_registry(make_gate11j_manifest(), make_gate11j_status(), status)
        compare = gate11k.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11j_run_id"], "gate11j_run")
        self.assertEqual(compare[0]["one_admissible_later_source_explicit_presence_status"], "not_yet_present")


if __name__ == "__main__":
    unittest.main()