#!/usr/bin/env python3
"""Regression tests for Gate11H one later-source explicit-naming helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11h_one_later_source_explicit_naming_audit as gate11h


def make_gate11g_manifest(run_id: str = "gate11g_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11g_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    gate11d_not_yet_declared_state_preservation_status: str = "preserved",
    gate11e_path_defined_state_preservation_status: str = "preserved",
    gate11f_not_yet_admissible_state_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_later_source_marker_shape_status: str = "defined",
    single_later_source_singularity_status: str = "defined",
    full_path_attachment_shape_status: str = "defined",
    anti_shortcut_boundary_status: str = "confirmed",
    later_source_naming_surface_status: str = "surface_defined",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "gate11f_not_yet_admissible_state_preservation_status": gate11f_not_yet_admissible_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_later_source_marker_shape_status": explicit_later_source_marker_shape_status,
        "single_later_source_singularity_status": single_later_source_singularity_status,
        "full_path_attachment_shape_status": full_path_attachment_shape_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_naming_surface_status": later_source_naming_surface_status,
    }


class RunGate11HOneLaterSourceExplicitNamingAuditTest(unittest.TestCase):
    def test_not_yet_named_when_no_explicit_later_source_marker_exists(self) -> None:
        status = gate11h.build_status_payload(
            make_gate11g_manifest(),
            make_gate11g_status(),
            "Gate11G fixes naming surface but names no later source instance.",
        )

        self.assertEqual(status["gate11g_naming_surface_preservation_status"], "preserved")
        self.assertEqual(status["explicit_later_source_marker_status"], "absent")
        self.assertEqual(status["later_source_singularity_status"], "none")
        self.assertEqual(status["full_path_attachment_status"], "not_attached")
        self.assertEqual(status["one_later_source_explicit_naming_status"], "not_yet_named")
        self.assertEqual(status["next_named_blocker"], "no_explicit_later_source_marker")

    def test_named_when_one_explicit_later_source_and_full_path_are_present(self) -> None:
        report_text = "\n".join(
            [
                "later_source_id = gate11i_future_run",
                "one declaration marker",
                "one candidate id",
                "one class",
                "one explicit host-failure sentence",
                "matched status, registry, and read surfaces",
            ]
        )
        status = gate11h.build_status_payload(
            make_gate11g_manifest(), make_gate11g_status(), report_text
        )

        self.assertEqual(status["explicit_later_source_marker_status"], "present")
        self.assertEqual(status["later_source_singularity_status"], "single")
        self.assertEqual(status["full_path_attachment_status"], "attached")
        self.assertEqual(status["one_later_source_explicit_naming_status"], "named")
        self.assertEqual(status["next_named_blocker"], "")

    def test_multiple_later_sources_defer(self) -> None:
        report_text = "\n".join(
            [
                "later_source_id = gate11i_future_run",
                "later_source_id = gate11j_future_run",
                "one declaration marker",
                "one candidate id",
                "one class",
                "one explicit host-failure sentence",
                "matched status, registry, and read surfaces",
            ]
        )
        status = gate11h.build_status_payload(
            make_gate11g_manifest(), make_gate11g_status(), report_text
        )

        self.assertEqual(status["later_source_singularity_status"], "multiple")
        self.assertEqual(status["full_path_attachment_status"], "deferred")
        self.assertEqual(status["one_later_source_explicit_naming_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_later_sources")

    def test_denied_when_anti_shortcut_boundary_breaks(self) -> None:
        status = gate11h.build_status_payload(
            make_gate11g_manifest(),
            make_gate11g_status(anti_shortcut_boundary_status="denied"),
            "later_source_id = gate11i_future_run",
        )

        self.assertEqual(status["anti_shortcut_boundary_status"], "denied")
        self.assertEqual(status["one_later_source_explicit_naming_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "anti_shortcut_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11h.build_status_payload(
            make_gate11g_manifest(run_id=""),
            make_gate11g_status(),
            "",
        )

        self.assertEqual(status["explicit_later_source_marker_status"], "deferred")
        self.assertEqual(status["later_source_singularity_status"], "deferred")
        self.assertEqual(status["full_path_attachment_status"], "deferred")
        self.assertEqual(status["one_later_source_explicit_naming_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11h.build_status_payload(
            make_gate11g_manifest(),
            make_gate11g_status(),
            "Gate11G fixes naming surface but names no later source instance.",
        )
        registry = gate11h.build_registry(make_gate11g_manifest(), make_gate11g_status(), status)
        compare = gate11h.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11g_run_id"], "gate11g_run")
        self.assertEqual(compare[0]["one_later_source_explicit_naming_status"], "not_yet_named")


if __name__ == "__main__":
    unittest.main()