#!/usr/bin/env python3
"""Regression tests for Gate11G later-source naming surface helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11g_later_source_naming_surface_audit as gate11g


def make_gate11f_manifest(run_id: str = "gate11f_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11f_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    gate11c_declaration_surface_preservation_status: str = "preserved",
    gate11d_not_yet_declared_state_preservation_status: str = "preserved",
    gate11e_path_defined_state_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    later_source_naming_status: str = "absent",
    later_source_cardinality_status: str = "none",
    same_source_path_attachment_status: str = "not_attached",
    anti_shortcut_boundary_status: str = "confirmed",
    later_source_instantiation_admissibility_status: str = "not_yet_admissible",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "later_source_naming_status": later_source_naming_status,
        "later_source_cardinality_status": later_source_cardinality_status,
        "same_source_path_attachment_status": same_source_path_attachment_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_instantiation_admissibility_status": later_source_instantiation_admissibility_status,
    }


class RunGate11GLaterSourceNamingSurfaceAuditTest(unittest.TestCase):
    def test_surface_defined_for_current_frozen_not_yet_admissible_source(self) -> None:
        status = gate11g.build_status_payload(
            make_gate11f_manifest(),
            make_gate11f_status(),
            "Gate11F preserves no-later-source state while the naming surface remains to be fixed.",
        )

        self.assertEqual(status["gate11f_not_yet_admissible_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_later_source_marker_shape_status"], "defined")
        self.assertEqual(status["single_later_source_singularity_status"], "defined")
        self.assertEqual(status["full_path_attachment_shape_status"], "defined")
        self.assertEqual(status["anti_shortcut_boundary_status"], "confirmed")
        self.assertEqual(status["later_source_naming_surface_status"], "surface_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_denied_when_anti_shortcut_boundary_breaks(self) -> None:
        status = gate11g.build_status_payload(
            make_gate11f_manifest(),
            make_gate11f_status(anti_shortcut_boundary_status="denied"),
            "Gate11F source breaks anti-shortcut boundary.",
        )

        self.assertEqual(status["anti_shortcut_boundary_status"], "denied")
        self.assertEqual(status["later_source_naming_surface_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "anti_shortcut_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11g.build_status_payload(
            make_gate11f_manifest(run_id=""),
            make_gate11f_status(),
            "",
        )

        self.assertEqual(status["explicit_later_source_marker_shape_status"], "deferred")
        self.assertEqual(status["single_later_source_singularity_status"], "deferred")
        self.assertEqual(status["full_path_attachment_shape_status"], "deferred")
        self.assertEqual(status["anti_shortcut_boundary_status"], "deferred")
        self.assertEqual(status["later_source_naming_surface_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_denied_when_gate11f_state_is_not_preserved(self) -> None:
        status = gate11g.build_status_payload(
            make_gate11f_manifest(),
            make_gate11f_status(later_source_instantiation_admissibility_status="instantiation_admissible"),
            "Gate11F source has moved past not_yet_admissible.",
        )

        self.assertEqual(status["gate11f_not_yet_admissible_state_preservation_status"], "not_preserved")
        self.assertEqual(status["later_source_naming_surface_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "gate11f_not_yet_admissible_state_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11g.build_status_payload(
            make_gate11f_manifest(),
            make_gate11f_status(),
            "Gate11F preserves no-later-source state while the naming surface remains to be fixed.",
        )
        registry = gate11g.build_registry(make_gate11f_manifest(), make_gate11f_status(), status)
        compare = gate11g.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11f_run_id"], "gate11f_run")
        self.assertEqual(compare[0]["later_source_naming_surface_status"], "surface_defined")


if __name__ == "__main__":
    unittest.main()