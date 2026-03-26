#!/usr/bin/env python3
"""Regression tests for Gate11M one admissible later-source explicit-presence path-instantiation helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11m_one_admissible_later_source_explicit_presence_path_instantiation_audit as gate11m


def make_gate11l_manifest(run_id: str = "gate11l_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11l_status(
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
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    missing_explicit_presence_component_naming_status: str = "named",
    minimal_same_source_admissible_presence_instantiation_rule_status: str = "defined",
    admissibility_boundary_status: str = "confirmed",
    admissible_later_source_explicit_presence_instantiation_path_status: str = "path_defined",
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "missing_explicit_presence_component_naming_status": missing_explicit_presence_component_naming_status,
        "minimal_same_source_admissible_presence_instantiation_rule_status": minimal_same_source_admissible_presence_instantiation_rule_status,
        "admissibility_boundary_status": admissibility_boundary_status,
        "admissible_later_source_explicit_presence_instantiation_path_status": admissible_later_source_explicit_presence_instantiation_path_status,
    }


class RunGate11MOneAdmissibleLaterSourceExplicitPresencePathInstantiationAuditTest(unittest.TestCase):
    def test_not_yet_present_for_current_frozen_path_defined_source(self) -> None:
        status = gate11m.build_status_payload(
            make_gate11l_manifest(),
            make_gate11l_status(),
            "Gate11L preserves a path-defined source but does not name any admissible later source.",
        )

        self.assertEqual(status["gate11l_path_defined_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_admissible_later_source_presence_marker_status"], "absent")
        self.assertEqual(status["later_source_singularity_status"], "none")
        self.assertEqual(status["same_source_fixed_gate11l_path_instantiation_status"], "not_instantiated")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_path_instantiation_status"], "not_yet_present")
        self.assertEqual(status["next_named_blocker"], "no_explicit_admissible_later_source_presence_marker")

    def test_present_when_one_later_source_is_named_and_instantiated(self) -> None:
        report_text = """
one later source is explicitly named: runs/gate11n_future
- one explicit later_source_id or later_frozen_run_id
- one later source and only one later source
- one declaration marker
- one candidate id
- one class
- one explicit host-failure sentence
- matched status, registry, and read surfaces
"""
        status = gate11m.build_status_payload(make_gate11l_manifest(), make_gate11l_status(), report_text)

        self.assertEqual(status["explicit_admissible_later_source_presence_marker_status"], "present")
        self.assertEqual(status["later_source_singularity_status"], "single")
        self.assertEqual(status["same_source_fixed_gate11l_path_instantiation_status"], "instantiated")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_path_instantiation_status"], "present")
        self.assertEqual(status["next_named_blocker"], "")

    def test_deferred_when_multiple_later_sources_compete(self) -> None:
        report_text = """
one later source is explicitly named: runs/gate11n_future_a
one later frozen run is explicitly named: runs/gate11n_future_b
"""
        status = gate11m.build_status_payload(make_gate11l_manifest(), make_gate11l_status(), report_text)

        self.assertEqual(status["later_source_singularity_status"], "multiple")
        self.assertEqual(status["same_source_fixed_gate11l_path_instantiation_status"], "deferred")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_path_instantiation_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_later_sources")

    def test_denied_when_admissibility_boundary_breaks(self) -> None:
        status = gate11m.build_status_payload(
            make_gate11l_manifest(),
            make_gate11l_status(admissibility_boundary_status="not_confirmed"),
            "Gate11L source breaks the admissibility boundary.",
        )

        self.assertEqual(status["admissibility_boundary_status"], "denied")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_path_instantiation_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "admissibility_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11m.build_status_payload(make_gate11l_manifest(run_id=""), make_gate11l_status(), "")

        self.assertEqual(status["explicit_admissible_later_source_presence_marker_status"], "deferred")
        self.assertEqual(status["later_source_singularity_status"], "deferred")
        self.assertEqual(status["same_source_fixed_gate11l_path_instantiation_status"], "deferred")
        self.assertEqual(status["one_admissible_later_source_explicit_presence_path_instantiation_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11m.build_status_payload(
            make_gate11l_manifest(),
            make_gate11l_status(),
            "Gate11L preserves a path-defined source but does not name any admissible later source.",
        )
        registry = gate11m.build_registry(make_gate11l_manifest(), make_gate11l_status(), status)
        compare = gate11m.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11l_run_id"], "gate11l_run")
        self.assertEqual(compare[0]["one_admissible_later_source_explicit_presence_path_instantiation_status"], "not_yet_present")


if __name__ == "__main__":
    unittest.main()