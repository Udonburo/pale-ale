#!/usr/bin/env python3
"""Regression tests for Gate11C bounded-line insufficiency declaration-surface helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11c_bounded_line_insufficiency_declaration_surface_audit as gate11c


def make_gate11b_manifest(run_id: str = "gate11b_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11b_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    bounded_line_insufficiency_candidate_status: str = "absent",
    bounded_line_insufficiency_class_status: str = "none",
    settlement_inflation_pressure_status: str = "absent",
    graph_wide_operator_leap_pressure_status: str = "absent",
    bounded_line_insufficiency_declarability_status: str = "not_yet_declarable",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "bounded_line_insufficiency_candidate_status": bounded_line_insufficiency_candidate_status,
        "bounded_line_insufficiency_class_status": bounded_line_insufficiency_class_status,
        "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
        "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        "bounded_line_insufficiency_declarability_status": bounded_line_insufficiency_declarability_status,
    }


class RunGate11CBoundedLineInsufficiencyDeclarationSurfaceAuditTest(unittest.TestCase):
    def test_surface_defined_when_absence_state_is_preserved(self) -> None:
        status = gate11c.build_status_payload(
            make_gate11b_manifest(),
            make_gate11b_status(),
            "Gate11B preserves absence and defines no declaration markers in this run.",
        )

        self.assertEqual(status["gate10_closeout_preservation_status"], "preserved")
        self.assertEqual(status["gate11a_absence_result_preservation_status"], "preserved")
        self.assertEqual(status["gate11b_no_candidate_state_preservation_status"], "preserved")
        self.assertEqual(status["explicit_marker_shape_status"], "defined")
        self.assertEqual(status["single_candidate_singularity_status"], "defined")
        self.assertEqual(status["bounded_line_insufficiency_evidence_shape_status"], "defined")
        self.assertEqual(status["anti_inflation_boundary_status"], "defined")
        self.assertEqual(status["bounded_line_insufficiency_declaration_surface_status"], "surface_defined")
        self.assertEqual(status["next_named_blocker"], "")

    def test_surface_denied_when_inflation_boundary_is_broken(self) -> None:
        status = gate11c.build_status_payload(
            make_gate11b_manifest(),
            make_gate11b_status(settlement_inflation_pressure_status="present"),
            "Gate11B source is preserved but inflation pressure is explicit.",
        )

        self.assertEqual(status["anti_inflation_boundary_status"], "denied")
        self.assertEqual(status["bounded_line_insufficiency_declaration_surface_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "settlement_inflation_pressure")

    def test_incomplete_source_defers_audit(self) -> None:
        status = gate11c.build_status_payload(
            make_gate11b_manifest(run_id=""),
            make_gate11b_status(),
            "",
        )

        self.assertEqual(status["explicit_marker_shape_status"], "deferred")
        self.assertEqual(status["bounded_line_insufficiency_declaration_surface_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_explicit_declaration_markers_in_source_defer_worker_resolution(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate_declaration_status = declared",
                "bounded_line_insufficiency_candidate_id = candidate_alpha",
                "bounded_line_insufficiency_class_status = current_bounded_line_insufficiency",
                "bounded_line_host_failure_status = explicit",
                "one bounded-line insufficiency candidate is explicitly declared: candidate_alpha",
            ]
        )
        status = gate11c.build_status_payload(
            make_gate11b_manifest(), make_gate11b_status(), report_text
        )

        self.assertEqual(status["explicit_marker_shape_status"], "deferred")
        self.assertEqual(status["single_candidate_singularity_status"], "deferred")
        self.assertEqual(status["bounded_line_insufficiency_declaration_surface_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "declaration_surface_requires_worker_resolution")

    def test_narrative_mentions_without_markers_still_allow_surface_definition(self) -> None:
        report_text = (
            "The narrative mentions a future candidate_id and declaration surface in prose, "
            "but no explicit declaration markers are present."
        )
        status = gate11c.build_status_payload(
            make_gate11b_manifest(), make_gate11b_status(), report_text
        )

        self.assertEqual(status["explicit_marker_shape_status"], "defined")
        self.assertEqual(status["bounded_line_insufficiency_declaration_surface_status"], "surface_defined")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11c.build_status_payload(
            make_gate11b_manifest(),
            make_gate11b_status(),
            "Gate11B preserves absence and defines no declaration markers in this run.",
        )
        registry = gate11c.build_registry(make_gate11b_manifest(), make_gate11b_status(), status)
        compare = gate11c.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11b_run_id"], "gate11b_run")
        self.assertEqual(compare[0]["bounded_line_insufficiency_declaration_surface_status"], "surface_defined")


if __name__ == "__main__":
    unittest.main()